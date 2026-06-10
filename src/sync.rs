use crate::api::{
    ManifestLoad, ScanHook, ScanOptions, SyncIndexResult, SyncProgress, SyncRequest, SyncStats,
};
use crate::constants::{
    EMBEDDING_BATCH_DOC_LIMIT, EMBEDDING_JOB_CHANNEL_CAPACITY, INDEX_BATCH_BYTE_LIMIT,
    INDEX_BATCH_DELETE_LIMIT, INDEX_BATCH_DOC_LIMIT, INDEX_EVENT_CHANNEL_CAPACITY,
    MANIFEST_BATCH_ENTRY_LIMIT, PROGRESS_POLL_INTERVAL,
};
use crate::error::{SearchxError, SearchxResult};
use crate::index::{
    apply_index_batch, configure_index, document_id_for_path, embedded_document_vectors,
    new_heed_options,
};
use crate::manifest::{
    Manifest, ManifestWorkingSet, clear_index_incomplete, commit_working_manifest, data_paths,
    load_manifest, load_working_manifest, mark_index_incomplete, open_manifest_connection,
    reset_data_dir,
};
use crate::scan::{
    EmbeddingJob, EmbeddingJobInput, IndexEvent, ProgressUpdate, ScanPipeline, scan_root,
};
use milli::update::IndexerConfig;
use milli::{Index, all_obkv_to_json};
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

#[derive(Default)]
struct PendingIndexBatch {
    upserts: BTreeMap<String, String>,
    deleted_ids: BTreeSet<String>,
    progress_updates: Vec<ProgressUpdate>,
    bytes: usize,
}

impl PendingIndexBatch {
    fn push(&mut self, event: IndexEvent) {
        match event {
            IndexEvent::Upsert {
                document_id,
                document,
                progress,
            } => {
                let document_len = document.len();
                self.deleted_ids.remove(&document_id);
                if let Some(previous) = self.upserts.insert(document_id, document) {
                    self.bytes = self.bytes.saturating_sub(previous.len());
                }
                self.bytes += document_len;
                if let Some(progress) = progress {
                    self.progress_updates.push(progress);
                }
            }
            IndexEvent::Delete {
                document_id,
                progress,
            } => {
                if let Some(previous) = self.upserts.remove(&document_id) {
                    self.bytes = self.bytes.saturating_sub(previous.len());
                }
                self.deleted_ids.insert(document_id);
                if let Some(progress) = progress {
                    self.progress_updates.push(progress);
                }
            }
            IndexEvent::Progress(progress) => self.progress_updates.push(progress),
        }
    }

    fn should_flush(&self) -> bool {
        self.upserts.len() >= INDEX_BATCH_DOC_LIMIT
            || self.deleted_ids.len() >= INDEX_BATCH_DELETE_LIMIT
            || self.bytes >= INDEX_BATCH_BYTE_LIMIT
            || self.progress_updates.len() >= MANIFEST_BATCH_ENTRY_LIMIT
    }

    fn flush(
        &mut self,
        index: &Index,
        indexer_config: &IndexerConfig,
        progress_manifest: Option<&mut ManifestWorkingSet>,
        data_paths: &crate::api::DataPaths,
    ) -> SearchxResult<()> {
        if self.upserts.is_empty()
            && self.deleted_ids.is_empty()
            && self.progress_updates.is_empty()
        {
            return Ok(());
        }

        if !self.upserts.is_empty() || !self.deleted_ids.is_empty() {
            let upserts = self
                .upserts
                .values()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let deleted_ids = self
                .deleted_ids
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            apply_index_batch(
                index,
                indexer_config,
                &data_paths.base,
                &upserts,
                &deleted_ids,
            )?;
        }

        if let Some(progress_manifest) = progress_manifest {
            progress_manifest.update_entries(&self.progress_updates)?;
        }

        self.upserts.clear();
        self.deleted_ids.clear();
        self.progress_updates.clear();
        self.bytes = 0;
        Ok(())
    }
}

struct StreamScanJob<'a> {
    index: &'a Index,
    indexer_config: &'a IndexerConfig,
    options: &'a ScanOptions,
    root: &'a Path,
    data_paths: &'a crate::api::DataPaths,
    previous_manifest: Manifest,
    resume_existing_progress: bool,
}

fn panic_message(payload: &(dyn Any + Send + 'static)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn drain_scan_errors<F>(error_rx: &mpsc::Receiver<crate::api::ScanError>, on_progress: &mut F)
where
    F: FnMut(SyncProgress),
{
    while let Ok(error) = error_rx.try_recv() {
        on_progress(SyncProgress::ScanError(error));
    }
}

fn report_sync_progress<F>(
    scan_hook: &ScanHook,
    last_reported_file: &mut Option<String>,
    on_progress: &mut F,
) where
    F: FnMut(SyncProgress),
{
    let current_file = scan_hook.current_file();
    if current_file != *last_reported_file {
        if let Some(path) = current_file.as_deref() {
            on_progress(SyncProgress::Indexing {
                path: path.to_string(),
            });
        }
        *last_reported_file = current_file;
    }
}

fn finish_scan_thread(
    scan_handle: thread::JoinHandle<SearchxResult<SyncStats>>,
) -> SearchxResult<SyncStats> {
    match scan_handle.join() {
        Ok(result) => result,
        Err(payload) => Err(SearchxError::ScanThreadPanicked {
            message: panic_message(payload.as_ref()),
        }),
    }
}

fn finish_embedding_thread(
    embedding_handle: thread::JoinHandle<SearchxResult<()>>,
) -> SearchxResult<()> {
    match embedding_handle.join() {
        Ok(result) => result,
        Err(payload) => Err(SearchxError::EmbeddingThreadPanicked {
            message: panic_message(payload.as_ref()),
        }),
    }
}

fn emit_embedded_document(
    event_tx: &mpsc::SyncSender<IndexEvent>,
    job: EmbeddingJob,
    vector: Option<Vec<f32>>,
) -> SearchxResult<()> {
    let mut document = job.document;
    let document_id = document.id.clone();
    if let Some(vector) = vector {
        document.vectors = embedded_document_vectors(vector);
    }
    event_tx
        .send(IndexEvent::Upsert {
            document_id,
            document: serde_json::to_string(&document)?,
            progress: Some(job.progress),
        })
        .map_err(|_| SearchxError::IndexingPipelineDisconnected)
}

fn embed_single_text(
    embedder: &mut crate::embedding::Embedder,
    text: &str,
) -> SearchxResult<Vec<f32>> {
    embedder
        .embed_texts(&[text])?
        .into_iter()
        .next()
        .ok_or_else(|| SearchxError::Embedding {
            message: "text embedder returned no vectors".to_string(),
        })
}

fn embed_single_image(
    embedder: &mut crate::embedding::Embedder,
    bytes: &[u8],
) -> SearchxResult<Vec<f32>> {
    embedder
        .embed_images(&[bytes])?
        .into_iter()
        .next()
        .ok_or_else(|| SearchxError::Embedding {
            message: "image embedder returned no vectors".to_string(),
        })
}

fn embed_job(
    embedder: &mut crate::embedding::Embedder,
    job: &EmbeddingJob,
) -> SearchxResult<Vec<f32>> {
    match &job.input {
        EmbeddingJobInput::DocumentContents => embed_single_text(embedder, &job.document.contents),
        EmbeddingJobInput::Text(text) => embed_single_text(embedder, text),
        EmbeddingJobInput::Image(bytes) => embed_single_image(embedder, bytes),
    }
}

fn embed_text_jobs(
    embedder: &mut crate::embedding::Embedder,
    pending_jobs: &[EmbeddingJob],
    vectors: &mut [Option<Vec<f32>>],
) {
    let mut text_indices = Vec::new();
    let mut text_inputs = Vec::new();

    for (index, job) in pending_jobs.iter().enumerate() {
        match &job.input {
            EmbeddingJobInput::DocumentContents => {
                text_indices.push(index);
                text_inputs.push(job.document.contents.as_str());
            }
            EmbeddingJobInput::Text(text) => {
                text_indices.push(index);
                text_inputs.push(text.as_str());
            }
            EmbeddingJobInput::Image(_) => {}
        }
    }

    if text_inputs.is_empty() {
        return;
    }

    match embedder.embed_texts(&text_inputs) {
        Ok(batch_vectors) => {
            for (index, vector) in text_indices.into_iter().zip(batch_vectors) {
                vectors[index] = Some(vector);
            }
        }
        Err(error) => {
            eprintln!(
                "embedding batch error for {} text documents: {}",
                text_inputs.len(),
                error
            );
            for index in text_indices {
                match embed_job(embedder, &pending_jobs[index]) {
                    Ok(vector) => vectors[index] = Some(vector),
                    Err(error) => {
                        eprintln!(
                            "embedding error for {}: {}",
                            pending_jobs[index].progress.path, error
                        );
                    }
                }
            }
        }
    }
}

fn embed_image_jobs(
    embedder: &mut crate::embedding::Embedder,
    pending_jobs: &[EmbeddingJob],
    vectors: &mut [Option<Vec<f32>>],
) {
    let mut image_indices = Vec::new();
    let mut image_inputs = Vec::new();

    for (index, job) in pending_jobs.iter().enumerate() {
        if let EmbeddingJobInput::Image(bytes) = &job.input {
            image_indices.push(index);
            image_inputs.push(bytes.as_slice());
        }
    }

    if image_inputs.is_empty() {
        return;
    }

    match embedder.embed_images(&image_inputs) {
        Ok(batch_vectors) => {
            for (index, vector) in image_indices.into_iter().zip(batch_vectors) {
                vectors[index] = Some(vector);
            }
        }
        Err(error) => {
            eprintln!(
                "embedding batch error for {} image documents: {}",
                image_inputs.len(),
                error
            );
            for index in image_indices {
                match embed_job(embedder, &pending_jobs[index]) {
                    Ok(vector) => vectors[index] = Some(vector),
                    Err(error) => {
                        eprintln!(
                            "embedding error for {}: {}",
                            pending_jobs[index].progress.path, error
                        );
                    }
                }
            }
        }
    }
}

fn flush_embedding_jobs(
    embedder: &mut crate::embedding::Embedder,
    pending_jobs: &mut Vec<EmbeddingJob>,
    event_tx: &mpsc::SyncSender<IndexEvent>,
) -> SearchxResult<()> {
    if pending_jobs.is_empty() {
        return Ok(());
    }

    let mut vectors = vec![None; pending_jobs.len()];
    embed_text_jobs(embedder, pending_jobs, &mut vectors);
    embed_image_jobs(embedder, pending_jobs, &mut vectors);

    for (job, vector) in pending_jobs.drain(..).zip(vectors) {
        emit_embedded_document(event_tx, job, vector)?;
    }

    Ok(())
}

fn run_embedding_thread(
    job_rx: mpsc::Receiver<EmbeddingJob>,
    event_tx: mpsc::SyncSender<IndexEvent>,
    cancel_flag: Arc<AtomicBool>,
) -> SearchxResult<()> {
    let mut embedder = crate::embedding::Embedder::default();
    let mut pending_jobs = Vec::with_capacity(EMBEDDING_BATCH_DOC_LIMIT);

    loop {
        match job_rx.recv_timeout(PROGRESS_POLL_INTERVAL) {
            Ok(job) => {
                pending_jobs.push(job);

                let mut disconnected = false;
                while pending_jobs.len() < EMBEDDING_BATCH_DOC_LIMIT {
                    match job_rx.try_recv() {
                        Ok(job) => pending_jobs.push(job),
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            disconnected = true;
                            break;
                        }
                    }
                }

                flush_embedding_jobs(&mut embedder, &mut pending_jobs, &event_tx)?;
                if disconnected {
                    return Ok(());
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                flush_embedding_jobs(&mut embedder, &mut pending_jobs, &event_tx)?;
                if cancel_flag.load(Ordering::Relaxed) {
                    return Ok(());
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                flush_embedding_jobs(&mut embedder, &mut pending_jobs, &event_tx)?;
                return Ok(());
            }
        }
    }
}

fn repair_index_from_working_manifest(
    index: &Index,
    indexer_config: &IndexerConfig,
    data_paths: &crate::api::DataPaths,
    root: &Path,
) -> SearchxResult<()> {
    if !data_paths.manifest.exists() {
        return Ok(());
    }

    let conn = open_manifest_connection(&data_paths.manifest)?;
    let Some(working_manifest) = load_working_manifest(&conn, root)? else {
        return Ok(());
    };

    let indexed_paths = working_manifest
        .files
        .iter()
        .filter(|(_, entry)| entry.is_indexed())
        .map(|(path, _)| path.as_str())
        .collect::<HashSet<_>>();

    let rtxn = index.read_txn()?;
    let fields_ids_map = index.fields_ids_map(&rtxn)?;
    let mut orphaned_ids = Vec::new();

    for document in index.all_documents(&rtxn)? {
        let (_docid, obkv) = document?;
        let value = all_obkv_to_json(obkv, &fields_ids_map)?;
        let path = value.get("path").and_then(|entry| entry.as_str());
        let id = value.get("id").and_then(|entry| entry.as_str());

        match path {
            Some(path) if indexed_paths.contains(path) => {}
            Some(path) => orphaned_ids.push(document_id_for_path(path)),
            None => {
                if let Some(id) = id {
                    orphaned_ids.push(id.to_string());
                }
            }
        }
    }
    drop(rtxn);

    if !orphaned_ids.is_empty() {
        apply_index_batch(
            index,
            indexer_config,
            &data_paths.base,
            &[] as &[&str],
            &orphaned_ids,
        )?;
    }

    Ok(())
}

fn stream_scan_into_index<F>(
    job: StreamScanJob<'_>,
    scan_hook: &Arc<ScanHook>,
    on_progress: &mut F,
) -> SearchxResult<SyncStats>
where
    F: FnMut(SyncProgress),
{
    let StreamScanJob {
        index,
        indexer_config,
        options,
        root,
        data_paths,
        previous_manifest,
        resume_existing_progress,
    } = job;

    let mut progress_manifest =
        ManifestWorkingSet::open(&data_paths.manifest, root, resume_existing_progress)?;
    let (event_tx, event_rx) = mpsc::sync_channel(INDEX_EVENT_CHANNEL_CAPACITY);
    let (embedding_job_tx, embedding_job_rx) = mpsc::sync_channel(EMBEDDING_JOB_CHANNEL_CAPACITY);
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let scan_options = options.clone();
    let scan_root_path = root.to_path_buf();
    let scan_data_dir = data_paths.base.clone();
    let scan_hook_for_thread = Arc::clone(scan_hook);
    let cancel_flag_for_thread = Arc::clone(&cancel_flag);
    let cancel_flag_for_embedding = Arc::clone(&cancel_flag);
    let embedding_event_tx = event_tx.clone();
    let (scan_error_tx, scan_error_rx) = mpsc::channel();

    let embedding_handle = thread::Builder::new()
        .name("Embedding Thread".to_string())
        .spawn(move || {
            run_embedding_thread(
                embedding_job_rx,
                embedding_event_tx,
                cancel_flag_for_embedding,
            )
        })?;

    let scan_handle = thread::Builder::new()
        .name("Scan Thread".to_string())
        .spawn(move || {
            let pipeline = ScanPipeline {
                error_sender: Some(scan_error_tx),
                event_sender: event_tx,
                embedding_sender: Some(embedding_job_tx),
                cancel_flag: cancel_flag_for_thread,
            };

            scan_root(
                &scan_options,
                &scan_root_path,
                &scan_data_dir,
                &previous_manifest,
                Some(scan_hook_for_thread.as_ref()),
                &pipeline,
            )
        })?;

    let mut last_reported_file = None;
    let mut pending_batch = PendingIndexBatch::default();

    let scan_result = loop {
        drain_scan_errors(&scan_error_rx, on_progress);
        report_sync_progress(scan_hook, &mut last_reported_file, on_progress);

        match event_rx.recv_timeout(PROGRESS_POLL_INTERVAL) {
            Ok(event) => {
                pending_batch.push(event);
                if pending_batch.should_flush()
                    && let Err(error) = pending_batch.flush(
                        index,
                        indexer_config,
                        Some(&mut progress_manifest),
                        data_paths,
                    )
                {
                    cancel_flag.store(true, Ordering::Relaxed);
                    drop(event_rx);
                    let _ = finish_scan_thread(scan_handle);
                    let _ = finish_embedding_thread(embedding_handle);
                    return Err(error);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Err(error) = pending_batch.flush(
                    index,
                    indexer_config,
                    Some(&mut progress_manifest),
                    data_paths,
                ) {
                    cancel_flag.store(true, Ordering::Relaxed);
                    drop(event_rx);
                    let _ = finish_scan_thread(scan_handle);
                    let _ = finish_embedding_thread(embedding_handle);
                    return Err(error);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let scan_result = finish_scan_thread(scan_handle);
                let embedding_result = finish_embedding_thread(embedding_handle);
                break match (scan_result, embedding_result) {
                    (Err(error), _) => Err(error),
                    (Ok(_), Err(error)) => Err(error),
                    (Ok(stats), Ok(())) => Ok(stats),
                }?;
            }
        }
    };

    drain_scan_errors(&scan_error_rx, on_progress);

    pending_batch.flush(
        index,
        indexer_config,
        Some(&mut progress_manifest),
        data_paths,
    )?;

    Ok(scan_result)
}

pub fn sync_index(request: &SyncRequest) -> SearchxResult<SyncIndexResult> {
    sync_index_with_progress(request, |_| {})
}

pub fn sync_index_with_progress<F>(
    request: &SyncRequest,
    mut on_progress: F,
) -> SearchxResult<SyncIndexResult>
where
    F: FnMut(SyncProgress),
{
    let root = fs::canonicalize(&request.root)?;
    let data_paths = data_paths(&request.data_dir)?;

    let ManifestLoad {
        manifest,
        rebuild_reason,
        resume_from_incomplete,
    } = load_manifest(&data_paths, &root, request.options.rebuild)?;

    let recovering_incomplete = data_paths.incomplete_marker.exists() && rebuild_reason.is_none();

    if let Some(reason) = rebuild_reason.as_ref() {
        on_progress(SyncProgress::Rebuilding {
            reason: reason.clone(),
        });
        reset_data_dir(&data_paths)?;
    } else {
        fs::create_dir_all(&data_paths.index)?;
    }

    mark_index_incomplete(&data_paths)?;

    let heed_options = new_heed_options();
    let create_or_open = if data_paths.index.join("data.mdb").exists() {
        milli::CreateOrOpen::Open
    } else {
        milli::CreateOrOpen::create_without_shards()
    };

    let index = Index::new(heed_options, &data_paths.index, create_or_open)?;
    let indexer_config = IndexerConfig::default();
    configure_index(&index, &indexer_config)?;

    let scan_hook = Arc::new(ScanHook::new());
    let stats = stream_scan_into_index(
        StreamScanJob {
            index: &index,
            indexer_config: &indexer_config,
            options: &request.options,
            root: &root,
            data_paths: &data_paths,
            previous_manifest: manifest,
            resume_existing_progress: resume_from_incomplete,
        },
        &scan_hook,
        &mut on_progress,
    )?;

    if recovering_incomplete {
        repair_index_from_working_manifest(&index, &indexer_config, &data_paths, &root)?;
    }

    commit_working_manifest(&data_paths.manifest, &root)?;

    clear_index_incomplete(&data_paths)?;

    Ok(SyncIndexResult {
        root,
        data_paths,
        index,
        stats,
        rebuild_reason,
    })
}
