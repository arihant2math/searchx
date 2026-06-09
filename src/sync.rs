use crate::api::{
    ManifestLoad, ScanHook, ScanOptions, SyncIndexResult, SyncProgress, SyncRequest, SyncStats,
};
use crate::constants::{
    INDEX_BATCH_BYTE_LIMIT, INDEX_BATCH_DELETE_LIMIT, INDEX_BATCH_DOC_LIMIT,
    INDEX_EVENT_CHANNEL_CAPACITY, MANIFEST_BATCH_ENTRY_LIMIT, PROGRESS_POLL_INTERVAL,
};
use crate::error::{SearchxError, SearchxResult};
use crate::index::{apply_index_batch, configure_index, document_id_for_path, new_heed_options};
use crate::manifest::{
    Manifest, ManifestWorkingSet, clear_index_incomplete, commit_working_manifest, data_paths,
    load_manifest, load_working_manifest, mark_index_incomplete, open_manifest_connection,
    reset_data_dir,
};
use crate::scan::{IndexEvent, ProgressUpdate, ScanPipeline, scan_root};
use milli::update::IndexerConfig;
use milli::{Index, all_obkv_to_json};
use std::any::Any;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

#[derive(Default)]
struct PendingIndexBatch {
    upserts: Vec<String>,
    deleted_ids: Vec<String>,
    progress_updates: Vec<ProgressUpdate>,
    bytes: usize,
}

impl PendingIndexBatch {
    fn push(&mut self, event: IndexEvent) {
        match event {
            IndexEvent::Upsert { document, progress } => {
                self.bytes += document.len();
                self.upserts.push(document);
                self.progress_updates.push(progress);
            }
            IndexEvent::Delete {
                document_id,
                progress,
            } => {
                self.deleted_ids.push(document_id);
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
        progress_manifest: Option<&ManifestWorkingSet>,
        data_paths: &crate::api::DataPaths,
    ) -> SearchxResult<()> {
        if self.upserts.is_empty()
            && self.deleted_ids.is_empty()
            && self.progress_updates.is_empty()
        {
            return Ok(());
        }

        if !self.upserts.is_empty() || !self.deleted_ids.is_empty() {
            apply_index_batch(
                index,
                indexer_config,
                &data_paths.base,
                &self.upserts,
                &self.deleted_ids,
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

fn cancel_scan_thread(
    scan_handle: thread::JoinHandle<SearchxResult<SyncStats>>,
    cancel_flag: &AtomicBool,
) {
    cancel_flag.store(true, Ordering::Relaxed);
    let _ = scan_handle.join();
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
        apply_index_batch(index, indexer_config, &data_paths.base, &[], &orphaned_ids)?;
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

    let progress_manifest =
        ManifestWorkingSet::open(&data_paths.manifest, root, resume_existing_progress)?;
    let (event_tx, event_rx) = mpsc::sync_channel(INDEX_EVENT_CHANNEL_CAPACITY);
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let scan_options = options.clone();
    let scan_root_path = root.to_path_buf();
    let scan_data_dir = data_paths.base.clone();
    let scan_hook_for_thread = Arc::clone(scan_hook);
    let cancel_flag_for_thread = Arc::clone(&cancel_flag);
    let (scan_error_tx, scan_error_rx) = mpsc::channel();

    let scan_handle = thread::spawn(move || {
        let pipeline = ScanPipeline {
            error_sender: Some(scan_error_tx),
            event_sender: event_tx,
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
    });

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
                        Some(&progress_manifest),
                        data_paths,
                    )
                {
                    drop(event_rx);
                    cancel_scan_thread(scan_handle, &cancel_flag);
                    return Err(error);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Err(error) =
                    pending_batch.flush(index, indexer_config, Some(&progress_manifest), data_paths)
                {
                    drop(event_rx);
                    cancel_scan_thread(scan_handle, &cancel_flag);
                    return Err(error);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                break finish_scan_thread(scan_handle)?;
            }
        }
    };

    drain_scan_errors(&scan_error_rx, on_progress);

    pending_batch.flush(index, indexer_config, Some(&progress_manifest), data_paths)?;

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
