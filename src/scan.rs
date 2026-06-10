use crate::api::{ScanError, ScanHook, ScanOptions, SyncStats};
use crate::error::{SearchxError, SearchxResult};
use crate::index::{IndexedDocument, document_id_for_path, empty_document_vectors};
use crate::manifest::{FileFingerprint, FileState, Manifest, ManifestEntry, SkipReason};
use crossbeam_channel as channel;
use ignore::{DirEntry, WalkBuilder};
use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tempfile::NamedTempFile;

#[derive(Debug, Clone)]
pub(crate) struct ProgressUpdate {
    pub(crate) path: String,
    pub(crate) entry: ManifestEntry,
}

#[derive(Debug)]
pub(crate) enum IndexEvent {
    Upsert {
        document: IndexedDocument,
        progress: Option<ProgressUpdate>,
    },
    Delete {
        document_id: String,
        progress: Option<ProgressUpdate>,
    },
    Progress(ProgressUpdate),
}

#[derive(Debug)]
pub(crate) enum EmbeddingJobInput {
    DocumentContents,
    Text(String),
    Image(Vec<u8>),
}

#[derive(Debug)]
pub(crate) struct EmbeddingJob {
    pub(crate) document: IndexedDocument,
    pub(crate) input: EmbeddingJobInput,
    pub(crate) progress: ProgressUpdate,
}

#[derive(Debug)]
pub(crate) struct ScanPipeline {
    pub(crate) error_sender: Option<channel::Sender<ScanError>>,
    pub(crate) event_sender: channel::Sender<IndexEvent>,
    pub(crate) embedding_sender: Option<channel::Sender<EmbeddingJob>>,
    pub(crate) cancel_flag: Arc<AtomicBool>,
}

pub(crate) fn scan_root(
    options: &ScanOptions,
    root: &Path,
    data_dir: &Path,
    previous: &Manifest,
    scan_hook: Option<&ScanHook>,
    pipeline: &ScanPipeline,
) -> SearchxResult<SyncStats> {
    ScanContext::new(options, root, previous, scan_hook, pipeline)?.run(data_dir)
}

struct ScanContext<'a> {
    options: &'a ScanOptions,
    root: &'a Path,
    previous: &'a Manifest,
    scan_hook: Option<&'a ScanHook>,
    error_sender: Option<&'a channel::Sender<ScanError>>,
    event_sender: &'a channel::Sender<IndexEvent>,
    embedding_sender: Option<&'a channel::Sender<EmbeddingJob>>,
    cancel_flag: &'a AtomicBool,
    seen_paths: HashSet<String>,
    stats: SyncStats,
}

impl<'a> ScanContext<'a> {
    fn new(
        options: &'a ScanOptions,
        root: &'a Path,
        previous: &'a Manifest,
        scan_hook: Option<&'a ScanHook>,
        pipeline: &'a ScanPipeline,
    ) -> SearchxResult<Self> {
        Ok(Self {
            options,
            root,
            previous,
            scan_hook,
            error_sender: pipeline.error_sender.as_ref(),
            event_sender: &pipeline.event_sender,
            embedding_sender: pipeline.embedding_sender.as_ref(),
            cancel_flag: pipeline.cancel_flag.as_ref(),
            seen_paths: HashSet::with_capacity(previous.files.len().saturating_mul(2)),
            stats: SyncStats::default(),
        })
    }

    fn run(mut self, data_dir: &Path) -> SearchxResult<SyncStats> {
        let result = self.run_inner(data_dir);
        if let Some(scan_hook) = self.scan_hook {
            scan_hook.clear_current_file();
        }
        result.map(|()| self.stats)
    }

    fn run_inner(&mut self, data_dir: &Path) -> SearchxResult<()> {
        let mut walker_builder = WalkBuilder::new(self.root);
        walker_builder
            .hidden(false)
            .require_git(false)
            .current_dir(self.root);

        let custom_ignore_file = build_ignore_file(data_dir, &self.options.ignore_rules)?;
        if let Some(ignore_file) = &custom_ignore_file
            && let Some(error) = walker_builder.add_ignore(ignore_file.path())
        {
            return Err(error.into());
        }

        let data_dir = data_dir.to_path_buf();
        let walker = walker_builder
            .filter_entry(move |entry| should_walk_entry(entry, &data_dir))
            .build();

        for result in walker {
            self.ensure_not_canceled()?;

            match result {
                Ok(entry) => self.scan_entry(&entry)?,
                Err(error) => {
                    self.stats.walk_errors += 1;
                    self.report_error(ScanError::walk(error.to_string()));
                }
            }
        }

        self.delete_missing_documents()
    }

    fn ensure_not_canceled(&self) -> SearchxResult<()> {
        if self.cancel_flag.load(Ordering::Relaxed) {
            Err(SearchxError::ScanCanceled)
        } else {
            Ok(())
        }
    }

    fn scan_entry(&mut self, entry: &DirEntry) -> SearchxResult<()> {
        let Some(file_type) = entry.file_type() else {
            return Ok(());
        };
        if !file_type.is_file() {
            return Ok(());
        }

        let path = entry.path();
        let relative_path = normalize_relative_path(path, self.root)?;
        self.seen_paths.insert(relative_path.clone());
        if let Some(scan_hook) = self.scan_hook {
            scan_hook.set_current_file(relative_path.clone());
        }
        self.stats.scanned_files += 1;

        let Some(metadata) = self.read_metadata(&relative_path, path)? else {
            return Ok(());
        };

        let fingerprint = FileFingerprint::from_metadata(&metadata);
        if let Some(previous_entry) = self.previous.files.get(&relative_path).cloned()
            && previous_entry.matches(fingerprint)
            && !previous_entry.embedding_pending()
        {
            self.mark_unchanged(&relative_path, &previous_entry)?;
            return Ok(());
        }

        if metadata.len() > self.options.max_file_bytes {
            self.index_metadata_only_document(
                &relative_path,
                path,
                fingerprint,
                SkipReason::TooLarge,
            )?;
            return Ok(());
        }

        let Some(bytes) = self.read_bytes(&relative_path, path)? else {
            return Ok(());
        };

        let bytes = match supported_binary_embedding_input(&relative_path, path, bytes) {
            Ok(embedding_input) => {
                return self.index_binary_document(
                    &relative_path,
                    path,
                    fingerprint,
                    embedding_input,
                );
            }
            Err(bytes) => bytes,
        };

        if bytes.contains(&0) {
            self.index_metadata_only_document(
                &relative_path,
                path,
                fingerprint,
                SkipReason::Binary,
            )?;
            return Ok(());
        }

        let Ok(contents) = String::from_utf8(bytes) else {
            self.index_metadata_only_document(
                &relative_path,
                path,
                fingerprint,
                SkipReason::Binary,
            )?;
            return Ok(());
        };

        self.index_document(&relative_path, path, fingerprint, contents)
    }

    fn read_metadata(
        &mut self,
        relative_path: &str,
        path: &Path,
    ) -> SearchxResult<Option<fs::Metadata>> {
        match fs::metadata(path) {
            Ok(metadata) => Ok(Some(metadata)),
            Err(error) => {
                self.stats.read_errors += 1;
                self.report_error(ScanError::metadata(
                    path.display().to_string(),
                    error.to_string(),
                ));
                self.preserve_previous_entry(relative_path)?;
                Ok(None)
            }
        }
    }

    fn read_bytes(&mut self, relative_path: &str, path: &Path) -> SearchxResult<Option<Vec<u8>>> {
        match fs::read(path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) => {
                self.stats.read_errors += 1;
                self.report_error(ScanError::read(
                    path.display().to_string(),
                    error.to_string(),
                ));
                self.preserve_previous_entry(relative_path)?;
                Ok(None)
            }
        }
    }

    fn preserve_previous_entry(&self, relative_path: &str) -> SearchxResult<()> {
        if let Some(previous_entry) = self.previous.files.get(relative_path) {
            send_progress_event(
                self.event_sender,
                self.cancel_flag,
                relative_path,
                previous_entry,
            )?;
        }
        Ok(())
    }

    fn mark_unchanged(
        &mut self,
        relative_path: &str,
        previous_entry: &ManifestEntry,
    ) -> SearchxResult<()> {
        send_progress_event(
            self.event_sender,
            self.cancel_flag,
            relative_path,
            previous_entry,
        )?;
        if previous_entry.skips_contents() {
            self.stats.unchanged_skipped += 1;
        } else if previous_entry.is_indexed() {
            self.stats.unchanged_indexed += 1;
        } else {
            self.stats.unchanged_skipped += 1;
        }
        Ok(())
    }

    fn index_metadata_only_document(
        &mut self,
        relative_path: &str,
        path: &Path,
        fingerprint: FileFingerprint,
        reason: SkipReason,
    ) -> SearchxResult<()> {
        match reason {
            SkipReason::TooLarge => self.stats.skipped_too_large += 1,
            SkipReason::Binary => self.stats.skipped_binary += 1,
        }

        self.upsert_document(
            relative_path,
            path,
            fingerprint,
            FileState::IndexedMetadata { reason },
            String::new(),
            Some(EmbeddingJobInput::Text(metadata_embedding_text(
                relative_path,
                path,
            ))),
        )
    }

    fn index_binary_document(
        &mut self,
        relative_path: &str,
        path: &Path,
        fingerprint: FileFingerprint,
        embedding_input: EmbeddingJobInput,
    ) -> SearchxResult<()> {
        self.upsert_document(
            relative_path,
            path,
            fingerprint,
            FileState::Indexed,
            String::new(),
            Some(embedding_input),
        )
    }

    fn index_document(
        &mut self,
        relative_path: &str,
        path: &Path,
        fingerprint: FileFingerprint,
        contents: String,
    ) -> SearchxResult<()> {
        self.upsert_document(
            relative_path,
            path,
            fingerprint,
            FileState::Indexed,
            contents,
            Some(EmbeddingJobInput::DocumentContents),
        )
    }

    fn upsert_document(
        &mut self,
        relative_path: &str,
        path: &Path,
        fingerprint: FileFingerprint,
        final_state: FileState,
        contents: String,
        embedding_input: Option<EmbeddingJobInput>,
    ) -> SearchxResult<()> {
        let embed_async = embedding_input.is_some() && self.embedding_sender.is_some();
        let initial_state = if embed_async {
            pending_embedding_state(&final_state)
        } else {
            final_state.clone()
        };
        let initial_entry = ManifestEntry::from_fingerprint(fingerprint, initial_state);
        let final_entry = ManifestEntry::from_fingerprint(fingerprint, final_state);

        let document = IndexedDocument {
            id: document_id_for_path(relative_path),
            path: relative_path.to_string(),
            file_name: path
                .file_name()
                .and_then(OsStr::to_str)
                .unwrap_or_default()
                .to_string(),
            extension: path.extension().and_then(OsStr::to_str).map(str::to_string),
            contents,
            vectors: empty_document_vectors(),
        };

        if let (Some(embedding_sender), Some(input)) = (self.embedding_sender, embedding_input) {
            send_progress_event(
                self.event_sender,
                self.cancel_flag,
                relative_path,
                &initial_entry,
            )?;
            send_embedding_job(
                embedding_sender,
                self.cancel_flag,
                EmbeddingJob {
                    document,
                    input,
                    progress: ProgressUpdate {
                        path: relative_path.to_string(),
                        entry: final_entry,
                    },
                },
            )?;
        } else {
            send_index_event(
                self.event_sender,
                self.cancel_flag,
                IndexEvent::Upsert {
                    document,
                    progress: Some(ProgressUpdate {
                        path: relative_path.to_string(),
                        entry: initial_entry,
                    }),
                },
            )?;
        }

        self.stats.indexed_or_updated += 1;
        Ok(())
    }

    fn delete_missing_documents(&mut self) -> SearchxResult<()> {
        for (path, entry) in &self.previous.files {
            if entry.is_indexed() && !self.seen_paths.contains(path) {
                send_index_event(
                    self.event_sender,
                    self.cancel_flag,
                    IndexEvent::Delete {
                        document_id: document_id_for_path(path),
                        progress: None,
                    },
                )?;
                self.stats.deleted_missing += 1;
            }
        }
        Ok(())
    }

    fn report_error(&self, error: ScanError) {
        report_scan_error(self.error_sender, error);
    }
}

fn send_index_event(
    event_sender: &channel::Sender<IndexEvent>,
    cancel_flag: &AtomicBool,
    event: IndexEvent,
) -> SearchxResult<()> {
    if cancel_flag.load(Ordering::Relaxed) {
        return Err(SearchxError::IndexingCanceled);
    }

    event_sender
        .send(event)
        .map_err(|_| SearchxError::IndexingPipelineDisconnected)
}

fn send_progress_event(
    event_sender: &channel::Sender<IndexEvent>,
    cancel_flag: &AtomicBool,
    relative_path: &str,
    entry: &ManifestEntry,
) -> SearchxResult<()> {
    send_index_event(
        event_sender,
        cancel_flag,
        IndexEvent::Progress(ProgressUpdate {
            path: relative_path.to_string(),
            entry: entry.clone(),
        }),
    )
}

fn send_embedding_job(
    embedding_sender: &channel::Sender<EmbeddingJob>,
    cancel_flag: &AtomicBool,
    job: EmbeddingJob,
) -> SearchxResult<()> {
    if cancel_flag.load(Ordering::Relaxed) {
        return Err(SearchxError::IndexingCanceled);
    }

    embedding_sender
        .send(job)
        .map_err(|_| SearchxError::IndexingPipelineDisconnected)
}

fn report_scan_error(error_sender: Option<&channel::Sender<ScanError>>, error: ScanError) {
    if let Some(error_sender) = error_sender {
        let _ = error_sender.send(error);
    } else {
        eprintln!("{error}");
    }
}

fn build_ignore_file(
    data_dir: &Path,
    ignore_rules: &[String],
) -> SearchxResult<Option<NamedTempFile>> {
    if ignore_rules.is_empty() {
        return Ok(None);
    }

    let mut file = NamedTempFile::new_in(data_dir)?;
    for rule in ignore_rules {
        writeln!(file, "{rule}")?;
    }
    file.as_file_mut().flush()?;

    Ok(Some(file))
}

fn pending_embedding_state(state: &FileState) -> FileState {
    match state {
        FileState::Indexed => FileState::IndexedPendingEmbedding,
        FileState::IndexedPendingEmbedding => FileState::IndexedPendingEmbedding,
        FileState::IndexedMetadata { reason } => {
            FileState::IndexedMetadataPendingEmbedding { reason: *reason }
        }
        FileState::IndexedMetadataPendingEmbedding { reason } => {
            FileState::IndexedMetadataPendingEmbedding { reason: *reason }
        }
        FileState::Skipped { reason } => FileState::Skipped { reason: *reason },
    }
}

fn metadata_embedding_text(relative_path: &str, path: &Path) -> String {
    let file_name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
    let extension = path.extension().and_then(OsStr::to_str).unwrap_or_default();

    if extension.is_empty() {
        format!("{relative_path}\n{file_name}")
    } else {
        format!("{relative_path}\n{file_name}\n{extension}")
    }
}

fn supported_binary_embedding_input(
    relative_path: &str,
    path: &Path,
    bytes: Vec<u8>,
) -> Result<EmbeddingJobInput, Vec<u8>> {
    let Some(extension) = path.extension().and_then(OsStr::to_str) else {
        return Err(bytes);
    };

    if extension.eq_ignore_ascii_case("pdf") {
        return Ok(EmbeddingJobInput::Text(metadata_embedding_text(
            relative_path,
            path,
        )));
    }

    if ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff"]
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    {
        return Ok(EmbeddingJobInput::Image(bytes));
    }

    Err(bytes)
}

fn should_walk_entry(entry: &DirEntry, data_dir: &Path) -> bool {
    !entry.path().starts_with(data_dir)
}

fn normalize_relative_path(path: &Path, root: &Path) -> SearchxResult<String> {
    let relative = path.strip_prefix(root)?;
    Ok(relative
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/"))
}
