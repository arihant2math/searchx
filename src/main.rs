use milli::update::IndexerConfig;
use milli::{CreateOrOpen, Index};
use searchx::{
    DataPaths, IndexEvent, ScanError, ScanHook, ScanOptions, ScanPipeline, SyncStats,
    apply_index_batch, clear_index_incomplete, commit_working_manifest, configure_index,
    data_paths, default_ignore_rules, discard_working_manifest, load_manifest,
    mark_index_incomplete, new_heed_options, reset_data_dir, scan_root, search_index,
};
use std::any::Any;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

const DATA_DIR_NAME: &str = ".searchx-data";
const DEFAULT_ROOT: &str = "/Users/anaren/Documents/";
const PROGRESS_POLL_INTERVAL: Duration = Duration::from_millis(250);
const INDEX_EVENT_CHANNEL_CAPACITY: usize = 32;
const INDEX_BATCH_DOC_LIMIT: usize = 128;
const INDEX_BATCH_DELETE_LIMIT: usize = 512;
const INDEX_BATCH_BYTE_LIMIT: usize = 8 * 1024 * 1024;

fn print_summary(root: &Path, data_paths: &DataPaths, stats: &SyncStats, max_file_bytes: u64) {
    println!("Indexed root: {}", root.display());
    println!("Index data: {}", data_paths.index.display());
    println!("Scanned files: {}", stats.scanned_files);
    println!("Indexed/updated documents: {}", stats.indexed_or_updated);
    println!("Deleted documents: {}", stats.deleted_total());
    println!(
        "Skipped unchanged indexed files: {}",
        stats.unchanged_indexed
    );
    println!(
        "Skipped unchanged unsupported files: {}",
        stats.unchanged_skipped
    );
    println!(
        "Skipped oversized files (> {max_file_bytes} bytes): {}",
        stats.skipped_too_large
    );
    println!("Skipped binary/non-UTF8 files: {}", stats.skipped_binary);
    println!("Read errors: {}", stats.read_errors);
    println!("Walk errors: {}", stats.walk_errors);
}

fn drain_scan_errors(error_rx: &mpsc::Receiver<ScanError>) {
    while let Ok(error) = error_rx.try_recv() {
        eprintln!("{error}");
    }
}

fn discard_working_manifest_logged(path: &Path, context: &str) {
    if let Err(error) = discard_working_manifest(path) {
        eprintln!("manifest cleanup error after {context}: {error}");
    }
}

fn panic_message(payload: Box<dyn Any + Send + 'static>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn report_progress(scan_hook: &ScanHook, last_reported_file: &mut Option<String>) {
    let current_file = scan_hook.current_file();
    if current_file != *last_reported_file {
        if let Some(path) = current_file.as_deref() {
            println!("Indexing: {path}");
        }
        *last_reported_file = current_file;
    }
}

fn flush_index_batch(
    index: &Index,
    indexer_config: &IndexerConfig,
    data_paths: &DataPaths,
    pending_upserts: &mut Vec<String>,
    pending_deleted_ids: &mut Vec<String>,
    pending_bytes: &mut usize,
) -> Result<(), Box<dyn Error>> {
    if pending_upserts.is_empty() && pending_deleted_ids.is_empty() {
        return Ok(());
    }

    apply_index_batch(
        index,
        indexer_config,
        &data_paths.base,
        pending_upserts,
        pending_deleted_ids,
    )?;

    pending_upserts.clear();
    pending_deleted_ids.clear();
    *pending_bytes = 0;
    Ok(())
}

fn stream_scan_into_index(
    index: &Index,
    indexer_config: &IndexerConfig,
    options: &ScanOptions,
    root: &Path,
    data_paths: &DataPaths,
    previous_manifest: searchx::Manifest,
    scan_hook: Arc<ScanHook>,
) -> Result<SyncStats, Box<dyn Error>> {
    let (event_tx, event_rx) = mpsc::sync_channel(INDEX_EVENT_CHANNEL_CAPACITY);
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let scan_options = options.clone();
    let scan_root_path = root.to_path_buf();
    let scan_data_dir = data_paths.base.clone();
    let scan_manifest_path = data_paths.manifest.clone();
    let scan_hook_for_thread = scan_hook.clone();
    let cancel_flag_for_thread = cancel_flag.clone();
    let (scan_error_tx, scan_error_rx) = mpsc::channel();

    let scan_handle = thread::spawn(move || {
        scan_root(
            &scan_options,
            &scan_root_path,
            &scan_data_dir,
            &previous_manifest,
            Some(scan_hook_for_thread),
            ScanPipeline {
                progress_manifest_db: Some(scan_manifest_path),
                error_sender: Some(scan_error_tx),
                event_sender: event_tx,
                cancel_flag: cancel_flag_for_thread,
            },
        )
        .map_err(|error| error.to_string())
    });

    let mut last_reported_file = None;
    let mut pending_upserts = Vec::new();
    let mut pending_deleted_ids = Vec::new();
    let mut pending_bytes = 0usize;

    let scan_result = loop {
        drain_scan_errors(&scan_error_rx);
        report_progress(&scan_hook, &mut last_reported_file);

        match event_rx.recv_timeout(PROGRESS_POLL_INTERVAL) {
            Ok(event) => {
                match event {
                    IndexEvent::Upsert(document) => {
                        pending_bytes += document.len();
                        pending_upserts.push(document);
                    }
                    IndexEvent::Delete(document_id) => pending_deleted_ids.push(document_id),
                }

                if (pending_upserts.len() >= INDEX_BATCH_DOC_LIMIT
                    || pending_deleted_ids.len() >= INDEX_BATCH_DELETE_LIMIT
                    || pending_bytes >= INDEX_BATCH_BYTE_LIMIT)
                    && let Err(error) = flush_index_batch(
                        index,
                        indexer_config,
                        data_paths,
                        &mut pending_upserts,
                        &mut pending_deleted_ids,
                        &mut pending_bytes,
                    )
                {
                    cancel_flag.store(true, Ordering::Relaxed);
                    drop(event_rx);
                    let _ = scan_handle.join();
                    discard_working_manifest_logged(
                        &data_paths.manifest,
                        "index batch update failure",
                    );
                    return Err(error);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Err(error) = flush_index_batch(
                    index,
                    indexer_config,
                    data_paths,
                    &mut pending_upserts,
                    &mut pending_deleted_ids,
                    &mut pending_bytes,
                ) {
                    cancel_flag.store(true, Ordering::Relaxed);
                    drop(event_rx);
                    let _ = scan_handle.join();
                    discard_working_manifest_logged(
                        &data_paths.manifest,
                        "index batch update failure",
                    );
                    return Err(error);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                break match scan_handle.join() {
                    Ok(Ok(summary)) => summary,
                    Ok(Err(error)) => {
                        discard_working_manifest_logged(&data_paths.manifest, "scan failure");
                        return Err(error.into());
                    }
                    Err(payload) => {
                        discard_working_manifest_logged(&data_paths.manifest, "scan panic");
                        return Err(
                            format!("scan thread panicked: {}", panic_message(payload)).into()
                        );
                    }
                };
            }
        }
    };

    drain_scan_errors(&scan_error_rx);

    if let Err(error) = flush_index_batch(
        index,
        indexer_config,
        data_paths,
        &mut pending_upserts,
        &mut pending_deleted_ids,
        &mut pending_bytes,
    ) {
        discard_working_manifest_logged(&data_paths.manifest, "final index batch update failure");
        return Err(error);
    }

    Ok(scan_result)
}

fn run() -> Result<(), Box<dyn Error>> {
    let options = ScanOptions {
        rebuild: false,
        max_file_bytes: 1024 * 1024 * 50,
        ignore_rules: default_ignore_rules(),
    };
    let query = Some("Hi".to_string());
    let root = fs::canonicalize(DEFAULT_ROOT)?;
    let data_paths = data_paths(DATA_DIR_NAME)?;

    let manifest_load = load_manifest(&data_paths, &root, options.rebuild)?;
    if let Some(reason) = &manifest_load.rebuild_reason {
        eprintln!("Rebuilding index: {reason}");
        reset_data_dir(&data_paths)?;
    } else {
        fs::create_dir_all(&data_paths.index)?;
    }

    mark_index_incomplete(&data_paths)?;

    let heed_options = new_heed_options();

    let create_or_open = if data_paths.index.join("data.mdb").exists() {
        CreateOrOpen::Open
    } else {
        CreateOrOpen::create_without_shards()
    };

    let index = Index::new(heed_options, &data_paths.index, create_or_open)?;
    let indexer_config = IndexerConfig::default();

    configure_index(&index, &indexer_config)?;

    let scan_hook = Arc::new(ScanHook::new());
    let stats = stream_scan_into_index(
        &index,
        &indexer_config,
        &options,
        &root,
        &data_paths,
        manifest_load.manifest,
        scan_hook,
    )?;

    if let Err(error) = commit_working_manifest(&data_paths.manifest, &root) {
        discard_working_manifest_logged(&data_paths.manifest, "manifest commit failure");
        return Err(error);
    }

    clear_index_incomplete(&data_paths)?;
    print_summary(&root, &data_paths, &stats, options.max_file_bytes);

    if let Some(query) = query.as_deref().filter(|query| !query.trim().is_empty()) {
        search_index(&index, query, 10)?;
    }

    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
