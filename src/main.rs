use milli::update::IndexerConfig;
use milli::{CreateOrOpen, Index};
use searchx::{
    DataPaths, ScanError, ScanHook, ScanOptions, SyncStats, apply_index_changes,
    commit_working_manifest, configure_index, data_paths, default_ignore_rules,
    discard_working_manifest, load_manifest, new_heed_options, reset_data_dir, scan_root,
    search_index,
};
use std::any::Any;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

const DATA_DIR_NAME: &str = ".searchx-data";
const DEFAULT_ROOT: &str = "/Users/anaren/Documents/";
const PROGRESS_POLL_INTERVAL: Duration = Duration::from_millis(250);

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
    let (error_tx, error_rx) = mpsc::channel();
    let scan_options = options.clone();
    let scan_root_path = root.clone();
    let scan_data_dir = data_paths.base.clone();
    let scan_manifest_path = data_paths.manifest.clone();
    let previous_manifest = manifest_load.manifest.clone();
    let scan_hook_for_thread = scan_hook.clone();

    let scan_handle = thread::spawn(move || {
        scan_root(
            &scan_options,
            &scan_root_path,
            &scan_data_dir,
            &previous_manifest,
            Some(scan_hook_for_thread),
            Some(scan_manifest_path),
            Some(error_tx),
        )
        .map_err(|error| error.to_string())
    });

    let mut last_reported_file = None;

    while !scan_handle.is_finished() {
        drain_scan_errors(&error_rx);

        let current_file = scan_hook.current_file();
        if current_file != last_reported_file {
            if let Some(path) = current_file.as_deref() {
                println!("Indexing: {path}");
            }
            last_reported_file = current_file;
        }

        thread::sleep(PROGRESS_POLL_INTERVAL);
    }

    let scan = match scan_handle.join() {
        Ok(Ok(scan)) => scan,
        Ok(Err(error)) => {
            drain_scan_errors(&error_rx);
            discard_working_manifest_logged(&data_paths.manifest, "scan failure");
            return Err(error.into());
        }
        Err(payload) => {
            drain_scan_errors(&error_rx);
            discard_working_manifest_logged(&data_paths.manifest, "scan panic");
            return Err(format!("scan thread panicked: {}", panic_message(payload)).into());
        }
    };

    drain_scan_errors(&error_rx);

    if scan.updated_count > 0 || !scan.deleted_ids.is_empty() {
        if let Err(error) = apply_index_changes(
            &index,
            &indexer_config,
            &scan.updates_file,
            scan.updated_count,
            &scan.deleted_ids,
        ) {
            discard_working_manifest_logged(&data_paths.manifest, "index update failure");
            return Err(error);
        }
    }

    if let Err(error) = commit_working_manifest(&data_paths.manifest, &root) {
        discard_working_manifest_logged(&data_paths.manifest, "manifest commit failure");
        return Err(error);
    }

    print_summary(&root, &data_paths, &scan.stats, options.max_file_bytes);

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
