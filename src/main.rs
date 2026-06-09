use milli::update::IndexerConfig;
use milli::{CreateOrOpen, Index};
use searchx::{
    DataPaths, ScanOptions, SyncStats, apply_index_changes, configure_index, data_paths,
    load_manifest, new_heed_options, reset_data_dir, save_manifest, scan_root, search_index,
};
use std::error::Error;
use std::fs;
use std::path::Path;

const DATA_DIR_NAME: &str = ".searchx-data";
const DEFAULT_ROOT: &str = "/Users/anaren/Documents/oss/memchr";

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

fn run() -> Result<(), Box<dyn Error>> {
    let options = ScanOptions {
        rebuild: false,
        max_file_bytes: 1024 * 1024 * 50,
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

    let scan = scan_root(&options, &root, &data_paths.base, &manifest_load.manifest)?;
    if scan.updated_count > 0 || !scan.deleted_ids.is_empty() {
        apply_index_changes(
            &index,
            &indexer_config,
            &scan.updates_file,
            scan.updated_count,
            &scan.deleted_ids,
        )?;
        save_manifest(&data_paths.manifest, &scan.next_manifest)?;
    } else if manifest_load.rebuild_reason.is_some() || !data_paths.manifest.exists() {
        save_manifest(&data_paths.manifest, &scan.next_manifest)?;
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
