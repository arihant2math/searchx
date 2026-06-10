use clap::Parser;
use searchx::{
    DEFAULT_DATA_DIR_NAME, DEFAULT_MAX_FILE_BYTES, ScanOptions, SearchxResult, SyncIndexResult,
    SyncProgress, SyncRequest, default_ignore_rules, sync_index_with_progress,
};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(version, about = "Build or update a searchx index")]
struct Args {
    /// Root directory to index
    root: PathBuf,
    /// Directory used to store index data
    #[arg(short = 'd', long, default_value = DEFAULT_DATA_DIR_NAME)]
    data_dir: PathBuf,
    /// Skip files larger than this many bytes
    #[arg(short = 'm', long, default_value_t = DEFAULT_MAX_FILE_BYTES)]
    max_file_bytes: u64,
    /// Force a full rebuild before indexing
    #[arg(short = 'r', long)]
    rebuild: bool,
    #[arg(short = 'i', long = "ignore")]
    ignore_rules: Vec<String>,
}

fn print_summary(result: &SyncIndexResult, max_file_bytes: u64) {
    println!("Indexed root: {}", result.root.display());
    println!("Index data: {}", result.data_paths.index.display());
    println!("Scanned files: {}", result.stats.scanned_files);
    println!(
        "Indexed/updated documents: {}",
        result.stats.indexed_or_updated
    );
    println!("Deleted documents: {}", result.stats.deleted_total());
    println!(
        "Skipped unchanged indexed files: {}",
        result.stats.unchanged_indexed
    );
    println!(
        "Skipped unchanged unsupported files: {}",
        result.stats.unchanged_skipped
    );
    println!(
        "Metadata-only files due to size limit (> {max_file_bytes} bytes): {}",
        result.stats.skipped_too_large
    );
    println!(
        "Metadata-only binary/non-UTF8 files: {}",
        result.stats.skipped_binary
    );
    println!("Read errors: {}", result.stats.read_errors);
    println!("Walk errors: {}", result.stats.walk_errors);
}

fn run() -> SearchxResult<()> {
    let args = Args::parse();

    let mut ignore_rules = default_ignore_rules();
    ignore_rules.append(&mut args.ignore_rules.clone());

    let request = SyncRequest::new(&args.root)
        .with_data_dir(&args.data_dir)
        .with_options(ScanOptions {
            rebuild: args.rebuild,
            max_file_bytes: args.max_file_bytes,
            ignore_rules,
        });

    let result = sync_index_with_progress(&request, |progress| match progress {
        SyncProgress::Rebuilding { reason } => eprintln!("Rebuilding index: {reason}"),
        SyncProgress::Indexing { path } => eprintln!("Indexing: {path}"),
        SyncProgress::ScanError(error) => eprintln!("{error}"),
    })?;

    print_summary(&result, args.max_file_bytes);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
