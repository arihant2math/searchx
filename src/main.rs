use searchx::{
    SearchResults, SyncIndexResult, SyncProgress, SyncRequest, search_index,
    sync_index_with_progress,
};
use std::error::Error;

const DEFAULT_ROOT: &str = "/Users/anaren/Documents/";

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
        "Skipped oversized files (> {max_file_bytes} bytes): {}",
        result.stats.skipped_too_large
    );
    println!(
        "Skipped binary/non-UTF8 files: {}",
        result.stats.skipped_binary
    );
    println!("Read errors: {}", result.stats.read_errors);
    println!("Walk errors: {}", result.stats.walk_errors);
}

fn print_search_results(results: &SearchResults) {
    println!();
    println!("Query: {}", results.query);
    println!("Found {} matching documents.", results.candidate_count);

    for hit in &results.hits {
        println!("{}. {}", hit.rank, hit.path);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let query = Some("Hi".to_string());
    let request = SyncRequest::new(DEFAULT_ROOT);
    let max_file_bytes = request.options.max_file_bytes;

    let result = sync_index_with_progress(&request, |progress| match progress {
        SyncProgress::Rebuilding { reason } => eprintln!("Rebuilding index: {reason}"),
        SyncProgress::Indexing { path } => println!("Indexing: {path}"),
        SyncProgress::ScanError(error) => eprintln!("{error}"),
    })?;

    print_summary(&result, max_file_bytes);

    if let Some(query) = query.as_deref().filter(|query| !query.trim().is_empty()) {
        let results = search_index(&result.index, query, 10)?;
        print_search_results(&results);
    }

    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
