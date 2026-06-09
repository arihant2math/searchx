use clap::Parser;
use milli::Index;
use searchx::{
    DEFAULT_DATA_DIR_NAME, SearchResults, SearchxResult, data_paths, new_heed_options, search_index,
};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(version, about = "Search an existing searchx index")]
struct Args {
    /// Search query
    query: String,
    /// Directory containing index data
    #[arg(short = 'd', long, default_value = DEFAULT_DATA_DIR_NAME)]
    data_dir: PathBuf,
    /// Maximum number of search hits to print
    #[arg(short = 'l', long, default_value_t = 10)]
    limit: usize,
}

fn open_existing_index(data_dir: &Path) -> SearchxResult<Index> {
    let data_paths = data_paths(data_dir)?;
    let data_file = data_paths.index.join("data.mdb");

    if !data_file.exists() {
        return Err(io::Error::new(
            ErrorKind::NotFound,
            format!("index not found at {}", data_file.display()),
        )
        .into());
    }

    Index::new(
        new_heed_options(),
        &data_paths.index,
        milli::CreateOrOpen::Open,
    )
    .map_err(Into::into)
}

fn print_search_results(results: &SearchResults) {
    println!("Query: {}", results.query);
    println!("Found {} matching documents.", results.candidate_count);
    for (timing, value) in &results.timings {
        println!("{timing}: {value}");
    }

    for hit in &results.hits {
        println!("{}. {}", hit.rank, hit.path);
    }
}

fn run() -> SearchxResult<()> {
    let args = Args::parse();
    let query = args.query.trim();
    if query.is_empty() {
        return Err(io::Error::new(ErrorKind::InvalidInput, "query must not be empty").into());
    }

    let index = open_existing_index(&args.data_dir)?;
    let results = search_index(&index, query, args.limit)?;
    print_search_results(&results);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
