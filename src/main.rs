use searchx::{
    DEFAULT_DATA_DIR_NAME, DEFAULT_MAX_FILE_BYTES, ScanOptions, SearchResults, SyncIndexResult,
    SyncProgress, SyncRequest, search_index, sync_index_with_progress,
};
use std::env;
use std::error::Error;
use std::path::PathBuf;

struct CliOptions {
    root: PathBuf,
    query: Option<String>,
    data_dir: PathBuf,
    max_file_bytes: u64,
    rebuild: bool,
    limit: usize,
}

enum Command {
    Help(String),
    Run(CliOptions),
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} [options] <root> [query]\n\n\
         Options:\n\
           -d, --data-dir PATH         Directory used to store the index [{DEFAULT_DATA_DIR_NAME}]\n\
           -m, --max-file-bytes BYTES  Skip files larger than this [{DEFAULT_MAX_FILE_BYTES}]\n\
           -l, --limit N               Maximum number of search hits to print [10]\n\
           -r, --rebuild               Force a full rebuild before indexing\n\
           -h, --help                  Show this help\n"
    )
}

fn parse_u64_flag(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn parse_usize_flag(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn next_flag_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_args() -> Result<Command, String> {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "searchx".to_string());
    let mut args = args.peekable();

    let mut data_dir = PathBuf::from(DEFAULT_DATA_DIR_NAME);
    let mut max_file_bytes = DEFAULT_MAX_FILE_BYTES;
    let mut rebuild = false;
    let mut limit = 10usize;
    let mut positional = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help(usage(&program))),
            "-d" | "--data-dir" => {
                data_dir = PathBuf::from(next_flag_value(&mut args, arg.as_str())?);
            }
            "-m" | "--max-file-bytes" => {
                let value = next_flag_value(&mut args, arg.as_str())?;
                max_file_bytes = parse_u64_flag(arg.as_str(), &value)?;
            }
            "-l" | "--limit" => {
                let value = next_flag_value(&mut args, arg.as_str())?;
                limit = parse_usize_flag(arg.as_str(), &value)?;
            }
            "-r" | "--rebuild" => rebuild = true,
            _ if arg.starts_with('-') => {
                return Err(format!("unknown option: {arg}\n\n{}", usage(&program)));
            }
            _ => positional.push(arg),
        }
    }

    match positional.as_slice() {
        [] => Err(usage(&program)),
        [root] => Ok(Command::Run(CliOptions {
            root: PathBuf::from(root),
            query: None,
            data_dir,
            max_file_bytes,
            rebuild,
            limit,
        })),
        [root, query] => Ok(Command::Run(CliOptions {
            root: PathBuf::from(root),
            query: Some(query.clone()),
            data_dir,
            max_file_bytes,
            rebuild,
            limit,
        })),
        _ => Err(format!(
            "too many positional arguments\n\n{}",
            usage(&program)
        )),
    }
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
    let command = match parse_args() {
        Ok(command) => command,
        Err(error) if error.starts_with("Usage:") => {
            println!("{error}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };

    let cli = match command {
        Command::Help(help) => {
            println!("{help}");
            return Ok(());
        }
        Command::Run(cli) => cli,
    };

    let request = SyncRequest::new(&cli.root)
        .with_data_dir(&cli.data_dir)
        .with_options(ScanOptions {
            rebuild: cli.rebuild,
            max_file_bytes: cli.max_file_bytes,
            ..ScanOptions::default()
        });

    let result = sync_index_with_progress(&request, |progress| match progress {
        SyncProgress::Rebuilding { reason } => eprintln!("Rebuilding index: {reason}"),
        SyncProgress::Indexing { path } => eprintln!("Indexing: {path}"),
        SyncProgress::ScanError(error) => eprintln!("{error}"),
    })?;

    print_summary(&result, cli.max_file_bytes);

    if let Some(query) = cli
        .query
        .as_deref()
        .filter(|query| !query.trim().is_empty())
    {
        let results = search_index(&result.index, query, cli.limit)?;
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
