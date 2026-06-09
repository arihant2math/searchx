use bumpalo::Bump;
use http_client::policy::IpPolicy;
use ignore::{DirEntry, WalkBuilder};
use memmap2::Mmap;
use milli::heed::{EnvOpenOptions, WithoutTls};
use milli::progress::{EmbedderStats, Progress};
use milli::update::new::indexer;
use milli::update::{IndexerConfig, MissingDocumentPolicy, Setting, Settings};
use milli::vector::VectorStoreBackend;
use milli::vector::settings::{EmbedderSource, EmbeddingSettings};
use milli::{Index, all_obkv_to_json};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::error::Error;
use std::ffi::OsStr;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread;
use std::time::{Duration, UNIX_EPOCH};
use tempfile::NamedTempFile;

const MANIFEST_VERSION: u32 = 1;
const MANIFEST_FILE_NAME: &str = "manifest.sqlite3";
const INCOMPLETE_FILE_NAME: &str = "indexing-incomplete";
const DEFAULT_MAP_SIZE_BYTES: usize = 10 * 1024 * 1024 * 1024;
const PRIMARY_KEY: &str = "id";
pub const DEFAULT_DATA_DIR_NAME: &str = ".searchx-data";
pub const DEFAULT_MAX_FILE_BYTES: u64 = 50 * 1024 * 1024;
pub const VECTOR_EMBEDDER_NAME: &str = "default";
pub const VECTOR_DIMENSIONS: usize = 1536;
const VECTOR_STORE_BACKEND: VectorStoreBackend = VectorStoreBackend::Arroy;
const SEARCHABLE_FIELDS: [&str; 4] = ["file_name", "path", "contents", "extension"];
const PROGRESS_POLL_INTERVAL: Duration = Duration::from_millis(250);
const INDEX_EVENT_CHANNEL_CAPACITY: usize = 32;
const INDEX_BATCH_DOC_LIMIT: usize = 128;
const INDEX_BATCH_DELETE_LIMIT: usize = 512;
const INDEX_BATCH_BYTE_LIMIT: usize = 8 * 1024 * 1024;
const DEFAULT_IGNORE_RULES: &[&str] = &[
    ".git/",
    "node_modules/",
    "target/",
    "dist/",
    "build/",
    ".next/",
    ".turbo/",
    ".cache/",
    "coverage/",
    "__pycache__/",
    ".venv/",
    "venv/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
];

#[derive(Debug, Clone)]
pub struct ScanOptions {
    pub rebuild: bool,
    pub max_file_bytes: u64,
    pub ignore_rules: Vec<String>,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            rebuild: false,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            ignore_rules: default_ignore_rules(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SyncRequest {
    pub root: PathBuf,
    pub data_dir: PathBuf,
    pub options: ScanOptions,
}

impl SyncRequest {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            data_dir: PathBuf::from(DEFAULT_DATA_DIR_NAME),
            options: ScanOptions::default(),
        }
    }

    pub fn with_data_dir(mut self, data_dir: impl Into<PathBuf>) -> Self {
        self.data_dir = data_dir.into();
        self
    }

    pub fn with_options(mut self, options: ScanOptions) -> Self {
        self.options = options;
        self
    }
}

#[derive(Debug, Clone)]
pub enum SyncProgress {
    Rebuilding { reason: String },
    Indexing { path: String },
    ScanError(ScanError),
}

pub struct SyncIndexResult {
    pub root: PathBuf,
    pub data_paths: DataPaths,
    pub index: Index,
    pub stats: SyncStats,
    pub rebuild_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub rank: usize,
    pub path: String,
    pub document: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct SearchResults {
    pub query: String,
    pub candidate_count: u64,
    pub hits: Vec<SearchHit>,
}

pub fn default_ignore_rules() -> Vec<String> {
    DEFAULT_IGNORE_RULES
        .iter()
        .map(|rule| (*rule).to_string())
        .collect()
}

#[derive(Debug, Clone)]
pub struct DataPaths {
    pub base: PathBuf,
    pub index: PathBuf,
    pub manifest: PathBuf,
    pub incomplete_marker: PathBuf,
}

#[derive(Debug)]
pub struct OptionalCell<T> {
    value: Mutex<Option<T>>,
}

impl<T> Default for OptionalCell<T> {
    fn default() -> Self {
        Self {
            value: Mutex::new(None),
        }
    }
}

impl<T> OptionalCell<T> {
    pub fn new() -> Self {
        Self {
            value: Mutex::new(None),
        }
    }

    pub fn set(&self, value: T) {
        *lock_unpoisoned(&self.value) = Some(value);
    }

    pub fn clear(&self) {
        *lock_unpoisoned(&self.value) = None;
    }
}

impl<T: Clone> OptionalCell<T> {
    pub fn get(&self) -> Option<T> {
        lock_unpoisoned(&self.value).clone()
    }
}

#[derive(Debug, Default)]
pub struct ScanHook {
    current_file: OptionalCell<String>,
}

impl ScanHook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_current_file<S: Into<String>>(&self, path: S) {
        self.current_file.set(path.into());
    }

    pub fn clear_current_file(&self) {
        self.current_file.clear();
    }

    pub fn current_file(&self) -> Option<String> {
        self.current_file.get()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ScanErrorKind {
    Walk,
    Metadata,
    Read,
}

impl Display for ScanErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Walk => "walk",
            Self::Metadata => "metadata",
            Self::Read => "read",
        };
        write!(f, "{label}")
    }
}

#[derive(Debug, Clone)]
pub struct ScanError {
    pub kind: ScanErrorKind,
    pub path: Option<String>,
    pub message: String,
}

impl ScanError {
    fn walk(message: impl Into<String>) -> Self {
        Self {
            kind: ScanErrorKind::Walk,
            path: None,
            message: message.into(),
        }
    }

    fn metadata(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: ScanErrorKind::Metadata,
            path: Some(path.into()),
            message: message.into(),
        }
    }

    fn read(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: ScanErrorKind::Read,
            path: Some(path.into()),
            message: message.into(),
        }
    }
}

impl Display for ScanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(path) = &self.path {
            write!(f, "{} error for {}: {}", self.kind, path, self.message)
        } else {
            write!(f, "{} error: {}", self.kind, self.message)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq, Eq)]
pub struct Manifest {
    version: u32,
    root: String,
    files: BTreeMap<String, ManifestEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
struct ManifestEntry {
    size: u64,
    modified_secs: u64,
    modified_nanos: u32,
    state: FileState,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
enum FileState {
    Indexed,
    Skipped { reason: SkipReason },
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
enum SkipReason {
    TooLarge,
    Binary,
}

#[derive(Debug, Clone, Copy)]
struct FileFingerprint {
    size: u64,
    modified_secs: u64,
    modified_nanos: u32,
}

type IndexedVectors = BTreeMap<String, Option<Vec<f32>>>;

#[derive(Serialize, Debug)]
struct IndexedDocument {
    id: String,
    path: String,
    file_name: String,
    extension: Option<String>,
    contents: String,
    #[serde(rename = "_vectors")]
    vectors: IndexedVectors,
}

#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    pub scanned_files: u64,
    pub unchanged_indexed: u64,
    pub unchanged_skipped: u64,
    pub indexed_or_updated: u64,
    pub deleted_missing: u64,
    pub deleted_became_unsupported: u64,
    pub skipped_too_large: u64,
    pub skipped_binary: u64,
    pub read_errors: u64,
    pub walk_errors: u64,
}

#[derive(Debug)]
pub enum IndexEvent {
    Upsert(String),
    Delete(String),
}

#[derive(Debug)]
pub struct ScanPipeline {
    pub progress_manifest_db: Option<PathBuf>,
    pub error_sender: Option<mpsc::Sender<ScanError>>,
    pub event_sender: mpsc::SyncSender<IndexEvent>,
    pub cancel_flag: Arc<AtomicBool>,
}

#[derive(Debug)]
pub struct ManifestLoad {
    pub manifest: Manifest,
    pub rebuild_reason: Option<String>,
}

impl Manifest {
    pub fn new(root: &Path) -> Self {
        Self {
            version: MANIFEST_VERSION,
            root: root.display().to_string(),
            files: BTreeMap::new(),
        }
    }
}

impl ManifestEntry {
    fn from_fingerprint(fingerprint: FileFingerprint, state: FileState) -> Self {
        Self {
            size: fingerprint.size,
            modified_secs: fingerprint.modified_secs,
            modified_nanos: fingerprint.modified_nanos,
            state,
        }
    }

    fn matches(&self, fingerprint: FileFingerprint) -> bool {
        self.size == fingerprint.size
            && self.modified_secs == fingerprint.modified_secs
            && self.modified_nanos == fingerprint.modified_nanos
    }

    fn is_indexed(&self) -> bool {
        matches!(self.state, FileState::Indexed)
    }
}

impl FileFingerprint {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .unwrap_or_default();

        Self {
            size: metadata.len(),
            modified_secs: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
        }
    }
}

impl SyncStats {
    pub fn deleted_total(&self) -> u64 {
        self.deleted_missing + self.deleted_became_unsupported
    }
}

struct ManifestWorkingSet {
    conn: Connection,
}

impl ManifestWorkingSet {
    fn open(path: &Path) -> Result<Self, Box<dyn Error>> {
        let conn = open_manifest_connection(path)?;
        clear_working_manifest(&conn)?;
        Ok(Self { conn })
    }

    fn update_entry(
        &self,
        relative_path: &str,
        entry: &ManifestEntry,
    ) -> Result<(), Box<dyn Error>> {
        let (state, skip_reason) = file_state_to_db(&entry.state);
        self.conn.execute(
            "INSERT INTO working_manifest_files \
                (path, size, modified_secs, modified_nanos, state, skip_reason) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(path) DO UPDATE SET
                size = excluded.size,
                modified_secs = excluded.modified_secs,
                modified_nanos = excluded.modified_nanos,
                state = excluded.state,
                skip_reason = excluded.skip_reason",
            params![
                relative_path,
                i64_from_u64(entry.size, "size")?,
                i64_from_u64(entry.modified_secs, "modified_secs")?,
                i64::from(entry.modified_nanos),
                state,
                skip_reason,
            ],
        )?;
        Ok(())
    }
}

fn open_manifest_connection(path: &Path) -> Result<Connection, Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let conn = Connection::open(path)?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         CREATE TABLE IF NOT EXISTS manifest_info (
             singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
             version INTEGER NOT NULL,
             root TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS manifest_files (
             path TEXT PRIMARY KEY,
             size INTEGER NOT NULL,
             modified_secs INTEGER NOT NULL,
             modified_nanos INTEGER NOT NULL,
             state TEXT NOT NULL,
             skip_reason TEXT
         );
         CREATE TABLE IF NOT EXISTS working_manifest_files (
             path TEXT PRIMARY KEY,
             size INTEGER NOT NULL,
             modified_secs INTEGER NOT NULL,
             modified_nanos INTEGER NOT NULL,
             state TEXT NOT NULL,
             skip_reason TEXT
         );",
    )?;
    Ok(conn)
}

fn clear_working_manifest(conn: &Connection) -> Result<(), Box<dyn Error>> {
    conn.execute("DELETE FROM working_manifest_files", [])?;
    Ok(())
}

fn load_manifest_info(conn: &Connection) -> Result<Option<(u32, String)>, Box<dyn Error>> {
    let info = conn
        .query_row(
            "SELECT version, root FROM manifest_info WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()?;
    Ok(info)
}

fn load_manifest_files(
    conn: &Connection,
) -> Result<BTreeMap<String, ManifestEntry>, Box<dyn Error>> {
    let mut statement = conn.prepare(
        "SELECT path, size, modified_secs, modified_nanos, state, skip_reason
         FROM manifest_files",
    )?;
    let mut rows = statement.query([])?;
    let mut files = BTreeMap::new();

    while let Some(row) = rows.next()? {
        let path = row.get::<_, String>(0)?;
        let state_name = row.get::<_, String>(4)?;
        let skip_reason = row.get::<_, Option<String>>(5)?;
        let state = file_state_from_db(&state_name, skip_reason.as_deref())?;
        files.insert(
            path,
            ManifestEntry {
                size: u64_from_i64(row.get::<_, i64>(1)?, "size")?,
                modified_secs: u64_from_i64(row.get::<_, i64>(2)?, "modified_secs")?,
                modified_nanos: u32::try_from(row.get::<_, i64>(3)?)?,
                state,
            },
        );
    }

    Ok(files)
}

fn file_state_to_db(state: &FileState) -> (&'static str, Option<&'static str>) {
    match state {
        FileState::Indexed => ("indexed", None),
        FileState::Skipped {
            reason: SkipReason::TooLarge,
        } => ("skipped", Some("too_large")),
        FileState::Skipped {
            reason: SkipReason::Binary,
        } => ("skipped", Some("binary")),
    }
}

fn file_state_from_db(state: &str, skip_reason: Option<&str>) -> Result<FileState, Box<dyn Error>> {
    match (state, skip_reason) {
        ("indexed", None) => Ok(FileState::Indexed),
        ("skipped", Some("too_large")) => Ok(FileState::Skipped {
            reason: SkipReason::TooLarge,
        }),
        ("skipped", Some("binary")) => Ok(FileState::Skipped {
            reason: SkipReason::Binary,
        }),
        ("indexed", Some(reason)) => {
            Err(format!("indexed entry cannot have skip reason {reason}").into())
        }
        ("skipped", None) => Err("skipped entry is missing a skip reason".into()),
        (state, reason) => {
            Err(format!("invalid manifest state {state:?} with reason {reason:?}").into())
        }
    }
}

fn i64_from_u64(value: u64, field: &str) -> Result<i64, Box<dyn Error>> {
    i64::try_from(value)
        .map_err(|_| format!("manifest {field} value {value} exceeds sqlite INTEGER range").into())
}

fn u64_from_i64(value: i64, field: &str) -> Result<u64, Box<dyn Error>> {
    u64::try_from(value)
        .map_err(|_| format!("manifest {field} value {value} is negative and invalid").into())
}

pub fn data_paths<P: AsRef<Path>>(data_dir: P) -> Result<DataPaths, Box<dyn Error>> {
    let base = env::current_dir()?.join(data_dir);
    Ok(DataPaths {
        index: base.join("index"),
        manifest: base.join(MANIFEST_FILE_NAME),
        incomplete_marker: base.join(INCOMPLETE_FILE_NAME),
        base,
    })
}

pub fn mark_index_incomplete(data_paths: &DataPaths) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(&data_paths.base)?;
    fs::write(&data_paths.incomplete_marker, b"incomplete\n")?;
    Ok(())
}

pub fn clear_index_incomplete(data_paths: &DataPaths) -> Result<(), Box<dyn Error>> {
    if data_paths.incomplete_marker.exists() {
        fs::remove_file(&data_paths.incomplete_marker)?;
    }
    Ok(())
}

pub fn load_manifest(
    data_paths: &DataPaths,
    root: &Path,
    force_rebuild: bool,
) -> Result<ManifestLoad, Box<dyn Error>> {
    let empty_manifest = Manifest::new(root);

    if force_rebuild {
        return Ok(ManifestLoad {
            manifest: empty_manifest,
            rebuild_reason: Some("forced by --rebuild".to_string()),
        });
    }

    if data_paths.incomplete_marker.exists() {
        return Ok(ManifestLoad {
            manifest: empty_manifest,
            rebuild_reason: Some("previous indexing run did not complete cleanly".to_string()),
        });
    }

    if !data_paths.manifest.exists() {
        let existing_index = data_paths.index.join("data.mdb").exists();
        return Ok(ManifestLoad {
            manifest: empty_manifest,
            rebuild_reason: existing_index.then(|| "manifest is missing".to_string()),
        });
    }

    let conn = match open_manifest_connection(&data_paths.manifest) {
        Ok(conn) => conn,
        Err(error) => {
            return Ok(ManifestLoad {
                manifest: empty_manifest,
                rebuild_reason: Some(format!("manifest could not be opened: {error}")),
            });
        }
    };

    let Some((version, manifest_root)) = (match load_manifest_info(&conn) {
        Ok(info) => info,
        Err(error) => {
            return Ok(ManifestLoad {
                manifest: empty_manifest,
                rebuild_reason: Some(format!("manifest could not be read: {error}")),
            });
        }
    }) else {
        let existing_index = data_paths.index.join("data.mdb").exists();
        return Ok(ManifestLoad {
            manifest: empty_manifest,
            rebuild_reason: existing_index.then(|| "manifest is missing".to_string()),
        });
    };

    if version != MANIFEST_VERSION {
        return Ok(ManifestLoad {
            manifest: empty_manifest,
            rebuild_reason: Some(format!(
                "manifest version {} does not match {}",
                version, MANIFEST_VERSION
            )),
        });
    }

    let root_display = root.display().to_string();
    if manifest_root != root_display {
        return Ok(ManifestLoad {
            manifest: empty_manifest,
            rebuild_reason: Some(format!(
                "manifest root {} does not match {}",
                manifest_root, root_display
            )),
        });
    }

    let files = match load_manifest_files(&conn) {
        Ok(files) => files,
        Err(error) => {
            return Ok(ManifestLoad {
                manifest: empty_manifest,
                rebuild_reason: Some(format!("manifest could not be read: {error}")),
            });
        }
    };

    Ok(ManifestLoad {
        manifest: Manifest {
            version,
            root: manifest_root,
            files,
        },
        rebuild_reason: None,
    })
}

pub fn reset_data_dir(data_paths: &DataPaths) -> Result<(), Box<dyn Error>> {
    if data_paths.base.exists() {
        fs::remove_dir_all(&data_paths.base)?;
    }
    fs::create_dir_all(&data_paths.index)?;
    Ok(())
}

pub fn configure_index(
    index: &Index,
    indexer_config: &IndexerConfig,
) -> Result<(), Box<dyn Error>> {
    let desired_searchable_fields = SEARCHABLE_FIELDS
        .iter()
        .map(|field| (*field).to_string())
        .collect::<Vec<_>>();

    let mut embedders = BTreeMap::new();
    embedders.insert(
        VECTOR_EMBEDDER_NAME.to_string(),
        Setting::Set(EmbeddingSettings {
            source: Setting::Set(EmbedderSource::UserProvided),
            dimensions: Setting::Set(VECTOR_DIMENSIONS),
            ..EmbeddingSettings::default()
        }),
    );

    let mut wtxn = index.write_txn()?;
    let mut settings = Settings::new(&mut wtxn, index, indexer_config);
    settings.set_primary_key(PRIMARY_KEY.to_string());
    settings.set_searchable_fields(desired_searchable_fields);
    settings.set_embedder_settings(embedders);
    settings.set_vector_store(VECTOR_STORE_BACKEND);
    settings.execute(
        &|| false,
        &Progress::default(),
        &IpPolicy::danger_always_allow(),
        Arc::<EmbedderStats>::default(),
    )?;
    wtxn.commit()?;

    Ok(())
}

pub fn generate_document_vector(_relative_path: &str, _contents: &str) -> Option<Vec<f32>> {
    None
}

fn document_vectors(relative_path: &str, contents: &str) -> IndexedVectors {
    let mut vectors = BTreeMap::new();
    vectors.insert(
        VECTOR_EMBEDDER_NAME.to_string(),
        generate_document_vector(relative_path, contents),
    );
    vectors
}

pub fn scan_root(
    options: &ScanOptions,
    root: &Path,
    data_dir: &Path,
    previous: &Manifest,
    scan_hook: Option<Arc<ScanHook>>,
    pipeline: ScanPipeline,
) -> Result<SyncStats, Box<dyn Error>> {
    let mut seen_paths = HashSet::with_capacity(previous.files.len().saturating_mul(2));
    let mut stats = SyncStats::default();
    let progress_manifest = pipeline
        .progress_manifest_db
        .as_deref()
        .map(ManifestWorkingSet::open)
        .transpose()?;

    let mut walker_builder = WalkBuilder::new(root);
    walker_builder
        .hidden(false)
        .require_git(false)
        .current_dir(root);

    let custom_ignore_file = build_ignore_file(data_dir, &options.ignore_rules)?;
    if let Some(ignore_file) = &custom_ignore_file
        && let Some(error) = walker_builder.add_ignore(ignore_file.path())
    {
        return Err(Box::new(error));
    }

    let data_dir = data_dir.to_path_buf();
    let walker = walker_builder
        .filter_entry(move |entry| should_walk_entry(entry, &data_dir))
        .build();

    for result in walker {
        if pipeline.cancel_flag.load(Ordering::Relaxed) {
            return Err("scan canceled".into());
        }

        let entry = match result {
            Ok(entry) => entry,
            Err(error) => {
                stats.walk_errors += 1;
                report_scan_error(
                    pipeline.error_sender.as_ref(),
                    ScanError::walk(error.to_string()),
                );
                continue;
            }
        };

        let Some(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }

        let path = entry.path();
        let relative_path = normalize_relative_path(path, root)?;
        seen_paths.insert(relative_path.clone());
        if let Some(scan_hook) = scan_hook.as_ref() {
            scan_hook.set_current_file(relative_path.clone());
        }
        stats.scanned_files += 1;

        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) => {
                stats.read_errors += 1;
                report_scan_error(
                    pipeline.error_sender.as_ref(),
                    ScanError::metadata(path.display().to_string(), error.to_string()),
                );
                if let Some(previous_entry) = previous.files.get(&relative_path) {
                    update_progress_manifest_entry(
                        progress_manifest.as_ref(),
                        &relative_path,
                        previous_entry,
                    )?;
                }
                continue;
            }
        };

        let fingerprint = FileFingerprint::from_metadata(&metadata);
        if let Some(previous_entry) = previous.files.get(&relative_path)
            && previous_entry.matches(fingerprint)
        {
            update_progress_manifest_entry(
                progress_manifest.as_ref(),
                &relative_path,
                previous_entry,
            )?;
            if previous_entry.is_indexed() {
                stats.unchanged_indexed += 1;
            } else {
                stats.unchanged_skipped += 1;
            }
            continue;
        }

        if metadata.len() > options.max_file_bytes {
            handle_unsupported_file(
                &relative_path,
                fingerprint,
                SkipReason::TooLarge,
                previous,
                progress_manifest.as_ref(),
                &pipeline.event_sender,
                &pipeline.cancel_flag,
                &mut stats,
            )?;
            continue;
        }

        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                stats.read_errors += 1;
                report_scan_error(
                    pipeline.error_sender.as_ref(),
                    ScanError::read(path.display().to_string(), error.to_string()),
                );
                if let Some(previous_entry) = previous.files.get(&relative_path) {
                    update_progress_manifest_entry(
                        progress_manifest.as_ref(),
                        &relative_path,
                        previous_entry,
                    )?;
                }
                continue;
            }
        };

        if bytes.contains(&0) {
            handle_unsupported_file(
                &relative_path,
                fingerprint,
                SkipReason::Binary,
                previous,
                progress_manifest.as_ref(),
                &pipeline.event_sender,
                &pipeline.cancel_flag,
                &mut stats,
            )?;
            continue;
        }

        let contents = match String::from_utf8(bytes) {
            Ok(contents) => contents,
            Err(_) => {
                handle_unsupported_file(
                    &relative_path,
                    fingerprint,
                    SkipReason::Binary,
                    previous,
                    progress_manifest.as_ref(),
                    &pipeline.event_sender,
                    &pipeline.cancel_flag,
                    &mut stats,
                )?;
                continue;
            }
        };

        let vectors = document_vectors(&relative_path, &contents);
        let document = IndexedDocument {
            id: document_id_for_path(&relative_path),
            path: relative_path.clone(),
            file_name: path
                .file_name()
                .and_then(OsStr::to_str)
                .unwrap_or_default()
                .to_string(),
            extension: path.extension().and_then(OsStr::to_str).map(str::to_string),
            contents,
            vectors,
        };
        let entry = ManifestEntry::from_fingerprint(fingerprint, FileState::Indexed);
        update_progress_manifest_entry(progress_manifest.as_ref(), &relative_path, &entry)?;
        send_index_event(
            &pipeline.event_sender,
            &pipeline.cancel_flag,
            IndexEvent::Upsert(serde_json::to_string(&document)?),
        )?;
        stats.indexed_or_updated += 1;
    }

    for (path, entry) in &previous.files {
        if entry.is_indexed() && !seen_paths.contains(path) {
            send_index_event(
                &pipeline.event_sender,
                &pipeline.cancel_flag,
                IndexEvent::Delete(document_id_for_path(path)),
            )?;
            stats.deleted_missing += 1;
        }
    }

    if let Some(scan_hook) = scan_hook.as_ref() {
        scan_hook.clear_current_file();
    }

    Ok(stats)
}

#[allow(clippy::too_many_arguments)]
fn handle_unsupported_file(
    relative_path: &str,
    fingerprint: FileFingerprint,
    reason: SkipReason,
    previous: &Manifest,
    progress_manifest: Option<&ManifestWorkingSet>,
    event_sender: &mpsc::SyncSender<IndexEvent>,
    cancel_flag: &AtomicBool,
    stats: &mut SyncStats,
) -> Result<(), Box<dyn Error>> {
    match reason {
        SkipReason::TooLarge => stats.skipped_too_large += 1,
        SkipReason::Binary => stats.skipped_binary += 1,
    }

    if previous
        .files
        .get(relative_path)
        .is_some_and(ManifestEntry::is_indexed)
    {
        send_index_event(
            event_sender,
            cancel_flag,
            IndexEvent::Delete(document_id_for_path(relative_path)),
        )?;
        stats.deleted_became_unsupported += 1;
    }

    let entry = ManifestEntry::from_fingerprint(fingerprint, FileState::Skipped { reason });
    update_progress_manifest_entry(progress_manifest, relative_path, &entry)
}

fn send_index_event(
    event_sender: &mpsc::SyncSender<IndexEvent>,
    cancel_flag: &AtomicBool,
    event: IndexEvent,
) -> Result<(), Box<dyn Error>> {
    if cancel_flag.load(Ordering::Relaxed) {
        return Err("indexing canceled".into());
    }

    event_sender
        .send(event)
        .map_err(|_| -> Box<dyn Error> { "indexing pipeline disconnected".into() })
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

fn update_progress_manifest_entry(
    progress_manifest: Option<&ManifestWorkingSet>,
    relative_path: &str,
    entry: &ManifestEntry,
) -> Result<(), Box<dyn Error>> {
    if let Some(progress_manifest) = progress_manifest {
        progress_manifest.update_entry(relative_path, entry)?;
    }
    Ok(())
}

fn report_scan_error(error_sender: Option<&mpsc::Sender<ScanError>>, error: ScanError) {
    if let Some(error_sender) = error_sender {
        let _ = error_sender.send(error);
    } else {
        eprintln!("{error}");
    }
}

fn build_ignore_file(
    data_dir: &Path,
    ignore_rules: &[String],
) -> Result<Option<NamedTempFile>, Box<dyn Error>> {
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

fn should_walk_entry(entry: &DirEntry, data_dir: &Path) -> bool {
    !entry.path().starts_with(data_dir)
}

fn normalize_relative_path(path: &Path, root: &Path) -> Result<String, Box<dyn Error>> {
    let relative = path.strip_prefix(root)?;
    Ok(relative
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/"))
}

fn document_id_for_path(relative_path: &str) -> String {
    blake3::hash(relative_path.as_bytes()).to_hex().to_string()
}

pub fn apply_index_batch(
    index: &Index,
    indexer_config: &IndexerConfig,
    temp_dir: &Path,
    upserts: &[String],
    deleted_ids: &[String],
) -> Result<(), Box<dyn Error>> {
    if upserts.is_empty() && deleted_ids.is_empty() {
        return Ok(());
    }

    let mut updates_file = if upserts.is_empty() {
        None
    } else {
        let mut file = NamedTempFile::new_in(temp_dir)?;
        {
            let mut writer = BufWriter::new(file.as_file_mut());
            for line in upserts {
                writer.write_all(line.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
        }
        Some(file)
    };

    let mut wtxn = index.write_txn()?;
    let rtxn = index.read_txn()?;
    let db_fields_ids_map = index.fields_ids_map(&rtxn)?;
    let mut new_fields_ids_map = db_fields_ids_map.clone();
    let ip_policy = IpPolicy::danger_always_allow();
    let embedders = milli::update::InnerIndexSettings::from_index(index, &rtxn, &ip_policy, None)?
        .runtime_embedders;

    let mut operations = indexer::IndexOperations::new();
    let mmap = if let Some(file) = updates_file.as_ref() {
        Some(unsafe { Mmap::map(file.as_file())? })
    } else {
        None
    };

    if let Some(mmap) = &mmap {
        operations.replace_documents(mmap, MissingDocumentPolicy::default())?;
    }

    let deleted_refs = deleted_ids.iter().map(String::as_str).collect::<Vec<_>>();
    if !deleted_refs.is_empty() {
        operations.delete_documents_by_external_ids(&deleted_refs);
    }

    let indexer_alloc = Bump::new();
    let (document_changes, operation_stats, primary_key) = operations.into_changes(
        &indexer_alloc,
        index,
        &rtxn,
        None,
        &mut new_fields_ids_map,
        &|| false,
        Progress::default(),
        None,
    )?;

    if let Some(error) = operation_stats.into_iter().find_map(|stat| stat.error) {
        return Err(Box::new(error));
    }

    indexer::index(
        &mut wtxn,
        index,
        &indexer_config.thread_pool,
        indexer_config.grenad_parameters(),
        &db_fields_ids_map,
        new_fields_ids_map,
        primary_key,
        &document_changes,
        embedders,
        &|| false,
        &Progress::default(),
        &ip_policy,
        &EmbedderStats::default(),
    )?;
    wtxn.commit()?;
    drop(mmap);
    drop(updates_file.take());

    Ok(())
}

pub fn commit_working_manifest(path: &Path, root: &Path) -> Result<(), Box<dyn Error>> {
    let mut conn = open_manifest_connection(path)?;
    let tx = conn.transaction()?;
    tx.execute(
        "INSERT INTO manifest_info (singleton, version, root)
         VALUES (1, ?1, ?2)
         ON CONFLICT(singleton) DO UPDATE SET
             version = excluded.version,
             root = excluded.root",
        params![MANIFEST_VERSION, root.display().to_string()],
    )?;
    tx.execute("DELETE FROM manifest_files", [])?;
    tx.execute(
        "INSERT INTO manifest_files
            (path, size, modified_secs, modified_nanos, state, skip_reason)
         SELECT path, size, modified_secs, modified_nanos, state, skip_reason
         FROM working_manifest_files",
        [],
    )?;
    tx.execute("DELETE FROM working_manifest_files", [])?;
    tx.commit()?;
    Ok(())
}

pub fn discard_working_manifest(path: &Path) -> Result<(), Box<dyn Error>> {
    if !path.exists() {
        return Ok(());
    }

    let conn = open_manifest_connection(path)?;
    clear_working_manifest(&conn)
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

fn drain_scan_errors<F>(error_rx: &mpsc::Receiver<ScanError>, on_progress: &mut F)
where
    F: FnMut(SyncProgress),
{
    while let Ok(error) = error_rx.try_recv() {
        on_progress(SyncProgress::ScanError(error));
    }
}

fn discard_working_manifest_quietly(path: &Path) {
    let _ = discard_working_manifest(path);
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

fn stream_scan_into_index<F>(
    index: &Index,
    indexer_config: &IndexerConfig,
    options: &ScanOptions,
    root: &Path,
    data_paths: &DataPaths,
    previous_manifest: Manifest,
    scan_hook: Arc<ScanHook>,
    on_progress: &mut F,
) -> Result<SyncStats, Box<dyn Error>>
where
    F: FnMut(SyncProgress),
{
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
        drain_scan_errors(&scan_error_rx, on_progress);
        report_sync_progress(&scan_hook, &mut last_reported_file, on_progress);

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
                    discard_working_manifest_quietly(&data_paths.manifest);
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
                    discard_working_manifest_quietly(&data_paths.manifest);
                    return Err(error);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                break match scan_handle.join() {
                    Ok(Ok(summary)) => summary,
                    Ok(Err(error)) => {
                        discard_working_manifest_quietly(&data_paths.manifest);
                        return Err(error.into());
                    }
                    Err(payload) => {
                        discard_working_manifest_quietly(&data_paths.manifest);
                        return Err(
                            format!("scan thread panicked: {}", panic_message(payload)).into()
                        );
                    }
                };
            }
        }
    };

    drain_scan_errors(&scan_error_rx, on_progress);

    if let Err(error) = flush_index_batch(
        index,
        indexer_config,
        data_paths,
        &mut pending_upserts,
        &mut pending_deleted_ids,
        &mut pending_bytes,
    ) {
        discard_working_manifest_quietly(&data_paths.manifest);
        return Err(error);
    }

    Ok(scan_result)
}

pub fn sync_index(request: &SyncRequest) -> Result<SyncIndexResult, Box<dyn Error>> {
    sync_index_with_progress(request, |_| {})
}

pub fn sync_index_with_progress<F>(
    request: &SyncRequest,
    mut on_progress: F,
) -> Result<SyncIndexResult, Box<dyn Error>>
where
    F: FnMut(SyncProgress),
{
    let root = fs::canonicalize(&request.root)?;
    let data_paths = data_paths(&request.data_dir)?;

    let ManifestLoad {
        manifest,
        rebuild_reason,
    } = load_manifest(&data_paths, &root, request.options.rebuild)?;

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
        &index,
        &indexer_config,
        &request.options,
        &root,
        &data_paths,
        manifest,
        scan_hook,
        &mut on_progress,
    )?;

    if let Err(error) = commit_working_manifest(&data_paths.manifest, &root) {
        discard_working_manifest_quietly(&data_paths.manifest);
        return Err(error);
    }

    clear_index_incomplete(&data_paths)?;

    Ok(SyncIndexResult {
        root,
        data_paths,
        index,
        stats,
        rebuild_reason,
    })
}

pub fn search_index(
    index: &Index,
    query: &str,
    limit: usize,
) -> Result<SearchResults, Box<dyn Error>> {
    let rtxn = index.read_txn()?;
    let progress = Progress::default();
    let mut search = milli::Search::new(&rtxn, index, &progress);
    search.query(query);
    search.limit(limit);

    let result = search.execute()?;
    let candidate_count = result.candidates.len();
    let fields_ids_map = index.fields_ids_map(&rtxn)?;
    let mut hits = Vec::new();

    for (rank, (_docid, obkv)) in index
        .documents(&rtxn, result.documents_ids)?
        .into_iter()
        .enumerate()
    {
        let document = all_obkv_to_json(obkv, &fields_ids_map)?;
        let path = document
            .get("path")
            .and_then(|value| value.as_str())
            .unwrap_or("<unknown>")
            .to_string();
        hits.push(SearchHit {
            rank: rank + 1,
            path,
            document: serde_json::Value::Object(document),
        });
    }

    Ok(SearchResults {
        query: query.to_string(),
        candidate_count,
        hits,
    })
}

pub fn new_heed_options() -> EnvOpenOptions<WithoutTls> {
    let mut heed_options = milli::heed::EnvOpenOptions::new();
    heed_options.map_size(DEFAULT_MAP_SIZE_BYTES);
    heed_options.read_txn_without_tls()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, mpsc};
    use tempfile::TempDir;

    fn empty_manifest(root: &Path) -> Manifest {
        Manifest {
            version: MANIFEST_VERSION,
            root: root.display().to_string(),
            files: BTreeMap::new(),
        }
    }

    fn setup_scan_dirs() -> (TempDir, PathBuf, PathBuf) {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path().join("root");
        let data_dir = temp_dir.path().join(".searchx-data");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&data_dir).unwrap();
        (temp_dir, root, data_dir)
    }

    fn streamed_indexed_paths(
        options: &ScanOptions,
        root: &Path,
        data_dir: &Path,
    ) -> BTreeSet<String> {
        let (event_tx, event_rx) = mpsc::sync_channel(32);
        scan_root(
            options,
            root,
            data_dir,
            &empty_manifest(root),
            None,
            ScanPipeline {
                progress_manifest_db: None,
                error_sender: None,
                event_sender: event_tx,
                cancel_flag: Arc::new(AtomicBool::new(false)),
            },
        )
        .unwrap();

        event_rx
            .into_iter()
            .filter_map(|event| match event {
                IndexEvent::Upsert(document) => {
                    let value: serde_json::Value = serde_json::from_str(&document).unwrap();
                    Some(value["path"].as_str().unwrap().to_string())
                }
                IndexEvent::Delete(_) => None,
            })
            .collect()
    }

    #[test]
    fn scan_root_respects_gitignore_style_ignore_rules() {
        let (_temp_dir, root, data_dir) = setup_scan_dirs();
        fs::write(root.join("keep.log"), "keep me").unwrap();
        fs::write(root.join("skip.log"), "skip me").unwrap();
        fs::create_dir_all(root.join("build")).unwrap();
        fs::write(root.join("build").join("artifact.txt"), "artifact").unwrap();

        let options = ScanOptions {
            rebuild: false,
            max_file_bytes: u64::MAX,
            ignore_rules: vec![
                "*.log".to_string(),
                "!keep.log".to_string(),
                "build/".to_string(),
            ],
        };

        let indexed = streamed_indexed_paths(&options, &root, &data_dir);

        assert_eq!(indexed, BTreeSet::from(["keep.log".to_string()]));
    }

    #[test]
    fn scan_root_applies_dot_gitignore_rules_outside_git_repos() {
        let (_temp_dir, root, data_dir) = setup_scan_dirs();
        fs::write(root.join(".gitignore"), "*.tmp\n").unwrap();
        fs::write(root.join("visible.txt"), "visible").unwrap();
        fs::write(root.join("ignored.tmp"), "ignored").unwrap();
        fs::write(root.join(".env"), "SECRET=1\n").unwrap();

        let options = ScanOptions {
            rebuild: false,
            max_file_bytes: u64::MAX,
            ignore_rules: Vec::new(),
        };

        let indexed = streamed_indexed_paths(&options, &root, &data_dir);

        assert!(indexed.contains("visible.txt"));
        assert!(indexed.contains(".env"));
        assert!(!indexed.contains("ignored.tmp"));
    }

    #[test]
    fn load_manifest_requests_rebuild_when_previous_run_was_incomplete() {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path().join("root");
        let data_paths = DataPaths {
            base: temp_dir.path().join(".searchx-data"),
            index: temp_dir.path().join(".searchx-data").join("index"),
            manifest: temp_dir
                .path()
                .join(".searchx-data")
                .join(MANIFEST_FILE_NAME),
            incomplete_marker: temp_dir
                .path()
                .join(".searchx-data")
                .join(INCOMPLETE_FILE_NAME),
        };
        fs::create_dir_all(&root).unwrap();
        mark_index_incomplete(&data_paths).unwrap();

        let loaded = load_manifest(&data_paths, &root, false).unwrap();
        assert_eq!(
            loaded.rebuild_reason,
            Some("previous indexing run did not complete cleanly".to_string())
        );
    }

    #[test]
    fn manifest_persists_in_sqlite_without_exposing_uncommitted_working_state() {
        let temp_dir = tempfile::tempdir().unwrap();
        let root = temp_dir.path().join("root");
        let base = temp_dir.path().join(".searchx-data");
        let data_paths = DataPaths {
            base: base.clone(),
            index: base.join("index"),
            manifest: base.join(MANIFEST_FILE_NAME),
            incomplete_marker: base.join(INCOMPLETE_FILE_NAME),
        };
        fs::create_dir_all(&root).unwrap();

        let expected = Manifest {
            version: MANIFEST_VERSION,
            root: root.display().to_string(),
            files: BTreeMap::from([
                (
                    "indexed.txt".to_string(),
                    ManifestEntry::from_fingerprint(
                        FileFingerprint {
                            size: 12,
                            modified_secs: 34,
                            modified_nanos: 56,
                        },
                        FileState::Indexed,
                    ),
                ),
                (
                    "too-large.bin".to_string(),
                    ManifestEntry::from_fingerprint(
                        FileFingerprint {
                            size: 78,
                            modified_secs: 90,
                            modified_nanos: 12,
                        },
                        FileState::Skipped {
                            reason: SkipReason::TooLarge,
                        },
                    ),
                ),
            ]),
        };

        let working = ManifestWorkingSet::open(&data_paths.manifest).unwrap();
        for (path, entry) in &expected.files {
            working.update_entry(path, entry).unwrap();
        }
        commit_working_manifest(&data_paths.manifest, &root).unwrap();

        let loaded = load_manifest(&data_paths, &root, false).unwrap();
        assert!(loaded.rebuild_reason.is_none());
        assert_eq!(loaded.manifest, expected);

        let stale_working = ManifestWorkingSet::open(&data_paths.manifest).unwrap();
        let stale_entry = ManifestEntry::from_fingerprint(
            FileFingerprint {
                size: 1,
                modified_secs: 2,
                modified_nanos: 3,
            },
            FileState::Indexed,
        );
        stale_working
            .update_entry("stale.txt", &stale_entry)
            .unwrap();

        let loaded_again = load_manifest(&data_paths, &root, false).unwrap();
        assert_eq!(loaded_again.manifest, expected);

        discard_working_manifest(&data_paths.manifest).unwrap();
    }
}
