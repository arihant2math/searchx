use crate::constants::{DEFAULT_DATA_DIR_NAME, DEFAULT_IGNORE_RULES, DEFAULT_MAX_FILE_BYTES};
use crate::manifest::Manifest;
use milli::Index;
use std::fmt::{self, Display, Formatter};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

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
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            data_dir: PathBuf::from(DEFAULT_DATA_DIR_NAME),
            options: ScanOptions::default(),
        }
    }

    #[must_use]
    pub fn with_data_dir(mut self, data_dir: impl Into<PathBuf>) -> Self {
        self.data_dir = data_dir.into();
        self
    }

    #[must_use]
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

#[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn get(&self) -> Option<T> {
        lock_unpoisoned(&self.value).clone()
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Debug, Default)]
pub struct ScanHook {
    current_file: OptionalCell<String>,
}

impl ScanHook {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_current_file<S: Into<String>>(&self, path: S) {
        self.current_file.set(path.into());
    }

    pub fn clear_current_file(&self) {
        self.current_file.clear();
    }

    #[must_use]
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
    pub(crate) fn walk(message: impl Into<String>) -> Self {
        Self {
            kind: ScanErrorKind::Walk,
            path: None,
            message: message.into(),
        }
    }

    pub(crate) fn metadata(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: ScanErrorKind::Metadata,
            path: Some(path.into()),
            message: message.into(),
        }
    }

    pub(crate) fn read(path: impl Into<String>, message: impl Into<String>) -> Self {
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

impl SyncStats {
    #[must_use]
    pub fn deleted_total(&self) -> u64 {
        self.deleted_missing + self.deleted_became_unsupported
    }
}

#[derive(Debug)]
pub struct ManifestLoad {
    pub manifest: Manifest,
    pub rebuild_reason: Option<String>,
    pub resume_from_incomplete: bool,
}
