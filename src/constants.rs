use milli::vector::VectorStoreBackend;
use std::time::Duration;

pub(crate) const MANIFEST_VERSION: u32 = 2;
pub(crate) const MANIFEST_FILE_NAME: &str = "manifest.sqlite3";
pub(crate) const INCOMPLETE_FILE_NAME: &str = "indexing-incomplete";
pub(crate) const DEFAULT_MAP_SIZE_BYTES: usize = 10 * 1024 * 1024 * 1024;
pub(crate) const PRIMARY_KEY: &str = "id";
pub const DEFAULT_DATA_DIR_NAME: &str = ".searchx-data";
pub const DEFAULT_MAX_FILE_BYTES: u64 = 50 * 1024 * 1024;
pub const VECTOR_EMBEDDER_NAME: &str = "default";
pub const VECTOR_DIMENSIONS: usize = 1536;
pub(crate) const VECTOR_STORE_BACKEND: VectorStoreBackend = VectorStoreBackend::Arroy;
pub(crate) const SEARCHABLE_FIELDS: [&str; 4] = ["file_name", "path", "contents", "extension"];
pub(crate) const PROGRESS_POLL_INTERVAL: Duration = Duration::from_millis(250);
pub(crate) const INDEX_EVENT_CHANNEL_CAPACITY: usize = 32;
pub(crate) const INDEX_BATCH_DOC_LIMIT: usize = 128;
pub(crate) const INDEX_BATCH_DELETE_LIMIT: usize = 512;
pub(crate) const INDEX_BATCH_BYTE_LIMIT: usize = 8 * 1024 * 1024;
pub(crate) const MANIFEST_BATCH_ENTRY_LIMIT: usize = 512;
pub(crate) const DEFAULT_IGNORE_RULES: &[&str] = &[
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
