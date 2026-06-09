mod api;
mod constants;
mod embedding;
mod error;
mod index;
mod manifest;
mod scan;
mod sync;

pub use api::{
    DataPaths, ManifestLoad, OptionalCell, ScanError, ScanErrorKind, ScanHook, ScanOptions,
    SearchHit, SearchResults, SyncIndexResult, SyncProgress, SyncRequest, SyncStats,
    default_ignore_rules,
};
pub use constants::{
    DEFAULT_DATA_DIR_NAME, DEFAULT_MAX_FILE_BYTES, VECTOR_DIMENSIONS, VECTOR_EMBEDDER_NAME,
};
pub use embedding::EmbeddingInput;
pub use error::{SearchxError, SearchxResult};
pub use index::{
    apply_index_batch, configure_index, generate_document_vector, new_heed_options, search_index,
    search_index_vector,
};
pub use manifest::{
    Manifest, clear_index_incomplete, commit_working_manifest, data_paths,
    discard_working_manifest, load_manifest, mark_index_incomplete, reset_data_dir,
};
pub use sync::{sync_index, sync_index_with_progress};

#[cfg(test)]
pub(crate) use constants::{INCOMPLETE_FILE_NAME, MANIFEST_FILE_NAME, MANIFEST_VERSION};
#[cfg(test)]
pub(crate) use index::{IndexedDocument, document_id_for_path, document_vectors};
#[cfg(test)]
pub(crate) use manifest::{
    FileFingerprint, FileState, ManifestEntry, ManifestWorkingSet, SkipReason,
    open_manifest_connection,
};
#[cfg(test)]
pub(crate) use scan::{IndexEvent, ScanPipeline, scan_root};

#[cfg(test)]
mod tests;
