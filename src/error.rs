use std::io;
use std::num::TryFromIntError;
use std::path::StripPrefixError;
use thiserror::Error;

pub type SearchxResult<T> = Result<T, SearchxError>;

#[derive(Debug, Error)]
pub enum SearchxError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Ignore(#[from] ignore::Error),
    #[error(transparent)]
    Heed(#[from] milli::heed::Error),
    #[error(transparent)]
    Milli(Box<milli::Error>),
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
    #[error(transparent)]
    IntegerConversion(#[from] TryFromIntError),
    #[error(transparent)]
    StripPrefix(#[from] StripPrefixError),
    #[error("manifest {field} value {value} exceeds sqlite INTEGER range")]
    ManifestIntegerOverflow { field: &'static str, value: u64 },
    #[error("manifest {field} value {value} is negative and invalid")]
    ManifestNegativeValue { field: &'static str, value: i64 },
    #[error("indexed entry cannot have skip reason {reason}")]
    IndexedEntryWithSkipReason { reason: String },
    #[error("skipped entry is missing a skip reason")]
    MissingSkipReason,
    #[error("invalid manifest state {state:?} with reason {reason:?}")]
    InvalidManifestState {
        state: String,
        reason: Option<String>,
    },
    #[error("scan canceled")]
    ScanCanceled,
    #[error("indexing canceled")]
    IndexingCanceled,
    #[error("indexing pipeline disconnected")]
    IndexingPipelineDisconnected,
    #[error("scan thread panicked: {message}")]
    ScanThreadPanicked { message: String },
    #[error("embedding thread panicked: {message}")]
    EmbeddingThreadPanicked { message: String },
    #[error("embedding failed: {message}")]
    Embedding { message: String },
}

impl From<milli::Error> for SearchxError {
    fn from(error: milli::Error) -> Self {
        Self::Milli(Box::new(error))
    }
}
