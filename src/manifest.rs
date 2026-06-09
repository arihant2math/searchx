use crate::api::{DataPaths, ManifestLoad};
use crate::constants::{INCOMPLETE_FILE_NAME, MANIFEST_FILE_NAME, MANIFEST_VERSION};
use crate::error::{SearchxError, SearchxResult};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::UNIX_EPOCH;

#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub(crate) version: u32,
    pub(crate) root: String,
    pub(crate) files: BTreeMap<String, ManifestEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManifestEntry {
    pub(crate) size: u64,
    pub(crate) modified_secs: u64,
    pub(crate) modified_nanos: u32,
    pub(crate) state: FileState,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub(crate) enum FileState {
    Indexed,
    IndexedMetadata { reason: SkipReason },
    Skipped { reason: SkipReason },
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkipReason {
    TooLarge,
    Binary,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FileFingerprint {
    pub(crate) size: u64,
    pub(crate) modified_secs: u64,
    pub(crate) modified_nanos: u32,
}

pub(crate) struct ManifestWorkingSet {
    conn: Connection,
}

impl Manifest {
    #[must_use]
    pub fn new(root: &Path) -> Self {
        Self {
            version: MANIFEST_VERSION,
            root: root.display().to_string(),
            files: BTreeMap::new(),
        }
    }
}

impl ManifestEntry {
    pub(crate) fn from_fingerprint(fingerprint: FileFingerprint, state: FileState) -> Self {
        Self {
            size: fingerprint.size,
            modified_secs: fingerprint.modified_secs,
            modified_nanos: fingerprint.modified_nanos,
            state,
        }
    }

    pub(crate) fn matches(&self, fingerprint: FileFingerprint) -> bool {
        self.size == fingerprint.size
            && self.modified_secs == fingerprint.modified_secs
            && self.modified_nanos == fingerprint.modified_nanos
    }

    pub(crate) fn is_indexed(&self) -> bool {
        matches!(
            self.state,
            FileState::Indexed | FileState::IndexedMetadata { .. }
        )
    }

    pub(crate) fn skips_contents(&self) -> bool {
        matches!(
            self.state,
            FileState::IndexedMetadata { .. } | FileState::Skipped { .. }
        )
    }
}

impl FileFingerprint {
    pub(crate) fn from_metadata(metadata: &fs::Metadata) -> Self {
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

impl ManifestWorkingSet {
    pub(crate) fn open(path: &Path, root: &Path, resume_existing: bool) -> SearchxResult<Self> {
        let conn = open_manifest_connection(path)?;
        if resume_existing {
            ensure_working_manifest_info(&conn, root)?;
        } else {
            clear_working_manifest(&conn)?;
            store_working_manifest_info(&conn, root)?;
        }
        Ok(Self { conn })
    }

    pub(crate) fn update_entry(
        &self,
        relative_path: &str,
        entry: &ManifestEntry,
    ) -> SearchxResult<()> {
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

    pub(crate) fn update_entries(
        &self,
        updates: &[crate::scan::ProgressUpdate],
    ) -> SearchxResult<()> {
        for update in updates {
            self.update_entry(&update.path, &update.entry)?;
        }
        Ok(())
    }
}

enum StoredManifestState {
    Ready(Manifest),
    Missing,
    Rebuild(String),
}

pub(crate) fn open_manifest_connection(path: &Path) -> SearchxResult<Connection> {
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
         CREATE TABLE IF NOT EXISTS working_manifest_info (
             singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
             version INTEGER NOT NULL,
             root TEXT NOT NULL
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

fn store_manifest_info(conn: &Connection, table: &str, root: &Path) -> SearchxResult<()> {
    let sql = format!(
        "INSERT INTO {table} (singleton, version, root)
         VALUES (1, ?1, ?2)
         ON CONFLICT(singleton) DO UPDATE SET
             version = excluded.version,
             root = excluded.root"
    );
    conn.execute(&sql, params![MANIFEST_VERSION, root.display().to_string()])?;
    Ok(())
}

fn store_working_manifest_info(conn: &Connection, root: &Path) -> SearchxResult<()> {
    store_manifest_info(conn, "working_manifest_info", root)
}

fn load_manifest_info_from_table(
    conn: &Connection,
    table: &str,
) -> SearchxResult<Option<(u32, String)>> {
    let sql = format!("SELECT version, root FROM {table} WHERE singleton = 1");
    let info = conn
        .query_row(&sql, [], |row| {
            Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?))
        })
        .optional()?;
    Ok(info)
}

pub(crate) fn clear_working_manifest(conn: &Connection) -> SearchxResult<()> {
    conn.execute("DELETE FROM working_manifest_info", [])?;
    conn.execute("DELETE FROM working_manifest_files", [])?;
    Ok(())
}

fn ensure_working_manifest_info(conn: &Connection, root: &Path) -> SearchxResult<()> {
    let root_display = root.display().to_string();
    match load_manifest_info_from_table(conn, "working_manifest_info")? {
        Some((version, working_root))
            if version == MANIFEST_VERSION && working_root == root_display => {}
        _ => {
            clear_working_manifest(conn)?;
            store_working_manifest_info(conn, root)?;
        }
    }
    Ok(())
}

fn load_manifest_info(conn: &Connection) -> SearchxResult<Option<(u32, String)>> {
    load_manifest_info_from_table(conn, "manifest_info")
}

fn load_manifest_files_from_table(
    conn: &Connection,
    table: &str,
) -> SearchxResult<BTreeMap<String, ManifestEntry>> {
    let sql = format!(
        "SELECT path, size, modified_secs, modified_nanos, state, skip_reason
         FROM {table}"
    );
    let mut statement = conn.prepare(&sql)?;
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

fn load_manifest_files(conn: &Connection) -> SearchxResult<BTreeMap<String, ManifestEntry>> {
    load_manifest_files_from_table(conn, "manifest_files")
}

pub(crate) fn load_working_manifest(
    conn: &Connection,
    root: &Path,
) -> SearchxResult<Option<Manifest>> {
    let Some((version, manifest_root)) =
        load_manifest_info_from_table(conn, "working_manifest_info")?
    else {
        return Ok(None);
    };

    let root_display = root.display().to_string();
    if version != MANIFEST_VERSION || manifest_root != root_display {
        return Ok(None);
    }

    let files = load_manifest_files_from_table(conn, "working_manifest_files")?;
    Ok(Some(Manifest {
        version,
        root: manifest_root,
        files,
    }))
}

fn file_state_to_db(state: &FileState) -> (&'static str, Option<&'static str>) {
    match state {
        FileState::Indexed => ("indexed", None),
        FileState::IndexedMetadata {
            reason: SkipReason::TooLarge,
        } => ("indexed", Some("too_large")),
        FileState::IndexedMetadata {
            reason: SkipReason::Binary,
        } => ("indexed", Some("binary")),
        FileState::Skipped {
            reason: SkipReason::TooLarge,
        } => ("skipped", Some("too_large")),
        FileState::Skipped {
            reason: SkipReason::Binary,
        } => ("skipped", Some("binary")),
    }
}

fn file_state_from_db(state: &str, skip_reason: Option<&str>) -> SearchxResult<FileState> {
    match (state, skip_reason) {
        ("indexed", None) => Ok(FileState::Indexed),
        ("indexed", Some("too_large")) => Ok(FileState::IndexedMetadata {
            reason: SkipReason::TooLarge,
        }),
        ("indexed", Some("binary")) => Ok(FileState::IndexedMetadata {
            reason: SkipReason::Binary,
        }),
        ("skipped", Some("too_large")) => Ok(FileState::Skipped {
            reason: SkipReason::TooLarge,
        }),
        ("skipped", Some("binary")) => Ok(FileState::Skipped {
            reason: SkipReason::Binary,
        }),
        ("skipped", None) => Err(SearchxError::MissingSkipReason),
        (state, reason) => Err(SearchxError::InvalidManifestState {
            state: state.to_string(),
            reason: reason.map(str::to_string),
        }),
    }
}

fn i64_from_u64(value: u64, field: &'static str) -> SearchxResult<i64> {
    i64::try_from(value).map_err(|_| SearchxError::ManifestIntegerOverflow { field, value })
}

fn u64_from_i64(value: i64, field: &'static str) -> SearchxResult<u64> {
    u64::try_from(value).map_err(|_| SearchxError::ManifestNegativeValue { field, value })
}

fn manifest_load(
    manifest: Manifest,
    rebuild_reason: Option<String>,
    resume_from_incomplete: bool,
) -> ManifestLoad {
    ManifestLoad {
        manifest,
        rebuild_reason,
        resume_from_incomplete,
    }
}

fn manifest_load_with_reason(root: &Path, rebuild_reason: Option<String>) -> ManifestLoad {
    manifest_load(Manifest::new(root), rebuild_reason, false)
}

fn rebuild_manifest_load(root: &Path, reason: impl Into<String>) -> ManifestLoad {
    manifest_load_with_reason(root, Some(reason.into()))
}

fn missing_manifest_load(root: &Path, data_paths: &DataPaths) -> ManifestLoad {
    let rebuild_reason = data_paths
        .index
        .join("data.mdb")
        .exists()
        .then(|| "manifest is missing".to_string());
    manifest_load_with_reason(root, rebuild_reason)
}

fn load_stored_manifest_state(data_paths: &DataPaths, root: &Path) -> StoredManifestState {
    if !data_paths.manifest.exists() {
        return StoredManifestState::Missing;
    }

    let conn = match open_manifest_connection(&data_paths.manifest) {
        Ok(conn) => conn,
        Err(error) => {
            return StoredManifestState::Rebuild(format!("manifest could not be opened: {error}"));
        }
    };

    let Some((version, manifest_root)) = (match load_manifest_info(&conn) {
        Ok(info) => info,
        Err(error) => {
            return StoredManifestState::Rebuild(format!("manifest could not be read: {error}"));
        }
    }) else {
        return StoredManifestState::Missing;
    };

    if version != MANIFEST_VERSION {
        return StoredManifestState::Rebuild(format!(
            "manifest version {version} does not match {MANIFEST_VERSION}"
        ));
    }

    let root_display = root.display().to_string();
    if manifest_root != root_display {
        return StoredManifestState::Rebuild(format!(
            "manifest root {manifest_root} does not match {root_display}"
        ));
    }

    let files = match load_manifest_files(&conn) {
        Ok(files) => files,
        Err(error) => {
            return StoredManifestState::Rebuild(format!("manifest could not be read: {error}"));
        }
    };

    StoredManifestState::Ready(Manifest {
        version,
        root: manifest_root,
        files,
    })
}

pub fn data_paths<P: AsRef<Path>>(data_dir: P) -> SearchxResult<DataPaths> {
    let base = env::current_dir()?.join(data_dir);
    Ok(DataPaths {
        index: base.join("index"),
        manifest: base.join(MANIFEST_FILE_NAME),
        incomplete_marker: base.join(INCOMPLETE_FILE_NAME),
        base,
    })
}

pub fn mark_index_incomplete(data_paths: &DataPaths) -> SearchxResult<()> {
    fs::create_dir_all(&data_paths.base)?;
    fs::write(&data_paths.incomplete_marker, b"incomplete\n")?;
    Ok(())
}

pub fn clear_index_incomplete(data_paths: &DataPaths) -> SearchxResult<()> {
    if data_paths.incomplete_marker.exists() {
        fs::remove_file(&data_paths.incomplete_marker)?;
    }
    Ok(())
}

pub fn load_manifest(
    data_paths: &DataPaths,
    root: &Path,
    force_rebuild: bool,
) -> SearchxResult<ManifestLoad> {
    if force_rebuild {
        return Ok(rebuild_manifest_load(root, "forced by --rebuild"));
    }

    let stored_state = load_stored_manifest_state(data_paths, root);

    if data_paths.incomplete_marker.exists() {
        match &stored_state {
            StoredManifestState::Rebuild(reason) => {
                return Ok(rebuild_manifest_load(root, reason.clone()));
            }
            StoredManifestState::Ready(_) | StoredManifestState::Missing => {}
        }

        if data_paths.manifest.exists() {
            let conn = match open_manifest_connection(&data_paths.manifest) {
                Ok(conn) => conn,
                Err(error) => {
                    return Ok(rebuild_manifest_load(
                        root,
                        format!("manifest could not be opened: {error}"),
                    ));
                }
            };

            let working_manifest = match load_working_manifest(&conn, root) {
                Ok(manifest) => manifest,
                Err(error) => {
                    return Ok(rebuild_manifest_load(
                        root,
                        format!("working manifest could not be read: {error}"),
                    ));
                }
            };

            if let Some(working_manifest) = working_manifest {
                let manifest = match stored_state {
                    StoredManifestState::Ready(mut manifest) => {
                        manifest.files.extend(working_manifest.files);
                        manifest
                    }
                    StoredManifestState::Missing => working_manifest,
                    StoredManifestState::Rebuild(_) => unreachable!(),
                };
                return Ok(manifest_load(manifest, None, true));
            }
        }
    }

    Ok(match stored_state {
        StoredManifestState::Ready(manifest) => manifest_load(manifest, None, false),
        StoredManifestState::Missing => missing_manifest_load(root, data_paths),
        StoredManifestState::Rebuild(reason) => rebuild_manifest_load(root, reason),
    })
}

pub fn reset_data_dir(data_paths: &DataPaths) -> SearchxResult<()> {
    if data_paths.base.exists() {
        fs::remove_dir_all(&data_paths.base)?;
    }
    fs::create_dir_all(&data_paths.index)?;
    Ok(())
}

pub fn commit_working_manifest(path: &Path, root: &Path) -> SearchxResult<()> {
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
    tx.execute("DELETE FROM working_manifest_info", [])?;
    tx.execute("DELETE FROM working_manifest_files", [])?;
    tx.commit()?;
    Ok(())
}

pub fn discard_working_manifest(path: &Path) -> SearchxResult<()> {
    if !path.exists() {
        return Ok(());
    }

    let conn = open_manifest_connection(path)?;
    clear_working_manifest(&conn)
}
