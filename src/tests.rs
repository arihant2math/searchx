use super::*;
use rusqlite::params;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
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

fn streamed_indexed_paths(options: &ScanOptions, root: &Path, data_dir: &Path) -> BTreeSet<String> {
    let (event_tx, event_rx) = mpsc::sync_channel(32);
    scan_root(
        options,
        root,
        data_dir,
        &empty_manifest(root),
        None,
        &ScanPipeline {
            error_sender: None,
            event_sender: event_tx,
            embedding_sender: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
        },
    )
    .unwrap();

    event_rx
        .into_iter()
        .filter_map(|event| match event {
            IndexEvent::Upsert { document, .. } => {
                let value: serde_json::Value = serde_json::from_str(&document).unwrap();
                Some(value["path"].as_str().unwrap().to_string())
            }
            IndexEvent::Delete { .. } | IndexEvent::Progress(_) => None,
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
fn scan_root_still_indexes_binary_files_by_name() {
    let (_temp_dir, root, data_dir) = setup_scan_dirs();
    fs::write(root.join("archive.pdf"), [0, 1, 2, 3]).unwrap();

    let options = ScanOptions {
        rebuild: false,
        max_file_bytes: u64::MAX,
        ignore_rules: Vec::new(),
    };

    let indexed = streamed_indexed_paths(&options, &root, &data_dir);

    assert!(indexed.contains("archive.pdf"));
}

#[test]
fn load_manifest_resumes_from_working_state_when_previous_run_was_incomplete() {
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
    let previous = Manifest {
        version: MANIFEST_VERSION,
        root: root.display().to_string(),
        files: BTreeMap::from([(
            "old.txt".to_string(),
            ManifestEntry::from_fingerprint(
                FileFingerprint {
                    size: 1,
                    modified_secs: 2,
                    modified_nanos: 3,
                },
                FileState::Indexed,
            ),
        )]),
    };

    let mut conn = open_manifest_connection(&data_paths.manifest).unwrap();
    let tx = conn.transaction().unwrap();
    tx.execute(
        "INSERT INTO manifest_info (singleton, version, root) VALUES (1, ?1, ?2)",
        params![MANIFEST_VERSION, root.display().to_string()],
    )
    .unwrap();
    tx.execute(
        "INSERT INTO manifest_files (path, size, modified_secs, modified_nanos, state, skip_reason) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params!["old.txt", 1i64, 2i64, 3i64, "indexed", Option::<&str>::None],
    )
    .unwrap();
    tx.commit().unwrap();

    let working = ManifestWorkingSet::open(&data_paths.manifest, &root, false).unwrap();
    let resumed_entry = ManifestEntry::from_fingerprint(
        FileFingerprint {
            size: 7,
            modified_secs: 8,
            modified_nanos: 9,
        },
        FileState::Indexed,
    );
    working.update_entry("new.txt", &resumed_entry).unwrap();
    mark_index_incomplete(&data_paths).unwrap();

    let loaded = load_manifest(&data_paths, &root, false).unwrap();
    assert!(loaded.rebuild_reason.is_none());
    assert!(loaded.resume_from_incomplete);
    assert_eq!(
        loaded.manifest.files.get("old.txt"),
        previous.files.get("old.txt")
    );
    assert_eq!(loaded.manifest.files.get("new.txt"), Some(&resumed_entry));
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

    let working = ManifestWorkingSet::open(&data_paths.manifest, &root, false).unwrap();
    for (path, entry) in &expected.files {
        working.update_entry(path, entry).unwrap();
    }
    commit_working_manifest(&data_paths.manifest, &root).unwrap();

    let loaded = load_manifest(&data_paths, &root, false).unwrap();
    assert!(loaded.rebuild_reason.is_none());
    assert_eq!(loaded.manifest, expected);

    let stale_working = ManifestWorkingSet::open(&data_paths.manifest, &root, false).unwrap();
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

#[test]
fn sync_repairs_orphaned_documents_after_incomplete_run() {
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
    fs::write(root.join("live.txt"), "live").unwrap();
    fs::create_dir_all(&data_paths.index).unwrap();

    let index = milli::Index::new(
        new_heed_options(),
        &data_paths.index,
        milli::CreateOrOpen::create_without_shards(),
    )
    .unwrap();
    let indexer_config = milli::update::IndexerConfig::default();
    configure_index(&index, &indexer_config).unwrap();

    let stale_document = IndexedDocument {
        id: document_id_for_path("stale.txt"),
        path: "stale.txt".to_string(),
        file_name: "stale.txt".to_string(),
        extension: Some("txt".to_string()),
        contents: "stale".to_string(),
        vectors: document_vectors(EmbeddingInput::Text("stale")),
    };
    apply_index_batch(
        &index,
        &indexer_config,
        &data_paths.base,
        &[serde_json::to_string(&stale_document).unwrap()],
        &[],
    )
    .unwrap();
    drop(index);

    let _working = ManifestWorkingSet::open(&data_paths.manifest, &root, false).unwrap();
    mark_index_incomplete(&data_paths).unwrap();

    let result = sync_index(
        &SyncRequest::new(&root)
            .with_data_dir(&data_paths.base)
            .with_options(ScanOptions {
                max_file_bytes: u64::MAX,
                ..ScanOptions::default()
            }),
    )
    .unwrap();

    let stale_results = search_index(&result.index, "stale", 10).unwrap();
    assert_eq!(stale_results.candidate_count, 0);

    let live_results = search_index(&result.index, "live", 10).unwrap();
    assert_eq!(live_results.candidate_count, 1);
    assert_eq!(live_results.hits[0].path, "live.txt");
    assert_eq!(
        live_results.hits[0].document["_vectors"][VECTOR_EMBEDDER_NAME]["embeddings"][0]
            .as_array()
            .unwrap()
            .len(),
        VECTOR_DIMENSIONS
    );
}

#[test]
fn sync_indexes_supported_images_without_marking_them_as_skipped_binary() {
    let temp_dir = tempfile::tempdir().unwrap();
    let root = temp_dir.path().join("root");
    let data_dir = temp_dir.path().join(".searchx-data");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("pixel.gif"),
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02L\x01\x00;",
    )
    .unwrap();

    let result = sync_index(
        &SyncRequest::new(&root)
            .with_data_dir(&data_dir)
            .with_options(ScanOptions {
                max_file_bytes: u64::MAX,
                ..ScanOptions::default()
            }),
    )
    .unwrap();

    assert_eq!(result.stats.skipped_binary, 0);
    assert_eq!(result.stats.indexed_or_updated, 1);

    let image_results = search_index(&result.index, "pixel", 10).unwrap();
    assert_eq!(image_results.candidate_count, 1);
    assert_eq!(image_results.hits[0].path, "pixel.gif");
    assert_eq!(image_results.hits[0].document["contents"], "");
    assert_eq!(
        image_results.hits[0].document["_vectors"][VECTOR_EMBEDDER_NAME]["embeddings"][0]
            .as_array()
            .unwrap()
            .len(),
        VECTOR_DIMENSIONS
    );
}

#[test]
fn sync_indexes_binary_and_oversized_files_by_name() {
    let temp_dir = tempfile::tempdir().unwrap();
    let root = temp_dir.path().join("root");
    let data_dir = temp_dir.path().join(".searchx-data");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("archive.pdf"), [0, 1, 2, 3]).unwrap();
    fs::write(root.join("blob.bin"), [0, 1, 2, 3]).unwrap();
    fs::write(root.join("recording.mp4"), vec![b'x'; 32]).unwrap();

    let result = sync_index(
        &SyncRequest::new(&root)
            .with_data_dir(&data_dir)
            .with_options(ScanOptions {
                max_file_bytes: 8,
                ..ScanOptions::default()
            }),
    )
    .unwrap();

    assert_eq!(result.stats.skipped_binary, 1);
    assert_eq!(result.stats.skipped_too_large, 1);
    assert_eq!(result.stats.indexed_or_updated, 3);

    let archive_results = search_index(&result.index, "archive", 10).unwrap();
    assert_eq!(archive_results.candidate_count, 1);
    assert_eq!(archive_results.hits[0].path, "archive.pdf");
    assert_eq!(archive_results.hits[0].document["contents"], "");
    assert_eq!(
        archive_results.hits[0].document["_vectors"][VECTOR_EMBEDDER_NAME]["embeddings"][0]
            .as_array()
            .unwrap()
            .len(),
        VECTOR_DIMENSIONS
    );

    let blob_results = search_index(&result.index, "blob", 10).unwrap();
    assert_eq!(blob_results.candidate_count, 1);
    assert_eq!(blob_results.hits[0].path, "blob.bin");
    assert_eq!(blob_results.hits[0].document["contents"], "");

    let recording_results = search_index(&result.index, "recording", 10).unwrap();
    assert_eq!(recording_results.candidate_count, 1);
    assert_eq!(recording_results.hits[0].path, "recording.mp4");
    assert_eq!(recording_results.hits[0].document["contents"], "");
}

#[test]
fn vector_search_uses_vectorized_query_embeddings() {
    let temp_dir = tempfile::tempdir().unwrap();
    let root = temp_dir.path().join("root");
    let data_dir = temp_dir.path().join(".searchx-data");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("alpha.txt"), "alpha semantic target").unwrap();
    fs::write(root.join("beta.txt"), "beta semantic target").unwrap();

    let result = sync_index(
        &SyncRequest::new(&root)
            .with_data_dir(&data_dir)
            .with_options(ScanOptions {
                max_file_bytes: u64::MAX,
                ..ScanOptions::default()
            }),
    )
    .unwrap();

    let vector_results = search_index_vector(&result.index, "beta semantic target", 2).unwrap();

    assert_eq!(vector_results.query, "beta semantic target");
    assert!(vector_results.candidate_count >= 1);
    assert_eq!(vector_results.hits[0].path, "beta.txt");
    assert_eq!(
        vector_results.hits[0].document["_vectors"][VECTOR_EMBEDDER_NAME]["embeddings"][0]
            .as_array()
            .unwrap()
            .len(),
        VECTOR_DIMENSIONS
    );
}
