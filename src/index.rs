use crate::api::{SearchHit, SearchResults};
use crate::constants::{
    DEFAULT_MAP_SIZE_BYTES, PRIMARY_KEY, SEARCHABLE_FIELDS, VECTOR_DIMENSIONS,
    VECTOR_EMBEDDER_NAME, VECTOR_STORE_BACKEND,
};
use crate::error::SearchxResult;
use bumpalo::Bump;
use http_client::policy::IpPolicy;
use memmap2::Mmap;
use milli::heed::{EnvOpenOptions, WithoutTls};
use milli::progress::{EmbedderStats, Progress};
use milli::update::new::indexer;
use milli::update::{IndexerConfig, MissingDocumentPolicy, Setting, Settings};
use milli::vector::settings::{EmbedderSource, EmbeddingSettings};
use milli::{Index, all_obkv_to_json};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use fastembed::{EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions, TextEmbedding};
use tempfile::NamedTempFile;

pub(crate) type IndexedVectors = BTreeMap<String, Option<Vec<f32>>>;

#[derive(Serialize, Debug)]
pub(crate) struct IndexedDocument {
    pub(crate) id: String,
    pub(crate) path: String,
    pub(crate) file_name: String,
    pub(crate) extension: Option<String>,
    pub(crate) contents: String,
    #[serde(rename = "_vectors")]
    pub(crate) vectors: IndexedVectors,
}

pub fn configure_index(index: &Index, indexer_config: &IndexerConfig) -> SearchxResult<()> {
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

#[must_use]
pub fn generate_document_vector(relative_path: &str, contents: &str) -> Option<Vec<f32>> {
    None
}

/// Get all vectors for a document, keyed by embedder name.
///
/// Currently only supports one embedder, but this should be configurable
pub(crate) fn document_vectors(relative_path: &str, contents: &str) -> IndexedVectors {
    let mut vectors = BTreeMap::new();
    vectors.insert(
        VECTOR_EMBEDDER_NAME.to_string(),
        generate_document_vector(relative_path, contents),
    );
    vectors
}

pub(crate) fn document_id_for_path(relative_path: &str) -> String {
    blake3::hash(relative_path.as_bytes()).to_hex().to_string()
}

pub fn apply_index_batch(
    index: &Index,
    indexer_config: &IndexerConfig,
    temp_dir: &Path,
    upserts: &[String],
    deleted_ids: &[String],
) -> SearchxResult<()> {
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
        return Err(milli::Error::from(error).into());
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

pub fn search_index(index: &Index, query: &str, limit: usize) -> SearchxResult<SearchResults> {
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
        timings: progress.accumulated_durations(),
        candidate_count,
        hits,
    })
}

#[must_use]
pub fn new_heed_options() -> EnvOpenOptions<WithoutTls> {
    let mut heed_options = EnvOpenOptions::new();
    heed_options.map_size(DEFAULT_MAP_SIZE_BYTES);
    heed_options.read_txn_without_tls()
}
