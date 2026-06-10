use crate::api::{SearchHit, SearchResults};
use crate::constants::{
    DEFAULT_MAP_SIZE_BYTES, PRIMARY_KEY, SEARCHABLE_FIELDS, VECTOR_DIMENSIONS,
    VECTOR_EMBEDDER_NAME, VECTOR_STORE_BACKEND,
};
use crate::embedding::{Embedder, EmbeddingInput, OwnedEmbeddingInput};
use crate::error::{SearchxError, SearchxResult};
use bumpalo::Bump;
use http_client::policy::IpPolicy;
use memmap2::Mmap;
use milli::heed::{EnvOpenOptions, WithoutTls};
use milli::index::EmbeddingsWithMetadata;
use milli::progress::{EmbedderStats, Progress};
use milli::update::new::indexer;
use milli::update::{IndexerConfig, MissingDocumentPolicy, Setting, Settings};
use milli::vector::embedder::manual;
use milli::vector::parsed_vectors::ExplicitVectors;
use milli::vector::settings::{EmbedderSource, EmbeddingSettings};
use milli::vector::{Embedder as MilliSearchEmbedder, EmbedderOptions as MilliEmbedderOptions};
use milli::{Index, obkv_to_json};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use tempfile::NamedTempFile;

pub(crate) type IndexedVectors = BTreeMap<String, Option<Vec<f32>>>;

#[derive(Serialize, Debug, Clone)]
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

fn indexed_vectors(vector: Option<Vec<f32>>) -> IndexedVectors {
    let mut vectors = BTreeMap::new();
    vectors.insert(VECTOR_EMBEDDER_NAME.to_string(), vector);
    vectors
}

#[must_use]
pub(crate) fn empty_document_vectors() -> IndexedVectors {
    indexed_vectors(None)
}

#[must_use]
pub(crate) fn embedded_document_vectors(vector: Vec<f32>) -> IndexedVectors {
    indexed_vectors(Some(vector))
}

fn embed_input(input: EmbeddingInput<'_>) -> SearchxResult<Vec<f32>> {
    let input =
        OwnedEmbeddingInput::from_borrowed(input).ok_or_else(|| SearchxError::Embedding {
            message: "embedding input type is not supported".to_string(),
        })?;
    static EMBEDDER: OnceLock<Mutex<Embedder>> = OnceLock::new();
    let embedder = EMBEDDER.get_or_init(|| Mutex::new(Embedder::default()));

    embedder
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .embed(&input)
}

#[must_use]
pub fn generate_document_vector(input: EmbeddingInput<'_>) -> Option<Vec<f32>> {
    embed_input(input).ok()
}

/// Get all vectors for a document, keyed by embedder name.
///
/// Currently only supports one embedder, but this should be configurable.
#[allow(dead_code)]
pub(crate) fn document_vectors(input: EmbeddingInput<'_>) -> IndexedVectors {
    indexed_vectors(generate_document_vector(input))
}

pub(crate) fn document_id_for_path(relative_path: &str) -> String {
    blake3::hash(relative_path.as_bytes()).to_hex().to_string()
}

pub fn apply_index_batch<U, D>(
    index: &Index,
    indexer_config: &IndexerConfig,
    temp_dir: &Path,
    upserts: &[U],
    deleted_ids: &[D],
) -> SearchxResult<()>
where
    U: AsRef<str>,
    D: AsRef<str>,
{
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
                writer.write_all(line.as_ref().as_bytes())?;
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

    let deleted_refs = deleted_ids.iter().map(|id| id.as_ref()).collect::<Vec<_>>();
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

fn build_search_results(
    index: &Index,
    rtxn: &milli::heed::RoTxn<'_>,
    progress: &Progress,
    query: &str,
    result: milli::SearchResult,
) -> SearchxResult<SearchResults> {
    let candidate_count = result.candidates.len();
    let fields_ids_map = index.fields_ids_map(rtxn)?;
    let all_fields = fields_ids_map.iter().map(|(id, _)| id).collect::<Vec<_>>();
    let mut hits = Vec::new();

    for (rank, (docid, obkv)) in index
        .documents(rtxn, result.documents_ids)?
        .into_iter()
        .enumerate()
    {
        let mut document = obkv_to_json(&all_fields, &fields_ids_map, obkv)?;
        let mut vectors = match document.remove("_vectors") {
            Some(serde_json::Value::Object(map)) => map,
            _ => serde_json::Map::new(),
        };

        for (
            name,
            EmbeddingsWithMetadata {
                embeddings,
                regenerate,
                has_fragments: _,
            },
        ) in index.embeddings(rtxn, docid)?
        {
            let explicit_vectors = ExplicitVectors {
                embeddings: Some(embeddings.into()),
                regenerate,
            };
            vectors.insert(name, serde_json::to_value(explicit_vectors)?);
        }
        document.insert("_vectors".into(), serde_json::Value::Object(vectors));

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

fn vector_search_embedder(
    index: &Index,
    rtxn: &milli::heed::RoTxn<'_>,
    dimensions: usize,
) -> SearchxResult<(String, Arc<MilliSearchEmbedder>, bool)> {
    let config = index
        .embedding_configs()
        .embedding_configs(rtxn)?
        .into_iter()
        .find(|config| config.name == VECTOR_EMBEDDER_NAME)
        .ok_or_else(|| SearchxError::Embedding {
            message: format!("vector embedder `{VECTOR_EMBEDDER_NAME}` is not configured"),
        })?;
    let quantized = config.config.quantized();
    let distribution = match config.config.embedder_options {
        MilliEmbedderOptions::UserProvided(options) => options.distribution,
        _ => None,
    };
    let embedder = Arc::new(MilliSearchEmbedder::UserProvided(manual::Embedder::new(
        manual::EmbedderOptions {
            dimensions,
            distribution,
        },
    )));

    Ok((config.name, embedder, quantized))
}

pub fn search_index(index: &Index, query: &str, limit: usize) -> SearchxResult<SearchResults> {
    let rtxn = index.read_txn()?;
    let progress = Progress::default();
    let mut search = milli::Search::new(&rtxn, index, &progress);
    search.query(query);
    search.limit(limit);

    build_search_results(index, &rtxn, &progress, query, search.execute()?)
}

pub fn search_index_vector(
    index: &Index,
    query: &str,
    limit: usize,
) -> SearchxResult<SearchResults> {
    let vector = embed_input(EmbeddingInput::Text(query))?;
    let rtxn = index.read_txn()?;
    let progress = Progress::default();
    let mut search = milli::Search::new(&rtxn, index, &progress);
    let (embedder_name, embedder, quantized) = vector_search_embedder(index, &rtxn, vector.len())?;
    search.semantic(embedder_name, embedder, quantized, Some(vector), None);
    search.limit(limit);

    build_search_results(index, &rtxn, &progress, query, search.execute()?)
}

#[must_use]
pub fn new_heed_options() -> EnvOpenOptions<WithoutTls> {
    let mut heed_options = EnvOpenOptions::new();
    heed_options.map_size(DEFAULT_MAP_SIZE_BYTES);
    heed_options.read_txn_without_tls()
}
