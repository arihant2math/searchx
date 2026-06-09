use http_client::policy::IpPolicy;
use milli::documents::{DocumentsBatchBuilder, DocumentsBatchReader};
use milli::progress::{EmbedderStats, Progress};
use milli::update::{IndexDocuments, IndexDocumentsConfig, IndexDocumentsMethod, IndexerConfig};
use milli::{CreateOrOpen, Index};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::sync::Arc;
use tempfile::tempdir;

#[derive(Serialize, Deserialize, Debug)]
struct File {
    id: u32,
    path: String,
    contents: String,
}

fn main() {
    let dir = tempdir().unwrap();
    let mut options = milli::heed::EnvOpenOptions::new();
    options.map_size(100 * 1024 * 1024); // 100 MB
    let options = options.read_txn_without_tls();

    let index = Index::new(options, dir.path(), CreateOrOpen::Create { shards: None }).unwrap();

    let movies = vec![
        File {
            id: 1,
            path: String::from("test.txt"),
            contents: String::from("I love ice cream"),
        },
        File {
            id: 2,
            path: String::from("test2.txt"),
            contents: String::from("It is cold outside"),
        },
    ];
    let mut batch_builder = DocumentsBatchBuilder::new(Vec::new());
    for movie in movies {
        let value = serde_json::to_value(movie).unwrap();
        batch_builder.append_json_object(value.as_object().unwrap()).unwrap();
    }
    let batch = batch_builder.into_inner().unwrap();

    let mut wtxn = index.write_txn().unwrap();
    let indexer_config = IndexerConfig::default();
    let embedder_stats: Arc<EmbedderStats> = Default::default();
    let reader = DocumentsBatchReader::from_reader(Cursor::new(batch)).unwrap();
    let ip_policy = IpPolicy::danger_always_allow();
    let index_documents = IndexDocuments::new(
        &mut wtxn,
        &index,
        &indexer_config,
        IndexDocumentsConfig {
            update_method: IndexDocumentsMethod::ReplaceDocuments,
            ..Default::default()
        },
        |_| {},
        || false,
        &embedder_stats,
        &ip_policy,
    )
    .unwrap();
    let (index_documents, indexed_count) = index_documents.add_documents(reader).unwrap();
    println!("Indexed {} documents.", indexed_count.unwrap());
    index_documents.execute().unwrap();
    wtxn.commit().unwrap();

    let rtxn = index.read_txn().unwrap();
    let progress = Progress::default();
    let mut search = milli::Search::new(&rtxn, &index, &progress);
    search.query("cold");
    search.limit(10);

    let result = search.execute().unwrap();
    println!("Found {} documents:", result.candidates.len());

    for doc_id in result.candidates {
        println!("Document ID matched: {:?}", doc_id);
    }
}
