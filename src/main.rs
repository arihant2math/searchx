use milli::{CreateOrOpen, Index};
use serde::{Deserialize, Serialize};
use std::fs::File;
use milli::documents::DocumentsBatchBuilder;
use milli::heed::WithoutTls;
use milli::progress::Progress;
use tempfile::tempdir;

#[derive(Serialize, Deserialize, Debug)]
struct Movie {
    id: u32,
    title: String,
    description: String,
}

fn main() {
    let dir = tempdir().unwrap();
    let mut options = milli::heed::EnvOpenOptions::new();
    options.map_size(100 * 1024 * 1024); // 100 MB
    let options = options.read_txn_without_tls();

    let index = Index::new(options, dir.path(), CreateOrOpen::Create { shards: None }).unwrap();

    let movies = vec![
        Movie {
            id: 1,
            title: String::from("The Matrix"),
            description: String::from("A computer hacker learns about the true nature of his reality."),
        },
        Movie {
            id: 2,
            title: String::from("Inception"),
            description: String::from("A thief who enters the dreams of others to steal secrets."),
        },
    ];
    let batch_builder = DocumentsBatchBuilder::new(Vec::new());
    for movie in movies {
        batch_builder.append_json_object(serde_json::to_value(movie).unwrap().as_object().unwrap()).unwrap();
    }
    let batch = batch_builder.into_inner().unwrap();

    let mut wtxn = index.write_txn().unwrap();
    index.add_documents(&mut wtxn, batch_builder).unwrap();
    wtxn.commit().unwrap();

    let rtxn = index.read_txn().unwrap();
    let progress = Progress::default();
    let mut search = milli::Search::new(&rtxn, &index, &progress);
    search.query("Matrix");
    search.limit(10);

    let result = search.execute().unwrap();
    println!("Found {} documents:", result.candidates.len());

    for doc_id in result.candidates {
        println!("Document ID matched: {:?}", doc_id);
    }
}
