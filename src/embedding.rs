use fastembed::{EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions, TextEmbedding};

fn text_embedding_model() -> TextEmbedding {
    TextEmbedding::try_new(
        // TODO: Which model
        InitOptions::new(EmbeddingModel::ClipVitB32)
            .with_show_download_progress(true),
    ).unwrap()
}

fn image_embedding_model() -> ImageEmbedding {
    ImageEmbedding::try_new(
        // TODO: Which model
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
            .with_show_download_progress(true),
    ).unwrap()
}