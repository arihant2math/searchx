use crate::constants::VECTOR_DIMENSIONS;
use crate::error::{SearchxError, SearchxResult};
#[cfg(not(test))]
use fastembed::{
    EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, InitOptions,
    TextEmbedding,
};

#[derive(Debug, Clone, Copy)]
pub enum EmbeddingInput<'a> {
    Text(&'a str),
    Image(&'a [u8]),
    Pdf(&'a [u8]),
}

#[derive(Debug, Clone)]
pub(crate) enum OwnedEmbeddingInput {
    Text(String),
    Image(Vec<u8>),
}

impl OwnedEmbeddingInput {
    #[must_use]
    pub(crate) fn from_borrowed(input: EmbeddingInput<'_>) -> Option<Self> {
        match input {
            EmbeddingInput::Text(text) => Some(Self::Text(text.to_string())),
            EmbeddingInput::Image(bytes) => Some(Self::Image(bytes.to_vec())),
            EmbeddingInput::Pdf(_) => None,
        }
    }
}

#[derive(Default)]
pub(crate) struct Embedder {
    #[cfg(not(test))]
    text_model: Option<TextEmbedding>,
    #[cfg(not(test))]
    image_model: Option<ImageEmbedding>,
}

impl Embedder {
    pub(crate) fn embed(&mut self, input: &OwnedEmbeddingInput) -> SearchxResult<Vec<f32>> {
        match input {
            OwnedEmbeddingInput::Text(text) => self
                .embed_texts(&[text.as_str()])?
                .into_iter()
                .next()
                .ok_or_else(|| SearchxError::Embedding {
                    message: "text embedder returned no vectors".to_string(),
                }),
            OwnedEmbeddingInput::Image(bytes) => self
                .embed_images(&[bytes.as_slice()])?
                .into_iter()
                .next()
                .ok_or_else(|| SearchxError::Embedding {
                    message: "image embedder returned no vectors".to_string(),
                }),
        }
    }

    pub(crate) fn embed_texts(&mut self, texts: &[&str]) -> SearchxResult<Vec<Vec<f32>>> {
        #[cfg(test)]
        {
            return Ok(texts
                .iter()
                .map(|text| pseudo_embedding(text.as_bytes()))
                .collect());
        }

        #[cfg(not(test))]
        {
            let model = self.text_model.get_or_insert(text_embedding_model()?);
            let embeddings = model
                .embed(texts, None)
                .map_err(|error| SearchxError::Embedding {
                    message: error.to_string(),
                })?;
            validate_embeddings(embeddings, texts.len())
        }
    }

    pub(crate) fn embed_images(&mut self, images: &[&[u8]]) -> SearchxResult<Vec<Vec<f32>>> {
        #[cfg(test)]
        {
            return Ok(images.iter().map(|bytes| pseudo_embedding(bytes)).collect());
        }

        #[cfg(not(test))]
        {
            let model = self.image_model.get_or_insert(image_embedding_model()?);
            let embeddings =
                model
                    .embed_bytes(images, None)
                    .map_err(|error| SearchxError::Embedding {
                        message: error.to_string(),
                    })?;
            validate_embeddings(embeddings, images.len())
        }
    }
}

#[cfg_attr(test, allow(dead_code))]
fn validate_embeddings(
    embeddings: Vec<Vec<f32>>,
    expected_len: usize,
) -> SearchxResult<Vec<Vec<f32>>> {
    if embeddings.len() != expected_len {
        return Err(SearchxError::Embedding {
            message: format!(
                "embedder returned {} vectors, expected {}",
                embeddings.len(),
                expected_len
            ),
        });
    }

    for vector in &embeddings {
        if vector.len() != VECTOR_DIMENSIONS {
            return Err(SearchxError::Embedding {
                message: format!(
                    "embedder returned {} dimensions, expected {}",
                    vector.len(),
                    VECTOR_DIMENSIONS
                ),
            });
        }
    }

    Ok(embeddings)
}

#[cfg(not(test))]
fn text_embedding_model() -> SearchxResult<TextEmbedding> {
    TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::ClipVitB32).with_show_download_progress(true),
    )
    .map_err(|error| SearchxError::Embedding {
        message: error.to_string(),
    })
}

#[cfg(not(test))]
fn image_embedding_model() -> SearchxResult<ImageEmbedding> {
    ImageEmbedding::try_new(
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32).with_show_download_progress(true),
    )
    .map_err(|error| SearchxError::Embedding {
        message: error.to_string(),
    })
}

#[cfg(test)]
fn pseudo_embedding(bytes: &[u8]) -> Vec<f32> {
    let mut vector = Vec::with_capacity(VECTOR_DIMENSIONS);
    let mut counter = 0u64;

    while vector.len() < VECTOR_DIMENSIONS {
        let mut hasher = blake3::Hasher::new();
        hasher.update(bytes);
        hasher.update(&counter.to_le_bytes());
        let hash = hasher.finalize();

        for chunk in hash.as_bytes().chunks_exact(4) {
            let value = u32::from_le_bytes(chunk.try_into().expect("hash chunk size")) as f32
                / u32::MAX as f32;
            vector.push(value.mul_add(2.0, -1.0));
            if vector.len() == VECTOR_DIMENSIONS {
                break;
            }
        }

        counter += 1;
    }

    normalize(&mut vector);
    vector
}

#[cfg(test)]
fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    let divisor = if norm > 0.0 { norm } else { 1.0 };
    for value in vector {
        *value /= divisor;
    }
}
