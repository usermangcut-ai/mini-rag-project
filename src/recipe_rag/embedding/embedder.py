"""Convert chunks and queries into configurable embedding vectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


Chunk = dict[str, Any]
EmbeddedChunk = dict[str, Any]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Runtime settings for one embedding model profile."""

    profile_name: str
    model_name: str
    query_prefix: str
    passage_prefix: str
    normalize: bool
    batch_size: int


def load_embedding_config(config_path: str | Path) -> EmbeddingConfig:
    """Load the active embedding profile from a YAML config file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding config does not exist: {path}")

    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    embedding_config = raw_config.get("embedding", {})
    active_model = embedding_config.get("active_model")
    models = embedding_config.get("models", {})

    if active_model not in models:
        raise ValueError(f"Unknown active embedding model profile: {active_model}")

    model_config = models[active_model]
    return EmbeddingConfig(
        profile_name=active_model,
        model_name=model_config["model_name"],
        query_prefix=model_config.get("query_prefix", ""),
        passage_prefix=model_config.get("passage_prefix", ""),
        normalize=bool(embedding_config.get("normalize", True)),
        batch_size=int(embedding_config.get("batch_size", 32)),
    )


class TextEmbedder:
    """Apply one configured Sentence Transformers model to passages and queries."""

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is not installed; install project dependencies "
                "before running the embedding layer"
            ) from error

        self.config = config
        self.model = SentenceTransformer(config.model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed passage texts using the configured passage prefix."""
        if not texts:
            raise ValueError("Cannot embed an empty passage list")

        prepared_texts = [self.config.passage_prefix + text for text in texts]
        embeddings = self.model.encode(
            prepared_texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > self.config.batch_size,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed one retrieval query using the configured query prefix."""
        return self.embed_queries([query])[0]

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed retrieval queries in one batch."""
        if not queries or any(not query.strip() for query in queries):
            raise ValueError("Queries cannot be empty")

        prepared_queries = [self.config.query_prefix + query.strip() for query in queries]
        embeddings = self.model.encode(
            prepared_queries,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(queries) > self.config.batch_size,
        )
        return embeddings.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Attach an embedding vector and model metadata to every chunk."""
        if not chunks:
            raise ValueError("Cannot embed an empty chunk list")

        texts = [chunk["content"] for chunk in chunks]
        vectors = self.embed_passages(texts)
        embedded_chunks: list[EmbeddedChunk] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            embedded_chunks.append(
                {
                    **chunk,
                    "embedding": vector,
                    "embedding_metadata": {
                        "profile_name": self.config.profile_name,
                        "model_name": self.config.model_name,
                        "dimension": len(vector),
                        "normalized": self.config.normalize,
                    },
                }
            )

        return embedded_chunks


def save_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk], output_path: str | Path
) -> Path:
    """Save embedded chunks as UTF-8 JSONL and return the output path."""
    if not embedded_chunks:
        raise ValueError("Cannot save an empty embedded chunk list")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for chunk in embedded_chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return path
