"""Tests for embedding the complete chunk corpus."""

import math
from pathlib import Path

import pytest

pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers is required for the embedding integration test",
)

from recipe_rag.chunking.chunker import chunk_documents
from recipe_rag.cleaning.cleaner import clean_documents
from recipe_rag.embedding.embedder import TextEmbedder, load_embedding_config
from recipe_rag.ingestion.loader import load_documents


def test_embed_full_corpus() -> None:
    """Embed all chunks and validate vector shape and normalization."""
    chunks = chunk_documents(clean_documents(load_documents(Path("data/raw"))))
    config = load_embedding_config(Path("configs/embedding.yaml"))
    embedded_chunks = TextEmbedder(config).embed_chunks(chunks)

    assert len(chunks) == 272
    assert len(embedded_chunks) == 272

    dimensions = {len(chunk["embedding"]) for chunk in embedded_chunks}
    assert dimensions == {384}

    for original, embedded in zip(chunks, embedded_chunks, strict=True):
        assert embedded["chunk_id"] == original["chunk_id"]
        assert embedded["document_id"] == original["document_id"]
        assert embedded["content"] == original["content"]
        assert embedded["embedding_metadata"]["model_name"] == config.model_name
        assert embedded["embedding_metadata"]["dimension"] == 384
        assert all(math.isfinite(value) for value in embedded["embedding"])

        vector_norm = math.sqrt(sum(value * value for value in embedded["embedding"]))
        assert math.isclose(vector_norm, 1.0, rel_tol=1e-5, abs_tol=1e-5)

    print(f"\nModel: {config.model_name}")
    print(f"Chunks embedded: {len(embedded_chunks)}")
    print(f"Vector dimensions: {dimensions}")


# PowerShell: python -m pytest -s .\tests\embedding\test_embedder.py
