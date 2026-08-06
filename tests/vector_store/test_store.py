"""Tests for ChromaDB using all chunks in the recipe corpus."""

import hashlib
import math
from pathlib import Path

from recipe_rag.chunking.chunker import chunk_documents
from recipe_rag.cleaning.cleaner import clean_documents
from recipe_rag.ingestion.loader import load_documents
from recipe_rag.vector_store.store import ChromaVectorStore


TEST_PROFILE = "test_profile"
TEST_MODEL = "deterministic-test-embedder"
TEST_DIMENSION = 32


def _deterministic_vector(text: str) -> list[float]:
    """Create a repeatable unit vector for testing without an ML model."""
    values: list[float] = []
    counter = 0
    while len(values) < TEST_DIMENSION:
        digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        counter += 1
    vector = values[:TEST_DIMENSION]
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


def test_chroma_full_corpus(tmp_path: Path) -> None:
    """Build, reload, search, and filter a 272-record persistent store."""
    chunks = chunk_documents(clean_documents(load_documents(Path("data/raw"))))
    embedded_chunks = [
        {
            **chunk,
            "embedding": _deterministic_vector(chunk["chunk_id"]),
            "embedding_metadata": {
                "profile_name": TEST_PROFILE,
                "model_name": TEST_MODEL,
                "dimension": TEST_DIMENSION,
                "normalized": True,
            },
        }
        for chunk in chunks
    ]

    store_path = tmp_path / "chroma"
    store = ChromaVectorStore(store_path, TEST_PROFILE)
    assert store.build(embedded_chunks) == 272

    reloaded_store = ChromaVectorStore(store_path, TEST_PROFILE)
    reloaded_store.validate_model(TEST_MODEL, TEST_DIMENSION)
    assert reloaded_store.count() == 272

    target = embedded_chunks[0]
    results = reloaded_store.search(target["embedding"], top_k=5)
    assert len(results) == 5
    assert results[0]["chunk_id"] == target["chunk_id"]
    assert math.isclose(results[0]["score"], 1.0, abs_tol=1e-5)

    ingredient_results = reloaded_store.search(
        target["embedding"],
        top_k=5,
        filters={"section": "ingredients"},
    )
    assert ingredient_results
    assert all(
        result["metadata"]["section"] == "ingredients"
        for result in ingredient_results
    )

    print(f"\nChunks indexed: {reloaded_store.count()}")
    print(f"Top result: {results[0]['chunk_id']}")
    print(f"Top score: {results[0]['score']:.4f}")


# PowerShell: python -m pytest -s .\tests\vector_store\test_store.py
