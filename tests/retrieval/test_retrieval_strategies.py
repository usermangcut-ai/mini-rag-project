"""Test dense, BM25, and hybrid retrieval rules independently of config."""

from pathlib import Path
from types import SimpleNamespace

from recipe_rag.retrieval.bm25_retriever import BM25Retriever
from recipe_rag.retrieval.dense_retriever import DenseRetriever
from recipe_rag.retrieval.hybrid_retriever import HybridRetriever


def _result(chunk_id: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "score": score,
        "content": chunk_id,
        "metadata": {"source": f"{chunk_id}.md", "section": "ingredients"},
    }


class _EmbedderStub:
    def __init__(self) -> None:
        self.config = SimpleNamespace(model_name="test-model")
        self.received_query: str | None = None

    def embed_query(self, query: str) -> list[float]:
        self.received_query = query
        return [0.1, 0.2, 0.3]


class _StoreStub:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.validated: tuple[str, int] | None = None
        self.search_call: tuple[list[float], int, dict | None] | None = None

    def validate_model(self, model_name: str, dimension: int) -> None:
        self.validated = (model_name, dimension)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        self.search_call = (query_vector, top_k, filters)
        return self.results[:top_k]


class _DenseStub:
    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 5, filters=None) -> list[dict]:
        return self.results[:top_k]

    def retrieve_many(self, queries: list[str], top_k: int = 5) -> list[list[dict]]:
        return [self.results[:top_k] for _ in queries]


def test_dense_connects_query_embedding_to_matching_vector_store() -> None:
    embedder = _EmbedderStub()
    store = _StoreStub([_result("dense-result", 0.9)])
    retriever = DenseRetriever(embedder, store)

    results = retriever.retrieve(
        "roasted peanuts",
        top_k=1,
        filters={"section": "ingredients"},
    )

    assert results[0]["chunk_id"] == "dense-result"
    assert embedder.received_query == "roasted peanuts"
    assert store.validated == ("test-model", 3)
    assert store.search_call == (
        [0.1, 0.2, 0.3],
        1,
        {"section": "ingredients"},
    )


def test_bm25_retrieves_exact_numeric_recipe_from_full_corpus() -> None:
    retriever = BM25Retriever.from_jsonl(Path("data/processed/chunks.jsonl"))

    results = retriever.retrieve(
        "Which recipe uses exactly 1/4 cup of roasted peanuts?",
        top_k=5,
    )

    assert len(retriever.chunks) == 272
    assert any(
        result["metadata"]["source"] == "data/raw/pad-thai.md"
        and result["metadata"]["section"] == "ingredients"
        for result in results
    )


def test_hybrid_rrf_rewards_candidates_found_by_both_retrievers() -> None:
    dense = _DenseStub([_result("dense-only", 0.9), _result("shared", 0.8)])
    bm25 = BM25Retriever(
        [
            {
                "chunk_id": "shared",
                "content": "roasted peanuts",
                "metadata": {"source": "shared.md", "section": "ingredients"},
            },
            {
                "chunk_id": "bm25-only",
                "content": "unrelated text",
                "metadata": {"source": "bm25-only.md", "section": "ingredients"},
            },
        ]
    )
    hybrid = HybridRetriever(dense, bm25, dense_top_k=2, bm25_top_k=2)

    results = hybrid.retrieve("roasted peanuts", top_k=3)

    assert results[0]["chunk_id"] == "shared"
    assert results[0]["dense_rank"] == 2
    assert results[0]["bm25_rank"] == 1
    assert results[0]["retrieval_method"] == "hybrid"


def test_hybrid_rrf_applies_configurable_retriever_weights() -> None:
    dense = _DenseStub([_result("dense-choice", 0.9)])
    bm25 = BM25Retriever(
        [
            {
                "chunk_id": "bm25-choice",
                "content": "roasted peanuts",
                "metadata": {
                    "source": "bm25-choice.md",
                    "section": "ingredients",
                },
            }
        ]
    )
    hybrid = HybridRetriever(
        dense,
        bm25,
        dense_top_k=1,
        bm25_top_k=1,
        dense_weight=2.0,
        bm25_weight=1.0,
    )

    results = hybrid.retrieve("roasted peanuts", top_k=2)

    assert results[0]["chunk_id"] == "dense-choice"
    assert results[0]["score"] > results[1]["score"]


# PowerShell: python -m pytest -s .\tests\retrieval\test_retrieval_strategies.py
