"""Combine dense and BM25 candidates using Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Any

from recipe_rag.retrieval.bm25_retriever import BM25Retriever
from recipe_rag.retrieval.dense_retriever import DenseRetriever


SearchResult = dict[str, Any]


class HybridRetriever:
    """Fuse semantic and lexical rankings without comparing raw score scales."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        if dense_top_k <= 0 or bm25_top_k <= 0 or rrf_k <= 0:
            raise ValueError("Hybrid retrieval parameters must be greater than zero")
        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError("Hybrid retrieval weights cannot be negative")
        if dense_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one hybrid retrieval weight must be positive")
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve from both systems and fuse their candidate rankings."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=self.dense_top_k,
            filters=filters,
        )
        bm25_results = self.bm25_retriever.retrieve(
            query,
            top_k=self.bm25_top_k,
            filters=filters,
        )
        return self._fuse(dense_results, bm25_results, top_k)

    def retrieve_many(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[list[SearchResult]]:
        """Batch dense encoding, then fuse each query with its BM25 ranking."""
        if not queries:
            raise ValueError("Queries cannot be empty")
        dense_batches = self.dense_retriever.retrieve_many(
            queries,
            top_k=self.dense_top_k,
        )
        bm25_batches = self.bm25_retriever.retrieve_many(
            queries,
            top_k=self.bm25_top_k,
        )
        return [
            self._fuse(dense_results, bm25_results, top_k)
            for dense_results, bm25_results in zip(
                dense_batches,
                bm25_batches,
                strict=True,
            )
        ]

    def _fuse(
        self,
        dense_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Fuse ranks with RRF so incompatible raw scores are never mixed."""
        candidates: dict[str, SearchResult] = {}
        weights = {
            "dense": self.dense_weight,
            "bm25": self.bm25_weight,
        }

        for method, results in (("dense", dense_results), ("bm25", bm25_results)):
            for rank, result in enumerate(results, start=1):
                chunk_id = result["chunk_id"]
                candidate = candidates.setdefault(
                    chunk_id,
                    {
                        "chunk_id": chunk_id,
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "score": 0.0,
                        "retrieval_method": "hybrid",
                        "dense_rank": None,
                        "dense_score": None,
                        "bm25_rank": None,
                        "bm25_score": None,
                    },
                )
                candidate["score"] += weights[method] / (self.rrf_k + rank)
                candidate[f"{method}_rank"] = rank
                candidate[f"{method}_score"] = result["score"]

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: candidate["score"],
            reverse=True,
        )
        return ranked[:top_k]
