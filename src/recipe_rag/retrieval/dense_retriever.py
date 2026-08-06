"""Retrieve relevant chunks with dense embeddings and ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from recipe_rag.embedding.embedder import EmbeddingConfig, TextEmbedder
from recipe_rag.vector_store.store import ChromaVectorStore


SearchResult = dict[str, Any]


@dataclass(frozen=True)
class RetrievalConfig:
    """Runtime settings for dense, BM25, or hybrid retrieval."""

    strategy: str
    final_top_k: int
    dense_top_k: int
    bm25_top_k: int
    rrf_k: int
    dense_weight: float
    bm25_weight: float
    reranking_enabled: bool
    reranker_model_name: str
    reranker_candidate_top_k: int
    reranker_batch_size: int


def load_retrieval_config(config_path: str | Path) -> RetrievalConfig:
    """Load and validate retrieval strategy settings from YAML."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval config does not exist: {path}")

    raw_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    retrieval_config = raw_config.get("retrieval", {})
    strategy = retrieval_config.get("strategy")
    if strategy not in {"dense", "bm25", "hybrid"}:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    weights = retrieval_config.get("weights", {})
    reranking = retrieval_config.get("reranking", {})
    config = RetrievalConfig(
        strategy=strategy,
        final_top_k=int(retrieval_config.get("final_top_k", 5)),
        dense_top_k=int(retrieval_config.get("dense_top_k", 20)),
        bm25_top_k=int(retrieval_config.get("bm25_top_k", 20)),
        rrf_k=int(retrieval_config.get("rrf_k", 60)),
        dense_weight=float(weights.get("dense", 1.0)),
        bm25_weight=float(weights.get("bm25", 1.0)),
        reranking_enabled=bool(reranking.get("enabled", False)),
        reranker_model_name=str(reranking.get("model_name", "")),
        reranker_candidate_top_k=int(reranking.get("candidate_top_k", 20)),
        reranker_batch_size=int(reranking.get("batch_size", 32)),
    )
    if min(
        config.final_top_k,
        config.dense_top_k,
        config.bm25_top_k,
        config.rrf_k,
    ) <= 0:
        raise ValueError("Retrieval top-k and RRF values must be greater than zero")
    if config.dense_weight < 0 or config.bm25_weight < 0:
        raise ValueError("Retrieval weights cannot be negative")
    if config.dense_weight == 0 and config.bm25_weight == 0:
        raise ValueError("At least one retrieval weight must be positive")
    if config.reranking_enabled and not config.reranker_model_name.strip():
        raise ValueError("Enabled reranking requires a model_name")
    if config.reranker_candidate_top_k < config.final_top_k:
        raise ValueError("Reranker candidate_top_k must be at least final_top_k")
    if config.reranker_batch_size <= 0:
        raise ValueError("Reranker batch_size must be greater than zero")
    return config


class DenseRetriever:
    """Connect the configured query embedder to its matching vector store."""

    def __init__(self, embedder: TextEmbedder, store: ChromaVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Embed one query and return ranked chunks."""
        query_vector = self.embedder.embed_query(query)
        self.store.validate_model(self.embedder.config.model_name, len(query_vector))
        return self.store.search(query_vector, top_k=top_k, filters=filters)

    def retrieve_many(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[list[SearchResult]]:
        """Embed multiple queries in a batch and search each vector."""
        query_vectors = self.embedder.embed_queries(queries)
        self.store.validate_model(
            self.embedder.config.model_name,
            len(query_vectors[0]),
        )
        return [
            self.store.search(query_vector, top_k=top_k)
            for query_vector in query_vectors
        ]


def build_retriever(
    retrieval_config: RetrievalConfig,
    embedding_config: EmbeddingConfig,
    chunks_path: str | Path = "data/processed/chunks.jsonl",
    vector_store_directory: str | Path = "data/vector_store",
) -> Any:
    """Build the retrieval implementation selected in configuration."""
    from recipe_rag.retrieval.bm25_retriever import BM25Retriever

    bm25_retriever = None
    if retrieval_config.strategy in {"bm25", "hybrid"}:
        bm25_retriever = BM25Retriever.from_jsonl(chunks_path)
    if retrieval_config.strategy == "bm25":
        base_retriever = bm25_retriever
    else:
        dense_retriever = DenseRetriever(
            TextEmbedder(embedding_config),
            ChromaVectorStore(
                Path(vector_store_directory) / embedding_config.profile_name,
                embedding_config.profile_name,
            ),
        )
        if retrieval_config.strategy == "dense":
            base_retriever = dense_retriever
        else:
            from recipe_rag.retrieval.hybrid_retriever import HybridRetriever

            base_retriever = HybridRetriever(
                dense_retriever,
                bm25_retriever,
                dense_top_k=retrieval_config.dense_top_k,
                bm25_top_k=retrieval_config.bm25_top_k,
                rrf_k=retrieval_config.rrf_k,
                dense_weight=retrieval_config.dense_weight,
                bm25_weight=retrieval_config.bm25_weight,
            )

    if not retrieval_config.reranking_enabled:
        return base_retriever

    from recipe_rag.retrieval.reranker import (
        CrossEncoderReranker,
        RerankingRetriever,
    )

    return RerankingRetriever(
        base_retriever,
        CrossEncoderReranker(
            retrieval_config.reranker_model_name,
            batch_size=retrieval_config.reranker_batch_size,
        ),
        candidate_top_k=retrieval_config.reranker_candidate_top_k,
    )
