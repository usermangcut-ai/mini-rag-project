"""Re-rank retrieved candidates with a query-passage cross-encoder."""

from __future__ import annotations

from typing import Any


SearchResult = dict[str, Any]


class CrossEncoderReranker:
    """Score query and chunk pairs jointly, then reorder the candidates."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Reranker model name cannot be empty")
        if batch_size <= 0:
            raise ValueError("Reranker batch size must be greater than zero")

        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for cross-encoder reranking"
                ) from error
            model = CrossEncoder(model_name)

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Score one query against its candidates and return the best chunks."""
        return self.rerank_many([query], [candidates], top_k=top_k)[0]

    def rerank_many(
        self,
        queries: list[str],
        candidate_batches: list[list[SearchResult]],
        top_k: int = 5,
    ) -> list[list[SearchResult]]:
        """Score all query-candidate pairs in one model call and restore batches."""
        if not queries or any(not query.strip() for query in queries):
            raise ValueError("Reranking queries cannot be empty")
        if len(queries) != len(candidate_batches):
            raise ValueError("Each query must have one candidate batch")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        pairs = [
            (query.strip(), candidate["content"])
            for query, candidates in zip(queries, candidate_batches, strict=True)
            for candidate in candidates
        ]
        if not pairs:
            return [[] for _ in queries]

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) > self.batch_size,
        )
        scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

        reranked_batches: list[list[SearchResult]] = []
        offset = 0
        for candidates in candidate_batches:
            batch_scores = scores[offset : offset + len(candidates)]
            offset += len(candidates)
            scored_candidates = [
                {
                    **candidate,
                    "retrieval_score": candidate["score"],
                    "score": float(score),
                    "rerank_score": float(score),
                    "reranked": True,
                    "reranker_model": self.model_name,
                }
                for candidate, score in zip(candidates, batch_scores, strict=True)
            ]
            scored_candidates.sort(
                key=lambda candidate: candidate["rerank_score"],
                reverse=True,
            )
            reranked_batches.append(scored_candidates[:top_k])

        return reranked_batches


class RerankingRetriever:
    """Wrap any first-stage retriever with a cross-encoder reranking stage."""

    def __init__(
        self,
        base_retriever: Any,
        reranker: CrossEncoderReranker,
        candidate_top_k: int = 20,
    ) -> None:
        if candidate_top_k <= 0:
            raise ValueError("Reranking candidate_top_k must be greater than zero")
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve a broad candidate set, then rerank it to final top-k."""
        candidates = self.base_retriever.retrieve(
            query,
            top_k=self.candidate_top_k,
            filters=filters,
        )
        return self.reranker.rerank(query, candidates, top_k=top_k)

    def retrieve_many(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[list[SearchResult]]:
        """Retrieve and rerank multiple queries while batching model scoring."""
        candidate_batches = self.base_retriever.retrieve_many(
            queries,
            top_k=self.candidate_top_k,
        )
        return self.reranker.rerank_many(queries, candidate_batches, top_k=top_k)
