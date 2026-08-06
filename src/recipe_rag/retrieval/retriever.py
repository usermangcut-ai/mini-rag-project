"""Retrieve relevant chunks by embedding queries and searching ChromaDB."""

from __future__ import annotations

from typing import Any

from recipe_rag.embedding.embedder import TextEmbedder
from recipe_rag.vector_store.store import ChromaVectorStore


SearchResult = dict[str, Any]


class Retriever:
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
