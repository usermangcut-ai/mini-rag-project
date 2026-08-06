"""Persist and search externally generated embeddings with ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


EmbeddedChunk = dict[str, Any]
SearchResult = dict[str, Any]


class ChromaVectorStore:
    """A persistent Chroma collection tied to one embedding profile."""

    def __init__(self, persist_directory: str | Path, profile_name: str) -> None:
        self.persist_directory = Path(persist_directory)
        self.profile_name = profile_name
        self.collection_name = f"recipes-{profile_name.replace('_', '-')}"
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    def build(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Rebuild the profile collection from embedded chunks."""
        if not embedded_chunks:
            raise ValueError("Cannot build a vector store from an empty chunk list")

        first_embedding_metadata = embedded_chunks[0]["embedding_metadata"]
        model_name = first_embedding_metadata["model_name"]
        dimension = int(first_embedding_metadata["dimension"])
        normalized = bool(first_embedding_metadata["normalized"])

        for chunk in embedded_chunks:
            vector = chunk.get("embedding")
            embedding_metadata = chunk.get("embedding_metadata", {})
            if not isinstance(vector, list) or len(vector) != dimension:
                raise ValueError(f"Invalid embedding for chunk {chunk.get('chunk_id')}")
            if embedding_metadata.get("profile_name") != self.profile_name:
                raise ValueError("Embedding profile does not match vector-store profile")
            if embedding_metadata.get("model_name") != model_name:
                raise ValueError("Mixed embedding models are not allowed in one collection")

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={
                "profile_name": self.profile_name,
                "model_name": model_name,
                "dimension": dimension,
                "normalized": normalized,
            },
            configuration={"hnsw": {"space": "cosine"}},
        )

        batch_size = 100
        for start in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[start : start + batch_size]
            collection.add(
                ids=[chunk["chunk_id"] for chunk in batch],
                embeddings=[chunk["embedding"] for chunk in batch],
                documents=[chunk["content"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
            )

        return collection.count()

    def _get_collection(self):
        """Open the existing collection without a Chroma embedding function."""
        return self.client.get_collection(
            name=self.collection_name,
            embedding_function=None,
        )

    def validate_model(self, model_name: str, dimension: int) -> None:
        """Reject queries produced by a model incompatible with the index."""
        metadata = self._get_collection().metadata or {}
        if metadata.get("profile_name") != self.profile_name:
            raise ValueError("Vector-store profile metadata does not match")
        if metadata.get("model_name") != model_name:
            raise ValueError(
                f"Query model {model_name} does not match index model "
                f"{metadata.get('model_name')}"
            )
        if int(metadata.get("dimension", -1)) != dimension:
            raise ValueError("Query vector dimension does not match the index")

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return self._get_collection().count()

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the nearest chunks using cosine distance."""
        if not query_vector:
            raise ValueError("Query vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        collection = self._get_collection()
        metadata = collection.metadata or {}
        expected_dimension = int(metadata.get("dimension", -1))
        if len(query_vector) != expected_dimension:
            raise ValueError(
                f"Query dimension {len(query_vector)} does not match "
                f"index dimension {expected_dimension}"
            )

        query_result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        ids = query_result["ids"][0]
        documents = query_result["documents"][0]
        metadatas = query_result["metadatas"][0]
        distances = query_result["distances"][0]

        return [
            {
                "chunk_id": chunk_id,
                "score": 1.0 - float(distance),
                "content": document,
                "metadata": result_metadata,
            }
            for chunk_id, document, result_metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
