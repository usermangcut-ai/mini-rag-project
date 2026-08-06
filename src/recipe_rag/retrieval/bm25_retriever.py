"""Retrieve chunks by exact lexical overlap with BM25."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


Chunk = dict[str, Any]
SearchResult = dict[str, Any]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Normalize text while preserving useful tokens such as ``1/4``."""
    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """Build an in-memory BM25 index over already-created chunks."""

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 from an empty chunk list")
        if any(not chunk.get("chunk_id") or not chunk.get("content") for chunk in chunks):
            raise ValueError("Every BM25 chunk must contain chunk_id and content")
        if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
            raise ValueError("BM25 chunk IDs must be unique")

        self.chunks = chunks
        tokenized_corpus = [tokenize(chunk["content"]) for chunk in chunks]
        self.index = BM25Okapi(tokenized_corpus)

    @classmethod
    def from_jsonl(cls, chunks_path: str | Path) -> BM25Retriever:
        """Load chunk records from JSONL and build their lexical index."""
        path = Path(chunks_path)
        if not path.is_file():
            raise FileNotFoundError(f"Chunks file does not exist: {path}")

        chunks = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Rank chunks by BM25 score and return the shared retrieval schema."""
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        scores = self.index.get_scores(query_tokens)
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if self._matches_filters(self.chunks[index], filters)
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)

        return [
            {
                "chunk_id": self.chunks[index]["chunk_id"],
                "score": score,
                "content": self.chunks[index]["content"],
                "metadata": self.chunks[index]["metadata"],
                "retrieval_method": "bm25",
            }
            for index, score in candidates[:top_k]
        ]

    def retrieve_many(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[list[SearchResult]]:
        """Retrieve lexical candidates for multiple queries."""
        if not queries:
            raise ValueError("Queries cannot be empty")
        return [self.retrieve(query, top_k=top_k) for query in queries]

    @staticmethod
    def _matches_filters(
        chunk: Chunk,
        filters: dict[str, Any] | None,
    ) -> bool:
        """Apply simple metadata equality filters, matching Chroma's common case."""
        if not filters:
            return True
        metadata = chunk.get("metadata", {})
        return all(metadata.get(key) == value for key, value in filters.items())
