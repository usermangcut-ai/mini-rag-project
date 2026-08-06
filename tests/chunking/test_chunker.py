"""Tests for section-based chunking over the complete corpus."""

import json
from pathlib import Path

from recipe_rag.chunking.chunker import chunk_documents, save_chunks
from recipe_rag.cleaning.cleaner import clean_documents
from recipe_rag.ingestion.loader import load_documents


def test_chunk_full_corpus(tmp_path: Path) -> None:
    """Chunk all recipes and validate child-to-parent relationships."""
    documents = clean_documents(load_documents(Path("data/raw")))
    chunks = chunk_documents(documents)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    document_ids = {document["document_id"] for document in documents}

    assert len(documents) == 85
    assert len(chunks) > len(documents)
    assert len(chunk_ids) == len(set(chunk_ids))

    sections_by_document: dict[str, set[str]] = {}
    for chunk in chunks:
        document_id = chunk["document_id"]
        metadata = chunk["metadata"]
        section = metadata["section"]

        assert document_id in document_ids
        assert metadata["parent_document_id"] == document_id
        assert metadata["title"]
        assert section != "based_on"
        assert chunk["content"].startswith(f"Recipe: {metadata['title']}\n")
        assert f"Section: {section}\n" in chunk["content"]
        assert chunk["content"].strip()

        sections_by_document.setdefault(document_id, set()).add(section)

    for document_id in document_ids:
        assert "ingredients" in sections_by_document[document_id]
        assert "steps" in sections_by_document[document_id]

    output_path = tmp_path / "chunks.jsonl"
    saved_path = save_chunks(chunks, output_path)
    saved_chunks = [
        json.loads(line)
        for line in saved_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert saved_chunks == chunks

    print(f"\nParent documents: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved chunks verified: {len(saved_chunks)}")


# PowerShell: python -m pytest -s .\tests\chunking\test_chunker.py
