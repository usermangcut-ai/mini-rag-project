"""Tests for running cleaning over the complete corpus."""

import json
from pathlib import Path

from recipe_rag.cleaning.cleaner import clean_documents, save_cleaned_documents
from recipe_rag.ingestion.loader import load_documents


def test_clean_full_corpus(tmp_path: Path) -> None:
    """Clean and validate every document loaded from the raw corpus."""
    corpus_dir = Path("data/raw")
    ingested_documents = load_documents(corpus_dir)
    original_contents = [document["content"] for document in ingested_documents]

    cleaned_documents = clean_documents(ingested_documents)
    changed_count = 0

    assert len(cleaned_documents) == 85

    for original, cleaned in zip(ingested_documents, cleaned_documents, strict=True):
        assert cleaned["document_id"] == original["document_id"]
        assert cleaned["metadata"] == original["metadata"]
        assert cleaned["metadata"] is not original["metadata"]
        assert cleaned["content"].strip()
        assert "\r" not in cleaned["content"]
        assert "\n\n\n" not in cleaned["content"]
        assert all(line == line.rstrip() for line in cleaned["content"].splitlines())

        if cleaned["content"] != original["content"]:
            changed_count += 1

    assert [document["content"] for document in ingested_documents] == original_contents
    assert changed_count > 0

    output_path = tmp_path / "cleaned_documents.jsonl"
    saved_path = save_cleaned_documents(cleaned_documents, output_path)
    saved_records = [
        json.loads(line)
        for line in saved_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert saved_path == output_path
    assert len(saved_records) == 85
    assert saved_records == cleaned_documents

    print(f"\nDocuments cleaned: {len(cleaned_documents)}")
    print(f"Documents changed: {changed_count}")
    print(f"Saved records verified: {len(saved_records)}")


# PowerShell: python -m pytest -s .\tests\cleaning\test_cleaner.py
