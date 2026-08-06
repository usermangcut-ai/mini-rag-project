"""Tests for running the ingestion loader over the complete corpus."""

from pathlib import Path

from recipe_rag.ingestion.loader import load_documents


def test_load_full_corpus() -> None:
    """Load and validate every Markdown document in the raw corpus."""
    corpus_dir = Path("data/raw")

    documents = load_documents(corpus_dir)
    document_ids = [document["document_id"] for document in documents]

    print(f"\nCorpus: {corpus_dir}")
    print(f"Documents loaded: {len(documents)}")
    print(f"First document: {document_ids[0]}")
    print(f"Last document: {document_ids[-1]}")

    assert len(documents) == 85
    assert len(document_ids) == len(set(document_ids))

    for document in documents:
        metadata = document["metadata"]

        assert document["document_id"]
        assert document["content"].strip()
        assert metadata["extension"] == ".md"
        assert metadata["filename"] == f"{document['document_id']}.md"
        assert Path(metadata["source"]).is_file()


# PowerShell: python -m pytest -s .\tests\ingestion\test_loader.py
