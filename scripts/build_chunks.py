"""Build section-based chunks from the raw Markdown corpus."""

from pathlib import Path

from recipe_rag.chunking.chunker import chunk_documents, save_chunks
from recipe_rag.cleaning.cleaner import clean_documents
from recipe_rag.ingestion.loader import load_documents


RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")


def main() -> None:
    ingested_documents = load_documents(RAW_DATA_DIR)
    cleaned_documents = clean_documents(ingested_documents)
    chunks = chunk_documents(cleaned_documents)
    saved_path = save_chunks(chunks, OUTPUT_PATH)

    print(f"Parent documents: {len(cleaned_documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved to: {saved_path.as_posix()}")


if __name__ == "__main__":
    main()
