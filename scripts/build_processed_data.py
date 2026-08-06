"""Build the cleaned JSONL dataset from the raw Markdown corpus."""

from pathlib import Path

from recipe_rag.cleaning.cleaner import clean_documents, save_cleaned_documents
from recipe_rag.ingestion.loader import load_documents


RAW_DATA_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/cleaned_documents.jsonl")


def main() -> None:
    ingested_documents = load_documents(RAW_DATA_DIR)
    cleaned_documents = clean_documents(ingested_documents)
    saved_path = save_cleaned_documents(cleaned_documents, OUTPUT_PATH)

    print(f"Raw documents: {len(ingested_documents)}")
    print(f"Cleaned documents: {len(cleaned_documents)}")
    print(f"Saved to: {saved_path.as_posix()}")


if __name__ == "__main__":
    main()
