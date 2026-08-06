"""Manually inspect section chunks produced from one Markdown recipe."""

from pathlib import Path

from recipe_rag.chunking.chunker import chunk_document
from recipe_rag.cleaning.cleaner import clean_document
from recipe_rag.ingestion.loader import load_markdown_file


def main() -> None:
    input_value = input("Markdown file path: ").strip()
    input_path = Path(input_value)

    ingested_document = load_markdown_file(input_path)
    cleaned_document = clean_document(ingested_document)
    chunks = chunk_document(cleaned_document)

    print(f"\nPARENT DOCUMENT: {cleaned_document['document_id']}")
    print(f"TOTAL CHUNKS: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n===== CHUNK {index} =====")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"section: {chunk['metadata']['section']}")
        print("content:")
        print(chunk["content"])


if __name__ == "__main__":
    main()
