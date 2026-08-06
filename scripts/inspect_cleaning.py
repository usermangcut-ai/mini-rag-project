"""Manually inspect cleaning input and output."""

from pathlib import Path

from recipe_rag.cleaning.cleaner import clean_document
from recipe_rag.ingestion.loader import load_markdown_file


def main() -> None:
    input_value = input("Markdown file path: ").strip()
    input_path = Path(input_value)

    ingested_document = load_markdown_file(input_path)
    cleaned_document = clean_document(ingested_document)

    print("\nINPUT CONTENT:")
    print(repr(ingested_document["content"]))
    print("\nCLEANED CONTENT:")
    print(repr(cleaned_document["content"]))
    print("\nMETADATA:")
    print(cleaned_document["metadata"])


if __name__ == "__main__":
    main()
