"""Load raw Markdown documents into the RAG pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


Document = dict[str, Any]


def _source_path(file_path: Path) -> str:
    """Return a portable source path, relative to the current project if possible."""
    try:
        return file_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return file_path.resolve().as_posix()


def load_markdown_file(file_path: str | Path) -> Document:
    """Load one Markdown file and return its content with source metadata."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Markdown file does not exist: {path}")

    if path.suffix.lower() != ".md":
        raise ValueError(f"Expected a Markdown file, got: {path}")

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"Markdown file is empty: {path}")

    return {
        "document_id": path.stem,
        "content": content,
        "metadata": {
            "source": _source_path(path),
            "filename": path.name,
            "extension": path.suffix.lower(),
        },
    }


def load_documents(corpus_dir: str | Path) -> list[Document]:
    """Load every Markdown file in a corpus directory in deterministic order."""
    directory = Path(corpus_dir)

    if not directory.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {directory}")

    markdown_files = sorted(directory.glob("*.md"), key=lambda path: path.name.lower())
    if not markdown_files:
        raise ValueError(f"No Markdown files found in: {directory}")

    documents = [load_markdown_file(path) for path in markdown_files]
    document_ids = [document["document_id"] for document in documents]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate document_id values found in the corpus")

    return documents


def main() -> None:
    """Run the loader manually and display a compact, real output sample."""
    parser = argparse.ArgumentParser(description="Load a Markdown recipe corpus")
    parser.add_argument("corpus_dir", type=Path, help="Directory containing .md files")
    args = parser.parse_args()

    documents = load_documents(args.corpus_dir)
    first_document = documents[0]
    output_sample = {
        **first_document,
        "content": first_document["content"][:500],
    }

    print(f"Input directory: {args.corpus_dir.as_posix()}")
    print(f"Documents loaded: {len(documents)}")
    print("First output document (content limited to 500 characters):")
    print(json.dumps(output_sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
