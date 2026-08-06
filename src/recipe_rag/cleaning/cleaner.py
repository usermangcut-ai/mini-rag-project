"""Clean and normalize ingested documents."""

from __future__ import annotations

import re
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


Document = dict[str, Any]


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving Markdown structure and wording."""
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    lines_without_trailing_spaces = [
        line.rstrip() for line in normalized_newlines.split("\n")
    ]
    cleaned = "\n".join(lines_without_trailing_spaces)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def clean_document(document: Document) -> Document:
    """Return a cleaned copy of one ingested document."""
    if "document_id" not in document:
        raise ValueError("Document is missing document_id")
    if "content" not in document:
        raise ValueError(f"Document {document['document_id']} is missing content")
    if "metadata" not in document:
        raise ValueError(f"Document {document['document_id']} is missing metadata")

    content = document["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Document {document['document_id']} has empty content")

    cleaned_document = deepcopy(document)
    cleaned_document["content"] = clean_text(content)
    return cleaned_document


def clean_documents(documents: list[Document]) -> list[Document]:
    """Clean every ingested document without mutating the input list."""
    return [clean_document(document) for document in documents]


def save_cleaned_documents(
    documents: list[Document], output_path: str | Path
) -> Path:
    """Save cleaned documents as UTF-8 JSONL and return the output path."""
    if not documents:
        raise ValueError("Cannot save an empty document list")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for document in documents:
            output_file.write(json.dumps(document, ensure_ascii=False) + "\n")

    return path
