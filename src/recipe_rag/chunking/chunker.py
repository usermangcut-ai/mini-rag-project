"""Split cleaned Markdown documents into section-based chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


Document = dict[str, Any]
Chunk = dict[str, Any]
DEFAULT_EXCLUDED_SECTIONS = {"based_on"}
SECTION_PATTERN = re.compile(r"^##\s+(.+?)\s*$")


def _normalize_section(section_heading: str) -> str:
    """Convert a Markdown section heading into a stable metadata value."""
    normalized = section_heading.strip().rstrip(":").strip().lower()
    return re.sub(r"\s+", "_", normalized)


def _extract_title(content: str, document_id: str) -> str:
    """Read the H1 title, falling back to the document ID."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return document_id.replace("-", " ").title()


def chunk_document(
    document: Document,
    excluded_sections: set[str] | None = None,
) -> list[Chunk]:
    """Split one cleaned document into child chunks at level-two headings."""
    document_id = document.get("document_id")
    content = document.get("content")
    metadata = document.get("metadata")

    if not document_id:
        raise ValueError("Document is missing document_id")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Document {document_id} has empty content")
    if not isinstance(metadata, dict):
        raise ValueError(f"Document {document_id} is missing metadata")

    excluded = DEFAULT_EXCLUDED_SECTIONS if excluded_sections is None else excluded_sections
    title = _extract_title(content, document_id)
    sections: list[tuple[str, list[str]]] = []
    current_section: str | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        match = SECTION_PATTERN.match(line.strip())
        if match:
            if current_section is not None:
                sections.append((current_section, current_lines))
            current_section = _normalize_section(match.group(1))
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    if current_section is not None:
        sections.append((current_section, current_lines))

    chunks: list[Chunk] = []
    section_counts: dict[str, int] = {}

    for section, section_lines in sections:
        section_body = "\n".join(section_lines).strip()
        if section in excluded or not section_body:
            continue

        section_counts[section] = section_counts.get(section, 0) + 1
        occurrence = section_counts[section]
        chunk_id = f"{document_id}::{section}"
        if occurrence > 1:
            chunk_id = f"{chunk_id}::{occurrence}"

        chunk_content = f"Recipe: {title}\nSection: {section}\n\n{section_body}"
        chunk_metadata = {
            **metadata,
            "title": title,
            "section": section,
            "parent_document_id": document_id,
        }
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "content": chunk_content,
                "metadata": chunk_metadata,
            }
        )

    if not chunks:
        raise ValueError(f"Document {document_id} produced no chunks")

    return chunks


def chunk_documents(
    documents: list[Document],
    excluded_sections: set[str] | None = None,
) -> list[Chunk]:
    """Chunk every cleaned document and reject duplicate chunk IDs."""
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, excluded_sections)
    ]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk_id values found")

    return chunks


def save_chunks(chunks: list[Chunk], output_path: str | Path) -> Path:
    """Save chunks as UTF-8 JSONL and return the output path."""
    if not chunks:
        raise ValueError("Cannot save an empty chunk list")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return path
