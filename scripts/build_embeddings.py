"""Build embeddings for all section chunks using the active model config."""

from pathlib import Path

from recipe_rag.chunking.chunker import chunk_documents
from recipe_rag.cleaning.cleaner import clean_documents
from recipe_rag.embedding.embedder import (
    TextEmbedder,
    load_embedding_config,
    save_embedded_chunks,
)
from recipe_rag.ingestion.loader import load_documents


CONFIG_PATH = Path("configs/embedding.yaml")
RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/embeddings")


def main() -> None:
    config = load_embedding_config(CONFIG_PATH)
    chunks = chunk_documents(clean_documents(load_documents(RAW_DATA_DIR)))
    embedded_chunks = TextEmbedder(config).embed_chunks(chunks)
    output_path = OUTPUT_DIR / f"{config.profile_name}.jsonl"
    saved_path = save_embedded_chunks(embedded_chunks, output_path)

    print(f"Model: {config.model_name}")
    print(f"Chunks embedded: {len(embedded_chunks)}")
    print(f"Vector dimension: {embedded_chunks[0]['embedding_metadata']['dimension']}")
    print(f"Saved to: {saved_path.as_posix()}")


if __name__ == "__main__":
    main()
