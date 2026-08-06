"""Manually inspect embeddings for the chunks of one Markdown recipe."""

from pathlib import Path

from recipe_rag.chunking.chunker import chunk_document
from recipe_rag.cleaning.cleaner import clean_document
from recipe_rag.embedding.embedder import TextEmbedder, load_embedding_config
from recipe_rag.ingestion.loader import load_markdown_file


CONFIG_PATH = Path("configs/embedding.yaml")


def main() -> None:
    input_value = input("Markdown file path: ").strip()
    input_path = Path(input_value)

    document = clean_document(load_markdown_file(input_path))
    chunks = chunk_document(document)
    config = load_embedding_config(CONFIG_PATH)
    embedder = TextEmbedder(config)
    embedded_chunks = embedder.embed_chunks(chunks)

    print(f"\nMODEL PROFILE: {config.profile_name}")
    print(f"MODEL NAME: {config.model_name}")
    print(f"CHUNKS EMBEDDED: {len(embedded_chunks)}")

    for chunk in embedded_chunks:
        vector = chunk["embedding"]
        print(f"\n===== {chunk['chunk_id']} =====")
        print(f"dimension: {len(vector)}")
        print(f"first 10 values: {vector[:10]}")


if __name__ == "__main__":
    main()
