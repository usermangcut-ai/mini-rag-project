"""Build a persistent Chroma collection for the active embedding profile."""

import json
from pathlib import Path

from recipe_rag.embedding.embedder import load_embedding_config
from recipe_rag.vector_store.store import ChromaVectorStore


CONFIG_PATH = Path("configs/embedding.yaml")
EMBEDDINGS_DIR = Path("data/processed/embeddings")
VECTOR_STORE_DIR = Path("data/vector_store")


def main() -> None:
    config = load_embedding_config(CONFIG_PATH)
    embeddings_path = EMBEDDINGS_DIR / f"{config.profile_name}.jsonl"

    if not embeddings_path.is_file():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_path}. "
            "Run scripts/build_embeddings.py first."
        )

    embedded_chunks = [
        json.loads(line)
        for line in embeddings_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    store_path = VECTOR_STORE_DIR / config.profile_name
    store = ChromaVectorStore(store_path, config.profile_name)
    indexed_count = store.build(embedded_chunks)

    print(f"Profile: {config.profile_name}")
    print(f"Model: {config.model_name}")
    print(f"Chunks indexed: {indexed_count}")
    print(f"Persisted at: {store_path.as_posix()}")


if __name__ == "__main__":
    main()
