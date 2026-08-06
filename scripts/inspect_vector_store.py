"""Manually embed a query and inspect Chroma search results."""

from pathlib import Path

from recipe_rag.embedding.embedder import TextEmbedder, load_embedding_config
from recipe_rag.vector_store.store import ChromaVectorStore


CONFIG_PATH = Path("configs/embedding.yaml")
VECTOR_STORE_DIR = Path("data/vector_store")


def main() -> None:
    query = input("Search query: ").strip()
    config = load_embedding_config(CONFIG_PATH)
    embedder = TextEmbedder(config)
    query_vector = embedder.embed_query(query)

    store = ChromaVectorStore(
        VECTOR_STORE_DIR / config.profile_name,
        config.profile_name,
    )
    store.validate_model(config.model_name, len(query_vector))
    results = store.search(query_vector, top_k=5)

    print(f"\nPROFILE: {config.profile_name}")
    print(f"QUERY: {query}")
    for rank, result in enumerate(results, start=1):
        print(f"\n===== RESULT {rank} =====")
        print(f"chunk_id: {result['chunk_id']}")
        print(f"score: {result['score']:.4f}")
        print(f"section: {result['metadata']['section']}")
        print("content:")
        print(result["content"])


if __name__ == "__main__":
    main()
