"""Manually inspect retrieval results for one question."""

from pathlib import Path

from recipe_rag.embedding.embedder import TextEmbedder, load_embedding_config
from recipe_rag.retrieval.retriever import Retriever
from recipe_rag.vector_store.store import ChromaVectorStore


CONFIG_PATH = Path("configs/embedding.yaml")
VECTOR_STORE_DIR = Path("data/vector_store")


def main() -> None:
    query = input("Question: ").strip()
    config = load_embedding_config(CONFIG_PATH)
    retriever = Retriever(
        TextEmbedder(config),
        ChromaVectorStore(VECTOR_STORE_DIR / config.profile_name, config.profile_name),
    )

    results = retriever.retrieve(query, top_k=5)
    print(f"\nQUERY: {query}")
    for rank, result in enumerate(results, start=1):
        print(f"\n===== RANK {rank} =====")
        print(f"chunk_id: {result['chunk_id']}")
        print(f"score: {result['score']:.4f}")
        print(f"source: {result['metadata']['source']}")
        print(f"section: {result['metadata']['section']}")
        print("content:")
        print(result["content"])


if __name__ == "__main__":
    main()
