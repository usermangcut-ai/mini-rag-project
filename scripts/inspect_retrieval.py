"""Manually inspect retrieval results for one question."""

from pathlib import Path

from recipe_rag.embedding.embedder import load_embedding_config
from recipe_rag.retrieval.dense_retriever import (
    build_retriever,
    load_retrieval_config,
)


CONFIG_PATH = Path("configs/embedding.yaml")
RETRIEVAL_CONFIG_PATH = Path("configs/retrieval.yaml")


def main() -> None:
    query = input("Question: ").strip()
    embedding_config = load_embedding_config(CONFIG_PATH)
    retrieval_config = load_retrieval_config(RETRIEVAL_CONFIG_PATH)
    retriever = build_retriever(retrieval_config, embedding_config)

    results = retriever.retrieve(query, top_k=retrieval_config.final_top_k)
    print(f"\nSTRATEGY: {retrieval_config.strategy}")
    if retrieval_config.strategy != "bm25":
        print(f"EMBEDDING: {embedding_config.profile_name}")
    print(f"RERANKING: {retrieval_config.reranking_enabled}")
    if retrieval_config.reranking_enabled:
        print(f"RERANKER: {retrieval_config.reranker_model_name}")
    print(f"\nQUERY: {query}")
    for rank, result in enumerate(results, start=1):
        print(f"\n===== RANK {rank} =====")
        print(f"chunk_id: {result['chunk_id']}")
        print(f"score: {result['score']:.4f}")
        if result.get("reranked"):
            print(f"retrieval_score: {result['retrieval_score']:.4f}")
            print(f"rerank_score: {result['rerank_score']:.4f}")
        if result.get("retrieval_method") == "hybrid":
            print(f"dense_rank: {result['dense_rank']}")
            print(f"bm25_rank: {result['bm25_rank']}")
        print(f"source: {result['metadata']['source']}")
        print(f"section: {result['metadata']['section']}")
        print("content:")
        print(result["content"])


if __name__ == "__main__":
    main()
