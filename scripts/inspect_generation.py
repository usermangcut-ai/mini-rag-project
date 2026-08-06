"""Run retrieval and generation for one manually entered question."""

from pathlib import Path

from recipe_rag.embedding.embedder import load_embedding_config
from recipe_rag.generation.generator import (
    RAGGenerator,
    load_generation_config,
    load_llm_settings,
)
from recipe_rag.retrieval.dense_retriever import (
    build_retriever,
    load_retrieval_config,
)


def main() -> None:
    question = input("Question: ").strip()
    embedding_config = load_embedding_config(Path("configs/embedding.yaml"))
    retrieval_config = load_retrieval_config(Path("configs/retrieval.yaml"))
    generation_config = load_generation_config(Path("configs/generation.yaml"))

    retriever = build_retriever(retrieval_config, embedding_config)
    contexts = retriever.retrieve(question, top_k=retrieval_config.final_top_k)
    generator = RAGGenerator(
        generation_config,
        load_llm_settings(Path(".env")),
    )
    result = generator.generate(question, contexts)

    print(f"\nMODEL: {result['model']}")
    print(f"REFUSED: {result['refused']}")
    if result["guardrail_reason"]:
        print(f"GUARDRAIL: {result['guardrail_reason']}")
    print("\nANSWER:")
    print(result["answer"])
    print("\nCITATIONS:")
    for citation in result["citations"]:
        print(
            f"[{citation['index']}] {citation['source']} "
            f"({citation['section']})"
        )


if __name__ == "__main__":
    main()
