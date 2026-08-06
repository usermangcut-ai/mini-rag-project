"""Evaluate the full RAG path and print aggregate quality scores."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from recipe_rag.embedding.embedder import TextEmbedder, load_embedding_config
from recipe_rag.evaluation.evaluator import (
    RagasEvaluator,
    evaluate_deterministic,
    load_evaluation_config,
    summarize_deterministic,
)
from recipe_rag.generation.generator import (
    RAGGenerator,
    load_generation_config,
    load_llm_settings,
)
from recipe_rag.retrieval.dense_retriever import (
    build_retriever,
    load_retrieval_config,
)


def parse_args(default_limit: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAG evaluation and print aggregate scores."
    )
    parser.add_argument("--limit", type=int, default=default_limit)
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Skip paid RAGAS judge calls.",
    )
    return parser.parse_args()


def load_golden_records(path: Path, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("--limit must be greater than zero")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Golden dataset is empty: {path}")
    return records[:limit]


async def add_ragas_scores(
    rows: list[dict[str, Any]],
    evaluator: RagasEvaluator,
) -> None:
    """Judge answerable cases sequentially; each case runs three metrics together."""
    answerable_rows = [row for row in rows if row["answerable"]]
    for index, row in enumerate(answerable_rows, start=1):
        print(f"RAGAS {index}/{len(answerable_rows)}: {row['question_id']}")
        try:
            scores = await evaluator.evaluate(
                question=row["question"],
                response=row["generated_answer"],
                retrieved_contexts=row["retrieved_contexts"],
                reference=row["expected_answer"],
            )
            row.update(scores)
            row["ragas_error"] = ""
        except Exception as error:
            row["ragas_error"] = f"{type(error).__name__}: {error}"


def main() -> None:
    evaluation_config = load_evaluation_config(Path("configs/evaluation.yaml"))
    args = parse_args(evaluation_config.default_limit)
    records = load_golden_records(
        Path("data/evaluation/golden_recipes_100_en.jsonl"),
        args.limit,
    )

    embedding_config = load_embedding_config(Path("configs/embedding.yaml"))
    retrieval_config = load_retrieval_config(Path("configs/retrieval.yaml"))
    generation_config = load_generation_config(Path("configs/generation.yaml"))
    llm_settings = load_llm_settings(Path(".env"))

    print(
        f"Cases: {len(records)} | embedding: {embedding_config.profile_name} | "
        f"retrieval: {retrieval_config.strategy} | LLM: {llm_settings.model}"
    )
    # Reuse one in-memory embedding model for retrieval and AnswerRelevancy.
    embedder = TextEmbedder(embedding_config)
    retriever = build_retriever(
        retrieval_config,
        embedding_config,
        embedder=embedder,
    )
    questions = [record["question"] for record in records]
    context_batches = retriever.retrieve_many(
        questions,
        top_k=retrieval_config.final_top_k,
    )
    generator = RAGGenerator(generation_config, llm_settings)

    rows: list[dict[str, Any]] = []
    for index, (record, contexts) in enumerate(
        zip(records, context_batches, strict=True),
        start=1,
    ):
        print(f"Generation {index}/{len(records)}: {record['id']}")
        generated = generator.generate(record["question"], contexts)
        deterministic = evaluate_deterministic(record, generated, contexts)
        rows.append(
            {
                **deterministic,
                "question": record["question"],
                "expected_answer": record["expected_answer"],
                "generated_answer": generated["answer"],
                "retrieved_contexts": [context["content"] for context in contexts],
            }
        )

    run_ragas = evaluation_config.ragas_enabled and not args.deterministic_only
    if run_ragas:
        evaluator = RagasEvaluator.from_settings(
            llm_settings,
            embedder,
            timeout_seconds=evaluation_config.judge_timeout_seconds,
        )
        asyncio.run(add_ragas_scores(rows, evaluator))

    summary = summarize_deterministic(rows)
    print("\nDETERMINISTIC SUMMARY")
    for metric, score in summary.items():
        print(f"{metric}: {score:.4f}")
    if run_ragas:
        for metric in ("faithfulness", "answer_relevancy", "factual_correctness"):
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if values:
                print(f"{metric}: {sum(values) / len(values):.4f}")
        errors = [row for row in rows if row.get("ragas_error")]
        if errors:
            print(f"ragas_errors: {len(errors)}")
            for row in errors:
                print(f"- {row['question_id']}: {row['ragas_error']}")


if __name__ == "__main__":
    main()


# PowerShell, skip RAGAS judge calls (generation still calls the configured LLM):
# python .\scripts\inspect_evaluation.py --limit 10 --deterministic-only
# PowerShell, including paid RAGAS judges:
# python .\scripts\inspect_evaluation.py --limit 10
