"""Evaluate retrieval over the complete English golden dataset."""

import csv
import json
from pathlib import Path

from recipe_rag.embedding.embedder import TextEmbedder, load_embedding_config
from recipe_rag.retrieval.retriever import Retriever
from recipe_rag.vector_store.store import ChromaVectorStore


def _is_relevant(result: dict, record: dict) -> bool:
    metadata = result["metadata"]
    return (
        metadata["source"] in record["gold_sources"]
        and metadata["section"] in record["gold_sections"]
    )


def _write_error_report(
    answerable_pairs: list[tuple[dict, list[dict]]],
    profile_name: str,
    model_name: str,
) -> tuple[Path, int]:
    """Write top-five results for questions whose correct chunk is not rank one."""
    output_path = Path("data/processed/evaluation") / (
        f"retrieval_errors_{profile_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "embedding_profile",
        "embedding_model",
        "error_type",
        "question_id",
        "question",
        "expected_answer",
        "gold_sources",
        "gold_sections",
        "rank",
        "retrieved_source",
        "retrieved_section",
        "score",
        "is_relevant",
        "retrieved_content",
    ]
    error_count = 0

    # utf-8-sig lets Excel on Windows detect UTF-8 correctly.
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for record, results in answerable_pairs:
            relevant_ranks = [
                rank
                for rank, result in enumerate(results, start=1)
                if _is_relevant(result, record)
            ]
            if relevant_ranks and relevant_ranks[0] == 1:
                continue

            error_count += 1
            error_type = "miss@5" if not relevant_ranks else "miss@1"
            for rank, result in enumerate(results, start=1):
                metadata = result["metadata"]
                writer.writerow(
                    {
                        "embedding_profile": profile_name,
                        "embedding_model": model_name,
                        "error_type": error_type,
                        "question_id": record["id"],
                        "question": record["question"],
                        "expected_answer": record["expected_answer"],
                        "gold_sources": json.dumps(record["gold_sources"]),
                        "gold_sections": json.dumps(record["gold_sections"]),
                        "rank": rank,
                        "retrieved_source": metadata["source"],
                        "retrieved_section": metadata["section"],
                        "score": f'{result["score"]:.6f}',
                        "is_relevant": _is_relevant(result, record),
                        "retrieved_content": result["content"],
                    }
                )

    return output_path, error_count


def test_retrieval_golden_dataset() -> None:
    """Run all questions and report retrieval metrics for answerable records."""
    config = load_embedding_config(Path("configs/embedding.yaml"))
    store = ChromaVectorStore(
        Path("data/vector_store") / config.profile_name,
        config.profile_name,
    )
    retriever = Retriever(TextEmbedder(config), store)
    records = [
        json.loads(line)
        for line in Path("data/evaluation/golden_recipes_100_en.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    all_results = retriever.retrieve_many(
        [record["question"] for record in records],
        top_k=5,
    )
    answerable_pairs = [
        (record, results)
        for record, results in zip(records, all_results, strict=True)
        if record["answerable"]
    ]

    hit_at = {1: 0, 3: 0, 5: 0}
    reciprocal_rank_sum = 0.0
    source_recall_sum = 0.0

    for record, results in answerable_pairs:
        relevant_ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if _is_relevant(result, record)
        ]
        for k in hit_at:
            if any(rank <= k for rank in relevant_ranks):
                hit_at[k] += 1
        if relevant_ranks:
            reciprocal_rank_sum += 1.0 / relevant_ranks[0]

        found_sources = {
            result["metadata"]["source"]
            for result in results
            if _is_relevant(result, record)
        }
        source_recall_sum += len(found_sources) / len(set(record["gold_sources"]))

    evaluated = len(answerable_pairs)
    metrics = {
        "hit@1": hit_at[1] / evaluated,
        "hit@3": hit_at[3] / evaluated,
        "hit@5": hit_at[5] / evaluated,
        "mrr@5": reciprocal_rank_sum / evaluated,
        "source_recall@5": source_recall_sum / evaluated,
    }
    report_path, error_count = _write_error_report(
        answerable_pairs,
        config.profile_name,
        config.model_name,
    )

    assert len(records) == 100
    assert len(all_results) == 100
    assert evaluated > 0
    assert all(len(results) == 5 for results in all_results)
    assert metrics["hit@5"] >= 0.70
    assert report_path.is_file()
    assert error_count == evaluated - hit_at[1]

    print(f"\nEmbedding profile: {config.profile_name}")
    print(f"Embedding model: {config.model_name}")
    print(f"Questions: {len(records)}")
    print(f"Answerable questions evaluated: {evaluated}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Error questions: {error_count}")
    print(f"Error report: {report_path}")


# PowerShell: python -m pytest -s .\tests\retrieval\test_retriever.py
