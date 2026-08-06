"""Evaluate retrieval over the complete English golden dataset."""

import csv
import json
from pathlib import Path

from recipe_rag.embedding.embedder import load_embedding_config
from recipe_rag.retrieval.dense_retriever import (
    build_retriever,
    load_retrieval_config,
)


def _is_relevant(result: dict, record: dict) -> bool:
    metadata = result["metadata"]
    return (
        metadata["source"] in record["gold_sources"]
        and metadata["section"] in record["gold_sections"]
    )


def _write_error_report(
    answerable_pairs: list[tuple[dict, list[dict]]],
    strategy: str,
    profile_name: str,
) -> tuple[Path, int]:
    """Write one compact row for each question whose correct chunk is not rank one."""
    output_path = Path("data/processed/evaluation") / (
        f"retrieval_errors_{strategy}_{profile_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "error_type",
        "question",
        "expected_answer",
        "gold_source_sections",
        "first_relevant_rank",
        "retrieved_top_5",
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
            retrieved_top_5 = [
                {
                    "rank": rank,
                    "source": result["metadata"]["source"],
                    "section": result["metadata"]["section"],
                    "content": result["content"],
                }
                for rank, result in enumerate(results, start=1)
            ]
            gold_source_sections = [
                {"source": source, "sections": record["gold_sections"]}
                for source in record["gold_sources"]
            ]
            writer.writerow(
                {
                    "question_id": record["id"],
                    "error_type": error_type,
                    "question": record["question"],
                    "expected_answer": record["expected_answer"],
                    "gold_source_sections": json.dumps(gold_source_sections),
                    "first_relevant_rank": relevant_ranks[0] if relevant_ranks else "",
                    "retrieved_top_5": json.dumps(
                        retrieved_top_5,
                        ensure_ascii=False,
                    ),
                }
            )

    return output_path, error_count


def test_retrieval_golden_dataset() -> None:
    """Run all questions and report retrieval metrics for answerable records."""
    embedding_config = load_embedding_config(Path("configs/embedding.yaml"))
    retrieval_config = load_retrieval_config(Path("configs/retrieval.yaml"))
    retriever = build_retriever(
        retrieval_config,
        embedding_config,
    )
    records = [
        json.loads(line)
        for line in Path("data/evaluation/golden_recipes_100_en.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    all_results = retriever.retrieve_many(
        [record["question"] for record in records],
        top_k=retrieval_config.final_top_k,
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
    report_profile = (
        embedding_config.profile_name
        if retrieval_config.strategy != "bm25"
        else "lexical"
    )
    evaluation_strategy = retrieval_config.strategy
    if retrieval_config.reranking_enabled:
        evaluation_strategy += "_reranked"
    report_path, error_count = _write_error_report(
        answerable_pairs,
        evaluation_strategy,
        report_profile,
    )

    assert len(records) == 100
    assert len(all_results) == 100
    assert evaluated > 0
    assert all(len(results) == retrieval_config.final_top_k for results in all_results)
    assert metrics["hit@5"] >= 0.70
    assert report_path.is_file()
    assert error_count == evaluated - hit_at[1]

    print(f"\nRetrieval strategy: {retrieval_config.strategy}")
    if retrieval_config.strategy != "bm25":
        print(f"Embedding profile: {embedding_config.profile_name}")
        print(f"Embedding model: {embedding_config.model_name}")
    if retrieval_config.strategy == "hybrid":
        print(f"Dense weight: {retrieval_config.dense_weight}")
        print(f"BM25 weight: {retrieval_config.bm25_weight}")
    print(f"Reranking enabled: {retrieval_config.reranking_enabled}")
    if retrieval_config.reranking_enabled:
        print(f"Reranker model: {retrieval_config.reranker_model_name}")
        print(f"Reranker candidates: {retrieval_config.reranker_candidate_top_k}")
    print(f"Questions: {len(records)}")
    print(f"Answerable questions evaluated: {evaluated}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"Error questions: {error_count}")
    print(f"Error report: {report_path}")


# PowerShell: python -m pytest -s .\tests\retrieval\test_retriever.py
