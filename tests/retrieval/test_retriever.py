"""Evaluate retrieval over the complete English golden dataset."""

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

    assert len(records) == 100
    assert len(all_results) == 100
    assert evaluated > 0
    assert all(len(results) == 5 for results in all_results)
    assert metrics["hit@5"] >= 0.70

    print(f"\nQuestions: {len(records)}")
    print(f"Answerable questions evaluated: {evaluated}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


# PowerShell: python -m pytest -s .\tests\retrieval\test_retriever.py
