"""Test deterministic and RAGAS evaluation logic without real API calls."""

import asyncio
from types import SimpleNamespace

from recipe_rag.evaluation.evaluator import (
    RagasEvaluator,
    evaluate_deterministic,
    summarize_deterministic,
)


def _record(answerable: bool = True) -> dict:
    return {
        "id": "recipe-001",
        "answerable": answerable,
        "gold_sources": ["data/raw/garlic-bread.md"] if answerable else [],
        "gold_sections": ["info"] if answerable else [],
        "must_include": ["20 minutes"] if answerable else [],
    }


def _context() -> dict:
    return {
        "chunk_id": "garlic-bread::info",
        "content": "Garlic bread takes about 20 minutes.",
        "metadata": {
            "source": "data/raw/garlic-bread.md",
            "section": "info",
        },
    }


def test_deterministic_evaluation_scores_grounded_answer() -> None:
    result = evaluate_deterministic(
        _record(),
        {
            "answer": "It takes about 20 minutes [1].",
            "refused": False,
            "citations": [
                {
                    "chunk_id": "garlic-bread::info",
                    "source": "data/raw/garlic-bread.md",
                    "section": "info",
                }
            ],
        },
        [_context()],
    )

    assert result["refusal_correct"] is True
    assert result["citation_validity"] == 1.0
    assert result["citation_gold_precision"] == 1.0
    assert result["must_include_recall"] == 1.0


def test_deterministic_evaluation_scores_unanswerable_refusal() -> None:
    result = evaluate_deterministic(
        _record(answerable=False),
        {
            "answer": "I don't have enough information.",
            "refused": True,
            "citations": [],
        },
        [],
    )

    assert result["refusal_correct"] is True
    assert result["citation_validity"] is None
    assert result["must_include_recall"] is None


class _FakeMetric:
    async def ascore(self, **kwargs):
        return SimpleNamespace(value=0.75, reason=",".join(sorted(kwargs)))


def test_ragas_evaluator_maps_inputs_to_each_metric() -> None:
    evaluator = RagasEvaluator(
        {
            "faithfulness": _FakeMetric(),
            "answer_relevancy": _FakeMetric(),
            "factual_correctness": _FakeMetric(),
        }
    )

    result = asyncio.run(
        evaluator.evaluate(
            question="How long?",
            response="20 minutes.",
            retrieved_contexts=["About 20 minutes."],
            reference="About 20 minutes.",
        )
    )

    assert result["faithfulness"] == 0.75
    assert result["faithfulness_reason"] == (
        "response,retrieved_contexts,user_input"
    )
    assert result["answer_relevancy_reason"] == "response,user_input"
    assert result["factual_correctness_reason"] == "reference,response"


def test_deterministic_summary_ignores_non_applicable_values() -> None:
    summary = summarize_deterministic(
        [
            {
                "refusal_correct": True,
                "citation_validity": 1.0,
                "citation_gold_precision": 0.5,
                "must_include_recall": 1.0,
            },
            {
                "refusal_correct": False,
                "citation_validity": None,
                "citation_gold_precision": None,
                "must_include_recall": None,
            },
        ]
    )

    assert summary == {
        "refusal_accuracy": 0.5,
        "citation_validity": 1.0,
        "citation_gold_precision": 0.5,
        "must_include_recall": 1.0,
    }


# PowerShell: python -m pytest -s .\tests\evaluation\test_evaluator.py
