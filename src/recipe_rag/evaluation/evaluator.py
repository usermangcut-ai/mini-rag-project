"""Evaluate deterministic RAG rules and optional RAGAS judge metrics."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from recipe_rag.generation.generator import LLMSettings


@dataclass(frozen=True)
class EvaluationConfig:
    """Runtime limits for the end-to-end RAG evaluation script."""

    default_limit: int
    ragas_enabled: bool
    judge_timeout_seconds: float


def load_evaluation_config(config_path: str | Path) -> EvaluationConfig:
    """Load evaluation settings that are safe to commit."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = raw.get("evaluation", {})
    config = EvaluationConfig(
        default_limit=int(values.get("default_limit", 10)),
        ragas_enabled=bool(values.get("ragas_enabled", True)),
        judge_timeout_seconds=float(values.get("judge_timeout_seconds", 60)),
    )
    if config.default_limit <= 0 or config.judge_timeout_seconds <= 0:
        raise ValueError("Evaluation limit and timeout must be greater than zero")
    return config


def _normalize(text: str) -> str:
    """Normalize text for stable must-include matching."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def evaluate_deterministic(
    golden_record: dict[str, Any],
    generation_result: dict[str, Any],
    retrieved_contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score citations, refusal behavior, and required answer phrases."""
    answerable = bool(golden_record["answerable"])
    refused = bool(generation_result["refused"])
    citations = generation_result.get("citations", [])
    retrieved_by_id = {
        context.get("chunk_id"): context
        for context in retrieved_contexts
        if context.get("chunk_id")
    }

    citation_checks = []
    gold_citation_checks = []
    for citation in citations:
        context = retrieved_by_id.get(citation.get("chunk_id"))
        citation_checks.append(
            context is not None
            and citation.get("source") == context["metadata"].get("source")
            and citation.get("section") == context["metadata"].get("section")
        )
        gold_citation_checks.append(
            citation.get("source") in golden_record.get("gold_sources", [])
            and citation.get("section") in golden_record.get("gold_sections", [])
        )

    normalized_answer = _normalize(generation_result["answer"])
    must_include = golden_record.get("must_include", [])
    included_count = sum(
        _normalize(required_text) in normalized_answer
        for required_text in must_include
    )

    return {
        "question_id": golden_record["id"],
        "answerable": answerable,
        "refused": refused,
        "refusal_correct": refused != answerable,
        "citation_validity": (
            sum(citation_checks) / len(citation_checks) if citation_checks else None
        ),
        "citation_gold_precision": (
            sum(gold_citation_checks) / len(gold_citation_checks)
            if gold_citation_checks
            else None
        ),
        "must_include_recall": (
            included_count / len(must_include)
            if answerable and must_include
            else None
        ),
    }


def summarize_deterministic(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average deterministic metrics while ignoring non-applicable values."""
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation result")

    def average(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return mean(values) if values else 0.0

    return {
        "refusal_accuracy": average("refusal_correct"),
        "citation_validity": average("citation_validity"),
        "citation_gold_precision": average("citation_gold_precision"),
        "must_include_recall": average("must_include_recall"),
    }


class RagasEvaluator:
    """Run RAGAS collection metrics without coupling them to unit tests."""

    def __init__(self, metrics: dict[str, Any]) -> None:
        if not metrics:
            raise ValueError("At least one RAGAS metric is required")
        self.metrics = metrics

    @classmethod
    def from_settings(
        cls,
        settings: LLMSettings,
        embedding_model: Any,
        timeout_seconds: float = 60,
    ) -> RagasEvaluator:
        """Create RAGAS 0.4 collection metrics with an OpenAI-compatible client."""
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import BaseRagasEmbedding
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                Faithfulness,
                FactualCorrectness,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install evaluation dependencies with: "
                "python -m pip install -e '.[dev,evaluation]'"
            ) from error

        client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=timeout_seconds,
        )
        judge_llm = llm_factory(
            settings.model,
            provider="openai",
            client=client,
            temperature=0.0,
        )

        class LocalQueryEmbeddings(BaseRagasEmbedding):
            """Adapt the project's already-loaded query embedder to RAGAS."""

            def embed_text(self, text: str, **kwargs: Any) -> list[float]:
                return embedding_model.embed_query(text)

            async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
                return await asyncio.to_thread(embedding_model.embed_query, text)

        judge_embeddings = LocalQueryEmbeddings()
        return cls(
            {
                "faithfulness": Faithfulness(llm=judge_llm),
                "answer_relevancy": AnswerRelevancy(
                    llm=judge_llm,
                    embeddings=judge_embeddings,
                ),
                "factual_correctness": FactualCorrectness(llm=judge_llm),
            }
        )

    async def evaluate(
        self,
        question: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
    ) -> dict[str, Any]:
        """Evaluate one answer and retain metric reasons for case-level debugging."""
        metric_inputs = {
            "faithfulness": {
                "user_input": question,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            },
            "answer_relevancy": {
                "user_input": question,
                "response": response,
            },
            "factual_correctness": {
                "response": response,
                "reference": reference,
            },
        }

        async def run_metric(name: str, metric: Any) -> tuple[str, Any]:
            result = await metric.ascore(**metric_inputs[name])
            return name, result

        results = await asyncio.gather(
            *(run_metric(name, metric) for name, metric in self.metrics.items())
        )
        output: dict[str, Any] = {}
        for name, result in results:
            output[name] = float(result.value)
            output[f"{name}_reason"] = result.reason
        return output
