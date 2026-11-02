"""
Evaluation helpers covering all metric categories from the RAG Evaluation Metrics
infographic (Retrieval + Generator + End-to-End).

Retrieval metrics (require labeled eval set with relevant_chunk_ids):
  mrr              — Mean Reciprocal Rank
  ndcg_at_k        — Normalized Discounted Cumulative Gain @ K
  precision_at_k   — Fraction of top-K retrieved that are relevant
  recall_at_k      — Fraction of all relevant docs found in top-K

Generator metrics (reference-based, require ground_truth answers):
  rouge_scores     — ROUGE-1/2/L via rouge_score library
  bertscore        — Semantic similarity via bert_score

End-to-end metrics (LLM-as-judge, no ground truth needed for faithfulness/relevancy):
  run_ragas        — faithfulness, answer_relevancy, context_precision, context_recall
                     via the RAGAS library (uses OpenAI GPT-4o as judge by default)

LangSmith integration:
  log_to_langsmith — log per-query scores as feedback on LangSmith run traces
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np


# ------------------------------------------------------------------
# Retrieval metrics
# ------------------------------------------------------------------


def _reciprocal_rank(relevances: list[int]) -> float:
    """1 / rank of the first relevant document, or 0 if none found."""
    for i, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def mrr(all_relevances: list[list[int]]) -> float:
    """Mean Reciprocal Rank over multiple queries."""
    if not all_relevances:
        return 0.0
    return float(np.mean([_reciprocal_rank(r) for r in all_relevances]))


def _dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(relevances: list[int], k: int) -> float:
    """Normalized DCG at K. relevances is binary (1=relevant, 0=not) in ranked order."""
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    return _dcg_at_k(relevances, k) / idcg if idcg > 0 else 0.0


def precision_at_k(relevances: list[int], k: int) -> float:
    """Fraction of top-K retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    return sum(relevances[:k]) / k


def recall_at_k(relevances: list[int], k: int, n_relevant: int) -> float:
    """Fraction of all relevant documents retrieved in top-K."""
    if n_relevant == 0:
        return 0.0
    return sum(relevances[:k]) / n_relevant


def retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 10,
) -> dict[str, float]:
    """
    Compute all retrieval metrics for a single query.

    Args:
        retrieved_ids: Chunk IDs returned by the retriever in ranked order.
        relevant_ids:  Ground-truth set of relevant chunk IDs for this query.
        k:             Cutoff for Precision@K, Recall@K, NDCG@K.

    Returns dict with mrr_single, ndcg, precision, recall.
    """
    binary = [1 if cid in relevant_ids else 0 for cid in retrieved_ids]
    n_rel = len(relevant_ids)
    return {
        "mrr_single": _reciprocal_rank(binary),
        f"ndcg@{k}": ndcg_at_k(binary, k),
        f"precision@{k}": precision_at_k(binary, k),
        f"recall@{k}": recall_at_k(binary, k, n_rel),
    }


# ------------------------------------------------------------------
# Generator metrics (reference-based)
# ------------------------------------------------------------------


def rouge_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Compute average ROUGE-1, ROUGE-2, ROUGE-L across prediction/reference pairs.

    Requires: pip install rouge-score
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rl)),
    }


def bertscore_f1(predictions: list[str], references: list[str], lang: str = "en") -> float:
    """
    Compute average BERTScore F1 across prediction/reference pairs.

    Requires: pip install bert-score
    Uses distilbert-base-uncased by default for speed.
    """
    from bert_score import score as bert_score

    _, _, f1 = bert_score(predictions, references, lang=lang, verbose=False)
    return float(f1.mean().item())


# ------------------------------------------------------------------
# End-to-end metrics via RAGAS
# ------------------------------------------------------------------


def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """
    Run RAGAS evaluation and return a dict of metric → mean score.

    Metrics always computed (no ground truth needed):
      faithfulness       — is the answer grounded in the retrieved context?
      answer_relevancy   — does the answer address the question?

    Metrics computed when ground_truths provided:
      context_precision  — are retrieved chunks actually relevant? (≈ Precision@K)
      context_recall     — are all relevant facts covered by context? (≈ Recall@K)

    Args:
        judge_model: OpenAI model name for LLM-as-judge (default: gpt-4o-mini).
                     Set RAGAS_JUDGE_MODEL env var to override globally.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    judge_model = judge_model or os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")

    data: dict[str, list] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    metrics = [faithfulness, answer_relevancy]

    if ground_truths is not None:
        data["ground_truth"] = ground_truths
        metrics += [context_precision, context_recall]

    dataset = Dataset.from_dict(data)

    # Use a LangChain-wrapped OpenAI model as judge so LangSmith traces it
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm_wrapper = LangchainLLMWrapper(ChatOpenAI(model=judge_model))
    result = evaluate(dataset, metrics=metrics, llm=llm_wrapper)

    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}


# ------------------------------------------------------------------
# LangSmith feedback logging
# ------------------------------------------------------------------


def log_to_langsmith(
    run_ids: list[str],
    per_query_scores: list[dict[str, float]],
) -> None:
    """
    Log per-query metric scores as feedback on LangSmith run traces.

    Requires LANGCHAIN_API_KEY to be set. Silently skips if not available.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return

    try:
        from langsmith import Client

        ls_client = Client(api_key=api_key)
        for run_id, scores in zip(run_ids, per_query_scores):
            for key, value in scores.items():
                ls_client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=value,
                    source_info={"evaluator": "rag-eval-suite"},
                )
    except Exception as exc:
        print(f"[LangSmith] Could not log feedback: {exc}")
