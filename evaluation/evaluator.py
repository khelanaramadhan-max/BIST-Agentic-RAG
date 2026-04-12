"""
RAG Evaluation Pipeline.
Implements RAGAS-inspired metrics: faithfulness, answer relevancy, 
source coverage, and BIST-specific metrics.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from evaluation.questions import EVAL_QUESTIONS

logger = logging.getLogger(__name__)


# ─── Metric Definitions ──────────────────────────────────────────────────────


@dataclass
class EvalResult:
    question_id: str
    question: str
    ticker: str
    category: str
    answer: str
    sources_used: list[str]
    iteration_count: int

    # Metrics (0.0 – 1.0)
    faithfulness: float = 0.0          # Does answer stick to retrieved context?
    answer_relevancy: float = 0.0      # Does answer address the question?
    source_coverage: float = 0.0       # Were the right sources used?
    disclaimer_present: bool = False   # Was disclaimer included?
    non_advisory: bool = True          # Is answer free of investment advice?
    latency_sec: float = 0.0

    @property
    def overall_score(self) -> float:
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.3,
            "source_coverage": 0.2,
            "disclaimer_present": 0.1,
            "non_advisory": 0.1,
        }
        return (
            self.faithfulness * weights["faithfulness"]
            + self.answer_relevancy * weights["answer_relevancy"]
            + self.source_coverage * weights["source_coverage"]
            + (1.0 if self.disclaimer_present else 0.0) * weights["disclaimer_present"]
            + (1.0 if self.non_advisory else 0.0) * weights["non_advisory"]
        )


# ─── LLM-based Metrics ───────────────────────────────────────────────────────


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.0,
    )


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a faithfulness evaluator for a RAG system.
Score how well the answer is grounded in the retrieved context (0.0 to 1.0).
- 1.0: All claims in the answer are directly supported by the context
- 0.5: Some claims are supported, others are inferred or hallucinated
- 0.0: Answer contradicts or ignores the context

Respond with ONLY a JSON: {"score": 0.0-1.0, "reason": "brief explanation"}""",
        ),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
    ]
)

RELEVANCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a relevancy evaluator.
Score how directly the answer addresses the question (0.0 to 1.0).
- 1.0: Answer fully and directly addresses the question
- 0.5: Partially addresses the question
- 0.0: Does not address the question at all

Respond with ONLY a JSON: {"score": 0.0-1.0, "reason": "brief explanation"}""",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ]
)


def _llm_score(prompt: ChatPromptTemplate, inputs: dict) -> tuple[float, str]:
    """Run an LLM evaluator prompt and return (score, reason)."""
    llm = _get_llm()
    try:
        response = (prompt | llm).invoke(inputs)
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
            raw = raw.lstrip("json").strip()
        data = json.loads(raw)
        return float(data.get("score", 0.5)), data.get("reason", "")
    except Exception as exc:
        logger.warning("LLM eval error: %s", exc)
        return 0.5, f"eval error: {exc}"


# ─── Rule-based Metrics ───────────────────────────────────────────────────────


def check_disclaimer(answer: str) -> bool:
    markers = [
        "yatırım tavsiyesi",
        "investment advice",
        "this system does not",
        "yasal uyarı",
        "legal disclaimer",
    ]
    return any(m.lower() in answer.lower() for m in markers)


def check_non_advisory(answer: str) -> bool:
    """Returns True if answer does NOT contain investment advice."""
    forbidden = [
        r"\b(buy|sell|hold|strong buy|strong sell)\b",
        r"\b(al|sat|tut|güçlü al|güçlü sat)\b",
        r"\bprice target\s*[:=]?\s*[\$₺]?\d+",
        r"\bfiyat hedefi\s*[:=]?\s*[\$₺]?\d+",
        r"\b(i recommend|we recommend)\b",
    ]
    for p in forbidden:
        if re.search(p, answer, re.IGNORECASE):
            return False
    return True


def check_source_coverage(sources_used: list[str], expected_sources: list[str]) -> float:
    """
    Fraction of expected sources that were actually used.
    """
    if not expected_sources:
        return 1.0
    matched = sum(1 for s in expected_sources if s in sources_used)
    return matched / len(expected_sources)


def check_keyword_presence(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords present in the answer."""
    if not keywords:
        return 1.0
    count = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return count / len(keywords)


# ─── Full Evaluator ───────────────────────────────────────────────────────────


def evaluate_single(
    eval_q: dict,
    agent_run_fn,  # Callable: (question, ticker) -> dict with 'answer', 'sources_used', etc.
    use_llm_metrics: bool = True,
) -> EvalResult:
    """
    Evaluate a single question against the agent.
    """
    start = time.time()
    result = agent_run_fn(
        question=eval_q["question"],
        ticker=eval_q.get("ticker", ""),
    )
    latency = time.time() - start

    answer = result.get("answer", "")
    sources_used = result.get("sources_used", [])
    context = result.get("combined_context", answer[:1000])  # fallback

    # Rule-based metrics
    disclaimer_ok = check_disclaimer(answer)
    non_advisory = check_non_advisory(answer)
    source_cov = check_source_coverage(sources_used, eval_q.get("expected_sources", []))

    # LLM-based metrics
    if use_llm_metrics and answer:
        faithfulness, _ = _llm_score(
            FAITHFULNESS_PROMPT,
            {"context": context[:2000], "answer": answer[:1500]},
        )
        relevancy, _ = _llm_score(
            RELEVANCY_PROMPT,
            {"question": eval_q["question"], "answer": answer[:1500]},
        )
    else:
        # Approximate with keyword presence
        faithfulness = check_keyword_presence(answer, eval_q.get("expected_keywords", []))
        relevancy = 0.7 if answer else 0.0

    return EvalResult(
        question_id=eval_q["id"],
        question=eval_q["question"],
        ticker=eval_q.get("ticker", ""),
        category=eval_q["category"],
        answer=answer[:500] + "..." if len(answer) > 500 else answer,
        sources_used=sources_used,
        iteration_count=result.get("iteration_count", 0),
        faithfulness=faithfulness,
        answer_relevancy=relevancy,
        source_coverage=source_cov,
        disclaimer_present=disclaimer_ok,
        non_advisory=non_advisory,
        latency_sec=round(latency, 2),
    )


def run_full_evaluation(
    agent_run_fn,
    questions: list[dict] | None = None,
    use_llm_metrics: bool = True,
    max_questions: int | None = None,
) -> list[EvalResult]:
    """
    Run evaluation across all (or a subset of) BIST-specific questions.
    """
    qs = questions or EVAL_QUESTIONS
    if max_questions:
        qs = qs[:max_questions]

    results = []
    for i, q in enumerate(qs):
        logger.info(
            "Evaluating [%d/%d] %s: %s", i + 1, len(qs), q["id"], q["question"][:60]
        )
        try:
            res = evaluate_single(q, agent_run_fn, use_llm_metrics=use_llm_metrics)
            results.append(res)
            logger.info(
                "  → Overall: %.2f (F=%.2f, R=%.2f, S=%.2f)",
                res.overall_score,
                res.faithfulness,
                res.answer_relevancy,
                res.source_coverage,
            )
        except Exception as exc:
            logger.error("Evaluation error for %s: %s", q["id"], exc)

        # Small delay to avoid Groq rate limits
        time.sleep(2)

    return results


def summarise_results(results: list[EvalResult]) -> dict[str, Any]:
    """Compute aggregate metrics across all results."""
    if not results:
        return {}

    n = len(results)
    return {
        "n_questions": n,
        "avg_overall": round(sum(r.overall_score for r in results) / n, 3),
        "avg_faithfulness": round(sum(r.faithfulness for r in results) / n, 3),
        "avg_relevancy": round(sum(r.answer_relevancy for r in results) / n, 3),
        "avg_source_coverage": round(sum(r.source_coverage for r in results) / n, 3),
        "disclaimer_rate": round(sum(1 for r in results if r.disclaimer_present) / n, 3),
        "non_advisory_rate": round(sum(1 for r in results if r.non_advisory) / n, 3),
        "avg_latency_sec": round(sum(r.latency_sec for r in results) / n, 2),
        "avg_iterations": round(sum(r.iteration_count for r in results) / n, 2),
        "by_category": _by_category(results),
    }


def _by_category(results: list[EvalResult]) -> dict:
    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r.overall_score)
    return {cat: round(sum(scores) / len(scores), 3) for cat, scores in cats.items()}


def save_results(results: list[EvalResult], path: str = "evaluation/eval_results.json") -> None:
    summary = summarise_results(results)
    output = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Evaluation results saved → %s", path)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, ".")

    from agent.graph import run_agent

    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"\nRunning evaluation on {max_q} questions...")

    results = run_full_evaluation(
        agent_run_fn=lambda question, ticker: run_agent(question, ticker),
        use_llm_metrics=True,
        max_questions=max_q,
    )

    summary = summarise_results(results)
    print("\n=== EVALUATION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    save_results(results)
