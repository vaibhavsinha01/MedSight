"""
evaluation.py
=============
Benchmark dataset + evaluation harness for the agentic system.

Evaluates, per benchmark item, run through the real LangGraph orchestrator:
  - RAG relevance          (retrieval_relevance_avg, deterministic, from rerank scores)
  - Groundedness           (LLM-as-judge)
  - Answer correctness     (LLM-as-judge, vs. an optional reference note)
  - Hallucination          (LLM-as-judge, inverted scale)
  - Agent/tool selection   (deterministic: does route match expected_route?)
  - Final response quality (LLM-as-judge)

Deterministic metrics are always computed. LLM-as-judge metrics use the same
EvaluationAgent as the live system so eval reflects production behavior.

Usage:
    python evaluation.py                # run full benchmark, print + save report
    python evaluation.py --n 5          # run only first 5 cases (cheaper/faster)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.graph import MedicalAgentOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("medagent.eval")

REPORT_PATH = Path(__file__).parent / "logs" / "evaluation_report.json"


# --------------------------------------------------------------------------
# Benchmark dataset
# --------------------------------------------------------------------------

@dataclass
class BenchmarkCase:
    id: str
    query: str
    expected_route: str  # "research_agent" | "vision_agent" | "both" | "direct_answer"
    category: str
    reference_note: Optional[str] = None  # loose reference answer for correctness judging
    tabular_features: Optional[dict[str, Any]] = None
    tabular_model: Optional[str] = None


BENCHMARK: list[BenchmarkCase] = [
    BenchmarkCase(
        id="rag-001",
        query="What is the evidence for metformin as a first-line treatment for type 2 diabetes?",
        expected_route="research_agent",
        category="rag_medical_knowledge",
        reference_note="Metformin is widely recommended as first-line pharmacotherapy for type 2 "
                        "diabetes per major guidelines (e.g. ADA), due to efficacy, safety, low cost, "
                        "and cardiovascular-neutral/beneficial profile.",
    ),
    BenchmarkCase(
        id="rag-002",
        query="Does regular aspirin use reduce the risk of colorectal cancer?",
        expected_route="research_agent",
        category="rag_medical_knowledge",
        reference_note="Multiple studies suggest long-term low-dose aspirin use is associated with "
                        "reduced colorectal cancer incidence and mortality, though guidance on routine "
                        "use varies due to bleeding risk trade-offs.",
    ),
    BenchmarkCase(
        id="rag-003",
        query="What are current PubMed findings on GLP-1 receptor agonists and cardiovascular outcomes?",
        expected_route="research_agent",
        category="rag_medical_knowledge",
        reference_note="GLP-1 receptor agonists (e.g. semaglutide, liraglutide) have shown cardiovascular "
                        "benefit in trials such as LEADER and SUSTAIN-6, reducing major adverse "
                        "cardiovascular events in high-risk type 2 diabetes patients.",
    ),
    BenchmarkCase(
        id="tab-001",
        query="A patient has respiratory rate 28, oxygen saturation 89%, systolic BP 85, heart rate 122, "
              "temperature 38.9, is not alert, and is on supplemental oxygen. What is their risk level and why?",
        expected_route="both",
        category="tabular_vitals_risk",
        tabular_features={
            "Respiratory_Rate": 28, "Oxygen_Saturation": 89, "O2_Scale": 2, "Systolic_BP": 85,
            "Heart_Rate": 122, "Temperature": 38.9, "Consciousness": 0, "On_Oxygen": 1,
        },
        tabular_model="vitals_risk",
        reference_note="These vitals (tachypnea, hypoxia, hypotension, tachycardia, fever, altered "
                        "consciousness) are consistent with a high-risk / deteriorating patient (e.g. sepsis "
                        "picture) and should trigger urgent clinical escalation.",
    ),
    BenchmarkCase(
        id="tab-002",
        query="35-year-old office worker, 82kg, 175cm, low exercise, 5 hours sleep, high sugar intake, "
              "smokes, drinks alcohol, unmarried, BMI 26.8 — what's their lifestyle health risk?",
        expected_route="vision_agent",
        category="tabular_lifestyle_risk",
        tabular_features={
            "age": 35, "weight": 82, "height": 175, "exercise": "low", "sleep": 5,
            "sugar_intake": "high", "smoking": "yes", "alcohol": "yes", "married": "no",
            "profession": "office_worker", "bmi": 26.8,
        },
        tabular_model="lifestyle_risk",
        reference_note="Low exercise, short sleep, high sugar intake, smoking, and elevated BMI are all "
                        "established risk factors for elevated overall lifestyle-related health risk.",
    ),
    BenchmarkCase(
        id="direct-001",
        query="Hi, what can you help me with?",
        expected_route="direct_answer",
        category="meta_conversational",
        reference_note="A friendly overview of the system's capabilities: medical research Q&A, "
                        "image-based model predictions (e.g. X-rays), and vitals/lifestyle risk scoring.",
    ),
    BenchmarkCase(
        id="rag-004",
        query="Is there evidence that intermittent fasting improves insulin sensitivity?",
        expected_route="research_agent",
        category="rag_medical_knowledge",
        reference_note="Several clinical studies suggest intermittent fasting can improve insulin "
                        "sensitivity and related metabolic markers in some populations, though effect "
                        "sizes and long-term evidence vary.",
    ),
    BenchmarkCase(
        id="direct-002",
        query="Thanks, that's all for now!",
        expected_route="direct_answer",
        category="meta_conversational",
        reference_note="A brief, polite closing acknowledgment.",
    ),
]


# --------------------------------------------------------------------------
# Evaluation harness
# --------------------------------------------------------------------------

@dataclass
class CaseResult:
    id: str
    category: str
    query: str
    expected_route: str
    actual_route: str
    tool_selection_correct: bool
    final_answer: str
    metrics: dict[str, Any]
    latency_ms: int
    error: Optional[str] = None


@dataclass
class EvalReport:
    timestamp: float
    n_cases: int
    results: list[CaseResult] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)


def _aggregate(results: list[CaseResult]) -> dict[str, Any]:
    def _mean(vals: list[float]) -> Optional[float]:
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 3) if vals else None

    tool_acc = _mean([1.0 if r.tool_selection_correct else 0.0 for r in results])
    groundedness = _mean([r.metrics.get("groundedness_score") for r in results])
    correctness = _mean([r.metrics.get("correctness_score") for r in results])
    hallucination = _mean([r.metrics.get("hallucination_score") for r in results])
    relevance = _mean([r.metrics.get("relevance_score") for r in results])
    quality = _mean([r.metrics.get("response_quality_score") for r in results])
    overall = _mean([r.metrics.get("overall_score") for r in results])
    retrieval_relevance = _mean([r.metrics.get("retrieval_relevance_avg") for r in results])
    avg_latency = _mean([float(r.latency_ms) for r in results])

    by_category: dict[str, Any] = {}
    categories = {r.category for r in results}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        by_category[cat] = {
            "n": len(cat_results),
            "tool_selection_accuracy": _mean([1.0 if r.tool_selection_correct else 0.0 for r in cat_results]),
            "overall_score": _mean([r.metrics.get("overall_score") for r in cat_results]),
        }

    return {
        "tool_selection_accuracy": tool_acc,
        "groundedness_score": groundedness,
        "correctness_score": correctness,
        "hallucination_score": hallucination,
        "relevance_score": relevance,
        "response_quality_score": quality,
        "retrieval_relevance_avg": retrieval_relevance,
        "overall_score": overall,
        "avg_latency_ms": avg_latency,
        "by_category": by_category,
        "errors": [r.id for r in results if r.error],
    }


async def run_benchmark(n: Optional[int] = None) -> EvalReport:
    cases = BENCHMARK[:n] if n else BENCHMARK
    orch = MedicalAgentOrchestrator()
    results: list[CaseResult] = []

    for case in cases:
        logger.info(f"Running case {case.id}: {case.query[:60]}...")
        start = time.time()
        try:
            state = await orch.run(
                user_query=case.query,
                tabular_features=case.tabular_features,
                tabular_model_hint=case.tabular_model,
            )
            latency_ms = int((time.time() - start) * 1000)
            actual_route = state.get("route", "unknown")
            metrics = dict(state.get("evaluation") or {})

            # Correctness judging benefits from a reference note if provided; re-run a
            # lightweight correctness check against it if present (keeps main eval fast).
            if case.reference_note:
                correctness = await _judge_against_reference(
                    orch, case.query, state.get("final_answer", ""), case.reference_note
                )
                metrics["correctness_vs_reference_score"] = correctness

            results.append(CaseResult(
                id=case.id, category=case.category, query=case.query,
                expected_route=case.expected_route, actual_route=actual_route,
                tool_selection_correct=(actual_route == case.expected_route),
                final_answer=state.get("final_answer", ""),
                metrics=metrics, latency_ms=latency_ms,
            ))
        except Exception as e:
            logger.exception(f"Case {case.id} failed")
            results.append(CaseResult(
                id=case.id, category=case.category, query=case.query,
                expected_route=case.expected_route, actual_route="error",
                tool_selection_correct=False, final_answer="",
                metrics={}, latency_ms=int((time.time() - start) * 1000), error=str(e),
            ))

    report = EvalReport(timestamp=time.time(), n_cases=len(results), results=results)
    report.aggregate = _aggregate(results)
    return report


async def _judge_against_reference(orch: MedicalAgentOrchestrator, query: str, answer: str, reference: str) -> float:
    prompt = f"""Compare the assistant's answer to a reference note for factual alignment on a 0.0-1.0
scale (1.0 = fully aligned/correct, 0.0 = contradicts or ignores the reference facts).

Question: {query}
Reference note: {reference}
Assistant's answer: {answer}

Respond ONLY with a single float number, nothing else."""
    try:
        raw = await orch.llm.complete(prompt)
        return float(raw.strip().split()[0])
    except Exception as e:
        logger.warning(f"Reference-correctness judging failed: {e}")
        return 0.5


def save_report(report: EvalReport, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    logger.info(f"Evaluation report saved to {path}")


def print_summary(report: EvalReport) -> None:
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY — {report.n_cases} cases")
    print("=" * 70)
    for k, v in report.aggregate.items():
        if k in {"by_category", "errors"}:
            continue
        print(f"  {k:35s}: {v}")
    print("\n  By category:")
    for cat, stats in report.aggregate.get("by_category", {}).items():
        print(f"    - {cat:30s} n={stats['n']:<3} tool_acc={stats['tool_selection_accuracy']} overall={stats['overall_score']}")
    if report.aggregate.get("errors"):
        print(f"\n  Errored cases: {report.aggregate['errors']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the medical agent evaluation benchmark.")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N benchmark cases.")
    args = parser.parse_args()

    report = asyncio.run(run_benchmark(n=args.n))
    print_summary(report)
    save_report(report)
