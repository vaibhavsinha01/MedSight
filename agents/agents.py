"""
agents/agents.py
=================
Agent implementations + a minimal Agent-to-Agent (A2A) communication layer.

Agents:
  - SupervisorAgent : routes a user request to the right specialist agent(s)
  - ResearchAgent    : Agentic RAG over PubMed/PMC (rag/rag.py)
  - VisionAgent      : runs trained CNN/XGBoost models + Grad-CAM XAI (models/models.py)
  - EvaluationAgent  : LLM-as-judge + deterministic metrics over the produced answer

A2A protocol: agents communicate via simple typed `A2AMessage` envelopes routed
through an in-process `A2ABus`. This is intentionally lightweight (no external
transport) but follows the A2A shape: sender, receiver, performative
(request/inform/evaluate), payload, and a correlation id — enough to
demonstrate genuine independent-agent messaging rather than plain function calls,
while staying dependency-free. The VisionAgent and ResearchAgent are fully
independent (no shared state) and only interact via the bus; the Supervisor and
Evaluation agents subscribe to the bus to observe traffic for logging/eval.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from models.models import ModelRegistry, PredictionResult, predict_image, predict_tabular
from rag.rag import RAGResult, agentic_rag_query

logger = logging.getLogger("medagent.agents")


# --------------------------------------------------------------------------
# A2A protocol primitives
# --------------------------------------------------------------------------

@dataclass
class A2AMessage:
    sender: str
    receiver: str
    performative: str  # "request" | "inform" | "evaluate" | "error"
    payload: dict[str, Any]
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)


class A2ABus:
    """In-process pub/sub bus. Every message is logged for observability/eval traceability."""

    def __init__(self):
        self.history: list[A2AMessage] = []
        self._subscribers: dict[str, list[Callable[[A2AMessage], Awaitable[None]]]] = {}

    def subscribe(self, agent_name: str, handler: Callable[[A2AMessage], Awaitable[None]]):
        self._subscribers.setdefault(agent_name, []).append(handler)

    async def send(self, message: A2AMessage) -> None:
        self.history.append(message)
        logger.info(
            f"[A2A] {message.sender} -> {message.receiver} "
            f"({message.performative}, corr={message.correlation_id})"
        )
        for handler in self._subscribers.get(message.receiver, []):
            await handler(message)

    def trace(self, correlation_id: Optional[str] = None) -> list[dict[str, Any]]:
        msgs = self.history if correlation_id is None else [
            m for m in self.history if m.correlation_id == correlation_id
        ]
        return [
            {
                "sender": m.sender, "receiver": m.receiver, "performative": m.performative,
                "correlation_id": m.correlation_id, "timestamp": m.timestamp,
                "payload_summary": _summarize_payload(m.payload),
            }
            for m in msgs
        ]


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep large blobs (base64 images etc.) out of logs/traces."""
    out = {}
    for k, v in payload.items():
        if isinstance(v, (bytes,)) or (isinstance(v, str) and len(v) > 300):
            out[k] = f"<{len(v)} bytes/chars omitted>"
        elif isinstance(v, dict):
            out[k] = _summarize_payload(v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# LLM client abstraction (Anthropic) — shared by all agents
# --------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper around the Anthropic API so agents can call `await llm.complete(prompt)`."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1500):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic()  # picks up ANTHROPIC_API_KEY from env
        return self._client

    async def complete(self, prompt: str, system: Optional[str] = None) -> str:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = await client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")


# --------------------------------------------------------------------------
# Research Agent (Agentic RAG)
# --------------------------------------------------------------------------

class ResearchAgent:
    NAME = "research_agent"

    def __init__(self, bus: A2ABus, llm: LLMClient):
        self.bus = bus
        self.llm = llm
        self.bus.subscribe(self.NAME, self._on_message)

    async def _on_message(self, message: A2AMessage) -> None:
        """Handle inbound A2A requests (e.g. from the Vision Agent asking for supporting literature)."""
        if message.performative != "request":
            return
        query = message.payload.get("query", "")
        result = await self.run(query)
        await self.bus.send(A2AMessage(
            sender=self.NAME, receiver=message.sender, performative="inform",
            payload={"rag_result": _rag_result_to_dict(result)},
            correlation_id=message.correlation_id,
        ))

    async def run(self, query: str) -> RAGResult:
        logger.info(f"[{self.NAME}] Running agentic RAG for: {query!r}")
        return await agentic_rag_query(query, llm_call=self.llm.complete)


def _rag_result_to_dict(r: RAGResult) -> dict[str, Any]:
    return {
        "query": r.query,
        "reformulated_queries": r.reformulated_queries,
        "answer": r.answer,
        "citations": r.citations,
        "retrieval_rounds": r.retrieval_rounds,
        "sufficient_evidence_found": r.sufficient_evidence_found,
        "metadata": r.metadata,
    }


# --------------------------------------------------------------------------
# Vision Agent (trained CNN/XGBoost models + XAI)
# --------------------------------------------------------------------------

AVAILABLE_IMAGE_MODELS = ModelRegistry.IMAGE_MODELS
AVAILABLE_TABULAR_MODELS = ModelRegistry.TABULAR_MODELS


class VisionAgent:
    NAME = "vision_agent"

    def __init__(self, bus: A2ABus, llm: LLMClient):
        self.bus = bus
        self.llm = llm
        self.bus.subscribe(self.NAME, self._on_message)

    async def _on_message(self, message: A2AMessage) -> None:
        if message.performative != "request":
            return
        kind = message.payload.get("kind")
        try:
            if kind == "image":
                result = self.run_image(message.payload["model_name"], message.payload["image_bytes"])
            else:
                result = self.run_tabular(message.payload["model_name"], message.payload["features"])
            await self.bus.send(A2AMessage(
                sender=self.NAME, receiver=message.sender, performative="inform",
                payload={"prediction": _prediction_to_dict(result)},
                correlation_id=message.correlation_id,
            ))
        except Exception as e:
            logger.error(f"[{self.NAME}] prediction failed: {e}")
            await self.bus.send(A2AMessage(
                sender=self.NAME, receiver=message.sender, performative="error",
                payload={"error": str(e)}, correlation_id=message.correlation_id,
            ))

    def run_image(self, model_name: str, image_bytes: bytes, with_gradcam: bool = True) -> PredictionResult:
        logger.info(f"[{self.NAME}] Running image model '{model_name}' (gradcam={with_gradcam})")
        return predict_image(model_name, image_bytes, with_gradcam=with_gradcam)

    def run_tabular(self, model_name: str, features: dict[str, Any]) -> PredictionResult:
        logger.info(f"[{self.NAME}] Running tabular model '{model_name}'")
        return predict_tabular(model_name, features)

    async def explain_in_context(self, prediction: PredictionResult, clinical_question: str) -> str:
        """Ask the LLM to translate a raw prediction + Grad-CAM/feature-importance explanation
        into a clinician-readable narrative. Pure narration — never overrides the model's output."""
        expl_summary = {k: v for k, v in prediction.explanation.items() if k != "gradcam_png_base64"}
        prompt = (
            f"A trained {prediction.modality} classification model ('{prediction.model_name}') was run "
            f"for the task: {prediction.task}.\n"
            f"Predicted label: {prediction.predicted_label} (confidence {prediction.confidence:.2%})\n"
            f"Class probabilities: {prediction.class_probabilities}\n"
            f"Explainability data: {expl_summary}\n"
            f"User's clinical question: {clinical_question}\n\n"
            "In 3-5 sentences, explain what this prediction means, reference the explainability "
            "evidence (important region / top features) in plain language, and note this is decision "
            "support only, not a diagnosis. Do not contradict the model's numeric output."
        )
        return await self.llm.complete(prompt)


def _prediction_to_dict(p: PredictionResult) -> dict[str, Any]:
    d = {
        "model_name": p.model_name, "task": p.task, "predicted_label": p.predicted_label,
        "confidence": round(p.confidence, 4), "class_probabilities": {k: round(v, 4) for k, v in p.class_probabilities.items()},
        "modality": p.modality, "warnings": p.warnings,
        "explanation": {k: v for k, v in p.explanation.items()},  # includes gradcam b64 if present
    }
    return d


# --------------------------------------------------------------------------
# Evaluation Agent (deterministic metrics + LLM-as-judge)
# --------------------------------------------------------------------------

class EvaluationAgent:
    NAME = "evaluation_agent"

    def __init__(self, bus: A2ABus, llm: LLMClient):
        self.bus = bus
        self.llm = llm
        self.bus.subscribe(self.NAME, self._on_message)

    async def _on_message(self, message: A2AMessage) -> None:
        if message.performative != "evaluate":
            return
        report = await self.evaluate_turn(**message.payload)
        await self.bus.send(A2AMessage(
            sender=self.NAME, receiver=message.sender, performative="inform",
            payload={"evaluation": report}, correlation_id=message.correlation_id,
        ))

    async def evaluate_turn(
        self,
        user_query: str,
        final_answer: str,
        route: str,
        citations: Optional[list[dict[str, Any]]] = None,
        rag_answer: Optional[str] = None,
        evidence_texts: Optional[list[str]] = None,
        prediction: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Runs the full evaluation suite on one conversational turn."""
        citations = citations or []
        evidence_texts = evidence_texts or []

        metrics: dict[str, Any] = {}

        # ---- deterministic metrics ----
        metrics["citation_count"] = len(citations)
        metrics["tool_selection_valid"] = route in {
            "research_agent", "vision_agent", "both", "direct_answer",
        }
        metrics["retrieval_relevance_avg"] = (
            round(sum(c.get("relevance_score", 0.0) for c in citations) / len(citations), 3)
            if citations else None
        )
        metrics["answer_cites_sources"] = bool(re.search(r"\[\d+\]", final_answer)) if final_answer else False
        metrics["answer_length_chars"] = len(final_answer or "")

        # ---- LLM-as-judge metrics (groundedness, correctness, hallucination, quality) ----
        judge = await self._llm_judge(
            user_query=user_query, final_answer=final_answer,
            evidence_texts=evidence_texts, prediction=prediction,
        )
        metrics.update(judge)

        # ---- aggregate score ----
        judge_scores = [v for k, v in judge.items() if k.endswith("_score") and isinstance(v, (int, float))]
        metrics["overall_score"] = round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None

        return metrics

    async def _llm_judge(
        self, user_query: str, final_answer: str,
        evidence_texts: list[str], prediction: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        evidence_block = "\n---\n".join(evidence_texts) if evidence_texts else "(no retrieved evidence for this turn)"
        prediction_block = str(prediction) if prediction else "(no model prediction for this turn)"
        prompt = f"""You are an impartial medical AI evaluator (LLM-as-judge). Evaluate the assistant's
response on these axes, each scored 0.0-1.0:

1. groundedness_score: Is every factual claim in the answer supported by the evidence/prediction provided? (1.0 = fully grounded, 0.0 = fabricated)
2. correctness_score: Is the answer medically reasonable and correct given the evidence?
3. hallucination_score: INVERTED scale — 1.0 = no hallucination detected, 0.0 = severe hallucination.
4. relevance_score: Does the answer actually address the user's question?
5. response_quality_score: Overall clarity, completeness, appropriate hedging/safety caveats.

User question: {user_query}

Retrieved evidence:
{evidence_block}

Model prediction data (if any): {prediction_block}

Assistant's final answer:
{final_answer}

Respond ONLY as JSON with exactly these keys: groundedness_score, correctness_score,
hallucination_score, relevance_score, response_quality_score, judge_rationale (1-2 sentences).
No markdown, no other text."""
        try:
            raw = await self.llm.complete(prompt)
            import json
            cleaned = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            for k in ["groundedness_score", "correctness_score", "hallucination_score",
                      "relevance_score", "response_quality_score"]:
                parsed[k] = float(parsed.get(k, 0.5))
            return parsed
        except Exception as e:
            logger.warning(f"LLM-as-judge failed ({e}), returning neutral defaults.")
            return {
                "groundedness_score": 0.5, "correctness_score": 0.5, "hallucination_score": 0.5,
                "relevance_score": 0.5, "response_quality_score": 0.5,
                "judge_rationale": f"LLM judge unavailable: {e}",
            }


import re  # noqa: E402  (kept near usage above for clarity)
