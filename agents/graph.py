"""
agents/graph.py
================
LangGraph orchestration. The SupervisorAgent is realized as a router node in
a StateGraph that decides — per user turn — whether to invoke the Research
Agent (Agentic RAG), the Vision Agent (trained models + XAI), both, or answer
directly. It then always routes through the Evaluation Agent before returning.

Graph shape:

    entry -> supervisor_route -> [research_node | vision_node | direct_node | both]
                                        \\-> ... -> synthesize -> evaluate -> END

All inter-agent calls are made through the A2A bus (agents.py) so the Research
and Vision agents remain independently addressable / independently testable,
not just plain function calls.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.agents import (
    A2ABus, A2AMessage, EvaluationAgent, LLMClient, ResearchAgent, VisionAgent,
    AVAILABLE_IMAGE_MODELS, AVAILABLE_TABULAR_MODELS, _prediction_to_dict, _rag_result_to_dict,
)
from models.models import ModelNotAvailableError

logger = logging.getLogger("medagent.graph")


# --------------------------------------------------------------------------
# Graph state
# --------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    user_query: str
    image_bytes: Optional[bytes]
    image_model_hint: Optional[str]
    tabular_features: Optional[dict[str, Any]]
    tabular_model_hint: Optional[str]

    route: str  # "research_agent" | "vision_agent" | "both" | "direct_answer"
    route_rationale: str

    rag_result: Optional[dict[str, Any]]
    prediction: Optional[dict[str, Any]]
    prediction_narrative: Optional[str]

    final_answer: str
    evaluation: Optional[dict[str, Any]]
    a2a_trace: list[dict[str, Any]]
    correlation_id: str


# --------------------------------------------------------------------------
# Orchestrator: builds agents + bus once, exposes a compiled graph
# --------------------------------------------------------------------------

class MedicalAgentOrchestrator:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.bus = A2ABus()
        self.llm = LLMClient(model=model)
        self.research_agent = ResearchAgent(self.bus, self.llm)
        self.vision_agent = VisionAgent(self.bus, self.llm)
        self.evaluation_agent = EvaluationAgent(self.bus, self.llm)
        self.graph = self._build_graph()

    # ---- nodes -------------------------------------------------------

    async def _supervisor_route(self, state: AgentState) -> AgentState:
        """LangGraph router node = the Supervisor Agent's core decision logic."""
        has_image = state.get("image_bytes") is not None
        has_tabular = state.get("tabular_features") is not None
        query = state.get("user_query", "")

        if has_image or has_tabular:
            # Deterministic routing when structured input is present: always run the model,
            # and additionally consult research if the question asks for explanation/context.
            wants_context = any(
                kw in query.lower()
                for kw in ["why", "explain", "evidence", "research", "literature", "guideline", "should i", "risk factor"]
            )
            route = "both" if wants_context else "vision_agent"
            rationale = (
                f"Structured input detected ({'image' if has_image else 'tabular vitals/lifestyle data'}); "
                f"routing to vision_agent{' + research_agent (contextual question detected)' if wants_context else ''}."
            )
        else:
            # Free-text query: ask the LLM supervisor to classify intent.
            route, rationale = await self._llm_route(query)

        state["route"] = route
        state["route_rationale"] = rationale
        logger.info(f"[supervisor] route={route} | {rationale}")
        return state

    async def _llm_route(self, query: str) -> tuple[str, str]:
        prompt = f"""You are the Supervisor Agent of a medical AI system. Classify the user's message
into exactly one route:

- "research_agent": user is asking a medical knowledge/research question answerable via literature (PubMed/PMC).
- "vision_agent": user wants a model prediction but hasn't attached data yet (rare — usually means clarify).
- "direct_answer": casual/greeting/meta question not requiring retrieval or models.

User message: {query}

Respond ONLY as JSON: {{"route": "...", "rationale": "one sentence"}}"""
        try:
            raw = await self.llm.complete(prompt)
            import json
            cleaned = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(cleaned)
            route = parsed.get("route", "research_agent")
            if route not in {"research_agent", "vision_agent", "direct_answer"}:
                route = "research_agent"
            return route, parsed.get("rationale", "")
        except Exception as e:
            logger.warning(f"Supervisor LLM routing failed ({e}); defaulting to research_agent.")
            return "research_agent", f"Fallback routing (LLM error: {e})"

    async def _research_node(self, state: AgentState) -> AgentState:
        corr_id = state["correlation_id"]
        await self.bus.send(A2AMessage(
            sender="supervisor", receiver="research_agent", performative="request",
            payload={"query": state["user_query"]}, correlation_id=corr_id,
        ))
        result = await self.research_agent.run(state["user_query"])
        state["rag_result"] = _rag_result_to_dict(result)
        return state

    async def _vision_node(self, state: AgentState) -> AgentState:
        corr_id = state["correlation_id"]
        is_image = state.get("image_bytes") is not None
        model_name = (
            (state.get("image_model_hint") or AVAILABLE_IMAGE_MODELS[0]) if is_image
            else (state.get("tabular_model_hint") or AVAILABLE_TABULAR_MODELS[0])
        )

        await self.bus.send(A2AMessage(
            sender="supervisor", receiver="vision_agent", performative="request",
            payload=(
                {"kind": "image", "model_name": model_name, "image_bytes": state["image_bytes"]}
                if is_image else
                {"kind": "tabular", "model_name": model_name, "features": state["tabular_features"]}
            ),
            correlation_id=corr_id,
        ))

        try:
            if is_image:
                prediction = self.vision_agent.run_image(model_name, state["image_bytes"])
            else:
                prediction = self.vision_agent.run_tabular(model_name, state["tabular_features"])
        except ModelNotAvailableError as e:
            logger.warning(f"[vision_node] model '{model_name}' unavailable: {e}")
            state["prediction"] = None
            state["prediction_narrative"] = (
                f"The '{model_name}' model isn't available yet (its trained weight file hasn't "
                f"been added to the weights/ directory). {e}"
            )
            return state

        state["prediction"] = _prediction_to_dict(prediction)
        state["prediction_narrative"] = await self.vision_agent.explain_in_context(
            prediction, state.get("user_query") or "Explain this result."
        )
        return state

    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """Combines whatever branches ran into one final natural-language answer."""
        rag = state.get("rag_result")
        pred = state.get("prediction")
        narrative = state.get("prediction_narrative")
        vision_attempted = narrative is not None  # set even when the model was unavailable

        if vision_attempted and rag:
            parts = [
                f"**Model finding:** {narrative}",
                f"\n**Supporting research:** {rag['answer']}",
            ]
            state["final_answer"] = "\n".join(parts)
        elif vision_attempted:
            state["final_answer"] = narrative or "Prediction complete."
        elif rag:
            state["final_answer"] = rag["answer"]
        else:
            # direct_answer route
            state["final_answer"] = await self.llm.complete(
                f"Respond helpfully and briefly as a medical AI assistant to: {state['user_query']}\n"
                "If this needs clinical judgment, recommend consulting a healthcare professional."
            )
        return state

    async def _evaluate_node(self, state: AgentState) -> AgentState:
        corr_id = state["correlation_id"]
        citations = (state.get("rag_result") or {}).get("citations", [])
        evidence_texts = [
            f"{c['title']} ({c.get('journal', '')}, {c.get('pub_date', '')})" for c in citations
        ]
        payload = {
            "user_query": state["user_query"],
            "final_answer": state["final_answer"],
            "route": state["route"],
            "citations": citations,
            "evidence_texts": evidence_texts,
            "prediction": state.get("prediction"),
        }
        await self.bus.send(A2AMessage(
            sender="supervisor", receiver="evaluation_agent", performative="evaluate",
            payload=payload, correlation_id=corr_id,
        ))
        state["evaluation"] = await self.evaluation_agent.evaluate_turn(**payload)
        state["a2a_trace"] = self.bus.trace(corr_id)
        return state

    # ---- routing edges -------------------------------------------------

    @staticmethod
    def _route_selector(state: AgentState) -> str:
        route = state["route"]
        if route == "both":
            return "both"
        if route == "vision_agent":
            return "vision_only"
        if route == "research_agent":
            return "research_only"
        return "direct"

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("supervisor", self._supervisor_route)
        graph.add_node("research", self._research_node)
        graph.add_node("vision", self._vision_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("evaluate", self._evaluate_node)

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor", self._route_selector,
            {
                "both": "vision",       # vision -> research -> synthesize chain for combined route
                "vision_only": "vision",
                "research_only": "research",
                "direct": "synthesize",
            },
        )

        # For "both", after vision we still need research before synthesizing.
        def _after_vision(state: AgentState) -> str:
            return "research" if state["route"] == "both" else "synthesize"

        graph.add_conditional_edges("vision", _after_vision, {"research": "research", "synthesize": "synthesize"})
        graph.add_edge("research", "synthesize")
        graph.add_edge("synthesize", "evaluate")
        graph.add_edge("evaluate", END)

        return graph.compile()

    # ---- public entrypoint -------------------------------------------

    async def run(
        self,
        user_query: str,
        image_bytes: Optional[bytes] = None,
        image_model_hint: Optional[str] = None,
        tabular_features: Optional[dict[str, Any]] = None,
        tabular_model_hint: Optional[str] = None,
    ) -> AgentState:
        import uuid
        initial_state: AgentState = {
            "user_query": user_query,
            "image_bytes": image_bytes,
            "image_model_hint": image_model_hint,
            "tabular_features": tabular_features,
            "tabular_model_hint": tabular_model_hint,
            "correlation_id": str(uuid.uuid4())[:8],
        }
        final_state = await self.graph.ainvoke(initial_state)
        return final_state
