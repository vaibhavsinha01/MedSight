"""
api.py
======
FastAPI backend exposing the agentic medical AI system.

Endpoints:
  POST /chat      - full LangGraph workflow (supervisor -> agents -> synthesis -> evaluation)
  POST /predict    - direct model inference (image or tabular) + XAI, bypassing the graph
  POST /research    - direct agentic RAG query, bypassing the graph
  POST /upload      - upload an image, get back a file handle to reference in /chat or /predict
  GET  /health      - liveness + model/agent status report
  GET  /models      - catalog of available models and whether real weights are loaded
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.agents import AVAILABLE_IMAGE_MODELS, AVAILABLE_TABULAR_MODELS
from agents.graph import MedicalAgentOrchestrator
from models.models import ModelNotAvailableError, ModelRegistry, predict_image, predict_tabular
from rag.rag import agentic_rag_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("medagent.api")

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Medical Agentic AI System",
    description="LangGraph-orchestrated multi-agent medical assistant: Agentic RAG + trained "
                 "vision/tabular models with Grad-CAM XAI + LLM-as-judge evaluation.",
    version="1.0.0",
)


FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


@app.get("/", response_class=RedirectResponse)
async def root() -> RedirectResponse:
    if FRONTEND_DIR.is_dir() and (FRONTEND_DIR / "login.html").is_file():
        return RedirectResponse(url="/static/login.html")
    return RedirectResponse(url="/docs")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_orchestrator: Optional[MedicalAgentOrchestrator] = None


def get_orchestrator() -> MedicalAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MedicalAgentOrchestrator()
    return _orchestrator


# --------------------------------------------------------------------------
# Request/response schemas
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's natural-language message.")
    upload_id: Optional[str] = Field(None, description="Optional file id from /upload to attach an image.")
    image_model: Optional[str] = Field(None, description="Force a specific image model, e.g. 'chest_xray'.")
    tabular_features: Optional[dict[str, Any]] = Field(None, description="Optional structured vitals/lifestyle input.")
    tabular_model: Optional[str] = Field(None, description="'vitals_risk' or 'lifestyle_risk'.")


class ChatResponse(BaseModel):
    answer: str
    route: str
    route_rationale: str
    prediction: Optional[dict[str, Any]] = None
    research: Optional[dict[str, Any]] = None
    evaluation: Optional[dict[str, Any]] = None
    a2a_trace: list[dict[str, Any]] = []
    latency_ms: int


class PredictImageRequest(BaseModel):
    upload_id: str
    model_name: str = Field(..., description=f"One of {AVAILABLE_IMAGE_MODELS}")
    with_gradcam: bool = True


class PredictTabularRequest(BaseModel):
    model_name: str = Field(..., description=f"One of {AVAILABLE_TABULAR_MODELS}")
    features: dict[str, Any]


class ResearchRequest(BaseModel):
    query: str
    max_rounds: int = 2


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    size_bytes: int


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": time.time(),
        "models": ModelRegistry.status(),
    }


@app.get("/models")
async def list_models() -> dict[str, Any]:
    return {
        "image_models": AVAILABLE_IMAGE_MODELS,
        "tabular_models": AVAILABLE_TABULAR_MODELS,
        "status": ModelRegistry.status(),
    }


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file.")
    upload_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{upload_id}_{file.filename}"
    dest.write_bytes(contents)
    logger.info(f"Uploaded file saved: {dest} ({len(contents)} bytes)")
    return UploadResponse(upload_id=f"{upload_id}_{file.filename}", filename=file.filename, size_bytes=len(contents))


def _load_upload(upload_id: str) -> bytes:
    path = UPLOAD_DIR / upload_id
    if not path.exists():
        raise HTTPException(404, f"upload_id '{upload_id}' not found. Call /upload first.")
    return path.read_bytes()


@app.post("/predict")
async def predict(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Unified predict endpoint. Body shape depends on modality:
      Image:   {"modality": "image", "upload_id": "...", "model_name": "chest_xray", "with_gradcam": true}
      Tabular: {"modality": "tabular", "model_name": "vitals_risk", "features": {...}}
    """
    modality = payload.get("modality")
    if modality == "image":
        req = PredictImageRequest(**{k: v for k, v in payload.items() if k != "modality"})
        image_bytes = _load_upload(req.upload_id)
        try:
            result = predict_image(req.model_name, image_bytes, with_gradcam=req.with_gradcam)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except ModelNotAvailableError as e:
            raise HTTPException(503, str(e))
        return _prediction_response(result)
    elif modality == "tabular":
        req = PredictTabularRequest(**{k: v for k, v in payload.items() if k != "modality"})
        try:
            result = predict_tabular(req.model_name, req.features)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except ModelNotAvailableError as e:
            raise HTTPException(503, str(e))
        return _prediction_response(result)
    else:
        raise HTTPException(400, "payload.modality must be 'image' or 'tabular'.")


def _prediction_response(result) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "task": result.task,
        "predicted_label": result.predicted_label,
        "confidence": round(result.confidence, 4),
        "class_probabilities": {k: round(v, 4) for k, v in result.class_probabilities.items()},
        "modality": result.modality,
        "warnings": result.warnings,
        "explanation": result.explanation,
    }


@app.post("/research")
async def research(req: ResearchRequest) -> dict[str, Any]:
    orch = get_orchestrator()
    result = await agentic_rag_query(req.query, llm_call=orch.llm.complete, max_rounds=req.max_rounds)
    return {
        "query": result.query,
        "reformulated_queries": result.reformulated_queries,
        "answer": result.answer,
        "citations": result.citations,
        "retrieval_rounds": result.retrieval_rounds,
        "sufficient_evidence_found": result.sufficient_evidence_found,
        "metadata": result.metadata,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    start = time.time()
    orch = get_orchestrator()

    image_bytes = _load_upload(req.upload_id) if req.upload_id else None

    try:
        state = await orch.run(
            user_query=req.message,
            image_bytes=image_bytes,
            image_model_hint=req.image_model,
            tabular_features=req.tabular_features,
            tabular_model_hint=req.tabular_model,
        )
    except Exception as e:
        logger.exception("Graph execution failed")
        raise HTTPException(500, f"Agent workflow failed: {e}")

    latency_ms = int((time.time() - start) * 1000)
    return ChatResponse(
        answer=state.get("final_answer", ""),
        route=state.get("route", "unknown"),
        route_rationale=state.get("route_rationale", ""),
        prediction=state.get("prediction"),
        research=state.get("rag_result"),
        evaluation=state.get("evaluation"),
        a2a_trace=state.get("a2a_trace", []),
        latency_ms=latency_ms,
    )
