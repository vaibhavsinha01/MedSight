"""Small, local prediction agent and optional concise chat client."""
from __future__ import annotations

import base64
from typing import Any

from models.models import ModelRegistry, PredictionResult, predict_image, predict_tabular

AVAILABLE_IMAGE_MODELS = ModelRegistry.IMAGE_MODELS
AVAILABLE_TABULAR_MODELS = ModelRegistry.TABULAR_MODELS


def prediction_to_dict(p: PredictionResult) -> dict[str, Any]:
    return {"model_name": p.model_name, "task": p.task, "predicted_label": p.predicted_label,
            "confidence": round(p.confidence, 4), "class_probabilities": p.class_probabilities,
            "modality": p.modality, "warnings": p.warnings, "explanation": p.explanation}


class MedicalAgent:
    """Thin agent boundary that routes only to the three registered models."""
    async def run(self, query: str, image_bytes: bytes | None = None, image_model_hint: str | None = None,
                  tabular_features: dict[str, Any] | None = None,
                  tabular_model_hint: str | None = None) -> dict[str, Any]:
        if image_bytes is not None:
            name = image_model_hint or AVAILABLE_IMAGE_MODELS[0]
            result = predict_image(name, image_bytes)
            route = "chest_xray"
        elif tabular_features is not None:
            name = tabular_model_hint or AVAILABLE_TABULAR_MODELS[0]
            result = predict_tabular(name, tabular_features)
            route = name
        else:
            result = None
            route = "direct_answer"
        answer = (f"{result.predicted_label} ({result.confidence:.1%} model confidence). "
                  "This is an automated estimate for research and education, not a diagnosis." if result
                  else "Use the Research agent for PubMed retrieval, or provide a chest X-ray or model features.")
        return {"user_query": query, "route": route, "route_rationale": "Selected from supplied input.",
                "prediction": prediction_to_dict(result) if result else None, "final_answer": answer,
                "a2a_trace": [{"agent": "medical_agent", "model": route}] if result else []}
