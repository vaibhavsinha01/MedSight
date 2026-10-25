"""Minimal async facade retained for API compatibility; inference runs locally."""
from __future__ import annotations
from typing import Any
from agents.agents import MedicalAgent


class MedicalAgentOrchestrator:
    def __init__(self, **_):
        self.agent = MedicalAgent()

    async def run(self, user_query: str, image_bytes: bytes | None = None,
                  image_model_hint: str | None = None, tabular_features: dict[str, Any] | None = None,
                  tabular_model_hint: str | None = None) -> dict[str, Any]:
        return await self.agent.run(user_query, image_bytes, image_model_hint, tabular_features, tabular_model_hint)
