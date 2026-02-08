from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gateway_cache import CacheStore


@dataclass
class ModelSpec:
    provider: str
    model_id: str
    task_types: list[str]
    cost_per_1k_tokens: float
    latency_ms: int
    quality_score: float
    capabilities: list[str] | None = None


@dataclass
class RoutingRequest:
    intent: str
    task_type: str
    latency_budget_ms: int | None = None
    cost_budget: float | None = None
    quality: str | None = None

    def constraint_key(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "latency_budget_ms": self.latency_budget_ms,
            "cost_budget": self.cost_budget,
            "quality": self.quality or "medium",
        }


@dataclass
class RoutingDecision:
    model_id: str
    provider: str
    score: float
    rationale: str
    source: str


class ModelRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self._models = self._load()

    def _load(self) -> list[ModelSpec]:
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return [ModelSpec(**item) for item in raw.get("models", [])]

    def all_models(self) -> list[ModelSpec]:
        return list(self._models)


def _call_llm(prompt: str, model: str) -> str:
    try:
        from litellm import completion
        response = completion(model=model, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"[LLM error: {e}]"


class GatewayRouter:
    def __init__(
        self,
        registry: ModelRegistry,
        cache: CacheStore | None = None,
        llm_model: str = "gpt-4o",
    ):
        self.registry = registry
        self.cache = cache
        self.llm_model = llm_model

    def _rule_filter(self, request: RoutingRequest) -> list[ModelSpec]:
        candidates = []
        for model in self.registry.all_models():
            if request.task_type not in model.task_types:
                continue
            if request.latency_budget_ms is not None and model.latency_ms > request.latency_budget_ms:
                continue
            if request.cost_budget is not None and model.cost_per_1k_tokens > request.cost_budget:
                continue
            candidates.append(model)
        return candidates

    def _score_candidates(self, candidates: list[ModelSpec], quality: str) -> list[tuple[ModelSpec, float]]:
        if not candidates:
            return []

        max_latency = max(m.latency_ms for m in candidates)
        max_cost = max(m.cost_per_1k_tokens for m in candidates)
        max_latency = max_latency if max_latency > 0 else 1
        max_cost = max_cost if max_cost > 0 else 1

        quality = (quality or "medium").lower()
        if quality == "high":
            wq, wl, wc = 0.6, 0.2, 0.2
        elif quality == "low":
            wq, wl, wc = 0.2, 0.4, 0.4
        else:
            wq, wl, wc = 0.4, 0.3, 0.3

        scored = []
        for model in candidates:
            latency_norm = 1 - (model.latency_ms / max_latency)
            cost_norm = 1 - (model.cost_per_1k_tokens / max_cost)
            quality_norm = model.quality_score
            score = (wq * quality_norm) + (wl * latency_norm) + (wc * cost_norm)
            scored.append((model, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _llm_route(self, request: RoutingRequest, candidates: list[ModelSpec]) -> RoutingDecision:
        if not candidates:
            candidates = self.registry.all_models()
        model_lines = []
        for m in candidates:
            model_lines.append(
                f"- {m.model_id} (provider={m.provider}, latency_ms={m.latency_ms}, "
                f"cost_per_1k_tokens={m.cost_per_1k_tokens}, quality={m.quality_score})"
            )
        prompt = (
            "You are selecting the best model for a task.\n"
            f"Intent: {request.intent}\n"
            f"Task type: {request.task_type}\n"
            f"Latency budget (ms): {request.latency_budget_ms}\n"
            f"Cost budget: {request.cost_budget}\n"
            f"Quality target: {request.quality}\n"
            "Candidates:\n"
            + "\n".join(model_lines)
            + "\nReturn only the model_id."
        )
        choice = _call_llm(prompt, self.llm_model).strip()
        picked = next((m for m in candidates if m.model_id == choice), None)
        if not picked:
            picked = candidates[0]
        return RoutingDecision(
            model_id=picked.model_id,
            provider=picked.provider,
            score=0.0,
            rationale="LLM fallback router",
            source="llm",
        )

    def route(self, request: RoutingRequest) -> RoutingDecision:
        constraints = request.constraint_key()
        if self.cache:
            match = self.cache.find_match(request.intent, constraints)
            if match:
                decision = match.decision
                return RoutingDecision(
                    model_id=decision["model_id"],
                    provider=decision["provider"],
                    score=decision.get("score", 0.0),
                    rationale=decision.get("rationale", "Cache hit"),
                    source="cache",
                )

        candidates = self._rule_filter(request)
        scored = self._score_candidates(candidates, request.quality or "medium")
        if scored:
            top_model, top_score = scored[0]
            if len(scored) == 1 or (len(scored) > 1 and (top_score - scored[1][1]) >= 0.01):
                decision = RoutingDecision(
                    model_id=top_model.model_id,
                    provider=top_model.provider,
                    score=top_score,
                    rationale="Rule-based routing",
                    source="rules",
                )
                if self.cache:
                    self.cache.write_entry(
                        request.intent,
                        constraints,
                        {
                            "model_id": decision.model_id,
                            "provider": decision.provider,
                            "score": decision.score,
                            "rationale": decision.rationale,
                        },
                    )
                return decision

        decision = self._llm_route(request, candidates)
        if self.cache:
            self.cache.write_entry(
                request.intent,
                constraints,
                {
                    "model_id": decision.model_id,
                    "provider": decision.provider,
                    "score": decision.score,
                    "rationale": decision.rationale,
                },
            )
        return decision
