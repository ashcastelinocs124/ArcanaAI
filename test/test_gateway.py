import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

BACKEND_PATH = Path(__file__).resolve().parents[1] / "backend"
GATEWAY_PATH = BACKEND_PATH / "gateway"
sys.path.insert(0, str(BACKEND_PATH))
sys.path.insert(0, str(GATEWAY_PATH))

from gateway import ModelRegistry, GatewayRouter, RoutingRequest
from gateway_cache import CacheStore


class TestGatewayRouting(unittest.TestCase):
    def _write_registry(self, path: Path, models: list[dict]) -> None:
        path.write_text(json.dumps({"models": models}, ensure_ascii=True), encoding="utf-8")

    def test_rule_based_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_path = tmp_path / "registry.json"
            cache_path = tmp_path / "cache.jsonl"

            self._write_registry(
                registry_path,
                [
                    {
                        "provider": "openai",
                        "model_id": "fast",
                        "task_types": ["summarization"],
                        "cost_per_1k_tokens": 0.001,
                        "latency_ms": 600,
                        "quality_score": 0.5,
                        "capabilities": ["fast"],
                    },
                    {
                        "provider": "openai",
                        "model_id": "best",
                        "task_types": ["summarization"],
                        "cost_per_1k_tokens": 0.002,
                        "latency_ms": 900,
                        "quality_score": 0.9,
                        "capabilities": ["quality"],
                    },
                ],
            )

            registry = ModelRegistry(registry_path)
            cache = CacheStore(cache_path, similarity_threshold=0.9)
            router = GatewayRouter(registry=registry, cache=cache)

            decision = router.route(
                RoutingRequest(
                    intent="Summarize document",
                    task_type="summarization",
                    latency_budget_ms=1200,
                    cost_budget=0.003,
                    quality="high",
                )
            )

            self.assertEqual(decision.model_id, "best")
            self.assertEqual(decision.source, "rules")

    def test_llm_fallback_when_no_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_path = tmp_path / "registry.json"
            cache_path = tmp_path / "cache.jsonl"

            self._write_registry(
                registry_path,
                [
                    {
                        "provider": "openai",
                        "model_id": "m1",
                        "task_types": ["summarization"],
                        "cost_per_1k_tokens": 0.001,
                        "latency_ms": 600,
                        "quality_score": 0.5,
                        "capabilities": ["fast"],
                    },
                    {
                        "provider": "openai",
                        "model_id": "m2",
                        "task_types": ["summarization"],
                        "cost_per_1k_tokens": 0.002,
                        "latency_ms": 900,
                        "quality_score": 0.9,
                        "capabilities": ["quality"],
                    },
                ],
            )

            registry = ModelRegistry(registry_path)
            cache = CacheStore(cache_path, similarity_threshold=0.9)
            router = GatewayRouter(registry=registry, cache=cache)

            with patch("backend.gateway._call_llm", return_value="m2"):
                decision = router.route(
                    RoutingRequest(
                        intent="Solve math proof",
                        task_type="reasoning",
                        latency_budget_ms=1000,
                        cost_budget=0.003,
                        quality="high",
                    )
                )

            self.assertEqual(decision.model_id, "m2")
            self.assertEqual(decision.source, "llm")

    def test_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_path = tmp_path / "registry.json"
            cache_path = tmp_path / "cache.jsonl"

            self._write_registry(
                registry_path,
                [
                    {
                        "provider": "openai",
                        "model_id": "fast",
                        "task_types": ["summarization"],
                        "cost_per_1k_tokens": 0.001,
                        "latency_ms": 600,
                        "quality_score": 0.5,
                        "capabilities": ["fast"],
                    }
                ],
            )

            registry = ModelRegistry(registry_path)
            cache = CacheStore(cache_path, similarity_threshold=0.8)
            router = GatewayRouter(registry=registry, cache=cache)

            cache.write_entry(
                intent="Book flight to Tokyo",
                constraints={
                    "task_type": "summarization",
                    "latency_budget_ms": 1000,
                    "cost_budget": 0.003,
                    "quality": "medium",
                },
                decision={
                    "model_id": "fast",
                    "provider": "openai",
                    "score": 0.75,
                    "rationale": "Cached decision",
                },
            )

            decision = router.route(
                RoutingRequest(
                    intent="Book a flight to Tokyo",
                    task_type="summarization",
                    latency_budget_ms=1000,
                    cost_budget=0.003,
                    quality="medium",
                )
            )

            self.assertEqual(decision.model_id, "fast")
            self.assertEqual(decision.source, "cache")


if __name__ == "__main__":
    unittest.main()
