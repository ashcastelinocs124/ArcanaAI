"""
Celery tasks for batch trace upload and evaluation.

These run in a separate worker process, so they duplicate some helpers
from views_v1.py to avoid circular imports.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from celery import shared_task
from django.utils import timezone

# Add parent dirs for imports (same as tasks.py)
_backend_dir = str(Path(__file__).resolve().parent.parent)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from semantic_pipeline import AgentDAG, CascadeEvaluator, TaskProgressMonitor, extract_agent_logs

from .models import AgentExecution, BatchUpload, EvaluationResult, ExecutionTrace


# ---------------------------------------------------------------------------
# Telemetry extraction (duplicated from views_v1 for process isolation)
# ---------------------------------------------------------------------------

def _extract_latency_ms(sample: dict) -> float:
    tel = sample.get("telemetry") or {}
    if "latency" in tel and isinstance(tel["latency"], (int, float)):
        return tel["latency"] * 1000
    if "latency_ms" in tel and isinstance(tel["latency_ms"], (int, float)):
        return tel["latency_ms"]
    return 0


def _extract_ttft_ms(sample: dict) -> float:
    tel = sample.get("telemetry") or {}
    if "ttft" in tel and isinstance(tel["ttft"], (int, float)):
        return tel["ttft"] * 1000
    if "ttft_ms" in tel and isinstance(tel["ttft_ms"], (int, float)):
        return tel["ttft_ms"]
    return 0


def _extract_tokens(sample: dict) -> int:
    tel = sample.get("telemetry") or {}
    tokens = tel.get("tokens") or {}
    return tokens.get("prompt_tokens", 0) + tokens.get("completion_tokens", 0)


def _extract_model(sample: dict) -> str:
    mp = sample.get("model_parameters") or {}
    return mp.get("model", "")


def _persist_journey(journey: dict) -> tuple[str, bool]:
    """Persist a single journey to DB."""
    trace_id = journey["trace_id"]
    samples = journey.get("samples", [])

    _, was_created = ExecutionTrace.objects.update_or_create(
        trace_id=trace_id,
        defaults={
            "journey_name": journey.get("journey_name", ""),
            "workflow_type": "uploaded",
            "status": "completed",
            "raw_journey": journey,
        },
    )

    trace_obj = ExecutionTrace.objects.get(trace_id=trace_id)
    trace_obj.agents.all().delete()

    for sample in samples:
        AgentExecution.objects.create(
            trace=trace_obj,
            agent_id=sample.get("agent_id", ""),
            agent_name=sample.get("agent_name", ""),
            parent_id=sample.get("parent_id") or "",
            input_text=sample.get("input", ""),
            output_text=sample.get("output", ""),
            latency_ms=_extract_latency_ms(sample),
            ttft_ms=_extract_ttft_ms(sample),
            tokens=_extract_tokens(sample),
            model=_extract_model(sample),
            status="completed",
        )

    return trace_id, was_created


# ---------------------------------------------------------------------------
# Celery Tasks
# ---------------------------------------------------------------------------

@shared_task(bind=True)
def process_batch_upload(self, batch_id: str, journeys: list[dict], auto_evaluate: bool = False, user_goal: str = ""):
    """Process a batch of journeys asynchronously."""
    try:
        batch = BatchUpload.objects.get(batch_id=batch_id)
    except BatchUpload.DoesNotExist:
        return {"error": f"Batch {batch_id} not found"}

    created = 0
    updated = 0
    failed = 0
    errors = []
    trace_ids = []

    for i, journey in enumerate(journeys):
        try:
            trace_id, was_created = _persist_journey(journey)
            trace_ids.append(trace_id)
            if was_created:
                created += 1
            else:
                updated += 1
        except Exception as e:
            failed += 1
            errors.append({"trace_id": journey.get("trace_id", "unknown"), "error": str(e)})

        # Report progress
        self.update_state(state="PROGRESS", meta={
            "phase": "uploading",
            "completed": i + 1,
            "total": len(journeys),
        })

    batch.created_count = created
    batch.updated_count = updated
    batch.failed_count = failed
    batch.errors = errors
    batch.status = "completed"
    batch.completed_at = timezone.now()
    batch.save()

    # Chain to evaluation if requested
    if auto_evaluate and user_goal and trace_ids:
        batch.status = "evaluating"
        batch.save()
        batch_evaluate_task.delay(trace_ids, user_goal)

    return {
        "batch_id": batch_id,
        "created": created,
        "updated": updated,
        "failed": failed,
    }


@shared_task
def batch_evaluate_task(trace_ids: list[str], user_goal: str):
    """Run evaluation on multiple traces in parallel (max 4 workers)."""

    def _evaluate_one(trace_id: str) -> dict:
        try:
            trace = ExecutionTrace.objects.get(trace_id=trace_id)
            raw_journey = trace.raw_journey
            if not raw_journey or "samples" not in raw_journey:
                return {"trace_id": trace_id, "error": "No journey data"}

            agent_logs = extract_agent_logs([raw_journey])
            dags = AgentDAG.from_agent_logs(agent_logs)
            if trace_id not in dags:
                return {"trace_id": trace_id, "error": "Could not build DAG"}

            dag = dags[trace_id]
            t0 = time.time()
            cascade = CascadeEvaluator(dag, user_goal)
            result = cascade.evaluate()
            duration_ms = (time.time() - t0) * 1000

            EvaluationResult.objects.create(
                trace=trace,
                user_goal=user_goal,
                evaluation_mode="cascade",
                result_data={"trace_id": trace_id, "cascade_evaluation": result},
                llm_calls_made=result.get("overall", {}).get("llm_calls_made", 0),
                duration_ms=round(duration_ms, 1),
            )

            return {"trace_id": trace_id, "status": "completed", "duration_ms": round(duration_ms, 1)}
        except Exception as e:
            return {"trace_id": trace_id, "error": str(e)}

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_evaluate_one, tid): tid for tid in trace_ids}
        for future in as_completed(futures):
            results.append(future.result())

    return {"evaluated": len(results), "results": results}
