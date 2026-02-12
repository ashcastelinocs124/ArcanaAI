"""
V1 Trace Ingestion REST API.

Endpoints:
  POST   /api/v1/traces                     — Create traces (3 formats, auto-normalizes)
  GET    /api/v1/traces                     — List traces with pagination + filtering
  GET    /api/v1/traces/:trace_id           — Get single trace with agents + DAG
  PATCH  /api/v1/traces/:trace_id           — Update trace metadata (workflow_type, journey_name)
  DELETE /api/v1/traces/:trace_id           — Delete trace and related data
  POST   /api/v1/traces/:trace_id/evaluate  — Trigger cascade evaluation
  GET    /api/v1/traces/:trace_id/evaluation— Get evaluation results
  POST   /api/v1/traces/batch               — Async batch upload via Celery
  GET    /api/v1/traces/batch/:batch_id     — Check batch status
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

from django.db.models import Q
from django.utils import timezone
from django.views.decorators.http import require_http_methods

# Add parent dirs to sys.path so imports work (same pattern as views.py)
_backend_dir = str(Path(__file__).resolve().parent.parent)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from semantic_pipeline import AgentDAG, CascadeEvaluator, TaskProgressMonitor, extract_agent_logs

from .models import AgentExecution, BatchUpload, EvaluationResult, ExecutionTrace, PromptOverride
from .response import api_error, api_response
from .validators import ValidationError, normalize_input

# Path to built-in sample data
SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "test" / "keywords_ai_agent_output_samples.json"


# ---------------------------------------------------------------------------
# Telemetry extraction helpers
# ---------------------------------------------------------------------------

def _extract_latency_ms(sample: dict) -> float:
    """Extract latency in ms from various telemetry formats."""
    tel = sample.get("telemetry") or {}
    if "latency" in tel and isinstance(tel["latency"], (int, float)):
        return tel["latency"] * 1000
    if "latency_ms" in tel and isinstance(tel["latency_ms"], (int, float)):
        return tel["latency_ms"]
    return 0


def _extract_ttft_ms(sample: dict) -> float:
    """Extract TTFT in ms from various telemetry formats."""
    tel = sample.get("telemetry") or {}
    if "ttft" in tel and isinstance(tel["ttft"], (int, float)):
        return tel["ttft"] * 1000
    if "ttft_ms" in tel and isinstance(tel["ttft_ms"], (int, float)):
        return tel["ttft_ms"]
    return 0


def _extract_tokens(sample: dict) -> int:
    """Extract total token count from sample."""
    tel = sample.get("telemetry") or {}
    tokens = tel.get("tokens") or {}
    return tokens.get("prompt_tokens", 0) + tokens.get("completion_tokens", 0)


def _extract_model(sample: dict) -> str:
    """Extract model name from sample."""
    mp = sample.get("model_parameters") or {}
    return mp.get("model", "")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _persist_journey(journey: dict) -> tuple[str, bool]:
    """Persist a single journey to DB. Returns (trace_id, was_created)."""
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

    # Rebuild agent rows on every upsert
    trace_obj = ExecutionTrace.objects.get(trace_id=trace_id)
    trace_obj.agents.all().delete()

    for sample in samples:
        AgentExecution.objects.create(
            trace=trace_obj,
            agent_id=sample.get("agent_id", ""),
            agent_name=sample.get("agent_name", ""),
            parent_id=sample.get("parent_id") or (sample.get("parent_ids", [None])[0] if sample.get("parent_ids") else None) or "",
            input_text=sample.get("input", ""),
            output_text=sample.get("output", ""),
            latency_ms=_extract_latency_ms(sample),
            ttft_ms=_extract_ttft_ms(sample),
            tokens=_extract_tokens(sample),
            model=_extract_model(sample),
            status="completed",
        )

    return trace_id, was_created


def _load_sample_data(trace_ids: list[str] | None = None) -> list[dict]:
    """Load journeys from the built-in sample data file.

    If trace_ids is provided, filters to only those traces.
    Raises ValidationError if a requested trace_id is not found.
    """
    with open(SAMPLE_DATA_PATH) as f:
        data = json.load(f)

    all_journeys = data.get("journeys", [])

    if trace_ids is None:
        return all_journeys

    # Build lookup
    available = {j["trace_id"]: j for j in all_journeys}
    missing = [tid for tid in trace_ids if tid not in available]
    if missing:
        raise ValidationError([f"trace_id not found in sample data: {', '.join(missing)}"])

    return [available[tid] for tid in trace_ids]


# ---------------------------------------------------------------------------
# DAG builder from stored journey
# ---------------------------------------------------------------------------

def _build_dag_info(raw_journey: dict) -> dict:
    """Build DAG structure from a stored raw_journey."""
    journeys = [raw_journey] if isinstance(raw_journey, dict) and "samples" in raw_journey else []
    if not journeys:
        return {"roots": [], "leaves": [], "edges": [], "node_count": 0}

    agent_logs = extract_agent_logs(journeys)
    if not agent_logs:
        return {"roots": [], "leaves": [], "edges": [], "node_count": 0}

    dags = AgentDAG.from_agent_logs(agent_logs)
    if not dags:
        return {"roots": [], "leaves": [], "edges": [], "node_count": 0}

    dag = list(dags.values())[0]
    return {
        "roots": dag.roots(),
        "leaves": dag.leaves(),
        "edges": [{"parent": p, "child": c} for p, c in dag.get_edges()],
        "node_count": len(dag.get_nodes()),
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _run_evaluation(trace_obj: ExecutionTrace, user_goal: str, evaluation_mode: str = "cascade") -> dict:
    """Run evaluation on a trace and persist the result."""
    raw_journey = trace_obj.raw_journey
    if not raw_journey or "samples" not in raw_journey:
        raise ValueError("Trace has no stored journey data to evaluate")

    agent_logs = extract_agent_logs([raw_journey])
    dags = AgentDAG.from_agent_logs(agent_logs)

    if trace_obj.trace_id not in dags:
        raise ValueError(f"Could not build DAG for trace {trace_obj.trace_id}")

    dag = dags[trace_obj.trace_id]
    t0 = time.time()
    result = {"trace_id": trace_obj.trace_id}
    llm_calls = 0

    if evaluation_mode in ("cascade", "both"):
        cascade = CascadeEvaluator(dag, user_goal)
        cascade_result = cascade.evaluate()
        result["cascade_evaluation"] = cascade_result
        llm_calls += cascade_result.get("overall", {}).get("llm_calls_made", 0)

    if evaluation_mode in ("checkpoints", "both"):
        monitor = TaskProgressMonitor(dag, user_goal=user_goal, checkpoint_interval=2)
        checkpoints, deviations = monitor.check_progress()
        result["checkpoints"] = [
            {
                "step_count": cp["step_count"],
                "summary": cp["summary"],
                "verdict": cp["llm_verdict"],
                "is_deviation": monitor._is_deviation(cp["llm_verdict"]),
            }
            for cp in checkpoints
        ]
        result["deviations"] = [
            {"step_count": d["step_count"], "verdict": d["verdict"]}
            for d in deviations
        ]
        llm_calls += len(checkpoints)

    duration_ms = (time.time() - t0) * 1000

    # Persist
    eval_obj = EvaluationResult.objects.create(
        trace=trace_obj,
        user_goal=user_goal,
        evaluation_mode=evaluation_mode,
        result_data=result,
        llm_calls_made=llm_calls,
        duration_ms=round(duration_ms, 1),
    )

    result["evaluation_id"] = eval_obj.pk
    result["duration_ms"] = round(duration_ms, 1)
    result["llm_calls_made"] = llm_calls

    return result


# ===========================================================================
# Endpoint: POST/GET /api/v1/traces
# ===========================================================================

@require_http_methods(["GET", "POST"])
def traces_list_create(request):
    """Dispatch to list or create based on HTTP method."""
    if request.method == "GET":
        return _list_traces(request)
    return _create_traces(request)


def _list_traces(request):
    """GET /api/v1/traces — paginated list with filtering."""
    qs = ExecutionTrace.objects.all()

    # Filters
    status = request.GET.get("status")
    if status:
        qs = qs.filter(status=status)

    workflow_type = request.GET.get("workflow_type")
    if workflow_type:
        qs = qs.filter(workflow_type=workflow_type)

    search = request.GET.get("search")
    if search:
        qs = qs.filter(Q(journey_name__icontains=search) | Q(trace_id__icontains=search))

    # Ordering
    order_by = request.GET.get("order_by", "-created_at")
    allowed_orderings = {"created_at", "-created_at", "journey_name", "-journey_name"}
    if order_by in allowed_orderings:
        qs = qs.order_by(order_by)

    # Pagination (offset-based)
    total = qs.count()
    try:
        offset = max(0, int(request.GET.get("offset", 0)))
        limit = min(100, max(1, int(request.GET.get("limit", 20))))
    except (ValueError, TypeError):
        offset, limit = 0, 20

    traces = qs[offset:offset + limit]

    data = []
    for t in traces:
        agent_count = t.agents.count()
        data.append({
            "trace_id": t.trace_id,
            "journey_name": t.journey_name,
            "workflow_type": t.workflow_type,
            "status": t.status,
            "agent_count": agent_count,
            "created_at": t.created_at.isoformat(),
        })

    return api_response(data, meta={
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total,
    })


def _create_traces(request):
    """POST /api/v1/traces — create traces from various formats."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return api_error("Invalid JSON body", code="invalid_json")

    # Check for sample data source
    source = body.get("source")
    auto_evaluate = body.get("auto_evaluate", False)
    user_goal = body.get("user_goal", "")

    if source == "sample":
        try:
            trace_ids_filter = body.get("trace_ids")
            journeys = _load_sample_data(trace_ids_filter)
            format_detected = "sample"
        except ValidationError as e:
            return api_error(e.errors[0], code="sample_not_found", status=400, details=e.errors)
        except FileNotFoundError:
            return api_error("Sample data file not found", code="sample_missing", status=500)
    else:
        try:
            journeys, format_detected = normalize_input(body)
        except ValidationError as e:
            return api_error(
                "Validation failed",
                code="validation_error",
                status=400,
                details=e.errors,
            )

    # Persist
    created = 0
    updated = 0
    trace_ids = []

    for journey in journeys:
        try:
            trace_id, was_created = _persist_journey(journey)
            trace_ids.append(trace_id)
            if was_created:
                created += 1
            else:
                updated += 1
        except Exception as e:
            return api_error(f"Failed to persist trace: {e}", code="persist_error", status=500)

    result = {
        "created": created,
        "updated": updated,
        "total": created + updated,
        "trace_ids": trace_ids,
        "format_detected": format_detected,
        "source": source or "user",
    }

    # Auto-evaluate if requested (sync for single trace, enqueue for multiple)
    if auto_evaluate and user_goal and trace_ids:
        if len(trace_ids) == 1:
            try:
                trace_obj = ExecutionTrace.objects.get(trace_id=trace_ids[0])
                eval_result = _run_evaluation(trace_obj, user_goal)
                result["evaluation"] = eval_result
            except Exception as e:
                result["evaluation_error"] = str(e)
        else:
            # Enqueue async batch evaluation
            try:
                from .tasks_v1 import batch_evaluate_task
                batch_evaluate_task.delay(trace_ids, user_goal)
                result["evaluation_status"] = "queued"
            except Exception:
                result["evaluation_status"] = "skipped (Celery unavailable)"

    return api_response(result, status=201)


# ===========================================================================
# Endpoint: GET/DELETE /api/v1/traces/:trace_id
# ===========================================================================

@require_http_methods(["GET", "PATCH", "DELETE"])
def trace_detail(request, trace_id):
    """Dispatch to get, patch, or delete."""
    if request.method == "DELETE":
        return _delete_trace(request, trace_id)
    if request.method == "PATCH":
        return _patch_trace(request, trace_id)
    return _get_trace(request, trace_id)


def _get_trace(request, trace_id):
    """GET /api/v1/traces/:trace_id — trace with agents + DAG."""
    try:
        trace = ExecutionTrace.objects.get(trace_id=trace_id)
    except ExecutionTrace.DoesNotExist:
        return api_error(f"Trace '{trace_id}' not found", code="not_found", status=404)

    agents = trace.agents.all()
    agents_data = [
        {
            "agent_id": a.agent_id,
            "agent_name": a.agent_name,
            "parent_id": a.parent_id or None,
            "input": a.input_text,
            "output": a.output_text,
            "latency_ms": a.latency_ms,
            "ttft_ms": a.ttft_ms,
            "tokens": a.tokens,
            "model": a.model,
            "status": a.status,
        }
        for a in agents
    ]

    dag_info = _build_dag_info(trace.raw_journey) if trace.raw_journey else {}

    return api_response({
        "trace_id": trace.trace_id,
        "journey_name": trace.journey_name,
        "workflow_type": trace.workflow_type,
        "status": trace.status,
        "goal": trace.goal,
        "created_at": trace.created_at.isoformat(),
        "completed_at": trace.completed_at.isoformat() if trace.completed_at else None,
        "agents": agents_data,
        "dag": dag_info,
        "raw_journey": trace.raw_journey,
    })


def _patch_trace(request, trace_id):
    """PATCH /api/v1/traces/:trace_id — update trace metadata (workflow_type, journey_name)."""
    try:
        trace = ExecutionTrace.objects.get(trace_id=trace_id)
    except ExecutionTrace.DoesNotExist:
        return api_error(f"Trace '{trace_id}' not found", code="not_found", status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return api_error("Invalid JSON body", code="invalid_json", status=400)

    update_fields = []

    if "workflow_type" in body:
        wt = body["workflow_type"]
        if not isinstance(wt, str) or not wt.strip():
            return api_error("workflow_type must be a non-empty string", code="invalid_field", status=400)
        if len(wt) > 100:
            return api_error("workflow_type exceeds 100 characters", code="invalid_field", status=400)
        trace.workflow_type = wt
        update_fields.append("workflow_type")

    if "journey_name" in body:
        jn = body["journey_name"]
        if not isinstance(jn, str) or not jn.strip():
            return api_error("journey_name must be a non-empty string", code="invalid_field", status=400)
        if len(jn) > 300:
            return api_error("journey_name exceeds 300 characters", code="invalid_field", status=400)
        trace.journey_name = jn
        update_fields.append("journey_name")

    # Inject workflow_type into raw_journey samples metadata
    if "workflow_type" in body and trace.raw_journey:
        raw = trace.raw_journey
        for sample in raw.get("samples", []):
            meta = sample.setdefault("metadata", {})
            meta["workflow_type"] = body["workflow_type"]
        trace.raw_journey = raw
        if "raw_journey" not in update_fields:
            update_fields.append("raw_journey")

    if update_fields:
        trace.save(update_fields=update_fields)

    return api_response({
        "trace_id": trace.trace_id,
        "workflow_type": trace.workflow_type,
        "journey_name": trace.journey_name,
        "updated_fields": update_fields,
    })


def _delete_trace(request, trace_id):
    """DELETE /api/v1/traces/:trace_id — cascading delete."""
    try:
        trace = ExecutionTrace.objects.get(trace_id=trace_id)
    except ExecutionTrace.DoesNotExist:
        return api_error(f"Trace '{trace_id}' not found", code="not_found", status=404)

    trace.delete()
    return api_response({"deleted": True, "trace_id": trace_id})


# ===========================================================================
# Endpoint: POST /api/v1/traces/:trace_id/evaluate
# ===========================================================================

@require_http_methods(["POST"])
def trace_evaluate(request, trace_id):
    """POST — trigger cascade/checkpoint evaluation."""
    try:
        trace = ExecutionTrace.objects.get(trace_id=trace_id)
    except ExecutionTrace.DoesNotExist:
        return api_error(f"Trace '{trace_id}' not found", code="not_found", status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return api_error("Invalid JSON body", code="invalid_json")

    user_goal = body.get("user_goal")
    if not user_goal:
        return api_error("'user_goal' is required", code="missing_field")

    evaluation_mode = body.get("evaluation_mode", "cascade")
    if evaluation_mode not in ("cascade", "checkpoints", "both"):
        return api_error(
            "evaluation_mode must be 'cascade', 'checkpoints', or 'both'",
            code="invalid_mode",
        )

    try:
        result = _run_evaluation(trace, user_goal, evaluation_mode)
        return api_response(result)
    except Exception as e:
        return api_error(str(e), code="evaluation_error", status=500)


# ===========================================================================
# Endpoint: GET /api/v1/traces/:trace_id/evaluation
# ===========================================================================

@require_http_methods(["GET"])
def trace_evaluation(request, trace_id):
    """GET — retrieve evaluation results."""
    try:
        trace = ExecutionTrace.objects.get(trace_id=trace_id)
    except ExecutionTrace.DoesNotExist:
        return api_error(f"Trace '{trace_id}' not found", code="not_found", status=404)

    show_all = request.GET.get("all", "").lower() in ("true", "1", "yes")

    evals = EvaluationResult.objects.filter(trace=trace)
    if not evals.exists():
        return api_error("No evaluations found for this trace", code="no_evaluations", status=404)

    if show_all:
        data = [
            {
                "evaluation_id": e.pk,
                "evaluation_mode": e.evaluation_mode,
                "user_goal": e.user_goal,
                "result": e.result_data,
                "llm_calls_made": e.llm_calls_made,
                "duration_ms": e.duration_ms,
                "created_at": e.created_at.isoformat(),
            }
            for e in evals
        ]
        return api_response(data, meta={"total": len(data)})

    # Latest only
    latest = evals.first()
    return api_response({
        "evaluation_id": latest.pk,
        "evaluation_mode": latest.evaluation_mode,
        "user_goal": latest.user_goal,
        "result": latest.result_data,
        "llm_calls_made": latest.llm_calls_made,
        "duration_ms": latest.duration_ms,
        "created_at": latest.created_at.isoformat(),
    })


# ===========================================================================
# Endpoint: POST /api/v1/traces/batch
# ===========================================================================

@require_http_methods(["POST"])
def batch_upload(request):
    """POST — async batch upload via Celery."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return api_error("Invalid JSON body", code="invalid_json")

    # Validate input synchronously (fail fast)
    source = body.get("source")
    auto_evaluate = body.get("auto_evaluate", False)
    user_goal = body.get("user_goal", "")

    if source == "sample":
        try:
            trace_ids_filter = body.get("trace_ids")
            journeys = _load_sample_data(trace_ids_filter)
        except ValidationError as e:
            return api_error(e.errors[0], code="sample_not_found", status=400, details=e.errors)
        except FileNotFoundError:
            return api_error("Sample data file not found", code="sample_missing", status=500)
    else:
        try:
            journeys, _ = normalize_input(body)
        except ValidationError as e:
            return api_error(
                "Validation failed",
                code="validation_error",
                status=400,
                details=e.errors,
            )

    # Create batch record
    batch_id = str(uuid.uuid4())
    batch = BatchUpload.objects.create(
        batch_id=batch_id,
        total_traces=len(journeys),
        status="processing",
        auto_evaluate=auto_evaluate,
        user_goal=user_goal,
    )

    # Enqueue Celery task
    try:
        from .tasks_v1 import process_batch_upload
        process_batch_upload.delay(batch_id, journeys, auto_evaluate, user_goal)
    except Exception as e:
        # Celery/Redis unavailable — process synchronously as fallback
        batch.status = "processing_sync"
        batch.save()
        _process_batch_sync(batch, journeys, auto_evaluate, user_goal)

    return api_response(
        {"batch_id": batch_id, "total_traces": len(journeys), "status": batch.status},
        status=202,
    )


def _process_batch_sync(batch: BatchUpload, journeys: list[dict], auto_evaluate: bool, user_goal: str):
    """Fallback: process batch synchronously when Celery is unavailable."""
    created = 0
    updated = 0
    failed = 0
    errors = []

    for journey in journeys:
        try:
            trace_id, was_created = _persist_journey(journey)
            if was_created:
                created += 1
            else:
                updated += 1
        except Exception as e:
            failed += 1
            errors.append({"trace_id": journey.get("trace_id", "unknown"), "error": str(e)})

    batch.created_count = created
    batch.updated_count = updated
    batch.failed_count = failed
    batch.errors = errors
    batch.status = "completed"
    batch.completed_at = timezone.now()
    batch.save()


# ===========================================================================
# Endpoint: GET /api/v1/traces/batch/:batch_id
# ===========================================================================

@require_http_methods(["GET"])
def batch_status(request, batch_id):
    """GET — check batch upload status."""
    try:
        batch = BatchUpload.objects.get(batch_id=batch_id)
    except BatchUpload.DoesNotExist:
        return api_error(f"Batch '{batch_id}' not found", code="not_found", status=404)

    return api_response({
        "batch_id": batch.batch_id,
        "status": batch.status,
        "total_traces": batch.total_traces,
        "created": batch.created_count,
        "updated": batch.updated_count,
        "failed": batch.failed_count,
        "errors": batch.errors,
        "auto_evaluate": batch.auto_evaluate,
        "created_at": batch.created_at.isoformat(),
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
    })


# ===========================================================================
# Endpoint: GET/POST /api/v1/traces/:trace_id/prompts
# ===========================================================================

@require_http_methods(["GET", "POST"])
def trace_prompts(request, trace_id):
    """GET — list active prompt overrides; POST — store a new override."""
    try:
        trace = ExecutionTrace.objects.get(trace_id=trace_id)
    except ExecutionTrace.DoesNotExist:
        return api_error(f"Trace '{trace_id}' not found", code="not_found", status=404)

    if request.method == "GET":
        return _list_prompt_overrides(trace)
    return _create_prompt_override(request, trace)


def _list_prompt_overrides(trace):
    """GET /api/v1/traces/:trace_id/prompts — active overrides for a trace."""
    overrides = PromptOverride.objects.filter(trace=trace, is_active=True)
    data = [
        {
            "id": o.pk,
            "agent_name": o.agent_name,
            "optimized_prompt": o.optimized_prompt,
            "original_prompt": o.original_prompt,
            "cascade_feedback": o.cascade_feedback,
            "created_at": o.created_at.isoformat(),
        }
        for o in overrides
    ]
    return api_response(data, meta={"total": len(data), "trace_id": trace.trace_id})


def _create_prompt_override(request, trace):
    """POST /api/v1/traces/:trace_id/prompts — store an optimized prompt."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return api_error("Invalid JSON body", code="invalid_json")

    agent_name = body.get("agent_name", "").strip()
    optimized_prompt = body.get("optimized_prompt", "").strip()

    if not agent_name:
        return api_error("'agent_name' is required", code="missing_field")
    if not optimized_prompt:
        return api_error("'optimized_prompt' is required", code="missing_field")

    # Deactivate any existing active override for this agent
    PromptOverride.objects.filter(
        trace=trace, agent_name=agent_name, is_active=True
    ).update(is_active=False)

    override = PromptOverride.objects.create(
        trace=trace,
        agent_name=agent_name,
        optimized_prompt=optimized_prompt,
        original_prompt=body.get("original_prompt", ""),
        cascade_feedback=body.get("cascade_feedback", ""),
    )

    return api_response({
        "id": override.pk,
        "agent_name": override.agent_name,
        "trace_id": trace.trace_id,
        "created_at": override.created_at.isoformat(),
    }, status=201)
