"""
Django views for ArcanaAI LLM Observability Platform.

Migrated from Flask api.py — all endpoints preserved at same URL paths.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_http_methods

# Add parent dirs to sys.path so imports work at new depth (backend/api/views.py)
_backend_dir = str(Path(__file__).resolve().parent.parent)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ai.prompt_optimizer import (
    OptimizationResult,
    _call_llm,
    _render_prompt,
    _similarity,
    load_excel,
    optimize_row,
    run_optimizer,
)
from gateway.gateway import GatewayRouter, ModelRegistry, RoutingRequest
from gateway.gateway_cache import CacheStore
from semantic_pipeline import (
    AgentDAG,
    CascadeEvaluator,
    TaskProgressMonitor,
    extract_agent_logs,
)
from workflow_engine import WORKFLOW_TOPOLOGIES, WorkflowEngine

from .models import AgentExecution, ExecutionTrace, OptimizationRun, PromptOverride, RoutingDecision

ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}


def _load_stored_overrides(source_trace_id: str) -> dict[str, str]:
    """Load active prompt overrides from DB for a given trace.

    Returns dict of {agent_name: optimized_prompt}.
    """
    if not source_trace_id:
        return {}
    try:
        overrides = PromptOverride.objects.filter(
            trace__trace_id=source_trace_id,
            is_active=True,
        )
        result = {o.agent_name: o.optimized_prompt for o in overrides}
        if result:
            print(f"[overrides] Loaded {len(result)} stored overrides from trace {source_trace_id}: {list(result.keys())}")
        return result
    except Exception:
        return {}


def _reconstruct_trace_topology(source_trace_id: str) -> list[dict] | None:
    """Reconstruct a workflow topology from a stored trace's AgentExecution records.

    Used when rerunning proxy traces that don't exist in WORKFLOW_TOPOLOGIES.
    Returns a topology list compatible with WorkflowEngine, or None if not found.
    """
    if not source_trace_id:
        return None
    try:
        agents_qs = (
            AgentExecution.objects
            .filter(trace__trace_id=source_trace_id)
            .order_by('id')
            .values('agent_id', 'agent_name', 'parent_id', 'model')
        )
        rows = list(agents_qs)
        if not rows:
            return None

        # Build agent_id -> index and deduplicate by name (preserve order)
        seen_names: list[str] = []
        agent_id_to_name: dict[str, str] = {}
        name_to_model: dict[str, str] = {}
        name_to_parent_ids: dict[str, list[str]] = {}

        for row in rows:
            name = row['agent_name']
            if not name:
                continue
            if row['agent_id']:
                agent_id_to_name[row['agent_id']] = name
            if name not in name_to_model:
                seen_names.append(name)
                name_to_model[name] = row['model'] or 'gpt-4o'
                name_to_parent_ids[name] = []
            if row['parent_id']:
                name_to_parent_ids[name].append(row['parent_id'])

        if not seen_names:
            return None

        # Resolve parent_id strings to indices
        name_to_idx = {n: i for i, n in enumerate(seen_names)}
        topology = []
        for name in seen_names:
            parent_indices = []
            for pid in name_to_parent_ids[name]:
                parent_name = agent_id_to_name.get(pid, '')
                if parent_name in name_to_idx:
                    parent_indices.append(name_to_idx[parent_name])
            topology.append({
                'name': name,
                'role': name,
                'model': name_to_model[name],
                'system_prompt': f'You are {name}. Complete your assigned task accurately and thoroughly.',
                'parent_indices': sorted(set(parent_indices)),
                'tools': [],
            })

        print(f"[reconstruct] Built topology from trace {source_trace_id}: {[(a['name'], a['parent_indices']) for a in topology]}")
        return topology
    except Exception as e:
        print(f"[reconstruct] Failed for trace {source_trace_id}: {e}")
        return None


def _persist_workflow_result(trace_id, journey_name, workflow_type, goal, samples, raw_journey=None):
    """Best-effort persistence of workflow execution to DB."""
    try:
        from django.utils import timezone
        trace_obj = ExecutionTrace.objects.create(
            trace_id=trace_id,
            journey_name=journey_name,
            workflow_type=workflow_type,
            goal=goal,
            status='completed',
            raw_journey=raw_journey or {},
            completed_at=timezone.now(),
        )
        for sample in samples:
            # parent_id (singular) for single-parent; fall back to first of parent_ids for fan-in
            pid = sample.get('parent_id', '')
            if not pid:
                pids = sample.get('parent_ids', [])
                pid = pids[0] if pids else ''
            AgentExecution.objects.create(
                trace=trace_obj,
                agent_id=sample.get('agent_id', ''),
                agent_name=sample.get('agent_name', ''),
                parent_id=pid,
                input_text=sample.get('input', ''),
                output_text=sample.get('output', ''),
                latency_ms=sample.get('telemetry', {}).get('latency', 0) * 1000
                    if isinstance(sample.get('telemetry', {}).get('latency'), (int, float))
                    else sample.get('telemetry', {}).get('latency_ms', 0),
                ttft_ms=sample.get('telemetry', {}).get('ttft', 0) * 1000
                    if isinstance(sample.get('telemetry', {}).get('ttft'), (int, float))
                    else sample.get('telemetry', {}).get('ttft_ms', 0),
                tokens=sample.get('telemetry', {}).get('tokens', {}).get('prompt_tokens', 0)
                    + sample.get('telemetry', {}).get('tokens', {}).get('completion_tokens', 0),
                model=sample.get('model_parameters', {}).get('model', ''),
                status='completed',
            )
    except Exception:
        pass  # Best-effort — never break API responses


def _secure_filename(filename: str) -> str:
    """Sanitize a filename (replaces werkzeug.utils.secure_filename)."""
    # Keep only alphanumerics, dots, hyphens, underscores
    filename = re.sub(r'[^\w.\-]', '_', filename)
    # Strip leading dots/underscores for safety
    filename = filename.lstrip('._')
    return filename or 'upload'


def _allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint."""
    return JsonResponse({
        'status': 'ok',
        'service': 'arcana-api',
        'version': '1.0.0',
    })


# ---------------------------------------------------------------------------
# Traces (DB-backed, replaces static data.json)
# ---------------------------------------------------------------------------

@require_http_methods(["GET"])
def list_traces(request):
    """Return only API-generated execution traces (proxy + workflow), not seeded/uploaded."""
    traces = (
        ExecutionTrace.objects
        .exclude(raw_journey={})
        .exclude(workflow_type__in=['imported', 'uploaded'])
        .values_list('raw_journey', flat=True)
    )
    journeys = list(traces)
    return JsonResponse({
        'description': 'ArcanaAI execution traces',
        'schema_version': '1.0',
        'journeys': journeys,
    })


@require_http_methods(["POST"])
def upload_traces(request):
    """Accept uploaded journey data and persist to DB."""
    try:
        data = json.loads(request.body)
        journeys = data.get('journeys', [])
        if not journeys:
            return JsonResponse({'error': 'No journeys provided'}, status=400)

        created = 0
        updated = 0
        for journey in journeys:
            trace_id = journey.get('trace_id', '')
            if not trace_id:
                continue
            _, was_created = ExecutionTrace.objects.update_or_create(
                trace_id=trace_id,
                defaults={
                    'journey_name': journey.get('journey_name', ''),
                    'workflow_type': 'uploaded',
                    'status': 'completed',
                    'raw_journey': journey,
                },
            )
            if was_created:
                created += 1
            else:
                updated += 1

        return JsonResponse({
            'success': True,
            'created': created,
            'updated': updated,
            'total': created + updated,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ---------------------------------------------------------------------------
# Prompt Optimizer (async via Celery)
# ---------------------------------------------------------------------------

@require_http_methods(["POST"])
def run_prompt_optimizer(request):
    """
    Enqueue prompt optimization as a Celery task.

    Accepts the same multipart/form-data as the old Flask endpoint.
    Returns {"task_id": "...", "status": "queued"} immediately.
    """
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        file = request.FILES['file']
        if file.name == '':
            return JsonResponse({'error': 'No file selected'}, status=400)

        if not _allowed_file(file.name):
            return JsonResponse({'error': 'Invalid file type. Must be .xlsx, .xls, or .csv'}, status=400)

        # Save uploaded file to temp dir
        filename = _secure_filename(file.name)
        filepath = settings.UPLOAD_FOLDER / filename
        with open(filepath, 'wb') as dest:
            for chunk in file.chunks():
                dest.write(chunk)

        # Extract form parameters
        prompt_template = request.POST.get('prompt_template', '')
        if not prompt_template:
            filepath.unlink(missing_ok=True)
            return JsonResponse({'error': 'prompt_template is required'}, status=400)

        if '{input}' not in prompt_template:
            filepath.unlink(missing_ok=True)
            return JsonResponse({'error': 'prompt_template must contain {input} placeholder'}, status=400)

        input_col = request.POST.get('input_col', 'input')
        gold_col = request.POST.get('gold_col', 'gold')
        comments_col = request.POST.get('comments_col', None)
        eval_model = request.POST.get('eval_model', 'gpt-4o')
        optimizer_model = request.POST.get('optimizer_model', 'gpt-4o')
        target_score = float(request.POST.get('target_score', 0.85))
        max_iters = int(request.POST.get('max_iters', 3))
        agent_filter = request.POST.get('agent_filter', None)
        cascade_feedback = request.POST.get('cascade_feedback', None)

        # Enqueue Celery task
        from .tasks import run_optimization_task

        task = run_optimization_task.delay(
            file_path=str(filepath),
            prompt_template=prompt_template,
            target_score=target_score,
            max_iters=max_iters,
            input_col=input_col,
            gold_col=gold_col,
            comments_col=comments_col,
            eval_model=eval_model,
            optimizer_model=optimizer_model,
            agent_filter=agent_filter,
            cascade_feedback=cascade_feedback,
        )

        return JsonResponse({'task_id': task.id, 'status': 'queued'})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@require_http_methods(["GET"])
def optimizer_status(request, task_id):
    """Poll for optimization task status and results."""
    try:
        from celery.result import AsyncResult

        result = AsyncResult(task_id)

        if result.state == 'PROGRESS':
            return JsonResponse({'state': 'PROGRESS', 'progress': result.info})
        elif result.state == 'SUCCESS':
            return JsonResponse({'state': 'SUCCESS', 'result': result.result})
        elif result.state == 'FAILURE':
            return JsonResponse(
                {'state': 'FAILURE', 'error': str(result.result)},
                status=500,
            )
        else:
            return JsonResponse({'state': result.state})
    except Exception as e:
        return JsonResponse({'state': 'FAILURE', 'error': str(e)}, status=500)


@require_http_methods(["POST"])
def optimizer_cancel(request, task_id):
    """Cancel a running optimization task."""
    try:
        from celery.result import AsyncResult

        result = AsyncResult(task_id)
        result.revoke(terminate=True)
        return JsonResponse({'status': 'cancelled'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ---------------------------------------------------------------------------
# Pipeline Analysis
# ---------------------------------------------------------------------------

@require_http_methods(["POST"])
def analyze_pipeline(request):
    """Run LLM goal verification on agent traces."""
    try:
        data = json.loads(request.body)

        if not data or 'traces' not in data:
            return JsonResponse({'error': 'traces array is required'}, status=400)

        traces = data['traces']
        trace_id = data.get('trace_id')
        user_goal = data.get('user_goal')
        evaluation_mode = data.get('evaluation_mode', 'cascade')

        if not user_goal:
            return JsonResponse({'error': 'user_goal is required'}, status=400)

        if evaluation_mode not in ('checkpoints', 'cascade', 'both'):
            return JsonResponse({'error': 'evaluation_mode must be checkpoints, cascade, or both'}, status=400)

        agent_logs = extract_agent_logs(traces)
        dags = AgentDAG.from_agent_logs(agent_logs)

        if trace_id and trace_id not in dags:
            return JsonResponse({'error': f'trace_id {trace_id} not found'}, status=404)

        target_traces = [trace_id] if trace_id else list(dags.keys())

        results = {}
        for tid in target_traces:
            dag = dags[tid]
            result = {'trace_id': tid}

            if evaluation_mode in ('checkpoints', 'both'):
                monitor = TaskProgressMonitor(dag, user_goal=user_goal, checkpoint_interval=2)
                checkpoints, deviations = monitor.check_progress()
                result['total_steps'] = len(monitor._get_ordered_steps())
                result['checkpoints'] = [
                    {
                        'step_count': cp['step_count'],
                        'summary': cp['summary'],
                        'verdict': cp['llm_verdict'],
                        'is_deviation': monitor._is_deviation(cp['llm_verdict']),
                        'agent_ids': [child_id for _, child_id, _ in cp['steps']],
                        'agent_names': [
                            child_node.get('agent_name', child_id)
                            for _, child_id, child_node in cp['steps']
                        ],
                    }
                    for cp in checkpoints
                ]
                result['deviations'] = [
                    {
                        'step_count': d['step_count'],
                        'agent_ids': d['agent_ids'],
                        'agent_names': d['agent_names'],
                        'verdict': d['verdict'],
                    }
                    for d in deviations
                ]

            if evaluation_mode in ('cascade', 'both'):
                cascade = CascadeEvaluator(dag, user_goal)
                result['cascade_evaluation'] = cascade.evaluate()

            results[tid] = result

        return JsonResponse({'success': True, 'results': results})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------

@require_http_methods(["POST"])
def run_evaluations(request):
    """Run custom evaluations on agent outputs (placeholder)."""
    try:
        data = json.loads(request.body)

        if not data or 'traces' not in data:
            return JsonResponse({'error': 'traces array is required'}, status=400)

        traces = data['traces']
        eval_type = data.get('eval_type', 'custom')

        results = []
        for trace in traces:
            agent_id = trace.get('agent_id', 'unknown')
            output = trace.get('output', '')
            score = 0.85

            results.append({
                'agent_id': agent_id,
                'eval_type': eval_type,
                'score': score,
                'output': output[:100] + '...' if len(output) > 100 else output,
            })

        return JsonResponse({
            'success': True,
            'results': results,
            'summary': {
                'total_evaluated': len(results),
                'avg_score': sum(r['score'] for r in results) / len(results) if results else 0,
                'eval_type': eval_type,
            },
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ---------------------------------------------------------------------------
# Gateway Routing
# ---------------------------------------------------------------------------

@require_http_methods(["POST"])
def route_gateway(request):
    """Route a task to the best model based on constraints."""
    try:
        data = json.loads(request.body)
        if not data:
            return JsonResponse({'error': 'JSON body is required'}, status=400)

        intent = data.get('intent')
        task_type = data.get('task_type')
        if not intent or not task_type:
            return JsonResponse({'error': 'intent and task_type are required'}, status=400)

        latency_budget_ms = data.get('latency_budget_ms')
        cost_budget = data.get('cost_budget')
        quality = data.get('quality')
        cache_threshold = data.get('cache_threshold', 0.9)

        # Resolve gateway paths relative to backend/ dir
        gateway_dir = Path(__file__).resolve().parent.parent / 'gateway'
        registry_path = gateway_dir / 'gateway_registry.json'
        cache_path = gateway_dir / 'gateway_cache.jsonl'

        registry = ModelRegistry(registry_path)
        cache = CacheStore(cache_path, similarity_threshold=float(cache_threshold))
        router = GatewayRouter(registry=registry, cache=cache)

        decision = router.route(
            RoutingRequest(
                intent=intent,
                task_type=task_type,
                latency_budget_ms=latency_budget_ms,
                cost_budget=cost_budget,
                quality=quality,
            )
        )

        # Best-effort DB persistence
        try:
            RoutingDecision.objects.create(
                intent=intent,
                task_type=task_type,
                latency_budget_ms=latency_budget_ms,
                cost_budget=cost_budget,
                quality=quality,
                model_id=decision.model_id,
                provider=decision.provider,
                score=decision.score,
                rationale=decision.rationale,
                source=decision.source,
            )
        except Exception:
            pass

        return JsonResponse({
            'success': True,
            'decision': {
                'model_id': decision.model_id,
                'provider': decision.provider,
                'score': decision.score,
                'rationale': decision.rationale,
                'source': decision.source,
            },
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ---------------------------------------------------------------------------
# Workflow Execution
# ---------------------------------------------------------------------------

@require_http_methods(["POST"])
def run_workflow(request):
    """Execute a multi-agent workflow with real LLM calls."""
    try:
        data = json.loads(request.body)

        if not data or 'goal' not in data:
            return JsonResponse({'error': 'goal is required'}, status=400)

        goal = data['goal']
        workflow_type = data.get('workflow_type', 'custom')
        config = data.get('config', {})

        # Merge stored overrides from source trace (explicit overrides take precedence)
        source_trace_id = data.get('source_trace_id', '')
        if source_trace_id:
            stored = _load_stored_overrides(source_trace_id)
            if stored:
                explicit = config.get('prompt_overrides', {})
                merged = {**stored, **explicit}  # explicit wins
                config['prompt_overrides'] = merged

        # Reconstruct topology from trace if workflow_type isn't predefined
        trace_topology = None
        if workflow_type not in WORKFLOW_TOPOLOGIES and source_trace_id:
            trace_topology = _reconstruct_trace_topology(source_trace_id)

        engine = WorkflowEngine(goal=goal, workflow_type=workflow_type, config=config, topology=trace_topology)
        result = engine.execute()

        # Best-effort DB persistence
        journey_data = result.get('journey', {})
        _persist_workflow_result(
            trace_id=result['trace_id'],
            journey_name=result['journey_name'],
            workflow_type=workflow_type,
            goal=goal,
            samples=journey_data.get('samples', []),
            raw_journey=journey_data,
        )

        return JsonResponse({
            'success': True,
            'trace_id': result['trace_id'],
            'journey_name': result['journey_name'],
            'journey': result['journey'],
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def _reconstruct_api_topologies() -> dict[str, list[dict]]:
    """Reconstruct workflow topologies from API proxy traces.

    Queries ExecutionTrace for workflow_types not in WORKFLOW_TOPOLOGIES
    (and not generic types like proxy/imported/uploaded/custom), then
    rebuilds agent lists from AgentExecution rows.
    """
    from collections import Counter

    skip_types = set(WORKFLOW_TOPOLOGIES.keys()) | {'proxy', 'imported', 'uploaded', 'custom', ''}

    api_types = (
        ExecutionTrace.objects
        .exclude(workflow_type__in=skip_types)
        .values_list('workflow_type', flat=True)
        .distinct()
    )

    result: dict[str, list[dict]] = {}
    for wf_type in api_types:
        # Single query: get all fields needed for dedup + parent resolution
        agents_qs = (
            AgentExecution.objects
            .filter(trace__workflow_type=wf_type)
            .values('agent_id', 'agent_name', 'parent_id', 'model')
        )

        # Deduplicate by agent_name, pick most common model
        agent_models: dict[str, Counter] = {}
        agent_parents: dict[str, set] = {}
        agent_id_to_name: dict[str, str] = {}
        seen_order: list[str] = []
        for row in agents_qs:
            name = row['agent_name']
            if not name:
                continue
            if name not in agent_models:
                agent_models[name] = Counter()
                agent_parents[name] = set()
                seen_order.append(name)
            agent_models[name][row['model'] or 'unknown'] += 1
            if row['parent_id']:
                agent_parents[name].add(row['parent_id'])
            if row['agent_id']:
                agent_id_to_name[row['agent_id']] = name

        if not seen_order:
            continue

        # Build agent list preserving discovery order
        name_to_idx = {name: i for i, name in enumerate(seen_order)}

        agents_list = []
        for name in seen_order:
            model = agent_models[name].most_common(1)[0][0]
            # Resolve parent indices
            parent_indices = []
            for pid in agent_parents[name]:
                parent_name = agent_id_to_name.get(pid, '')
                if parent_name in name_to_idx:
                    parent_indices.append(name_to_idx[parent_name])
            agents_list.append({
                'name': name,
                'role': name,
                'model': model,
                'system_prompt': '',
                'parent_indices': sorted(parent_indices),
                'source': 'api',
            })

        result[wf_type] = agents_list

    return result


@require_http_methods(["GET"])
def get_workflow_topologies(request):
    """Return available workflow types and their agent names/roles.

    Merges hardcoded WORKFLOW_TOPOLOGIES with topologies reconstructed
    from API proxy traces. Hardcoded takes precedence on name collisions.
    """
    result = {}

    # Reconstructed API topologies first (lower precedence)
    try:
        api_topos = _reconstruct_api_topologies()
        for wf_type, agents in api_topos.items():
            result[wf_type] = agents  # already in the right format
    except Exception as e:
        print(f"[topologies] Failed to reconstruct API topologies: {e}")

    # Hardcoded topologies override on collision
    for wf_type, agents in WORKFLOW_TOPOLOGIES.items():
        result[wf_type] = [
            {'name': a['name'], 'role': a['role'], 'model': a['model'], 'system_prompt': a['system_prompt']}
            for a in agents
        ]
    return JsonResponse(result)


@require_http_methods(["POST"])
def stream_workflow(request):
    """Execute a multi-agent workflow with SSE streaming."""
    data = json.loads(request.body)

    if not data or 'goal' not in data:
        return JsonResponse({'error': 'goal is required'}, status=400)

    goal = data['goal']
    workflow_type = data.get('workflow_type', 'custom')
    config = data.get('config', {})

    # Merge stored overrides from source trace (explicit overrides take precedence)
    source_trace_id = data.get('source_trace_id', '')
    if source_trace_id:
        stored = _load_stored_overrides(source_trace_id)
        if stored:
            explicit = config.get('prompt_overrides', {})
            merged = {**stored, **explicit}
            config['prompt_overrides'] = merged

    # Reconstruct topology from trace if workflow_type isn't predefined
    trace_topology = None
    if workflow_type not in WORKFLOW_TOPOLOGIES and source_trace_id:
        trace_topology = _reconstruct_trace_topology(source_trace_id)

    print(f"[stream_workflow] workflow_type={workflow_type!r}, config_keys={list(config.keys())}, prompt_overrides={list(config.get('prompt_overrides', {}).keys())}, trace_topology={'yes' if trace_topology else 'no'}")

    engine = WorkflowEngine(goal=goal, workflow_type=workflow_type, config=config, topology=trace_topology)

    def _streaming_with_persistence():
        """Wrap the SSE stream to capture the final event and persist to DB."""
        complete_data = None
        for chunk in engine.execute_streaming():
            yield chunk
            # Parse SSE chunks to capture the 'complete' event
            if 'data: ' in chunk:
                for line in chunk.split('\n'):
                    if line.startswith('data: '):
                        try:
                            evt = json.loads(line[6:])
                            if evt.get('type') == 'complete':
                                complete_data = evt.get('data', {})
                        except (json.JSONDecodeError, ValueError):
                            pass

        # After stream ends, persist the result
        if complete_data:
            journey_data = complete_data.get('journey', {})
            _persist_workflow_result(
                trace_id=complete_data.get('trace_id', ''),
                journey_name=complete_data.get('journey_name', ''),
                workflow_type=workflow_type,
                goal=goal,
                samples=journey_data.get('samples', []),
                raw_journey=journey_data,
            )

    response = StreamingHttpResponse(
        _streaming_with_persistence(),
        content_type='text/event-stream',
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response
