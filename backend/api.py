"""
Flask API server for ArcanaAI LLM Observability Platform.

Provides endpoints for:
- Prompt optimization with Excel uploads
- Multi-agent workflow execution
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.prompt_optimizer import (
    run_optimizer,
    OptimizationResult,
    load_excel,
    optimize_row,
    _render_prompt,
    _call_llm,
    _similarity,
)
from gateway.gateway import ModelRegistry, GatewayRouter, RoutingRequest
from gateway.gateway_cache import CacheStore
from workflow_engine import WorkflowEngine
from semantic_pipeline import (
    AgentDAG,
    CascadeEvaluator,
    TaskProgressMonitor,
    extract_agent_logs,
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "arcana_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'arcana-api',
        'version': '1.0.0'
    })


@app.route('/api/optimizer/run', methods=['POST'])
def run_prompt_optimizer():
    """
    Run prompt optimizer with uploaded Excel file.

    Expects multipart/form-data with:
    - file: Excel/CSV file with input, gold, and optional comments columns
    - prompt_template: Prompt template with {input} placeholder
    - input_col: Column name/pattern for input (default: "input")
    - gold_col: Column name/pattern for gold output (default: "gold")
    - comments_col: Optional column name/pattern for comments
    - eval_model: LLM model for evaluation (default: "gpt-4o")
    - optimizer_model: LLM model for optimization (default: "gpt-4o")
    - target_score: Target similarity score (default: 0.85)
    - max_iters: Max iterations per row (default: 3)
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Must be .xlsx, .xls, or .csv'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(filepath))

        # Extract form parameters
        prompt_template = request.form.get('prompt_template', '')
        if not prompt_template:
            return jsonify({'error': 'prompt_template is required'}), 400

        if '{input}' not in prompt_template:
            return jsonify({'error': 'prompt_template must contain {input} placeholder'}), 400

        input_col = request.form.get('input_col', 'input')
        gold_col = request.form.get('gold_col', 'gold')
        comments_col = request.form.get('comments_col', None)
        eval_model = request.form.get('eval_model', 'gpt-4o')
        optimizer_model = request.form.get('optimizer_model', 'gpt-4o')
        target_score = float(request.form.get('target_score', 0.85))
        max_iters = int(request.form.get('max_iters', 3))
        agent_filter = request.form.get('agent_filter', None)

        # Create temporary output paths
        learning_path = Path(app.config['UPLOAD_FOLDER']) / f"{filename}_learning.md"
        output_path = Path(app.config['UPLOAD_FOLDER']) / f"{filename}_results.csv"

        # Load Excel data first to get baseline results
        df, resolved_input, resolved_gold, resolved_comments = load_excel(
            filepath, input_col, gold_col, comments_col
        )

        # Apply agent filter to baseline too
        import re as _re
        if agent_filter:
            agent_col = None
            for c in df.columns:
                if _re.match(r'agent[_ ]?(name)?$', str(c), _re.IGNORECASE):
                    agent_col = c
                    break
            if agent_col:
                df = df[df[agent_col].astype(str).str.strip() == agent_filter].copy()

        # Compute original (baseline) results before optimization — with metrics
        import time as _time
        original_results = []
        for idx, row in df.iterrows():
            input_text = str(row[resolved_input])
            gold_text = str(row[resolved_gold])
            rendered = _render_prompt(prompt_template, input_text)
            t0 = _time.time()
            output = _call_llm(rendered, eval_model)
            baseline_latency = (_time.time() - t0) * 1000
            score = _similarity(output, gold_text)
            original_results.append({
                'row_index': int(idx),
                'output': output,
                'score': score,
                'latency_ms': round(baseline_latency, 1),
            })

        # Run optimizer (with agent filter + metrics)
        results = run_optimizer(
            excel_path=filepath,
            prompt_template=prompt_template,
            input_col=input_col,
            gold_col=gold_col,
            comments_col=comments_col,
            eval_model=eval_model,
            optimizer_model=optimizer_model,
            target_score=target_score,
            max_iters=max_iters,
            learning_path=learning_path,
            output_path=output_path,
            agent_filter=agent_filter,
        )

        # Convert results to JSON-serializable format with original results
        result_data = []
        for i, r in enumerate(results):
            orig = original_results[i] if i < len(original_results) else None
            result_data.append({
                'row_index': r.row_index,
                'input': r.input_text,
                'gold': r.gold_text,
                'output': r.output_text,
                'score': r.score,
                'iterations': r.iterations,
                'template': r.prompt_template,
                'comments': r.user_comments,
                'original_output': orig['output'] if orig else '',
                'original_score': orig['score'] if orig else 0,
                'original_latency_ms': orig['latency_ms'] if orig else 0,
                'latency_ms': round(r.latency_ms, 1),
                'prompt_tokens': r.prompt_tokens,
                'completion_tokens': r.completion_tokens,
                'total_tokens': r.total_tokens,
                'cost_usd': round(r.cost_usd, 6),
            })

        # Compute aggregated metrics
        original_avg = sum(r['original_score'] for r in result_data) / len(result_data) if result_data else 0
        optimized_avg = sum(r['score'] for r in result_data) / len(result_data) if result_data else 0
        total_latency = sum(r['latency_ms'] for r in result_data)
        total_cost = sum(r['cost_usd'] for r in result_data)
        total_tokens = sum(r['total_tokens'] for r in result_data)
        total_baseline_latency = sum(r['original_latency_ms'] for r in result_data)

        # Clean up uploaded file
        filepath.unlink(missing_ok=True)

        return jsonify({
            'success': True,
            'results': result_data,
            'summary': {
                'total_rows': len(results),
                'original_avg': original_avg,
                'optimized_avg': optimized_avg,
                'improvement': ((optimized_avg - original_avg) * 100) if original_avg > 0 else 0,
                'total_iterations': sum(r.iterations for r in results),
                'met_target': sum(1 for r in results if r.score >= target_score),
                'total_latency_ms': round(total_latency, 1),
                'total_baseline_latency_ms': round(total_baseline_latency, 1),
                'avg_latency_ms': round(total_latency / len(result_data), 1) if result_data else 0,
                'total_cost_usd': round(total_cost, 4),
                'total_tokens': total_tokens,
                'agent_filter': agent_filter,
                'model': eval_model,
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/pipeline/analyze', methods=['POST'])
def analyze_pipeline():
    """
    Run LLM goal verification on agent traces.

    Expects JSON body with:
    - traces: List of agent execution samples (from keywords_ai format)
    - trace_id: Optional specific trace to analyze
    - user_goal: User goal for task progress monitoring (required)
    """
    try:
        data = request.get_json()

        if not data or 'traces' not in data:
            return jsonify({'error': 'traces array is required'}), 400

        traces = data['traces']
        trace_id = data.get('trace_id')
        user_goal = data.get('user_goal')
        evaluation_mode = data.get('evaluation_mode', 'cascade')

        if not user_goal:
            return jsonify({'error': 'user_goal is required'}), 400

        if evaluation_mode not in ('checkpoints', 'cascade', 'both'):
            return jsonify({'error': 'evaluation_mode must be checkpoints, cascade, or both'}), 400

        # Extract agent logs
        agent_logs = extract_agent_logs(traces)

        # Build DAGs
        dags = AgentDAG.from_agent_logs(agent_logs)

        if trace_id and trace_id not in dags:
            return jsonify({'error': f'trace_id {trace_id} not found'}), 404

        # Analyze specific trace or all traces
        target_traces = [trace_id] if trace_id else list(dags.keys())

        results = {}
        for tid in target_traces:
            dag = dags[tid]
            result = {'trace_id': tid}

            # Run checkpoints if mode is checkpoints or both
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
                        'agent_names': [child_node.get('agent_name', child_id) for _, child_id, child_node in cp['steps']],
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

            # Run cascade evaluation if mode is cascade or both
            if evaluation_mode in ('cascade', 'both'):
                cascade = CascadeEvaluator(dag, user_goal)
                result['cascade_evaluation'] = cascade.evaluate()

            results[tid] = result

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/evaluations/run', methods=['POST'])
def run_evaluations():
    """
    Run custom evaluations on agent outputs.

    Expects JSON body with:
    - traces: List of agent execution samples
    - eval_type: Type of evaluation (e.g., "accuracy", "relevance", "safety")
    - eval_prompt: Custom evaluation prompt template
    - model: Optional LLM model for evaluation (default: "gpt-4o")
    """
    try:
        data = request.get_json()

        if not data or 'traces' not in data:
            return jsonify({'error': 'traces array is required'}), 400

        traces = data['traces']
        eval_type = data.get('eval_type', 'custom')
        eval_prompt = data.get('eval_prompt', '')
        model = data.get('model', 'gpt-4o')

        # TODO: Implement custom evaluation logic
        # For now, return placeholder results
        results = []
        for trace in traces:
            agent_id = trace.get('agent_id', 'unknown')
            output = trace.get('output', '')

            # Placeholder score
            score = 0.85

            results.append({
                'agent_id': agent_id,
                'eval_type': eval_type,
                'score': score,
                'output': output[:100] + '...' if len(output) > 100 else output,
            })

        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_evaluated': len(results),
                'avg_score': sum(r['score'] for r in results) / len(results) if results else 0,
                'eval_type': eval_type,
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/gateway/route', methods=['POST'])
def route_gateway():
    """
    Route a task to the best model based on constraints.

    Expects JSON body with:
    - intent: user intent or task description
    - task_type: task type (e.g., summarization, analysis)
    - latency_budget_ms: optional latency budget
    - cost_budget: optional cost budget per 1k tokens
    - quality: optional quality target (low, medium, high)
    - cache_threshold: optional similarity threshold for cache hit
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON body is required'}), 400

        intent = data.get('intent')
        task_type = data.get('task_type')
        if not intent or not task_type:
            return jsonify({'error': 'intent and task_type are required'}), 400

        latency_budget_ms = data.get('latency_budget_ms')
        cost_budget = data.get('cost_budget')
        quality = data.get('quality')
        cache_threshold = data.get('cache_threshold', 0.9)

        registry_path = Path(__file__).parent / "gateway" / "gateway_registry.json"
        cache_path = Path(__file__).parent / "gateway" / "gateway_cache.jsonl"

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

        return jsonify({
            'success': True,
            'decision': {
                'model_id': decision.model_id,
                'provider': decision.provider,
                'score': decision.score,
                'rationale': decision.rationale,
                'source': decision.source,
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/workflow/run', methods=['POST'])
def run_workflow():
    """
    Execute a multi-agent workflow with real LLM calls.

    Expects JSON body with:
    - goal: User goal/instruction
    - workflow_type: Type of workflow (e.g., "trip_booking", "refund_request")
    - config: Optional workflow configuration
    """
    try:
        data = request.get_json()

        if not data or 'goal' not in data:
            return jsonify({'error': 'goal is required'}), 400

        goal = data['goal']
        workflow_type = data.get('workflow_type', 'custom')
        config = data.get('config', {})

        engine = WorkflowEngine(goal=goal, workflow_type=workflow_type, config=config)
        result = engine.execute()

        return jsonify({
            'success': True,
            'trace_id': result['trace_id'],
            'journey_name': result['journey_name'],
            'journey': result['journey'],
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/workflow/stream', methods=['POST'])
def stream_workflow():
    """
    Execute a multi-agent workflow with SSE streaming.

    Same parameters as /api/workflow/run but returns a text/event-stream
    with real-time agent execution updates.
    """
    data = request.get_json()

    if not data or 'goal' not in data:
        return jsonify({'error': 'goal is required'}), 400

    goal = data['goal']
    workflow_type = data.get('workflow_type', 'custom')
    config = data.get('config', {})

    engine = WorkflowEngine(goal=goal, workflow_type=workflow_type, config=config)

    return Response(
        engine.execute_streaming(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


if __name__ == '__main__':
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY not set in environment. LLM calls will fail.")
        print("Set it in .env file or export OPENAI_API_KEY=your-key")

    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("Starting ArcanaAI API server on http://localhost:5000")
    app.run(debug=False, port=5000)
