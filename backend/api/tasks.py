"""
Celery tasks for async prompt optimization.

The run_optimization_task runs LLM calls in a Celery worker process,
freeing the Django request to return immediately with a task_id.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from celery import shared_task

# Add parent dirs for imports
_backend_dir = str(Path(__file__).resolve().parent.parent)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ai.prompt_optimizer import (
    _call_llm,
    _render_prompt,
    _similarity,
    load_excel,
    run_optimizer,
)


@shared_task(bind=True)
def run_optimization_task(
    self,
    file_path: str,
    prompt_template: str,
    target_score: float = 0.85,
    max_iters: int = 3,
    input_col: str = 'input',
    gold_col: str = 'gold',
    comments_col: str | None = None,
    eval_model: str = 'gpt-4o',
    optimizer_model: str = 'gpt-4o',
    agent_filter: str | None = None,
    cascade_feedback: str | None = None,
):
    """
    Run prompt optimization in a Celery worker.

    Updates progress via self.update_state() so the frontend can poll.
    Returns the full results dict (stored in Redis by Celery).
    """
    filepath = Path(file_path)
    upload_folder = filepath.parent

    try:
        # Create temporary output paths
        filename = filepath.name
        learning_path = upload_folder / f"{filename}_learning.md"
        output_path = upload_folder / f"{filename}_results.csv"

        # Load Excel data for baseline
        df, resolved_input, resolved_gold, resolved_comments = load_excel(
            filepath, input_col, gold_col, comments_col
        )

        # Apply agent filter
        if agent_filter:
            agent_col = None
            for c in df.columns:
                if re.match(r'agent[_ ]?(name)?$', str(c), re.IGNORECASE):
                    agent_col = c
                    break
            if agent_col:
                df = df[df[agent_col].astype(str).str.strip() == agent_filter].copy()

        total_rows = len(df)

        # Compute baseline results
        self.update_state(state='PROGRESS', meta={
            'phase': 'baseline',
            'completed': 0,
            'total': total_rows,
        })

        original_results = []
        for i, (idx, row) in enumerate(df.iterrows()):
            input_text = str(row[resolved_input])
            gold_text = str(row[resolved_gold])
            rendered = _render_prompt(prompt_template, input_text)
            t0 = time.time()
            output = _call_llm(rendered, eval_model)
            baseline_latency = (time.time() - t0) * 1000
            score = _similarity(output, gold_text)
            original_results.append({
                'row_index': int(idx),
                'output': output,
                'score': score,
                'latency_ms': round(baseline_latency, 1),
            })

            self.update_state(state='PROGRESS', meta={
                'phase': 'baseline',
                'completed': i + 1,
                'total': total_rows,
            })

        # Run optimizer
        self.update_state(state='PROGRESS', meta={
            'phase': 'optimizing',
            'completed': 0,
            'total': total_rows,
        })

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
            cascade_feedback=cascade_feedback,
        )

        # Convert results to JSON-serializable format
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

        # Best-effort DB persistence
        try:
            from api.models import OptimizationResult as OptResult, OptimizationRun
            import hashlib
            file_hash = hashlib.sha256(filepath.name.encode()).hexdigest()[:16]
            improvement = ((optimized_avg - original_avg) * 100) if original_avg > 0 else 0

            run_obj = OptimizationRun.objects.create(
                file_hash=file_hash,
                prompt_template=prompt_template,
                eval_model=eval_model,
                optimizer_model=optimizer_model,
                target_score=target_score,
                max_iters=max_iters,
                original_avg_score=original_avg,
                optimized_avg_score=optimized_avg,
                improvement_pct=improvement,
                total_cost_usd=round(total_cost, 4),
                agent_filter=agent_filter,
            )
            for rd in result_data:
                OptResult.objects.create(
                    run=run_obj,
                    row_index=rd['row_index'],
                    input_text=rd['input'],
                    gold_text=rd['gold'],
                    output_text=rd['output'],
                    score=rd['score'],
                    iterations=rd['iterations'],
                    latency_ms=rd['latency_ms'],
                    tokens=rd['total_tokens'],
                    cost_usd=rd['cost_usd'],
                )
        except Exception:
            pass  # Best-effort — never break task results

        return {
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
            },
        }

    finally:
        # Clean up uploaded file
        filepath.unlink(missing_ok=True)
