from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher
import re

import pandas as pd


@dataclass
class LLMCallMetrics:
    """Metrics from a single LLM call."""
    output: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


# Approximate per-token costs (USD) for common models
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4.1": (2.00 / 1_000_000, 8.00 / 1_000_000),
    "gpt-4.1-mini": (0.40 / 1_000_000, 1.60 / 1_000_000),
    "gpt-4.1-nano": (0.10 / 1_000_000, 0.40 / 1_000_000),
    "gpt-5": (2.50 / 1_000_000, 10.00 / 1_000_000),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single LLM call."""
    input_rate, output_rate = MODEL_COSTS.get(model, (2.50 / 1_000_000, 10.00 / 1_000_000))
    return prompt_tokens * input_rate + completion_tokens * output_rate


@dataclass
class OptimizationResult:
    row_index: int
    input_text: str
    gold_text: str
    output_text: str
    score: float
    iterations: int
    prompt_template: str
    user_comments: str | None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _render_prompt(template: str, user_input: str) -> str:
    if "{input}" not in template:
        raise ValueError("Prompt template must include '{input}' placeholder.")
    return template.replace("{input}", user_input)


def _call_llm(prompt: str, model: str) -> str:
    """Simple string-only LLM call (backward-compat)."""
    return _call_llm_tracked(prompt, model).output


def _call_llm_tracked(prompt: str, model: str) -> LLMCallMetrics:
    """LLM call that returns full metrics (latency, tokens, cost)."""
    t0 = time.time()
    try:
        from litellm import completion
        response = completion(model=model, messages=[{"role": "user", "content": prompt}])
        latency = (time.time() - t0) * 1000
        usage = response.usage
        p_tok = usage.prompt_tokens if usage else 0
        c_tok = usage.completion_tokens if usage else 0
        return LLMCallMetrics(
            output=response.choices[0].message.content or "",
            latency_ms=latency,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            model=model,
        )
    except ImportError:
        return LLMCallMetrics("[LiteLLM not installed - pip install litellm.]", 0, 0, 0, 0, model)
    except Exception as e:
        return LLMCallMetrics(f"[LLM error: {e}]", (time.time() - t0) * 1000, 0, 0, 0, model)


def _optimizer_prompt(
    current_template: str,
    input_text: str,
    gold_text: str,
    output_text: str,
    score: float,
    comments: str | None = None,
) -> str:
    base = (
        "You are optimizing a prompt template for an LLM agent.\n"
        "Return ONLY the improved prompt template.\n\n"
        f"Current prompt template:\n{current_template}\n\n"
        f"Input:\n{input_text}\n\n"
        f"Gold output:\n{gold_text}\n\n"
        f"Model output:\n{output_text}\n\n"
        f"Similarity score: {score:.4f}\n\n"
    )
    if comments and comments.strip():
        base += f"EXPERT EVALUATION FEEDBACK (use this to guide your improvements):\n{comments}\n\n"
    base += (
        "Improve the prompt to make the model output closer to the gold output.\n"
        "Keep the {input} placeholder in the template."
    )
    return base


def load_excel(
    file_path: Path,
    input_col: str,
    gold_col: str,
    comments_col: str | None,
) -> tuple[pd.DataFrame, str, str, str | None]:
    suffix = str(file_path).rsplit('.', 1)[-1].lower()
    if suffix == 'csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    columns = list(df.columns)

    def resolve_column(name_or_pattern: str, label: str) -> str:
        if name_or_pattern in columns:
            return name_or_pattern
        pattern = re.compile(name_or_pattern, re.IGNORECASE)
        matches = [c for c in columns if pattern.search(str(c))]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous {label} column pattern '{name_or_pattern}'. "
                f"Matches: {', '.join(matches)}"
            )
        raise ValueError(
            f"Missing required {label} column. Pattern '{name_or_pattern}' "
            f"did not match any columns: {', '.join(columns)}"
        )

    resolved_input = resolve_column(input_col, "input")
    resolved_gold = resolve_column(gold_col, "gold")

    resolved_comments = None
    if comments_col:
        try:
            resolved_comments = resolve_column(comments_col, "comments")
        except ValueError:
            # Comments column is optional - ignore if not found
            resolved_comments = None

    return df, resolved_input, resolved_gold, resolved_comments


def optimize_row(
    prompt_template: str,
    input_text: str,
    gold_text: str,
    eval_model: str,
    optimizer_model: str,
    target_score: float,
    max_iters: int,
    comments: str | None = None,
) -> tuple[str, str, float, int, float, int, int, int, float]:
    """Returns (template, output, score, iterations, latency_ms, prompt_tokens, completion_tokens, total_tokens, cost_usd)."""
    current_template = prompt_template
    last_output = ""
    last_score = 0.0
    iterations = 0
    total_latency = 0.0
    total_p_tok = 0
    total_c_tok = 0
    total_tok = 0
    total_cost = 0.0

    for i in range(max_iters):
        iterations = i + 1
        rendered = _render_prompt(current_template, input_text)
        m = _call_llm_tracked(rendered, eval_model)
        total_latency += m.latency_ms
        total_p_tok += m.prompt_tokens
        total_c_tok += m.completion_tokens
        total_tok += m.total_tokens
        total_cost += _estimate_cost(eval_model, m.prompt_tokens, m.completion_tokens)

        score = _similarity(m.output, gold_text)
        last_output = m.output
        last_score = score

        if score >= target_score:
            break

        optimizer_query = _optimizer_prompt(
            current_template=current_template,
            input_text=input_text,
            gold_text=gold_text,
            output_text=m.output,
            score=score,
            comments=comments,
        )
        om = _call_llm_tracked(optimizer_query, optimizer_model)
        total_latency += om.latency_ms
        total_p_tok += om.prompt_tokens
        total_c_tok += om.completion_tokens
        total_tok += om.total_tokens
        total_cost += _estimate_cost(optimizer_model, om.prompt_tokens, om.completion_tokens)

        new_template = om.output.strip()
        if not new_template or new_template == current_template:
            break
        # LLM sometimes drops the {input} placeholder — recover by appending it
        if "{input}" not in new_template:
            new_template += "\n\n{input}"
        current_template = new_template

    return current_template, last_output, last_score, iterations, total_latency, total_p_tok, total_c_tok, total_tok, total_cost


def append_learning(
    learning_path: Path,
    run_title: str,
    results: list[OptimizationResult],
) -> None:
    learning_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# {run_title}")
    lines.append("")
    for r in results:
        lines.append(f"## Row {r.row_index}")
        lines.append(f"- Score: {r.score:.4f}")
        lines.append(f"- Iterations: {r.iterations}")
        if r.user_comments:
            lines.append(f"- User comments: {r.user_comments}")
        lines.append("")
        lines.append("### Prompt Template")
        lines.append(r.prompt_template)
        lines.append("")
        lines.append("### Input")
        lines.append(r.input_text)
        lines.append("")
        lines.append("### Gold Output")
        lines.append(r.gold_text)
        lines.append("")
        lines.append("### Model Output")
        lines.append(r.output_text)
        lines.append("")

    content = "\n".join(lines).strip() + "\n"
    if learning_path.exists():
        content = "\n" + content
    learning_path.open("a", encoding="utf-8").write(content)


def run_optimizer(
    excel_path: Path,
    prompt_template: str,
    input_col: str,
    gold_col: str,
    comments_col: str | None,
    eval_model: str,
    optimizer_model: str,
    target_score: float,
    max_iters: int,
    learning_path: Path,
    output_path: Path,
    agent_filter: str | None = None,
    cascade_feedback: str | None = None,
) -> list[OptimizationResult]:
    df, input_col, gold_col, comments_col = load_excel(
        excel_path, input_col, gold_col, comments_col
    )

    # Filter by agent if requested
    if agent_filter:
        columns = list(df.columns)
        agent_col = None
        for c in columns:
            if re.match(r'agent[_ ]?(name)?$', str(c), re.IGNORECASE):
                agent_col = c
                break
        if agent_col:
            df = df[df[agent_col].astype(str).str.strip() == agent_filter].copy()
            if df.empty:
                raise ValueError(f"No rows found for agent '{agent_filter}'")

    results: list[OptimizationResult] = []
    current_template = prompt_template

    for idx, row in df.iterrows():
        input_text = str(row[input_col])
        gold_text = str(row[gold_col])
        user_comments = str(row[comments_col]) if comments_col else None
        if user_comments and user_comments.lower() in {"nan", "none"}:
            user_comments = None

        # Combine cascade feedback with per-row comments
        all_comments_parts = [cascade_feedback, user_comments]
        all_comments = "\n\n".join(p for p in all_comments_parts if p and p.strip())

        current_template, output, score, iters, lat, p_tok, c_tok, t_tok, cost = optimize_row(
            prompt_template=current_template,
            input_text=input_text,
            gold_text=gold_text,
            eval_model=eval_model,
            optimizer_model=optimizer_model,
            target_score=target_score,
            max_iters=max_iters,
            comments=all_comments or None,
        )

        results.append(
            OptimizationResult(
                row_index=int(idx),
                input_text=input_text,
                gold_text=gold_text,
                output_text=output,
                score=score,
                iterations=iters,
                prompt_template=current_template,
                user_comments=user_comments,
                latency_ms=lat,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
                cost_usd=cost,
            )
        )

    run_title = f"Run {datetime.utcnow().isoformat()}Z"
    append_learning(learning_path, run_title, results)

    out_rows = [
        {
            "row_index": r.row_index,
            "score": r.score,
            "iterations": r.iterations,
            "prompt_template": r.prompt_template,
            "input": r.input_text,
            "gold": r.gold_text,
            "output": r.output_text,
            "user_comments": r.user_comments,
        }
        for r in results
    ]
    pd.DataFrame(out_rows).to_csv(output_path, index=False)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize prompts against gold outputs from Excel.")
    parser.add_argument("--excel", required=True, help="Path to Excel file")
    parser.add_argument("--prompt", required=True, help="Prompt template with {input} placeholder")
    parser.add_argument("--input-col", default="input", help="Column name or regex for input text")
    parser.add_argument("--gold-col", default="gold", help="Column name or regex for gold output")
    parser.add_argument("--comments-col", default=None, help="Optional column name or regex for user comments")
    parser.add_argument("--eval-model", default="gpt-4o", help="LLM model for evaluation runs")
    parser.add_argument("--optimizer-model", default="gpt-4o", help="LLM model for prompt optimization")
    parser.add_argument("--target-score", type=float, default=0.85, help="Target similarity score")
    parser.add_argument("--max-iters", type=int, default=3, help="Max optimization iterations per row")
    parser.add_argument("--learning-path", default="learning.md", help="Path to learning.md")
    parser.add_argument("--out", default="prompt_optimizer_results.csv", help="Output CSV path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_optimizer(
        excel_path=Path(args.excel),
        prompt_template=args.prompt,
        input_col=args.input_col,
        gold_col=args.gold_col,
        comments_col=args.comments_col,
        eval_model=args.eval_model,
        optimizer_model=args.optimizer_model,
        target_score=args.target_score,
        max_iters=args.max_iters,
        learning_path=Path(args.learning_path),
        output_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
