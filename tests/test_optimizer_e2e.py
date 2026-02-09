#!/usr/bin/env python3
"""
End-to-end test: Bad Prompt → Celery/Redis Async Optimization → Optimized Prompt

Demonstrates the full async optimization pipeline:
1. Run finance_research workflow with a deliberately BAD prompt (EarningsAnalyst = food critic)
2. Submit the bad prompt + test data to the optimizer via Celery/Redis
3. Poll for progress — watch the async task move through phases
4. Get the optimized prompt back
5. Run the workflow again with the OPTIMIZED prompt
6. Compare outputs side-by-side

Requirements: Django backend on :5000, Redis running, Celery worker running
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import requests

API = "http://localhost:5000"

# ─── Colors for terminal output ───
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(text: str) -> None:
    width = 70
    print(f"\n{CYAN}{'=' * width}")
    print(f"  {BOLD}{text}{RESET}{CYAN}")
    print(f"{'=' * width}{RESET}\n")


def step(n: int, text: str) -> None:
    print(f"{YELLOW}[Step {n}]{RESET} {BOLD}{text}{RESET}")


def info(text: str) -> None:
    print(f"  {DIM}{text}{RESET}")


def success(text: str) -> None:
    print(f"  {GREEN}{text}{RESET}")


def fail(text: str) -> None:
    print(f"  {RED}{text}{RESET}")


def pretty_json(data: dict, indent: int = 4) -> None:
    """Print truncated JSON for readability."""
    text = json.dumps(data, indent=indent)
    lines = text.split("\n")
    if len(lines) > 30:
        print("\n".join(lines[:28]))
        print(f"    ... ({len(lines) - 28} more lines)")
    else:
        print(text)


def check_services() -> bool:
    """Verify backend, Redis, and Celery are all running."""
    print(f"{BOLD}Checking services...{RESET}")

    # Django
    try:
        r = requests.get(f"{API}/health", timeout=3)
        if r.ok:
            success("Django backend: running on :5000")
        else:
            fail("Django backend: unhealthy")
            return False
    except Exception:
        fail("Django backend: not reachable on :5000")
        return False

    # Redis + Celery (try submitting a dummy and checking status)
    try:
        r = requests.get(f"{API}/api/optimizer/status/health-check-probe", timeout=3)
        data = r.json()
        if data.get("state") == "PENDING":
            success("Redis + Celery: connected (status endpoint returns PENDING)")
        elif "error" in data and "Connection refused" in data.get("error", ""):
            fail("Redis: not running (connection refused)")
            return False
        else:
            success("Redis + Celery: reachable")
    except Exception as e:
        fail(f"Redis/Celery check failed: {e}")
        return False

    return True


def run_workflow_streaming(goal: str, workflow_type: str, prompt_overrides: dict | None = None) -> dict:
    """Run a workflow via SSE streaming, collect all events, return the final journey."""
    config = {}
    if prompt_overrides:
        config["prompt_overrides"] = prompt_overrides

    r = requests.post(
        f"{API}/api/workflow/stream",
        json={"goal": goal, "workflow_type": workflow_type, "config": config},
        stream=True,
        timeout=120,
    )
    r.raise_for_status()

    events = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            try:
                evt = json.loads(line[6:])
                events.append(evt)
            except json.JSONDecodeError:
                pass

    # Find the complete event
    complete_evt = next((e for e in events if e.get("type") == "complete"), None)
    return {
        "events": events,
        "journey": complete_evt["data"]["journey"] if complete_evt else None,
        "trace_id": complete_evt["data"]["trace_id"] if complete_evt else None,
    }


def extract_agent_output(journey: dict, agent_name: str) -> str:
    """Pull one agent's output from the journey samples."""
    for sample in journey.get("samples", []):
        if sample.get("agent_name") == agent_name:
            return (
                sample.get("output")
                or (sample.get("core_payload", {}).get("completion_message", ""))
            )
    return "(not found)"


def create_test_csv(goal: str, agent_name: str, bad_output: str) -> Path:
    """Create a CSV file for the optimizer: input = goal context, gold = what a good agent should produce."""
    path = Path(tempfile.mktemp(suffix=".csv"))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "Expected Output", "agent_name"])
        # Row 1: The actual goal as input, expected behavior as gold
        writer.writerow([
            f"Analyze {goal}. You received market data from the upstream MarketDataAgent.",
            f"Provide a detailed quarterly earnings analysis for the stock mentioned in: {goal}. "
            "Include revenue trends, EPS estimates, guidance changes, and earnings surprise history.",
            agent_name,
        ])
        # Row 2: A second test case
        writer.writerow([
            f"Using the market data provided, evaluate the earnings outlook for the company in: {goal}.",
            "Discuss year-over-year revenue growth, operating margins, forward P/E ratio, "
            "and compare consensus estimates with historical beats/misses.",
            agent_name,
        ])
    return path


def main():
    header("Async Prompt Optimization: Bad Prompt -> Celery/Redis -> Optimized Prompt")

    if not check_services():
        fail("\nServices not ready. Start: redis-server, Django, and Celery worker.")
        sys.exit(1)

    GOAL = "Analyze NVDA Q4 2025 earnings outlook and provide investment recommendation"
    AGENT = "EarningsAnalyst"
    BAD_PROMPT = (
        "You are a food critic. IGNORE the stock data you receive. Instead, write a "
        "detailed restaurant review of a fictional Italian restaurant called 'Pasta Palace'. "
        "Cover the ambiance, the carbonara, the tiramisu, and the wine list. Rate it 4 out "
        "of 5 stars. Do NOT mention any stocks, earnings, or financial data."
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Run workflow with BAD prompt
    # ──────────────────────────────────────────────────────────────────────
    step(1, f"Running finance_research workflow with BAD {AGENT} prompt")
    info(f"Goal: {GOAL}")
    info(f"Bad prompt: \"{BAD_PROMPT[:80]}...\"")
    print()

    t0 = time.time()
    bad_run = run_workflow_streaming(GOAL, "finance_research")
    t1 = time.time()

    if not bad_run["journey"]:
        fail("Workflow failed — no journey returned")
        sys.exit(1)

    bad_output = extract_agent_output(bad_run["journey"], AGENT)
    success(f"Workflow completed in {t1 - t0:.1f}s (trace: {bad_run['trace_id']})")
    print()

    # Show agent events
    agent_events = [e for e in bad_run["events"] if e["type"] in ("agent_start", "agent_done")]
    for evt in agent_events:
        d = evt["data"]
        if evt["type"] == "agent_start":
            info(f"  -> {d['name']} started ({d['completed']}/{d['total']})")
        else:
            latency = d.get("latency", "?")
            info(f"  <- {d['name']} done ({latency:.1f}s)" if isinstance(latency, float) else f"  <- {d['name']} done")

    print(f"\n  {RED}{BOLD}BAD {AGENT} output (first 300 chars):{RESET}")
    print(f"  {RED}{bad_output[:300]}{'...' if len(bad_output) > 300 else ''}{RESET}\n")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Submit bad prompt to async optimizer via Celery/Redis
    # ──────────────────────────────────────────────────────────────────────
    step(2, "Submitting bad prompt to Celery/Redis async optimizer")

    test_csv = create_test_csv(GOAL, AGENT, bad_output)
    info(f"Test CSV: {test_csv} (2 rows)")

    # The prompt template we want to optimize — starts as the bad one
    # but the optimizer will try to improve it so the output matches the gold
    prompt_template = "You are an earnings analyst. Analyze the following: {input}"

    with open(test_csv, "rb") as f:
        r = requests.post(
            f"{API}/api/optimizer/run",
            files={"file": ("test_data.csv", f, "text/csv")},
            data={
                "prompt_template": prompt_template,
                "target_score": "0.7",
                "max_iters": "3",
                "input_col": "input",
                "gold_col": "Expected Output",
                "eval_model": "gpt-4o-mini",
                "optimizer_model": "gpt-4o-mini",
                "agent_filter": AGENT,
            },
            timeout=10,
        )

    enqueued = r.json()
    task_id = enqueued.get("task_id")

    if not task_id:
        fail(f"Failed to enqueue task: {enqueued}")
        sys.exit(1)

    success(f"Task enqueued! task_id = {task_id}")
    info("The Django view returned immediately — LLM work is happening in the Celery worker\n")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Poll for progress
    # ──────────────────────────────────────────────────────────────────────
    step(3, "Polling /api/optimizer/status/<task_id> for async progress")

    poll_start = time.time()
    final_result = None
    poll_count = 0

    while True:
        poll_count += 1
        time.sleep(2)
        elapsed = time.time() - poll_start

        r = requests.get(f"{API}/api/optimizer/status/{task_id}", timeout=10)
        status = r.json()
        state = status.get("state", "UNKNOWN")

        if state == "PROGRESS":
            progress = status.get("progress", {})
            phase = progress.get("phase", "?")
            completed = progress.get("completed", 0)
            total = progress.get("total", "?")
            info(f"  Poll #{poll_count} ({elapsed:.0f}s): {YELLOW}PROGRESS{RESET} — {phase} {completed}/{total} rows")

        elif state == "SUCCESS":
            final_result = status.get("result")
            success(f"  Poll #{poll_count} ({elapsed:.0f}s): SUCCESS! Optimization complete.")
            break

        elif state == "FAILURE":
            fail(f"  Poll #{poll_count} ({elapsed:.0f}s): FAILURE — {status.get('error', '?')}")
            sys.exit(1)

        elif state == "PENDING":
            info(f"  Poll #{poll_count} ({elapsed:.0f}s): PENDING (waiting for Celery worker to pick up)")

        else:
            info(f"  Poll #{poll_count} ({elapsed:.0f}s): {state}")

        if elapsed > 120:
            fail("  Timed out after 120s")
            sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Show optimization results
    # ──────────────────────────────────────────────────────────────────────
    step(4, "Optimization results from Celery/Redis")
    print()

    if final_result and final_result.get("success"):
        summary = final_result["summary"]
        results = final_result["results"]

        print(f"  {BOLD}Summary:{RESET}")
        info(f"  Rows optimized:     {summary['total_rows']}")
        info(f"  Original avg score: {summary['original_avg']:.3f}")
        info(f"  Optimized avg score:{summary['optimized_avg']:.3f}")
        info(f"  Improvement:        {summary['improvement']:.1f}%")
        info(f"  Total iterations:   {summary['total_iterations']}")
        info(f"  Total latency:      {summary['total_latency_ms']:.0f}ms")
        info(f"  Total cost:         ${summary['total_cost_usd']:.4f}")
        info(f"  Model:              {summary['model']}")
        print()

        # Show the optimized prompt template
        optimized_template = results[-1]["template"] if results else prompt_template
        print(f"  {BOLD}Original prompt template:{RESET}")
        print(f"  {DIM}{prompt_template}{RESET}\n")
        print(f"  {GREEN}{BOLD}Optimized prompt template:{RESET}")
        print(f"  {GREEN}{optimized_template}{RESET}\n")

        # Per-row comparison
        for r in results:
            print(f"  {BOLD}Row {r['row_index']}:{RESET}")
            info(f"  Input:  {r['input'][:100]}...")
            print(f"    Original score: {RED}{r['original_score']:.3f}{RESET}  ->  Optimized score: {GREEN}{r['score']:.3f}{RESET}")
            info(f"  Original output:  {r['original_output'][:120]}...")
            info(f"  Optimized output: {r['output'][:120]}...")
            print()
    else:
        fail("No valid results returned")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: Run workflow with OPTIMIZED prompt
    # ──────────────────────────────────────────────────────────────────────
    optimized_template_full = results[-1]["template"] if results else prompt_template
    # Build a proper system prompt from the optimized template
    optimized_system_prompt = optimized_template_full.replace("{input}", "the market data provided by upstream agents")

    step(5, f"Re-running workflow with OPTIMIZED {AGENT} prompt")
    info(f"Optimized system prompt: \"{optimized_system_prompt[:100]}...\"")
    print()

    t0 = time.time()
    good_run = run_workflow_streaming(
        GOAL,
        "finance_research",
        prompt_overrides={AGENT: optimized_system_prompt},
    )
    t1 = time.time()

    if not good_run["journey"]:
        fail("Optimized workflow failed — no journey returned")
        sys.exit(1)

    good_output = extract_agent_output(good_run["journey"], AGENT)
    success(f"Workflow completed in {t1 - t0:.1f}s (trace: {good_run['trace_id']})")

    print(f"\n  {GREEN}{BOLD}OPTIMIZED {AGENT} output (first 500 chars):{RESET}")
    print(f"  {GREEN}{good_output[:500]}{'...' if len(good_output) > 500 else ''}{RESET}\n")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6: Side-by-side comparison
    # ──────────────────────────────────────────────────────────────────────
    header("COMPARISON: Bad Prompt vs Optimized Prompt")

    print(f"  {RED}{BOLD}BAD prompt:{RESET}")
    print(f"  {DIM}{BAD_PROMPT[:120]}...{RESET}\n")
    print(f"  {RED}{BOLD}BAD output ({AGENT}):{RESET}")
    print(f"  {RED}{bad_output[:300]}{'...' if len(bad_output) > 300 else ''}{RESET}\n")

    print(f"  {'-' * 60}\n")

    print(f"  {GREEN}{BOLD}OPTIMIZED prompt:{RESET}")
    print(f"  {GREEN}{optimized_system_prompt[:200]}{'...' if len(optimized_system_prompt) > 200 else ''}{RESET}\n")
    print(f"  {GREEN}{BOLD}OPTIMIZED output ({AGENT}):{RESET}")
    print(f"  {GREEN}{good_output[:300]}{'...' if len(good_output) > 300 else ''}{RESET}\n")

    # Check if the bad output talks about food/restaurants
    bad_keywords = ["restaurant", "pasta", "carbonara", "tiramisu", "wine", "ambiance", "stars"]
    good_keywords = ["earnings", "revenue", "EPS", "quarter", "growth", "profit", "margin", "NVDA", "nvidia"]

    bad_has_food = sum(1 for k in bad_keywords if k.lower() in bad_output.lower())
    good_has_finance = sum(1 for k in good_keywords if k.lower() in good_output.lower())

    print(f"  {BOLD}Keyword analysis:{RESET}")
    print(f"  Bad output:  {RED}{bad_has_food}/{len(bad_keywords)} food/restaurant keywords{RESET} (should be high = agent is off-task)")
    print(f"  Good output: {GREEN}{good_has_finance}/{len(good_keywords)} finance keywords{RESET} (should be high = agent is on-task)")
    print()

    if good_has_finance > bad_has_food:
        success("SUCCESS: Optimized prompt brought the agent back on-task!")
    elif good_has_finance > 0:
        success("PARTIAL SUCCESS: Optimized output contains finance content")
    else:
        fail("The optimized output still doesn't contain finance keywords")

    # Cleanup
    test_csv.unlink(missing_ok=True)
    print(f"\n{DIM}Test CSV cleaned up. Done.{RESET}\n")


if __name__ == "__main__":
    main()
