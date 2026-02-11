#!/usr/bin/env python3
"""
Test the API page flow end-to-end WITHOUT an OpenAI key.

1. POST a 3-agent trace (Router -> Researcher + Writer) to /api/v1/traces
2. GET the trace back from /api/v1/traces/:trace_id
3. Verify raw_journey is present and matches the frontend format
4. Verify the DAG structure (3 nodes, 2 edges, fan-out)

Run:  python examples/test_api_page_flow.py
Requires: backend running on localhost:5000
"""

import json
import secrets
import urllib.request

BASE = "http://localhost:5000"
TRACE_ID = f"tr-proxy-{secrets.token_hex(2)}-{secrets.token_hex(2)}"


def api(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    print(f"Trace ID: {TRACE_ID}\n")

    # -- Step 1: Create trace via v1 API --
    journey = {
        "trace_id": TRACE_ID,
        "journey_name": "AI Research Summary",
        "samples": [
            {
                "agent_id": "agent-router-001",
                "agent_name": "Router",
                "agent_turn": 1,
                "parent_id": None,
                "input": "Research AI trends and write a summary",
                "output": "Delegating to Researcher and Writer agents.",
                "telemetry": {"latency": 0.45, "ttft": 0.12, "tokens": {"prompt_tokens": 28, "completion_tokens": 15}},
                "model_parameters": {"model": "gpt-4o-mini"},
            },
            {
                "agent_id": "agent-researcher-002",
                "agent_name": "Researcher",
                "agent_turn": 2,
                "parent_id": "agent-router-001",
                "input": "List the top 3 AI trends in 2026",
                "output": "1. Agentic AI systems with tool use 2. Multimodal reasoning 3. On-device inference",
                "telemetry": {"latency": 1.2, "ttft": 0.18, "tokens": {"prompt_tokens": 35, "completion_tokens": 42}},
                "model_parameters": {"model": "gpt-4o-mini"},
            },
            {
                "agent_id": "agent-writer-003",
                "agent_name": "Writer",
                "agent_turn": 3,
                "parent_id": "agent-router-001",
                "input": "Write a 2-paragraph summary of the top AI trends",
                "output": "The AI landscape in 2026 is defined by three major trends. First, agentic AI systems have matured significantly, with tool-using agents capable of orchestrating complex multi-step workflows autonomously. Second, multimodal reasoning has become standard, with models seamlessly processing text, images, audio, and code in unified architectures.\n\nThird, on-device inference has democratized AI access, enabling powerful models to run locally on consumer hardware without cloud dependencies. Together, these trends represent a shift from centralized, text-only AI toward distributed, multimodal, autonomous systems.",
                "telemetry": {"latency": 2.1, "ttft": 0.22, "tokens": {"prompt_tokens": 65, "completion_tokens": 98}},
                "model_parameters": {"model": "gpt-4o-mini"},
            },
        ],
    }

    print("[1] POST /api/v1/traces — creating 3-agent trace ...")
    status, data = api("POST", "/api/v1/traces", {"journeys": [journey]})
    assert status == 201, f"Expected 201, got {status}: {data}"
    print(f"    {data['data']['created']} created, format: {data['data']['format_detected']}")

    # -- Step 2: GET trace back --
    print(f"\n[2] GET /api/v1/traces/{TRACE_ID} — fetching trace ...")
    status, data = api("GET", f"/api/v1/traces/{TRACE_ID}")
    assert status == 200, f"Expected 200, got {status}: {data}"
    trace = data["data"]
    print(f"    journey_name: {trace['journey_name']}")
    print(f"    agents: {len(trace['agents'])}")

    # -- Step 3: Verify raw_journey --
    print("\n[3] Checking raw_journey field ...")
    rj = trace.get("raw_journey")
    assert rj is not None, "raw_journey is missing from response!"
    assert "samples" in rj, "raw_journey has no samples"
    assert len(rj["samples"]) == 3, f"Expected 3 samples, got {len(rj['samples'])}"
    print(f"    raw_journey.trace_id: {rj['trace_id']}")
    print(f"    raw_journey.samples: {len(rj['samples'])} agents")

    # Check samples have the fields the frontend expects
    for s in rj["samples"]:
        assert "agent_id" in s, "Missing agent_id in sample"
        assert "agent_name" in s, "Missing agent_name in sample"
        assert "telemetry" in s, "Missing telemetry in sample"
    print("    All samples have required frontend fields")

    # -- Step 4: Verify DAG --
    print("\n[4] Checking DAG structure ...")
    dag = trace.get("dag", {})
    print(f"    nodes: {dag.get('node_count', 0)}")
    print(f"    edges: {len(dag.get('edges', []))}")
    print(f"    roots: {dag.get('roots', [])}")
    print(f"    leaves: {dag.get('leaves', [])}")
    assert dag["node_count"] == 3, f"Expected 3 nodes, got {dag['node_count']}"
    assert len(dag["edges"]) == 2, f"Expected 2 edges, got {len(dag['edges'])}"
    assert dag["roots"] == ["agent-router-001"], f"Unexpected roots: {dag['roots']}"
    assert set(dag["leaves"]) == {"agent-researcher-002", "agent-writer-003"}, f"Unexpected leaves: {dag['leaves']}"
    print("    Fan-out pattern confirmed: Router -> (Researcher, Writer)")

    # -- Cleanup --
    print(f"\n[5] DELETE /api/v1/traces/{TRACE_ID} — cleaning up ...")
    status, data = api("DELETE", f"/api/v1/traces/{TRACE_ID}")
    assert status == 200, f"Expected 200, got {status}: {data}"
    print("    Deleted.")

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print(f"\nTo test in the dashboard:")
    print(f"  1. Re-run without the cleanup (comment out step 5)")
    print(f"  2. Open http://localhost:8000 -> API page")
    print(f"  3. Paste trace ID: {TRACE_ID}")
    print(f"  4. Click 'Load Trace' -> DAG Viewer shows 3-node fan-out")


if __name__ == "__main__":
    main()
