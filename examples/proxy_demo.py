#!/usr/bin/env python3
"""
ArcanaAI Proxy Demo — 3-agent fan-out pattern.

Sends requests through the ArcanaAI proxy so the trace is captured
automatically. After running, load the printed trace ID in the
dashboard's API page to see the DAG.

Requirements:
    pip install openai

Environment:
    OPENAI_API_KEY    — your OpenAI key (required)
    ARCANA_PROXY_URL  — proxy base URL (default: http://localhost:5000/api/v1/proxy)
"""

import os
import secrets

from openai import OpenAI

PROXY_URL = os.environ.get("ARCANA_PROXY_URL", "http://localhost:5000/api/v1/proxy")
TRACE_ID = f"tr-proxy-{secrets.token_hex(2)}-{secrets.token_hex(2)}"

client = OpenAI(
    base_url=PROXY_URL,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)


def call_agent(*, agent_id, agent_name, parent_id=None, messages):
    """Make a proxied chat completion with tracing headers."""
    headers = {
        "X-Trace-Id": TRACE_ID,
        "X-Agent-Id": agent_id,
        "X-Agent-Name": agent_name,
        "X-Journey-Name": "AI Research Summary",
    }
    if parent_id:
        headers["X-Parent-Id"] = parent_id

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        extra_headers=headers,
    )
    return resp.choices[0].message.content


def main():
    print(f"Trace ID: {TRACE_ID}")
    print(f"Proxy:    {PROXY_URL}\n")

    # Agent 1: Router (root)
    print("[1/3] Router ...")
    router_out = call_agent(
        agent_id="agent-router-001",
        agent_name="Router",
        messages=[{"role": "user", "content": "Research AI trends and write a summary"}],
    )
    print(f"  -> {router_out[:120]}...\n")

    # Agent 2: Researcher (child of Router)
    print("[2/3] Researcher ...")
    researcher_out = call_agent(
        agent_id="agent-researcher-002",
        agent_name="Researcher",
        parent_id="agent-router-001",
        messages=[{"role": "user", "content": "List the top 3 AI trends in 2026"}],
    )
    print(f"  -> {researcher_out[:120]}...\n")

    # Agent 3: Writer (child of Router)
    print("[3/3] Writer ...")
    writer_out = call_agent(
        agent_id="agent-writer-003",
        agent_name="Writer",
        parent_id="agent-router-001",
        messages=[
            {"role": "user", "content": f"Write a 2-paragraph summary of: {researcher_out}"},
        ],
    )
    print(f"  -> {writer_out[:120]}...\n")

    print("=" * 50)
    print(f"Done! Load this trace in the dashboard:")
    print(f"  Trace ID: {TRACE_ID}")
    print(f"  Dashboard: http://localhost:8000 -> API -> paste trace ID -> Load Trace")


if __name__ == "__main__":
    main()
