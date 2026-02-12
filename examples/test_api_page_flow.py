#!/usr/bin/env python3
"""
Test the API page flow: 5-agent research paper system (Scope -> Outline -> Draft -> Critic -> Finalizer).

Uses OPENAI_API_KEY from .env (project root). TRACE_ID for dashboard load.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/api/v1/proxy",
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

TRACE_ID = "tr-proxy-a298-88f9"

def _content(r):
    return r.choices[0].message.content or ""

# Agent 1: Scope (root) — define research question and scope
scope = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "User wants to write a short research-style note on: 'Impact of LLM fine-tuning on instruction following.'\n"
                "Define the research question, target audience, and 3–5 key sub-questions to answer. Output as bullets."
            ),
        }
    ],
    extra_headers={
        "X-Trace-Id": TRACE_ID,
        "X-Agent-Id": "agent-scope-001",
        "X-Agent-Name": "Scope",
        "X-Journey-Name": "Research Paper",
    },
)

# Agent 2: Outline (child of Scope) — structure and literature areas
outline = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Turn this into a paper outline with sections: Abstract, Intro, Background, Methods/Approach, Results/Discussion, Conclusion, References.\n"
                "For each section, list 2–3 bullet points on what to cover.\n\n"
                "Scope:\n" + _content(scope)
            ),
        }
    ],
    extra_headers={
        "X-Trace-Id": TRACE_ID,
        "X-Agent-Id": "agent-outline-002",
        "X-Agent-Name": "Outline",
        "X-Parent-Id": "agent-scope-001",
    },
)

# Agent 3: Draft (child of Outline) — write a short draft
draft = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Write a 2-paragraph draft covering: (1) what fine-tuning is and why it matters for instruction following, "
                "(2) one key finding or trade-off from recent work. Keep it concise and academic in tone.\n\n"
                "Outline:\n" + _content(outline)
            ),
        }
    ],
    extra_headers={
        "X-Trace-Id": TRACE_ID,
        "X-Agent-Id": "agent-draft-003",
        "X-Agent-Name": "Draft",
        "X-Parent-Id": "agent-outline-002",
    },
)

# Agent 4: Critic (child of Draft) — suggest improvements
critic = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Review this draft and give 3–4 concrete improvements: clarity, missing points, structure, or citations. Be brief.\n\n"
                "Draft:\n" + _content(draft)
            ),
        }
    ],
    extra_headers={
        "X-Trace-Id": TRACE_ID,
        "X-Agent-Id": "agent-critic-004",
        "X-Agent-Name": "Critic",
        "X-Parent-Id": "agent-draft-003",
    },
)

# Agent 5: Finalizer (child of Critic) — polish into final summary
finalizer = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": (
                "Apply the critic's suggestions and produce a short, polished 2-paragraph version. "
                "Then add one sentence: 'Suggested next steps: [2–3 items].'\n\n"
                "Draft:\n" + _content(draft) + "\n\nCritic feedback:\n" + _content(critic)
            ),
        }
    ],
    extra_headers={
        "X-Trace-Id": TRACE_ID,
        "X-Agent-Id": "agent-finalizer-005",
        "X-Agent-Name": "Finalizer",
        "X-Parent-Id": "agent-critic-004",
    },
)

print("Trace ID:", TRACE_ID)
print("Finalizer output:", _content(finalizer)[:500])
