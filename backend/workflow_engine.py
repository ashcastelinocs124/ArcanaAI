"""
Workflow execution engine for multi-agent LLM workflows.

Executes real LLM-powered agent chains based on predefined topologies.
Each agent calls LiteLLM completion() with a role-specific system prompt,
records timing/tokens, and passes output to downstream agents.

Supports async parallel execution: agents on the same topological level
(i.e. all parents complete) run concurrently via asyncio.gather().
"""
from __future__ import annotations

import asyncio
import json
import queue
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Generator


WORKFLOW_TOPOLOGIES: dict[str, list[dict]] = {}


def _generate_trace_id() -> str:
    short = uuid.uuid4().hex[:16]
    return f"tr-{short[:4]}-{short[4:8]}-{short[8:12]}-{short[12:16]}"


def _generate_agent_id(trace_id: str, index: int) -> str:
    short = uuid.uuid4().hex[:8]
    return f"{trace_id}-a{index}-{short}"


def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    payload = json.dumps({"type": event_type, "data": data})
    return f"event: {event_type}\ndata: {payload}\n\n"


class WorkflowEngine:
    """Executes a multi-agent workflow by traversing agents in topological order."""

    def __init__(self, goal: str, workflow_type: str = "custom", config: dict | None = None, topology: list[dict] | None = None):
        self.goal = goal
        self.workflow_type = workflow_type or "custom"
        self.config = config or {}
        self.trace_id = _generate_trace_id()

        import copy
        if topology is not None:
            # Use caller-provided topology (e.g. reconstructed from a trace)
            self.topology = copy.deepcopy(topology)
        else:
            if self.workflow_type not in WORKFLOW_TOPOLOGIES:
                print(f"[WorkflowEngine] WARNING: workflow_type '{self.workflow_type}' not found in topologies, falling back to 'custom'")
                self.workflow_type = "custom"
            # Deep copy topology so prompt overrides don't mutate the global
            self.topology = copy.deepcopy(WORKFLOW_TOPOLOGIES[self.workflow_type])

        # Apply per-agent prompt overrides from config
        prompt_overrides = self.config.get("prompt_overrides", {})
        if prompt_overrides:
            for agent_spec in self.topology:
                name = agent_spec["name"]
                # Exact match first
                if name in prompt_overrides:
                    agent_spec["system_prompt"] = prompt_overrides[name]
                else:
                    # Fuzzy: try stripping or adding "Agent" suffix
                    stripped = name[:-len("Agent")] if name.endswith("Agent") else name
                    suffixed = name + "Agent" if not name.endswith("Agent") else name
                    for variant in [stripped, suffixed]:
                        if variant in prompt_overrides:
                            agent_spec["system_prompt"] = prompt_overrides[variant]
                            break

    def _build_levels(self) -> list[list[int]]:
        """Group agent indices into topological levels for parallel execution.

        Agents on the same level have all their parents in earlier levels,
        so they can run concurrently.
        """
        n = len(self.topology)
        completed: set[int] = set()
        levels: list[list[int]] = []
        remaining = set(range(n))

        while remaining:
            level = [
                i for i in sorted(remaining)
                if all(p in completed for p in self.topology[i]["parent_indices"])
            ]
            if not level:
                # Should never happen with a valid DAG, break to avoid infinite loop
                level = list(sorted(remaining))
            levels.append(level)
            completed.update(level)
            remaining -= set(level)

        return levels

    @staticmethod
    def _run_single_agent(
        agent_spec: dict,
        agent_input: str,
        model: str,
    ) -> dict:
        """Execute a single agent LLM call (blocking). Used inside threads."""
        from litellm import completion

        messages = [
            {"role": "system", "content": agent_spec["system_prompt"]},
            {"role": "user", "content": agent_input},
        ]

        t_start = time.time()
        try:
            response = completion(model=model, messages=messages)
            t_end = time.time()

            output_text = response.choices[0].message.content or ""
            usage = response.usage
            ttft = response._response_ms if hasattr(response, '_response_ms') else int((t_end - t_start) * 200)
            latency = t_end - t_start
            tokens_in = usage.prompt_tokens if usage else 0
            tokens_out = usage.completion_tokens if usage else 0
        except Exception as e:
            t_end = time.time()
            output_text = f"[Agent error: {e}]"
            ttft = 0
            latency = t_end - t_start
            tokens_in = 0
            tokens_out = 0

        return {
            "output": output_text,
            "ttft": ttft,
            "latency": latency,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    def execute_streaming(self) -> Generator[str, None, None]:
        """Execute workflow with SSE streaming.

        Agents at the same topological level run in parallel using threads.
        Yields SSE-formatted events:
          - agent_start: when an agent begins execution
          - agent_done: when an agent finishes
          - level_start: when a parallel level begins
          - complete: final result with full journey data
          - error: on failure
        """
        agent_ids: list[str] = [
            _generate_agent_id(self.trace_id, idx)
            for idx in range(len(self.topology))
        ]
        agent_outputs: dict[int, str] = {}
        samples: list[dict] = [None] * len(self.topology)  # type: ignore[list-item]
        levels = self._build_levels()

        total_agents = len(self.topology)
        completed_count = 0

        for level_num, level_indices in enumerate(levels):
            level_names = [self.topology[i]["name"] for i in level_indices]
            parallel = len(level_indices) > 1

            yield _sse_event("level_start", {
                "level": level_num + 1,
                "total_levels": len(levels),
                "agents": level_names,
                "parallel": parallel,
            })

            # Signal start for each agent in the level
            for idx in level_indices:
                yield _sse_event("agent_start", {
                    "name": self.topology[idx]["name"],
                    "role": self.topology[idx]["role"],
                    "index": idx,
                    "parallel": parallel,
                    "level_peers": level_names if parallel else [],
                    "completed": completed_count,
                    "total": total_agents,
                })

            # Prepare inputs for this level
            level_tasks: list[tuple[int, dict, str, str]] = []
            for idx in level_indices:
                agent_spec = self.topology[idx]
                if not agent_spec["parent_indices"]:
                    agent_input = self.goal
                else:
                    parent_texts = [
                        agent_outputs[pi]
                        for pi in agent_spec["parent_indices"]
                        if pi in agent_outputs
                    ]
                    agent_input = "\n\n---\n\n".join(parent_texts)
                model = self.config.get("model", agent_spec["model"])
                level_tasks.append((idx, agent_spec, agent_input, model))

            # Execute agents in parallel using ThreadPoolExecutor
            if parallel:
                with ThreadPoolExecutor(max_workers=len(level_tasks)) as pool:
                    futures = {
                        pool.submit(
                            self._run_single_agent, spec, inp, mdl
                        ): idx
                        for idx, spec, inp, mdl in level_tasks
                    }
                    for future in as_completed(futures):
                        agent_idx = futures[future]
                        result = future.result()
                        agent_outputs[agent_idx] = result["output"]
                        samples[agent_idx] = self._build_sample(
                            agent_idx, agent_ids, agent_outputs, result
                        )
                        completed_count += 1
                        yield _sse_event("agent_done", {
                            "name": self.topology[agent_idx]["name"],
                            "index": agent_idx,
                            "latency": round(result["latency"], 2),
                            "completed": completed_count,
                            "total": total_agents,
                        })
            else:
                # Single agent — run directly
                for idx, spec, inp, mdl in level_tasks:
                    result = self._run_single_agent(spec, inp, mdl)
                    agent_outputs[idx] = result["output"]
                    samples[idx] = self._build_sample(
                        idx, agent_ids, agent_outputs, result
                    )
                    completed_count += 1
                    yield _sse_event("agent_done", {
                        "name": self.topology[idx]["name"],
                        "index": idx,
                        "latency": round(result["latency"], 2),
                        "completed": completed_count,
                        "total": total_agents,
                    })

        # Build final journey
        journey_name = f"{self.workflow_type.replace('_', ' ').title()}: {self.goal[:60]}"
        journey = {
            "trace_id": self.trace_id,
            "journey_name": journey_name,
            "samples": [s for s in samples if s is not None],
        }

        yield _sse_event("complete", {
            "trace_id": self.trace_id,
            "journey_name": journey_name,
            "journey": journey,
        })

    def _build_sample(
        self, idx: int, agent_ids: list[str],
        agent_outputs: dict[int, str], result: dict
    ) -> dict:
        """Build a single agent sample dict in Keywords AI format."""
        agent_spec = self.topology[idx]
        parent_id_list = [
            agent_ids[pi] for pi in agent_spec["parent_indices"]
            if pi < len(agent_ids)
        ]

        tool_calls = [
            {"function_name": t, "arguments": {"query": result["output"][:100]}}
            for t in agent_spec.get("tools", [])
        ]

        sample: dict[str, Any] = {
            "agent_turn": idx + 1,
            "agent_name": agent_spec["name"],
            "agent_id": agent_ids[idx],
            "input": self.goal if not agent_spec["parent_indices"] else "\n\n---\n\n".join(
                agent_outputs[pi] for pi in agent_spec["parent_indices"] if pi in agent_outputs
            ),
            "output": result["output"],
            "core_payload": {
                "completion_message": result["output"],
                "tool_calls": tool_calls,
                "reasoning_blocks": [],
            },
            "model_parameters": {
                "model": self.config.get("model", agent_spec["model"]),
            },
            "telemetry": {
                "ttft": round(result["ttft"], 3) if isinstance(result["ttft"], float) else result["ttft"],
                "latency": round(result["latency"], 3),
                "tokens": {
                    "prompt_tokens": result["tokens_in"],
                    "completion_tokens": result["tokens_out"],
                },
            },
            "metadata": {
                "workflow_type": self.workflow_type,
                "role": agent_spec["role"],
            },
            "evaluation_signals": {},
        }

        if not parent_id_list:
            pass  # No parent fields
        elif len(parent_id_list) == 1:
            sample["parent_id"] = parent_id_list[0]
        else:
            sample["parent_ids"] = parent_id_list

        return sample

    def execute(self) -> dict:
        """Synchronous execute (legacy). Runs all levels sequentially via streaming."""
        final_result = None
        for event_str in self.execute_streaming():
            # Parse the SSE data line to check for 'complete'
            for line in event_str.split("\n"):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "complete":
                            final_result = data.get("data", {})
                    except (json.JSONDecodeError, KeyError):
                        pass
        if final_result:
            return final_result
        # Fallback — should not happen
        return {"trace_id": self.trace_id, "journey_name": "", "journey": {"trace_id": self.trace_id, "samples": []}}
