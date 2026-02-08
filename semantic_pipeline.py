"""
Foresnsic Semantic Pipeline which is responsible for taking the logs from every agent and finding whether there are similar issues that could be resolved
So lets say there are 100 agents running in the background
If one agent is talking to another agents and towards the end we figure out the issue was one in agent 3 or 4 we can fix it.

1)Extracting Data from the dataset
2)Mapping them out figure out the agent archietecture(Linear, Swarm)

"""
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from dataclasses import dataclass
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "test" / "keywords_ai_agent_output_samples.json"

_agent_output_samples = None

def _load_agent_samples():
    """Lazy load agent samples data."""
    global _agent_output_samples
    if _agent_output_samples is None:
        if DATA_PATH.exists():
            with open(DATA_PATH) as f:
                _agent_output_samples = json.load(f)
        else:
            _agent_output_samples = {"journeys": []}
    return _agent_output_samples


def extract_agent_logs(journeys_data=None):
    """Extract all agent outputs as a flat list from the nested journey structure."""
    if journeys_data is None:
        agent_output_samples = _load_agent_samples()
    else:
        agent_output_samples = journeys_data if isinstance(journeys_data, dict) else {"journeys": journeys_data}

    agent_logs = []
    for journey in agent_output_samples.get("journeys", []):
        trace_id = journey["trace_id"]
        journey_name = journey["journey_name"]
        for sample in journey["samples"]:
            agent_logs.append({
                "trace_id": trace_id,
                "journey_name": journey_name,
                **sample,
            })
    return agent_logs


# Extracted flat list of all agent outputs (one per agent turn)
agent_logs = extract_agent_logs()


class AgentDAG:
    """
    DAG representing agent parent/child relationships within a journey.
    Built from agent logs using parent_id and parent_ids.
    """

    def __init__(self, trace_id: str, journey_name: str, nodes: dict, edges: list, parents_of: dict, children_of: dict):
        self.trace_id = trace_id
        self.journey_name = journey_name
        self._nodes = nodes
        self._edges = edges
        self._parents_of = parents_of
        self._children_of = children_of

    @classmethod
    def from_agent_logs(cls, agent_logs: list[dict]) -> dict[str, "AgentDAG"]:
        """Build one AgentDAG per trace_id from a flat list of agent logs."""
        by_trace: dict[str, list[dict]] = {}
        for log in agent_logs:
            tid = log["trace_id"]
            if tid not in by_trace:
                by_trace[tid] = []
            by_trace[tid].append(log)

        dags = {}
        for trace_id, logs in by_trace.items():
            dag = cls._build_single(trace_id, logs)
            dags[trace_id] = dag
        return dags

    @classmethod
    def _build_single(cls, trace_id: str, logs: list[dict]) -> "AgentDAG":
        nodes = {}
        edges = []
        parents_of = {}
        children_of = {}

        for log in logs:
            agent_id = log["agent_id"]
            nodes[agent_id] = log

            parents = []
            if log.get("parent_ids"):
                parents = list(log["parent_ids"])
            elif log.get("parent_id"):
                parents = [log["parent_id"]]

            parents_of[agent_id] = parents

            for parent_id in parents:
                edges.append((parent_id, agent_id))
                if parent_id not in children_of:
                    children_of[parent_id] = []
                children_of[parent_id].append(agent_id)

        return cls(
            trace_id=trace_id,
            journey_name=logs[0]["journey_name"] if logs else "",
            nodes=nodes,
            edges=edges,
            parents_of=parents_of,
            children_of=children_of,
        )

    def get_parents(self, agent_id: str) -> list[str]:
        """Return parent agent IDs for the given agent."""
        return self._parents_of.get(agent_id, [])

    def get_children(self, agent_id: str) -> list[str]:
        """Return child agent IDs for the given agent."""
        return self._children_of.get(agent_id, [])

    def get_node(self, agent_id: str) -> dict | None:
        """Return full agent data for the given agent_id."""
        return self._nodes.get(agent_id)

    def get_edges(self) -> list[tuple[str, str]]:
        """Return all edges as (parent_id, child_id) tuples."""
        return list(self._edges)

    def get_nodes(self) -> dict[str, dict]:
        """Return all nodes (agent_id -> agent_data)."""
        return dict(self._nodes)

    def roots(self) -> list[str]:
        """Return agent IDs with no parents (root nodes)."""
        return [aid for aid in self._nodes if not self._parents_of.get(aid)]

    def leaves(self) -> list[str]:
        """Return agent IDs with no children (leaf nodes)."""
        return [aid for aid in self._nodes if not self._children_of.get(aid)]

    def has_edge(self, parent_id: str, child_id: str) -> bool:
        """Check if an edge exists from parent to child."""
        return (parent_id, child_id) in self._edges

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)


# DAG per journey: trace_id -> AgentDAG
agent_dags = AgentDAG.from_agent_logs(agent_logs)


def _text_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts (0-1) using term frequencies."""
    if not text_a or not text_b:
        return 0.0
    from collections import Counter
    import math

    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if not tokens_a or not tokens_b:
        return 0.0

    vec_a = Counter(tokens_a)
    vec_b = Counter(tokens_b)
    all_tokens = set(vec_a) | set(vec_b)

    dot = sum(vec_a[t] * vec_b[t] for t in all_tokens)
    norm_a = math.sqrt(sum((vec_a[t] ** 2) for t in all_tokens))
    norm_b = math.sqrt(sum((vec_b[t] ** 2) for t in all_tokens))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingMatchAnalyzer:
    """
    Analyzes input/output embedding similarity across agent DAG edges.
    For each edge (parent -> child): similarity = cos_sim(embed(input), embed(output)).
    Marks edges: RED if similarity < 0.8 (below threshold).
    """

    STATUS_GREEN = "green"
    STATUS_YELLOW = "yellow"
    STATUS_RED = "red"

    def __init__(self, dag: AgentDAG, similarity_fn=None, min_similarity: float = 0.8):
        self.dag = dag
        self._sim_fn = similarity_fn or _text_similarity
        self._min_similarity = min_similarity
        self._edge_results: list[dict] = []
        self._avg: float | None = None
        self._stdev: float | None = None

    def analyze(self) -> list[dict]:
        """
        Compute diff for each edge, running avg, then final stdev.
        Returns list of {edge, parent_name, child_name, diff, running_avg, status}.
        """
        edges = self.dag.get_edges()
        if not edges:
            return []

        diffs: list[float] = []
        results: list[dict] = []
        running_sum = 0.0

        for parent_id, child_id in edges:
            child_node = self.dag.get_node(child_id)
            if not child_node:
                continue

            input_text = child_node.get("input") or ""
            output_text = child_node.get("output") or ""

            similarity = self._sim_fn(input_text, output_text)
            diff = 1.0 - similarity
            diff = max(0.0, min(1.0, diff))

            diffs.append(diff)
            running_sum += diff
            running_avg = running_sum / len(diffs)

            parent_node = self.dag.get_node(parent_id)
            parent_name = parent_node.get("agent_name", parent_id) if parent_node else parent_id
            child_name = child_node.get("agent_name", child_id)

            results.append({
                "parent_id": parent_id,
                "child_id": child_id,
                "parent_name": parent_name,
                "child_name": child_name,
                "edge": f"{parent_name} -> {child_name}",
                "similarity": similarity,
                "diff": diff,
                "running_avg": running_avg,
            })

        self._edge_results = results
        self._avg = sum(diffs) / len(diffs) if diffs else 0.0
        self._stdev = (sum((d - self._avg) ** 2 for d in diffs) / len(diffs)) ** 0.5 if len(diffs) > 1 else 0.0

        # Apply status: RED if similarity < min_similarity, else GREEN
        for r in results:
            sim = r["similarity"]
            r["status"] = self.STATUS_RED if sim < self._min_similarity else self.STATUS_GREEN

        return results

    @property
    def avg(self) -> float:
        if self._avg is None:
            self.analyze()
        return self._avg or 0.0

    @property
    def stdev(self) -> float:
        if self._stdev is None:
            self.analyze()
        return self._stdev or 0.0

    def get_results(self) -> list[dict]:
        if not self._edge_results:
            self.analyze()
        return list(self._edge_results)

    def print_red_agents(self) -> None:
        """Print red agents with full context: agent, input, output, parent, and related data."""
        results = self.get_results()
        reds = [r for r in results if r["status"] == self.STATUS_RED]
        if not reds:
            print(f"No red agents for {self.dag.journey_name}.")
            return
        print(f"RED agents for {self.dag.journey_name} (similarity < {self._min_similarity}):")
        for r in reds:
            child_node = self.dag.get_node(r["child_id"])
            parent_node = self.dag.get_node(r["parent_id"])
            print(f"\n--- {r['edge']} (similarity={r['similarity']:.3f}) ---")
            print(f"  Agent: {r['child_name']} (id: {r['child_id']})")
            print(f"  Parent: {r['parent_name']} (id: {r['parent_id']})")
            print(f"  Input: {_truncate(child_node.get('input', ''), 200)}")
            print(f"  Output: {_truncate(child_node.get('output', ''), 200)}")
            if child_node:
                meta = child_node.get("metadata", {})
                if meta:
                    print(f"  Metadata: {meta}")
                tool_calls = child_node.get("core_payload", {}).get("tool_calls", [])
                if tool_calls:
                    print(f"  Tool calls: {len(tool_calls)}")
                    for tc in tool_calls[:3]:
                        print(f"    - {tc.get('function_name', '?')}({tc.get('arguments', {})})")
                parent_ids = child_node.get("parent_ids")
                if parent_ids:
                    print(f"  Parent IDs: {parent_ids}")
        print()


def _truncate(s: str, max_len: int) -> str:
    if not s:
        return "(empty)"
    return s[:max_len] + "..." if len(s) > max_len else s


def _call_llm(prompt: str, model: str = "gpt-4o") -> str:
    """
    Call LLM via LiteLLM. Set OPENAI_API_KEY or use model='ollama/llama2' for local.
    Falls back to mock when unavailable.
    """
    try:
        from litellm import completion
        response = completion(model=model, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content or ""
    except ImportError:
        return "[LiteLLM not installed - pip install litellm. Mock: on_track=unknown]"
    except Exception as e:
        return f"[LLM error: {e}. Set OPENAI_API_KEY or use model='ollama/llama2' for local.]"


class HallucinationDetector:
    """
    Second-pass LLM verification for red-flagged edges.
    Checks whether agent output aligns with the user's original goal,
    reducing false positives from pure similarity-based detection.
    """

    STATUS_VERIFIED_RED = "verified_red"
    STATUS_CLEARED = "cleared"
    STATUS_UNVERIFIED = "unverified"

    def __init__(self, dag: AgentDAG, user_goal: str, red_edges: list[dict], model: str = "gpt-4o"):
        self.dag = dag
        self.user_goal = user_goal
        self.red_edges = red_edges
        self.model = model
        self._results: list[dict] = []

    def _build_prompt(self, child_node: dict, parent_node: dict | None) -> str:
        agent_name = child_node.get("agent_name", "Unknown Agent")
        agent_output = _truncate(child_node.get("output", ""), 500)
        agent_input = _truncate(child_node.get("input", ""), 500)
        parent_name = parent_node.get("agent_name", "Unknown") if parent_node else "Root"

        tool_calls = child_node.get("core_payload", {}).get("tool_calls", [])
        tool_context = ""
        if tool_calls:
            tool_names = [tc.get("function_name", "?") for tc in tool_calls[:5]]
            tool_context = f"\nTools called by this agent: {', '.join(tool_names)}"

        return f"""You are an LLM observability analyst checking for hallucination in a multi-agent system.

User's original goal: {self.user_goal}

Agent being checked: {agent_name}
Parent agent: {parent_name}
Input received: {agent_input}
Output produced: {agent_output}{tool_context}

This agent was flagged because its output differs significantly from its input.
However, low similarity does NOT always mean hallucination — the agent may have correctly transformed the input or fulfilled a specialized subtask.

Question: Is this agent's output aligned with the user's goal, or is it hallucinating / producing irrelevant content?

Answer with exactly one of:
- "ALIGNED" if the output serves the user's goal (even if it looks different from the input)
- "HALLUCINATING" if the output is irrelevant, fabricated, or contradicts the user's goal
- "UNCLEAR" if you cannot determine alignment

Follow your answer with a brief one-sentence explanation."""

    def _parse_verdict(self, response: str) -> tuple[str, str]:
        if not response or response.startswith("["):
            return self.STATUS_UNVERIFIED, response or "No response"

        import re
        cleaned = response.strip().strip('"').strip("'")
        first_line = cleaned.split("\n")[0]
        first_upper = first_line.upper()

        def _extract_explanation(text: str, keyword: str) -> str:
            """Extract explanation after the keyword from the response."""
            # Try multi-line: explanation is on second line
            if "\n" in text:
                return text.split("\n", 1)[1].strip()
            # Single-line: strip keyword and surrounding punctuation
            idx = text.upper().find(keyword)
            if idx >= 0:
                after = text[idx + len(keyword):].strip().lstrip('"').lstrip("'").lstrip(':').lstrip(',').lstrip('.').lstrip(' —-').strip()
                if after:
                    return after
            return ""

        if "ALIGNED" in first_upper and "HALLUCINATING" not in first_upper:
            explanation = _extract_explanation(cleaned, "ALIGNED") or "Output aligns with user goal"
            return self.STATUS_CLEARED, explanation
        elif "HALLUCINATING" in first_upper:
            explanation = _extract_explanation(cleaned, "HALLUCINATING") or "Output does not align with user goal"
            return self.STATUS_VERIFIED_RED, explanation
        else:
            return self.STATUS_UNVERIFIED, cleaned

    def verify(self) -> list[dict]:
        """
        Run LLM verification on each red edge.
        Returns list of dicts with original edge data plus verification fields.
        """
        results = []
        for edge_result in self.red_edges:
            child_id = edge_result["child_id"]
            parent_id = edge_result["parent_id"]
            child_node = self.dag.get_node(child_id)
            parent_node = self.dag.get_node(parent_id)

            if not child_node:
                results.append({
                    **edge_result,
                    "verification_status": self.STATUS_UNVERIFIED,
                    "verification_explanation": "Child node not found",
                    "final_status": "red",
                })
                continue

            prompt = self._build_prompt(child_node, parent_node)
            response = _call_llm(prompt, self.model)
            status, explanation = self._parse_verdict(response)

            final_status = "green" if status == self.STATUS_CLEARED else "red"

            results.append({
                **edge_result,
                "verification_status": status,
                "verification_explanation": explanation,
                "final_status": final_status,
            })

        self._results = results
        return results

    @property
    def cleared_count(self) -> int:
        return sum(1 for r in self._results if r["verification_status"] == self.STATUS_CLEARED)

    @property
    def confirmed_count(self) -> int:
        return sum(1 for r in self._results if r["verification_status"] == self.STATUS_VERIFIED_RED)


class TaskProgressMonitor:
    """
    Takes user goal, walks the agent DAG, and every N steps summarizes progress
    and checks with an LLM whether agents are performing the task.
    """

    def __init__(self, dag: AgentDAG, user_goal: str, checkpoint_interval: int = 5, model: str = "gpt-4o"):
        self.dag = dag
        self.user_goal = user_goal
        self.checkpoint_interval = checkpoint_interval
        self.model = model

    def _get_ordered_steps(self) -> list[tuple[str, str, dict]]:
        """Return steps as (parent_id, child_id, child_data) in execution order."""
        edges = self.dag.get_edges()
        steps = []
        for parent_id, child_id in edges:
            child_node = self.dag.get_node(child_id)
            if child_node:
                steps.append((parent_id, child_id, child_node))
        return steps

    def _summarize_steps(self, steps: list[tuple[str, str, dict]]) -> str:
        """Build a short summary of agent inputs/outputs for the given steps."""
        lines = []
        for i, (parent_id, child_id, child_node) in enumerate(steps, 1):
            parent_node = self.dag.get_node(parent_id)
            parent_name = parent_node.get("agent_name", parent_id) if parent_node else parent_id
            child_name = child_node.get("agent_name", child_id)
            inp = _truncate(child_node.get("input", ""), 100)
            out = _truncate(child_node.get("output", ""), 100)
            lines.append(f"Step {i}: {parent_name} -> {child_name}")
            lines.append(f"  Input: {inp}")
            lines.append(f"  Output: {out}")
        return "\n".join(lines)

    def _is_deviation(self, verdict: str) -> bool:
        """Return True if LLM verdict indicates agents are deviating from user goal."""
        v = verdict.lower().strip()
        if v.startswith("[") or "error" in v:
            return False
        return v.startswith("no") or "not on track" in v or "deviating" in v or "off track" in v

    def check_progress(self) -> tuple[list[dict], list[dict]]:
        """
        Walk steps, every checkpoint_interval steps summarize cumulative progress
        and call LLM to verify agents are on-track with user goal.
        Returns (checkpoints, deviations) where deviations = [{step_count, agent_ids, agent_names, verdict, summary}, ...]
        """
        steps = self._get_ordered_steps()
        if not steps:
            return [], []

        checkpoints = []
        deviations = []

        def _do_checkpoint(chunk: list, step_count: int) -> None:
            summary = self._summarize_steps(chunk)
            prompt = f"""User task/goal: {self.user_goal}

Agent progress so far ({step_count} steps):
{summary}

Are the agents performing the user's task correctly? Answer in one short sentence: either "Yes, on track" or "No, [brief reason]."
"""
            verdict = _call_llm(prompt, self.model)
            cp = {
                "step_count": step_count,
                "summary": summary,
                "llm_verdict": verdict.strip(),
                "steps": chunk,
            }
            checkpoints.append(cp)

            if self._is_deviation(verdict):
                agent_ids = [child_id for _, child_id, _ in chunk]
                agent_names = []
                for _, child_id, child_node in chunk:
                    agent_names.append(child_node.get("agent_name", child_id))
                deviations.append({
                    "step_count": step_count,
                    "agent_ids": agent_ids,
                    "agent_names": agent_names,
                    "verdict": verdict.strip(),
                    "summary": summary,
                })

        for i in range(self.checkpoint_interval - 1, len(steps), self.checkpoint_interval):
            step_count = i + 1
            chunk = steps[: step_count]
            _do_checkpoint(chunk, step_count)

        if len(steps) % self.checkpoint_interval != 0 and (not checkpoints or checkpoints[-1]["step_count"] != len(steps)):
            _do_checkpoint(steps, len(steps))

        return checkpoints, deviations

    def run_and_print(self) -> None:
        """Run checkpoints, print results, and summarize deviations with culpable agents."""
        print(f"User goal: {self.user_goal}")
        print(f"Journey: {self.dag.journey_name}")
        print(f"Checkpoint every {self.checkpoint_interval} steps\n")
        checkpoints, deviations = self.check_progress()
        for cp in checkpoints:
            print(f"--- Checkpoint at step {cp['step_count']} ---")
            print(cp["summary"])
            print(f"LLM verdict: {cp['llm_verdict']}\n")

        if deviations:
            print("=" * 60)
            print("DEVIATION SUMMARY – Agents identified as cause:")
            print("=" * 60)
            seen = set()
            agent_list = []
            for d in deviations:
                print(f"\nAt step {d['step_count']}: {d['verdict']}")
                for aid, aname in zip(d["agent_ids"], d["agent_names"]):
                    if aid not in seen:
                        seen.add(aid)
                        agent_list.append(f"{aname} ({aid})")
                        print(f"  - {aname} ({aid})")
            print(f"\nRoot cause agents: {', '.join(agent_list)}")
        else:
            print("\nNo deviations detected – agents stayed on track.")


class AgentLog:
    pass


if __name__ == "__main__":
    import sys
    agent_logs = extract_agent_logs()
    dags = AgentDAG.from_agent_logs(agent_logs)
    user_goal = sys.argv[1] if len(sys.argv) > 1 else "Book a flight from London to Tokyo for 2 adults."
    trace_id = sys.argv[2] if len(sys.argv) > 2 else "tr-a7f2-9b3c-4e1d-8f6a"
    dag = dags.get(trace_id) or list(dags.values())[0]
    monitor = TaskProgressMonitor(dag, user_goal, checkpoint_interval=5)
    monitor.run_and_print()

