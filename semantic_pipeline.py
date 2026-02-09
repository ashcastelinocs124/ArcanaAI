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


class SentinelAnalyzer:
    """
    Tier 1: Zero-LLM regex-based checks for obvious agent output failures.
    Catches empty outputs, placeholders, malformed JSON, error patterns, etc.
    Score = (checks passed / total checks). Agents scoring < 0.5 escalate to Tier 2.
    """

    MAX_OUTPUT_LENGTH = 10000
    MIN_OUTPUT_LENGTH = 5
    MIN_INPUT_FOR_SHORT_CHECK = 20
    REPEAT_BLOCK_MIN = 50
    REPEAT_THRESHOLD = 0.6

    def __init__(self, dag: AgentDAG):
        self.dag = dag

    def _check_empty_output(self, node: dict) -> dict:
        output = node.get("output") or ""
        passed = bool(output and output.strip())
        return {"name": "empty_output", "passed": passed, "detail": "Output is empty or whitespace-only" if not passed else "Output is non-empty"}

    def _check_placeholder_text(self, node: dict) -> dict:
        import re
        output = node.get("output") or ""
        pattern = r'\b(TODO|FIXME|lorem ipsum|placeholder|TBD|\[FILL.?IN\]|insert .+ here)\b'
        match = re.search(pattern, output, re.IGNORECASE)
        passed = match is None
        return {"name": "placeholder_text", "passed": passed, "detail": f"Placeholder detected: '{match.group()}'" if match else "No placeholder text found"}

    def _check_excessive_length(self, node: dict) -> dict:
        output = node.get("output") or ""
        passed = len(output) <= self.MAX_OUTPUT_LENGTH
        return {"name": "excessive_length", "passed": passed, "detail": f"Output length {len(output)} exceeds {self.MAX_OUTPUT_LENGTH}" if not passed else f"Output length {len(output)} is acceptable"}

    def _check_too_short(self, node: dict) -> dict:
        output = node.get("output") or ""
        input_text = node.get("input") or ""
        if len(input_text) <= self.MIN_INPUT_FOR_SHORT_CHECK:
            return {"name": "too_short", "passed": True, "detail": "Input too short to require substantial output"}
        passed = len(output.strip()) >= self.MIN_OUTPUT_LENGTH
        return {"name": "too_short", "passed": passed, "detail": f"Output only {len(output.strip())} chars for substantial input" if not passed else "Output length is sufficient"}

    def _check_repeated_blocks(self, node: dict) -> dict:
        output = node.get("output") or ""
        if len(output) < self.REPEAT_BLOCK_MIN * 2:
            return {"name": "repeated_blocks", "passed": True, "detail": "Output too short for repetition check"}
        block_size = self.REPEAT_BLOCK_MIN
        blocks = [output[i:i+block_size] for i in range(0, len(output) - block_size + 1, block_size)]
        if not blocks:
            return {"name": "repeated_blocks", "passed": True, "detail": "No blocks to check"}
        from collections import Counter
        counts = Counter(blocks)
        most_common_count = counts.most_common(1)[0][1]
        repeat_ratio = (most_common_count * block_size) / len(output) if len(output) > 0 else 0
        passed = repeat_ratio < self.REPEAT_THRESHOLD
        return {"name": "repeated_blocks", "passed": passed, "detail": f"Repeated content covers {repeat_ratio:.0%} of output" if not passed else "No excessive repetition detected"}

    def _check_malformed_json(self, node: dict) -> dict:
        output = (node.get("output", "") or "").strip()
        if not output or (not output.startswith("{") and not output.startswith("[")):
            return {"name": "malformed_json", "passed": True, "detail": "Output is not JSON-like"}
        try:
            json.loads(output)
            return {"name": "malformed_json", "passed": True, "detail": "Valid JSON output"}
        except (json.JSONDecodeError, ValueError):
            return {"name": "malformed_json", "passed": False, "detail": "Output starts with {/[ but is not valid JSON"}

    def _check_error_patterns(self, node: dict) -> dict:
        import re
        output = node.get("output") or ""
        patterns = [
            r'\bError:\s',
            r'\bTraceback\b',
            r'\b500\s+Internal\b',
            r'\brate\s+limit\b',
            r'\bException\b',
            r'\bFailed to\b',
            r'\bconnection refused\b',
        ]
        for p in patterns:
            match = re.search(p, output, re.IGNORECASE)
            if match:
                return {"name": "error_patterns", "passed": False, "detail": f"Error pattern detected: '{match.group()}'"}
        return {"name": "error_patterns", "passed": True, "detail": "No error patterns found"}

    def analyze(self) -> dict[str, dict]:
        nodes = self.dag.get_nodes()
        results = {}
        checks_fns = [
            self._check_empty_output,
            self._check_placeholder_text,
            self._check_excessive_length,
            self._check_too_short,
            self._check_repeated_blocks,
            self._check_malformed_json,
            self._check_error_patterns,
        ]
        for agent_id, node in nodes.items():
            checks = [fn(node) for fn in checks_fns]
            passed_count = sum(1 for c in checks if c["passed"])
            score = passed_count / len(checks) if checks else 1.0
            fail_reasons = [c["detail"] for c in checks if not c["passed"]]
            results[agent_id] = {
                "status": "sentinel_pass" if score >= 0.5 else "sentinel_fail",
                "score": round(score, 3),
                "checks": checks,
                "fail_reasons": fail_reasons,
            }
        return results


class GlobalRulesJudge:
    """
    Tier 2: Single LLM call generates global rules, then applies them deterministically.
    Only evaluates agents escalated from Tier 1.
    """

    def __init__(self, dag: AgentDAG, user_goal: str, model: str = "gpt-4o-mini", score_threshold: float = 0.6):
        self.dag = dag
        self.user_goal = user_goal
        self.model = model
        self.score_threshold = score_threshold
        self._rules: list[dict] | None = None

    def _build_context_prompt(self) -> str:
        nodes = self.dag.get_nodes()
        agent_summaries = []
        for agent_id, node in nodes.items():
            name = node.get("agent_name", agent_id)
            inp = _truncate(node.get("input", ""), 200)
            out = _truncate(node.get("output", ""), 200)
            tools = node.get("core_payload", {}).get("tool_calls", [])
            tool_names = [tc.get("function_name", "?") for tc in tools[:3]]
            tool_str = f", tools: {', '.join(tool_names)}" if tool_names else ""
            agent_summaries.append(f"- {name} (id: {agent_id}): input='{inp}', output='{out}'{tool_str}")

        return f"""You are analyzing a multi-agent system. Given the user's goal and agent information below, generate 5-8 quality rules that all agents should follow.

User goal: {self.user_goal}

Agents in the system:
{chr(10).join(agent_summaries)}

Generate rules as a JSON array. Each rule must have:
- "description": what the rule checks
- "check_type": one of "output_contains", "output_not_contains", "output_relevance", "output_length", "tone_check"
- "parameters": dict with type-specific params (e.g. "terms" for contains, "min_words" for length, "negative_terms" for tone_check)

Example:
[
  {{"description": "Output should reference the user's destination", "check_type": "output_relevance", "parameters": {{"terms": ["flight", "booking", "travel"]}}}},
  {{"description": "Output should not contain error messages", "check_type": "output_not_contains", "parameters": {{"terms": ["error", "failed", "exception"]}}}}
]

Return ONLY the JSON array, no other text."""

    def _generate_rules(self) -> list[dict]:
        if self._rules is not None:
            return self._rules
        import re
        prompt = self._build_context_prompt()
        response = _call_llm(prompt, self.model)
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                self._rules = json.loads(match.group())
            else:
                self._rules = []
        except (json.JSONDecodeError, ValueError):
            self._rules = []
        return self._rules

    def _evaluate_agent_against_rules(self, node: dict, rules: list[dict]) -> dict:
        output = (node.get("output", "") or "").lower()
        output_words = output.split()
        passed = 0
        total = len(rules)
        rule_results = []

        for rule in rules:
            check_type = rule.get("check_type", "")
            params = rule.get("parameters", {})
            result = {"rule": rule.get("description", ""), "check_type": check_type, "passed": False}

            if check_type == "output_contains":
                terms = params.get("terms", [])
                found = any(t.lower() in output for t in terms)
                result["passed"] = found
            elif check_type == "output_not_contains":
                terms = params.get("terms", [])
                found_bad = any(t.lower() in output for t in terms)
                result["passed"] = not found_bad
            elif check_type == "output_relevance":
                terms = params.get("terms", [])
                if terms:
                    overlap = sum(1 for t in terms if t.lower() in output)
                    result["passed"] = (overlap / len(terms)) >= 0.3
                else:
                    result["passed"] = True
            elif check_type == "output_length":
                min_words = params.get("min_words", 5)
                result["passed"] = len(output_words) >= min_words
            elif check_type == "tone_check":
                negative_terms = params.get("negative_terms", [])
                found_neg = any(t.lower() in output for t in negative_terms)
                result["passed"] = not found_neg
            else:
                result["passed"] = True

            if result["passed"]:
                passed += 1
            rule_results.append(result)

        score = passed / total if total > 0 else 1.0
        return {
            "status": "rules_pass" if score >= self.score_threshold else "rules_fail",
            "score": round(score, 3),
            "rule_results": rule_results,
            "fail_reasons": [r["rule"] for r in rule_results if not r["passed"]],
        }

    def analyze(self, exclude_agents: set[str] | None = None) -> dict[str, dict]:
        rules = self._generate_rules()
        nodes = self.dag.get_nodes()
        results = {}
        exclude = exclude_agents or set()
        for agent_id, node in nodes.items():
            if agent_id in exclude:
                continue
            if not rules:
                results[agent_id] = {"status": "rules_pass", "score": 1.0, "rule_results": [], "fail_reasons": []}
                continue
            results[agent_id] = self._evaluate_agent_against_rules(node, rules)
        return results


class SpecialistEvaluator:
    """
    Tier 3: Per-agent LLM evaluation for deep quality assessment.
    Only runs on agents escalated from Tier 2.
    """

    def __init__(self, dag: AgentDAG, user_goal: str, model: str = "gpt-4o", score_threshold: float = 0.7):
        self.dag = dag
        self.user_goal = user_goal
        self.model = model
        self.score_threshold = score_threshold

    def _build_prompt(self, agent_id: str, node: dict) -> str:
        name = node.get("agent_name", agent_id)
        inp = _truncate(node.get("input", ""), 400)
        out = _truncate(node.get("output", ""), 800)
        tools = node.get("core_payload", {}).get("tool_calls", [])
        tool_str = ""
        if tools:
            tool_names = [f"{tc.get('function_name', '?')}({json.dumps(tc.get('arguments', {}))[:100]})" for tc in tools[:5]]
            tool_str = f"\nTool calls: {', '.join(tool_names)}"

        parent_context = ""
        parent_ids = self.dag.get_parents(agent_id)
        if parent_ids:
            parent_outputs = []
            for pid in parent_ids[:3]:
                pnode = self.dag.get_node(pid)
                if pnode:
                    pname = pnode.get("agent_name", pid)
                    pout = _truncate(pnode.get("output", ""), 200)
                    parent_outputs.append(f"  - {pname}: {pout}")
            if parent_outputs:
                parent_context = f"\nParent agent outputs:\n{chr(10).join(parent_outputs)}"

        return f"""You are an expert evaluator for a multi-agent system. Analyze this agent's performance.

User's goal: {self.user_goal}

Agent: {name}
Input: {inp}
Output: {out}{tool_str}{parent_context}

Evaluate this agent on:
1. Role fulfillment - Does it perform its designated role correctly?
2. Output quality - Is the output well-formed, complete, and useful?
3. Goal alignment - Does it contribute to the user's overall goal?

Return a JSON object with:
{{
  "score": <float 0.0-1.0>,
  "role_assessment": "<brief assessment>",
  "quality_assessment": "<brief assessment>",
  "goal_alignment": "<brief assessment>",
  "verdict": "PASS" or "FAIL",
  "explanation": "<one sentence summary>"
}}

Return ONLY the JSON object."""

    def _parse_response(self, response: str) -> dict:
        import re
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return {
                    "score": float(data.get("score", 0.5)),
                    "role_assessment": data.get("role_assessment", ""),
                    "quality_assessment": data.get("quality_assessment", ""),
                    "goal_alignment": data.get("goal_alignment", ""),
                    "verdict": data.get("verdict", "UNCLEAR"),
                    "explanation": data.get("explanation", ""),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return {
            "score": 0.5,
            "role_assessment": "Could not parse evaluation",
            "quality_assessment": "Could not parse evaluation",
            "goal_alignment": "Could not parse evaluation",
            "verdict": "UNCLEAR",
            "explanation": response[:200] if response else "No response",
        }

    def analyze(self, exclude_agents: set[str] | None = None) -> dict[str, dict]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        nodes = self.dag.get_nodes()
        results = {}
        exclude = exclude_agents or set()

        # Build prompts for all agents up front
        tasks = {}
        for agent_id, node in nodes.items():
            if agent_id in exclude:
                continue
            tasks[agent_id] = self._build_prompt(agent_id, node)

        # Run all LLM calls in parallel
        def _eval_agent(agent_id: str, prompt: str) -> tuple[str, str]:
            return agent_id, _call_llm(prompt, self.model)

        with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as pool:
            futures = {
                pool.submit(_eval_agent, aid, prompt): aid
                for aid, prompt in tasks.items()
            }
            for future in as_completed(futures):
                agent_id, response = future.result()
                parsed = self._parse_response(response)
                status = "specialist_pass" if parsed["score"] >= self.score_threshold else "specialist_fail"
                results[agent_id] = {
                    "status": status,
                    **parsed,
                    "fail_reasons": [parsed["explanation"]] if status == "specialist_fail" else [],
                }
        return results


class CascadeEvaluator:
    """
    Orchestrates the 3-tier cascade: Sentinel -> Rules Judge -> Specialist.
    ALL agents run through ALL 3 tiers. An agent only passes (green) if it
    clears every tier. If it fails any tier, it is marked with that tier's failure.
    """

    def __init__(self, dag: AgentDAG, user_goal: str, tier2_model: str = "gpt-4o-mini", tier3_model: str = "gpt-4o"):
        self.dag = dag
        self.user_goal = user_goal
        self.tier2_model = tier2_model
        self.tier3_model = tier3_model

    def evaluate(self) -> dict:
        all_agents = set(self.dag.get_nodes().keys())

        # Tier 1: Sentinel (no LLM) — run on ALL agents
        sentinel = SentinelAnalyzer(self.dag)
        t1_results = sentinel.analyze()

        tier1_passed = set()
        tier1_failed = set()
        for aid, res in t1_results.items():
            if res["score"] >= 0.5:
                tier1_passed.add(aid)
            else:
                tier1_failed.add(aid)

        # Tier 2: Rules Judge (1 LLM call for rules) — run on ALL agents
        judge = GlobalRulesJudge(self.dag, self.user_goal, model=self.tier2_model)
        t2_results = judge.analyze()
        rules_generated = len(judge._rules or [])

        tier2_passed = set()
        tier2_failed = set()
        for aid, res in t2_results.items():
            if res["score"] >= 0.6:
                tier2_passed.add(aid)
            else:
                tier2_failed.add(aid)

        # Tier 3: Specialist (1 LLM call per agent) — run on ALL agents
        specialist = SpecialistEvaluator(self.dag, self.user_goal, model=self.tier3_model)
        t3_results = specialist.analyze()

        tier3_passed = set()
        tier3_failed = set()
        for aid, res in t3_results.items():
            if res["status"] == "specialist_pass":
                tier3_passed.add(aid)
            else:
                tier3_failed.add(aid)

        # Build verdicts: pass ONLY if all 3 tiers pass
        agent_verdicts = {}
        passed_all = set()
        for aid in all_agents:
            t1 = t1_results.get(aid, {})
            t2 = t2_results.get(aid, {})
            t3 = t3_results.get(aid, {})

            all_pass = aid in tier1_passed and aid in tier2_passed and aid in tier3_passed

            if all_pass:
                passed_all.add(aid)
                agent_verdicts[aid] = {
                    "evaluated_by_tier": 3, "tier_name": "All Tiers",
                    "status": "pass",
                    "score": t3.get("score", 1.0),
                    "fail_reasons": [],
                    "details": {
                        "t1_score": t1.get("score", 0),
                        "t2_score": t2.get("score", 0),
                        "t3_score": t3.get("score", 0),
                        "role_assessment": t3.get("role_assessment", ""),
                        "quality_assessment": t3.get("quality_assessment", ""),
                        "goal_alignment": t3.get("goal_alignment", ""),
                        "explanation": t3.get("explanation", ""),
                    },
                }
            else:
                # Find the FIRST tier that failed — that's the blocking tier
                if aid in tier1_failed:
                    failed_tier = 1
                    tier_name = "Sentinel"
                    score = t1.get("score", 0)
                    fail_reasons = t1.get("fail_reasons", [])
                    details = {"checks": t1.get("checks", [])}
                elif aid in tier2_failed:
                    failed_tier = 2
                    tier_name = "Rules Judge"
                    score = t2.get("score", 0)
                    fail_reasons = t2.get("fail_reasons", [])
                    details = {"rule_results": t2.get("rule_results", [])}
                else:
                    failed_tier = 3
                    tier_name = "Specialist"
                    score = t3.get("score", 0)
                    fail_reasons = t3.get("fail_reasons", [])
                    details = {
                        "role_assessment": t3.get("role_assessment", ""),
                        "quality_assessment": t3.get("quality_assessment", ""),
                        "goal_alignment": t3.get("goal_alignment", ""),
                        "explanation": t3.get("explanation", ""),
                    }

                agent_verdicts[aid] = {
                    "evaluated_by_tier": failed_tier, "tier_name": tier_name,
                    "status": "fail",
                    "score": score,
                    "fail_reasons": fail_reasons,
                    "details": details,
                }

        llm_calls = 1 + len(t3_results)  # 1 for T2 rules + 1 per agent for T3

        return {
            "agent_verdicts": agent_verdicts,
            "tier_summaries": {
                "1": {"total_evaluated": len(t1_results), "passed": len(tier1_passed), "failed": len(tier1_failed)},
                "2": {"total_evaluated": len(t2_results), "passed": len(tier2_passed), "failed": len(tier2_failed), "rules_generated": rules_generated},
                "3": {"total_evaluated": len(t3_results), "passed": len(tier3_passed), "failed": len(tier3_failed)},
            },
            "overall": {
                "total_agents": len(all_agents),
                "passed_all": len(passed_all),
                "failed_tier1": len(tier1_failed),
                "failed_tier2": len(tier2_failed - tier1_failed),
                "failed_tier3": len(tier3_failed - tier2_failed - tier1_failed),
                "llm_calls_made": llm_calls,
            },
        }


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

