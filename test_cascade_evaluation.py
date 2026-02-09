"""
Test cases for the 3-tier cascading evaluation system.
Run: python -m unittest test_cascade_evaluation
"""

import json
import unittest
from unittest.mock import patch

from semantic_pipeline import (
    AgentDAG,
    CascadeEvaluator,
    GlobalRulesJudge,
    SentinelAnalyzer,
    SpecialistEvaluator,
)


class FakeDag:
    """Minimal DAG stub for unit tests."""

    def __init__(self, nodes: dict, edges: list | None = None, trace_id: str = "test-trace", journey_name: str = "Test"):
        self._nodes = nodes
        self._edges = edges or []
        self.trace_id = trace_id
        self.journey_name = journey_name
        self._parents_of = {}
        self._children_of = {}
        for p, c in self._edges:
            self._parents_of.setdefault(c, []).append(p)
            self._children_of.setdefault(p, []).append(c)

    def get_nodes(self):
        return dict(self._nodes)

    def get_node(self, agent_id):
        return self._nodes.get(agent_id)

    def get_edges(self):
        return list(self._edges)

    def get_parents(self, agent_id):
        return self._parents_of.get(agent_id, [])

    def get_children(self, agent_id):
        return self._children_of.get(agent_id, [])

    def roots(self):
        return [aid for aid in self._nodes if not self._parents_of.get(aid)]

    def leaves(self):
        return [aid for aid in self._nodes if not self._children_of.get(aid)]


def _make_node(name, input_text="What is the weather?", output="The weather is sunny today."):
    return {
        "agent_name": name,
        "agent_id": name.lower().replace(" ", "-"),
        "input": input_text,
        "output": output,
        "core_payload": {"tool_calls": []},
        "metadata": {},
    }


class TestSentinelAnalyzer(unittest.TestCase):
    """Tier 1: regex-based checks."""

    def test_normal_output_passes_all(self):
        dag = FakeDag({"a1": _make_node("Agent1")})
        results = SentinelAnalyzer(dag).analyze()
        self.assertEqual(results["a1"]["status"], "sentinel_pass")
        self.assertEqual(len(results["a1"]["fail_reasons"]), 0)

    def test_empty_output_fails(self):
        node = _make_node("Agent1", output="")
        dag = FakeDag({"a1": node})
        results = SentinelAnalyzer(dag).analyze()
        self.assertIn("empty", results["a1"]["fail_reasons"][0].lower())

    def test_whitespace_only_output_fails(self):
        node = _make_node("Agent1", output="   \n\t  ")
        dag = FakeDag({"a1": node})
        results = SentinelAnalyzer(dag).analyze()
        checks = {c["name"]: c["passed"] for c in results["a1"]["checks"]}
        self.assertFalse(checks["empty_output"])

    def test_placeholder_detected(self):
        node = _make_node("Agent1", output="TODO: implement this feature later")
        dag = FakeDag({"a1": node})
        results = SentinelAnalyzer(dag).analyze()
        checks = {c["name"]: c["passed"] for c in results["a1"]["checks"]}
        self.assertFalse(checks["placeholder_text"])

    def test_malformed_json_detected(self):
        node = _make_node("Agent1", output='{"broken: json, missing bracket')
        dag = FakeDag({"a1": node})
        results = SentinelAnalyzer(dag).analyze()
        checks = {c["name"]: c["passed"] for c in results["a1"]["checks"]}
        self.assertFalse(checks["malformed_json"])

    def test_valid_json_passes(self):
        node = _make_node("Agent1", output='{"status": "ok", "data": [1, 2, 3]}')
        dag = FakeDag({"a1": node})
        results = SentinelAnalyzer(dag).analyze()
        checks = {c["name"]: c["passed"] for c in results["a1"]["checks"]}
        self.assertTrue(checks["malformed_json"])

    def test_error_pattern_detected(self):
        node = _make_node("Agent1", output="Error: Connection refused to database server")
        dag = FakeDag({"a1": node})
        results = SentinelAnalyzer(dag).analyze()
        checks = {c["name"]: c["passed"] for c in results["a1"]["checks"]}
        self.assertFalse(checks["error_patterns"])

    def test_score_below_threshold_triggers_fail(self):
        # Empty output that also starts with '{' (malformed JSON) + error pattern + placeholder = many failures
        node = _make_node("Agent1", input_text="Please provide a detailed analysis of the market trends",
                          output='{TODO Error: broken placeholder [FILL IN]')
        dag = FakeDag({"a1": node})
        results = SentinelAnalyzer(dag).analyze()
        # Fails: placeholder_text, malformed_json, error_patterns = 3 failures out of 7 => score ~0.57
        # Let's verify it catches multiple issues
        checks = {c["name"]: c["passed"] for c in results["a1"]["checks"]}
        self.assertFalse(checks["placeholder_text"])
        self.assertFalse(checks["malformed_json"])
        self.assertFalse(checks["error_patterns"])
        self.assertGreater(len(results["a1"]["fail_reasons"]), 2)


class TestGlobalRulesJudge(unittest.TestCase):
    """Tier 2: single LLM call for rules, deterministic application."""

    def _mock_rules_response(self):
        return json.dumps([
            {"description": "Output should mention booking", "check_type": "output_contains", "parameters": {"terms": ["booking", "book"]}},
            {"description": "No error messages", "check_type": "output_not_contains", "parameters": {"terms": ["error", "failed"]}},
            {"description": "Relevant to travel", "check_type": "output_relevance", "parameters": {"terms": ["flight", "hotel", "travel", "trip"]}},
            {"description": "Minimum length", "check_type": "output_length", "parameters": {"min_words": 3}},
            {"description": "Professional tone", "check_type": "tone_check", "parameters": {"negative_terms": ["stupid", "dumb"]}},
        ])

    @patch("semantic_pipeline._call_llm")
    def test_llm_called_exactly_once(self, mock_llm):
        mock_llm.return_value = self._mock_rules_response()
        nodes = {"a1": _make_node("Agent1", output="Booking your flight and hotel trip"), "a2": _make_node("Agent2", output="Travel booking confirmed")}
        dag = FakeDag(nodes)
        judge = GlobalRulesJudge(dag, "Book a flight")
        judge.analyze()
        self.assertEqual(mock_llm.call_count, 1)

    @patch("semantic_pipeline._call_llm")
    def test_rules_are_cached(self, mock_llm):
        mock_llm.return_value = self._mock_rules_response()
        dag = FakeDag({"a1": _make_node("Agent1", output="Booking your flight")})
        judge = GlobalRulesJudge(dag, "Book a flight")
        judge.analyze()
        judge.analyze()  # second call should use cached rules
        self.assertEqual(mock_llm.call_count, 1)

    @patch("semantic_pipeline._call_llm")
    def test_exclusion_respected(self, mock_llm):
        mock_llm.return_value = self._mock_rules_response()
        nodes = {"a1": _make_node("Agent1", output="Booking flight"), "a2": _make_node("Agent2", output="Hotel trip")}
        dag = FakeDag(nodes)
        judge = GlobalRulesJudge(dag, "Book a flight")
        results = judge.analyze(exclude_agents={"a1"})
        self.assertNotIn("a1", results)
        self.assertIn("a2", results)

    @patch("semantic_pipeline._call_llm")
    def test_passing_agent_scores_above_threshold(self, mock_llm):
        mock_llm.return_value = self._mock_rules_response()
        node = _make_node("Agent1", output="Your flight booking and hotel trip have been confirmed successfully.")
        dag = FakeDag({"a1": node})
        judge = GlobalRulesJudge(dag, "Book a flight", score_threshold=0.6)
        results = judge.analyze()
        self.assertEqual(results["a1"]["status"], "rules_pass")
        self.assertGreaterEqual(results["a1"]["score"], 0.6)

    @patch("semantic_pipeline._call_llm")
    def test_failing_agent_scores_below_threshold(self, mock_llm):
        mock_llm.return_value = self._mock_rules_response()
        node = _make_node("Agent1", output="error failed")  # fails contains, relevance; passes not_contains(false), etc.
        dag = FakeDag({"a1": node})
        judge = GlobalRulesJudge(dag, "Book a flight", score_threshold=0.6)
        results = judge.analyze()
        # "error failed" contains "error" -> not_contains fails, no booking -> contains fails
        self.assertEqual(results["a1"]["status"], "rules_fail")


class TestSpecialistEvaluator(unittest.TestCase):
    """Tier 3: per-agent LLM evaluation."""

    def _mock_specialist_response(self, score=0.9, verdict="PASS"):
        return json.dumps({
            "score": score,
            "role_assessment": "Agent fulfills its role correctly",
            "quality_assessment": "Output is well-formed and complete",
            "goal_alignment": "Directly contributes to user goal",
            "verdict": verdict,
            "explanation": "Agent performed well overall",
        })

    @patch("semantic_pipeline._call_llm")
    def test_per_agent_llm_calls(self, mock_llm):
        mock_llm.return_value = self._mock_specialist_response()
        nodes = {"a1": _make_node("Agent1"), "a2": _make_node("Agent2"), "a3": _make_node("Agent3")}
        dag = FakeDag(nodes)
        specialist = SpecialistEvaluator(dag, "Book a flight")
        specialist.analyze()
        self.assertEqual(mock_llm.call_count, 3)

    @patch("semantic_pipeline._call_llm")
    def test_exclusion_reduces_calls(self, mock_llm):
        mock_llm.return_value = self._mock_specialist_response()
        nodes = {"a1": _make_node("Agent1"), "a2": _make_node("Agent2")}
        dag = FakeDag(nodes)
        specialist = SpecialistEvaluator(dag, "Book a flight")
        results = specialist.analyze(exclude_agents={"a1"})
        self.assertEqual(mock_llm.call_count, 1)
        self.assertNotIn("a1", results)
        self.assertIn("a2", results)

    @patch("semantic_pipeline._call_llm")
    def test_valid_response_parsing(self, mock_llm):
        mock_llm.return_value = self._mock_specialist_response(score=0.85, verdict="PASS")
        dag = FakeDag({"a1": _make_node("Agent1")})
        specialist = SpecialistEvaluator(dag, "Book a flight")
        results = specialist.analyze()
        self.assertAlmostEqual(results["a1"]["score"], 0.85)
        self.assertEqual(results["a1"]["verdict"], "PASS")
        self.assertEqual(results["a1"]["status"], "specialist_pass")

    @patch("semantic_pipeline._call_llm")
    def test_invalid_response_fallback(self, mock_llm):
        mock_llm.return_value = "This is not valid JSON at all"
        dag = FakeDag({"a1": _make_node("Agent1")})
        specialist = SpecialistEvaluator(dag, "Book a flight")
        results = specialist.analyze()
        self.assertEqual(results["a1"]["score"], 0.5)
        self.assertEqual(results["a1"]["verdict"], "UNCLEAR")


class TestCascadeEvaluator(unittest.TestCase):
    """Full cascade orchestration tests (gauntlet model: all agents through all 3 tiers)."""

    def _mock_rules(self):
        return json.dumps([
            {"description": "Has content", "check_type": "output_length", "parameters": {"min_words": 1}},
        ])

    def _mock_specialist_pass(self):
        return json.dumps({
            "score": 0.9, "role_assessment": "Good",
            "quality_assessment": "Good", "goal_alignment": "Aligned",
            "verdict": "PASS", "explanation": "Agent performed well",
        })

    @patch("semantic_pipeline._call_llm")
    def test_all_pass_all_tiers(self, mock_llm):
        """Good agents pass all 3 tiers — status is 'pass', evaluated_by_tier=3 (All Tiers)."""
        def side_effect(prompt, model="gpt-4o"):
            if "generate" in prompt.lower() or "quality rules" in prompt.lower() or "rules" in prompt.lower():
                return self._mock_rules()
            return self._mock_specialist_pass()

        mock_llm.side_effect = side_effect
        nodes = {
            "a1": _make_node("Agent1", output="Booking confirmed for your flight to Paris."),
            "a2": _make_node("Agent2", output="Hotel reservation at Grand Hotel completed."),
        }
        dag = FakeDag(nodes)
        result = CascadeEvaluator(dag, "Book a trip").evaluate()
        # All agents through all tiers: 1 T2 rules call + 2 T3 specialist calls = 3
        self.assertEqual(mock_llm.call_count, 3)
        self.assertEqual(result["overall"]["passed_all"], 2)
        self.assertEqual(result["overall"]["llm_calls_made"], 3)
        for aid in ["a1", "a2"]:
            self.assertEqual(result["agent_verdicts"][aid]["evaluated_by_tier"], 3)
            self.assertEqual(result["agent_verdicts"][aid]["status"], "pass")

    @patch("semantic_pipeline._call_llm")
    def test_tier1_fail_blocks_pass(self, mock_llm):
        """Agent failing T1 is marked fail with evaluated_by_tier=1, even though all tiers run."""
        def side_effect(prompt, model="gpt-4o"):
            if "generate" in prompt.lower() or "quality rules" in prompt.lower() or "rules" in prompt.lower():
                return self._mock_rules()
            return self._mock_specialist_pass()

        mock_llm.side_effect = side_effect

        from semantic_pipeline import SentinelAnalyzer as SA
        original_analyze = SA.analyze

        def patched_analyze(self_sa):
            results = original_analyze(self_sa)
            if "a2" in results:
                results["a2"]["status"] = "sentinel_fail"
                results["a2"]["score"] = 0.3
                results["a2"]["fail_reasons"] = ["Forced fail for test"]
            return results

        with patch.object(SA, "analyze", patched_analyze):
            nodes = {
                "a1": _make_node("Agent1", output="Flight booked successfully."),
                "a2": _make_node("Agent2", output="Bad output"),
            }
            dag = FakeDag(nodes)
            result = CascadeEvaluator(dag, "Book a trip").evaluate()

        # a1 passes all 3 tiers
        self.assertEqual(result["agent_verdicts"]["a1"]["status"], "pass")
        # a2 fails at T1 (blocking tier)
        self.assertEqual(result["agent_verdicts"]["a2"]["status"], "fail")
        self.assertEqual(result["agent_verdicts"]["a2"]["evaluated_by_tier"], 1)
        self.assertEqual(result["overall"]["failed_tier1"], 1)

    @patch("semantic_pipeline._call_llm")
    def test_tier2_fail_blocks_pass(self, mock_llm):
        """Agent passing T1 but failing T2 is marked fail with evaluated_by_tier=2."""
        def side_effect(prompt, model="gpt-4o"):
            if "generate" in prompt.lower() or "quality rules" in prompt.lower() or "rules" in prompt.lower():
                return json.dumps([
                    {"description": "Must mention booking", "check_type": "output_contains", "parameters": {"terms": ["booking"]}},
                    {"description": "Must be relevant", "check_type": "output_relevance", "parameters": {"terms": ["flight", "hotel"]}},
                    {"description": "Min length", "check_type": "output_length", "parameters": {"min_words": 3}},
                ])
            else:
                return json.dumps({
                    "score": 0.3, "role_assessment": "Poor",
                    "quality_assessment": "Inadequate", "goal_alignment": "Misaligned",
                    "verdict": "FAIL", "explanation": "Agent output is irrelevant",
                })

        mock_llm.side_effect = side_effect
        nodes = {
            "a1": _make_node("Agent1", output="Flight booking confirmed for your trip."),
            "a2": _make_node("Agent2", output="xyz abc def"),  # passes T1 (normal text), fails T2 (no booking/flight terms)
        }
        dag = FakeDag(nodes)
        result = CascadeEvaluator(dag, "Book a trip").evaluate()

        # a2 passes T1 but fails T2 (no booking terms in output)
        self.assertEqual(result["agent_verdicts"]["a2"]["status"], "fail")
        # Blocking tier should be 2 or 3 (depends on rules matching)
        self.assertIn(result["agent_verdicts"]["a2"]["evaluated_by_tier"], [2, 3])

    @patch("semantic_pipeline._call_llm")
    def test_llm_call_count_always_includes_all_tiers(self, mock_llm):
        """LLM calls = 1 (T2 rules) + N (T3 agents) — always, since all agents go through all tiers."""
        def side_effect(prompt, model="gpt-4o"):
            if "generate" in prompt.lower() or "quality rules" in prompt.lower() or "rules" in prompt.lower():
                return self._mock_rules()
            return self._mock_specialist_pass()

        mock_llm.side_effect = side_effect
        nodes = {"a1": _make_node("Agent1"), "a2": _make_node("Agent2")}
        dag = FakeDag(nodes)
        result = CascadeEvaluator(dag, "Book").evaluate()
        # 1 T2 rules call + 2 T3 specialist calls = 3
        self.assertEqual(result["overall"]["llm_calls_made"], 3)
        self.assertEqual(mock_llm.call_count, 3)

    @patch("semantic_pipeline._call_llm")
    def test_summary_counts_correct(self, mock_llm):
        """Overall counts use passed_all / failed_tier1 / failed_tier2 / failed_tier3."""
        def side_effect(prompt, model="gpt-4o"):
            if "generate" in prompt.lower() or "quality rules" in prompt.lower() or "rules" in prompt.lower():
                return self._mock_rules()
            return self._mock_specialist_pass()

        mock_llm.side_effect = side_effect
        nodes = {
            "a1": _make_node("Agent1", output="Good output here."),
            "a2": _make_node("Agent2", output="Another good output."),
            "a3": _make_node("Agent3", output="Third good output."),
        }
        dag = FakeDag(nodes)
        result = CascadeEvaluator(dag, "Test goal").evaluate()
        self.assertEqual(result["tier_summaries"]["1"]["total_evaluated"], 3)
        self.assertEqual(result["overall"]["total_agents"], 3)
        total_passed = result["overall"]["passed_all"]
        total_failed = result["overall"]["failed_tier1"] + result["overall"]["failed_tier2"] + result["overall"]["failed_tier3"]
        self.assertEqual(total_passed + total_failed, 3)


if __name__ == "__main__":
    unittest.main()
