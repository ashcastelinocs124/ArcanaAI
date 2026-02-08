"""
Test cases for AgentDAG class.
Run: python -m unittest test_agent_dag
"""

import unittest
from semantic_pipeline import AgentDAG, extract_agent_logs, agent_dags, EmbeddingMatchAnalyzer


class TestAgentDAG(unittest.TestCase):
    """Tests for AgentDAG."""

    def setUp(self):
        self.agent_logs = extract_agent_logs()
        self.dags = AgentDAG.from_agent_logs(self.agent_logs)
        self.trip_trace_id = "tr-a7f2-9b3c-4e1d-8f6a"
        self.dag = self.dags[self.trip_trace_id]

    def test_from_agent_logs_returns_dict_of_dags(self):
        """from_agent_logs returns a dict mapping trace_id to AgentDAG."""
        self.assertIsInstance(self.dags, dict)
        for trace_id, dag in self.dags.items():
            self.assertIsInstance(dag, AgentDAG)
            self.assertEqual(dag.trace_id, trace_id)

    def test_dag_has_trace_id_and_journey_name(self):
        """Each DAG has trace_id and journey_name."""
        self.assertEqual(self.dag.trace_id, self.trip_trace_id)
        self.assertEqual(self.dag.journey_name, "Trip Booking (London → Paris)")

    def test_get_parents_returns_empty_for_root(self):
        """Root agents (agent-1) have no parents."""
        root_id = f"{self.trip_trace_id}-agent-1"
        parents = self.dag.get_parents(root_id)
        self.assertEqual(parents, [])

    def test_get_parents_returns_single_parent(self):
        """Agents with single parent return one parent_id."""
        child_id = f"{self.trip_trace_id}-agent-2"
        parents = self.dag.get_parents(child_id)
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0], f"{self.trip_trace_id}-agent-1")

    def test_get_parents_returns_multiple_parents_for_fan_in(self):
        """ConfirmationAgent (agent-6) has two parents (fan-in)."""
        child_id = f"{self.trip_trace_id}-agent-6"
        parents = self.dag.get_parents(child_id)
        self.assertEqual(len(parents), 2)
        self.assertIn(f"{self.trip_trace_id}-agent-4", parents)
        self.assertIn(f"{self.trip_trace_id}-agent-5", parents)

    def test_get_children_returns_multiple_for_fan_out(self):
        """FlightSearchAgent (agent-1) has two children (fan-out)."""
        parent_id = f"{self.trip_trace_id}-agent-1"
        children = self.dag.get_children(parent_id)
        self.assertEqual(len(children), 2)
        self.assertIn(f"{self.trip_trace_id}-agent-2", children)
        self.assertIn(f"{self.trip_trace_id}-agent-3", children)

    def test_get_node_returns_agent_data(self):
        """get_node returns full agent data for valid agent_id."""
        agent_id = f"{self.trip_trace_id}-agent-1"
        node = self.dag.get_node(agent_id)
        self.assertIsNotNone(node)
        self.assertEqual(node["agent_name"], "FlightSearchAgent")
        self.assertEqual(node["agent_turn"], 1)

    def test_get_node_returns_none_for_invalid_id(self):
        """get_node returns None for unknown agent_id."""
        node = self.dag.get_node("nonexistent-agent-id")
        self.assertIsNone(node)

    def test_get_edges_returns_all_parent_child_pairs(self):
        """get_edges returns list of (parent_id, child_id) tuples."""
        edges = self.dag.get_edges()
        self.assertIsInstance(edges, list)
        self.assertGreater(len(edges), 0)
        for parent_id, child_id in edges:
            self.assertIsInstance(parent_id, str)
            self.assertIsInstance(child_id, str)

    def test_roots_returns_agents_with_no_parents(self):
        """roots returns agent IDs that have no parents."""
        roots = self.dag.roots()
        self.assertIn(f"{self.trip_trace_id}-agent-1", roots)
        for root_id in roots:
            self.assertEqual(self.dag.get_parents(root_id), [])

    def test_leaves_returns_agents_with_no_children(self):
        """leaves returns agent IDs that have no children."""
        leaves = self.dag.leaves()
        self.assertIn(f"{self.trip_trace_id}-agent-6", leaves)
        for leaf_id in leaves:
            self.assertEqual(self.dag.get_children(leaf_id), [])

    def test_has_edge_returns_true_for_valid_edge(self):
        """has_edge returns True when edge exists."""
        parent_id = f"{self.trip_trace_id}-agent-1"
        child_id = f"{self.trip_trace_id}-agent-2"
        self.assertTrue(self.dag.has_edge(parent_id, child_id))

    def test_has_edge_returns_false_for_invalid_edge(self):
        """has_edge returns False when edge does not exist."""
        parent_id = f"{self.trip_trace_id}-agent-2"
        child_id = f"{self.trip_trace_id}-agent-3"
        self.assertFalse(self.dag.has_edge(parent_id, child_id))

    def test_contains_operator(self):
        """AgentDAG supports 'in' operator for agent_id."""
        agent_id = f"{self.trip_trace_id}-agent-1"
        self.assertIn(agent_id, self.dag)
        self.assertNotIn("fake-agent-id", self.dag)

    def test_len_returns_node_count(self):
        """len(dag) returns number of nodes."""
        self.assertEqual(len(self.dag), 6)

    def test_dag_built_from_empty_logs(self):
        """Empty agent logs produce empty DAG."""
        empty_dags = AgentDAG.from_agent_logs([])
        self.assertEqual(len(empty_dags), 0)

    def test_refund_journey_fan_out_structure(self):
        """Refund journey has two children from root (RAGAgent)."""
        refund_dag = self.dags["tr-b8e3-0c4d-5f2e-9a7b"]
        root_id = "tr-b8e3-0c4d-5f2e-9a7b-agent-1"
        children = refund_dag.get_children(root_id)
        self.assertEqual(len(children), 2)

    def test_quote_journey_diamond_has_multi_parent_leaf(self):
        """Quote journey QuoteAggregator has two parents."""
        quote_dag = self.dags["tr-c9f4-1d5e-6a3f-0b8c"]
        aggregator_id = "tr-c9f4-1d5e-6a3f-0b8c-agent-4"
        parents = quote_dag.get_parents(aggregator_id)
        self.assertEqual(len(parents), 2)


class TestEmbeddingMatchAnalyzer(unittest.TestCase):
    """Tests for EmbeddingMatchAnalyzer."""

    def setUp(self):
        self.agent_logs = extract_agent_logs()
        self.dags = AgentDAG.from_agent_logs(self.agent_logs)
        self.trip_dag = self.dags["tr-a7f2-9b3c-4e1d-8f6a"]

    def test_analyze_returns_edge_results(self):
        """analyze returns one result per edge with diff and running_avg."""
        analyzer = EmbeddingMatchAnalyzer(self.trip_dag)
        results = analyzer.analyze()
        edges = self.trip_dag.get_edges()
        self.assertEqual(len(results), len(edges))
        for r in results:
            self.assertIn("edge", r)
            self.assertIn("diff", r)
            self.assertIn("running_avg", r)
            self.assertIn("status", r)

    def test_running_avg_cumulative(self):
        """First edge: avg=diff. Second edge: avg=(d1+d2)/2."""
        def mock_sim(a, b):
            return 0.5  # diff = 0.5
        analyzer = EmbeddingMatchAnalyzer(self.trip_dag, similarity_fn=mock_sim)
        results = analyzer.analyze()
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["running_avg"], 0.5)
        if len(results) >= 2:
            self.assertEqual(results[1]["running_avg"], 0.5)

    def test_status_red_when_similarity_below_threshold(self):
        """Edge with similarity < 0.8 gets RED."""
        def sim_low(a, b):
            return 0.7  # < 0.8
        analyzer = EmbeddingMatchAnalyzer(self.trip_dag, similarity_fn=sim_low, min_similarity=0.8)
        results = analyzer.analyze()
        statuses = [r["status"] for r in results]
        self.assertIn(EmbeddingMatchAnalyzer.STATUS_RED, statuses)

    def test_status_green_when_similarity_above_threshold(self):
        """Edge with similarity >= 0.8 gets GREEN."""
        def sim_high(a, b):
            return 0.9  # >= 0.8
        analyzer = EmbeddingMatchAnalyzer(self.trip_dag, similarity_fn=sim_high, min_similarity=0.8)
        results = analyzer.analyze()
        statuses = [r["status"] for r in results]
        self.assertTrue(all(s == EmbeddingMatchAnalyzer.STATUS_GREEN for s in statuses))

    def test_avg_and_stdev_properties(self):
        """avg and stdev are computed after analyze."""
        analyzer = EmbeddingMatchAnalyzer(self.trip_dag)
        _ = analyzer.analyze()
        self.assertIsInstance(analyzer.avg, (int, float))
        self.assertIsInstance(analyzer.stdev, (int, float))
        self.assertGreaterEqual(analyzer.avg, 0)
        self.assertGreaterEqual(analyzer.stdev, 0)

    def test_empty_edges_returns_empty_results(self):
        """DAG with no edges (single agent) produces no results."""
        support_logs = [l for l in self.agent_logs if l["trace_id"] == "tr-e1b6-3f7a-8c5b-2d0e"]
        dags = AgentDAG.from_agent_logs(support_logs)
        dag = dags.get("tr-e1b6-3f7a-8c5b-2d0e")
        if dag and len(dag.get_edges()) == 0:
            analyzer = EmbeddingMatchAnalyzer(dag)
            results = analyzer.analyze()
            self.assertEqual(results, [])

    def test_status_thresholds_with_known_similarities(self):
        """Status mapping uses min_similarity=0.8."""
        class FakeDag:
            def __init__(self, edges, nodes):
                self._edges = edges
                self._nodes = nodes

            def get_edges(self):
                return list(self._edges)

            def get_node(self, agent_id):
                return self._nodes.get(agent_id)

        # similarities: 0.8 (>=), 0.6 (<), 0.4 (<) with min_similarity=0.8
        edges = [("p1", "c1"), ("p2", "c2"), ("p3", "c3")]
        nodes = {
            "c1": {"agent_name": "C1", "input": "a1", "output": "b1"},
            "c2": {"agent_name": "C2", "input": "a2", "output": "b2"},
            "c3": {"agent_name": "C3", "input": "a3", "output": "b3"},
        }
        sim_map = {
            ("a1", "b1"): 0.8,  # >= 0.8 => GREEN
            ("a2", "b2"): 0.6,   # < 0.8 => RED
            ("a3", "b3"): 0.4,   # < 0.8 => RED
        }

        def sim_fn(a, b):
            return sim_map[(a, b)]

        analyzer = EmbeddingMatchAnalyzer(FakeDag(edges, nodes), similarity_fn=sim_fn, min_similarity=0.8)
        results = analyzer.analyze()

        status_map = {r["child_name"]: r["status"] for r in results}
        self.assertEqual(status_map["C1"], EmbeddingMatchAnalyzer.STATUS_GREEN)
        self.assertEqual(status_map["C2"], EmbeddingMatchAnalyzer.STATUS_RED)
        self.assertEqual(status_map["C3"], EmbeddingMatchAnalyzer.STATUS_RED)


if __name__ == "__main__":
    unittest.main()
