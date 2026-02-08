"""
Integration tests for the prompt optimizer backend pipeline.

These tests hit the real backend at http://localhost:5000 and make actual LLM
calls. They are NOT mocked -- they validate the full round-trip from file
upload through LLM execution to response serialization.

Run with:
    python -m pytest tests/test_backend_optimizer.py -v

Prerequisites:
    - Backend running: cd backend && python api.py
    - OPENAI_API_KEY set in environment or .env
"""
from __future__ import annotations

import io
import unittest

import requests

BASE_URL = "http://localhost:5000"
REQUEST_TIMEOUT = 60  # seconds -- LLM calls can be slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_csv_bytes(
    rows: list[dict],
    columns: list[str] | None = None,
) -> io.BytesIO:
    """Build an in-memory CSV file from a list of row dicts."""
    if not rows:
        raise ValueError("rows must not be empty")
    cols = columns or list(rows[0].keys())
    lines = [",".join(cols)]
    for row in rows:
        values = []
        for c in cols:
            val = str(row.get(c, ""))
            # Escape commas and quotes inside values
            if "," in val or '"' in val or "\n" in val:
                val = '"' + val.replace('"', '""') + '"'
            values.append(val)
        lines.append(",".join(values))
    buf = io.BytesIO("\n".join(lines).encode("utf-8"))
    buf.name = "test_data.csv"
    return buf


def _default_form_fields() -> dict:
    """Return the standard form fields shared by most tests."""
    return {
        "prompt_template": "Answer the following question concisely: {input}",
        "input_col": "User Input",
        "gold_col": "Expected Output",
        "eval_model": "gpt-4o",
        "optimizer_model": "gpt-4o",
        "target_score": "0.85",
        "max_iters": "1",
    }


def _default_csv_rows() -> list[dict]:
    """Return minimal test data rows."""
    return [
        {
            "Agent Name": "SummaryBot",
            "User Input": "What is the capital of France?",
            "Expected Output": "The capital of France is Paris.",
        },
    ]


def _post_optimizer(
    csv_rows: list[dict] | None = None,
    form_overrides: dict | None = None,
    include_file: bool = True,
    csv_columns: list[str] | None = None,
) -> requests.Response:
    """
    POST to /api/optimizer/run with sensible defaults.

    Parameters
    ----------
    csv_rows : list[dict] | None
        Rows for the CSV file.  Uses ``_default_csv_rows()`` when None.
    form_overrides : dict | None
        Keys to add/override on the default form fields.
    include_file : bool
        Whether to attach a file at all (set False to test missing-file error).
    csv_columns : list[str] | None
        Explicit column order for the CSV.  Inferred from row keys when None.
    """
    fields = _default_form_fields()
    if form_overrides:
        fields.update(form_overrides)

    files = {}
    if include_file:
        rows = csv_rows or _default_csv_rows()
        buf = _build_csv_bytes(rows, columns=csv_columns)
        files["file"] = ("test_data.csv", buf, "text/csv")

    return requests.post(
        f"{BASE_URL}/api/optimizer/run",
        data=fields,
        files=files,
        timeout=REQUEST_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestBackendOptimizer(unittest.TestCase):
    """Integration tests for POST /api/optimizer/run and GET /health."""

    # Required fields in each result object
    RESULT_FIELDS = {
        "row_index",
        "input",
        "gold",
        "output",
        "score",
        "iterations",
        "template",
        "comments",
        "original_output",
        "original_score",
        "latency_ms",
        "original_latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
    }

    # Required fields in the summary object
    SUMMARY_FIELDS = {
        "total_rows",
        "original_avg",
        "optimized_avg",
        "improvement",
        "total_iterations",
        "met_target",
        "total_latency_ms",
        "total_baseline_latency_ms",
        "avg_latency_ms",
        "total_cost_usd",
        "total_tokens",
        "agent_filter",
        "model",
    }

    # ------------------------------------------------------------------
    # Class-level setup: skip everything if backend is down
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        """Check that the backend is reachable before running any test."""
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Health check returned {resp.status_code}")
        except (requests.ConnectionError, requests.Timeout, ConnectionError) as exc:
            raise unittest.SkipTest(
                f"Backend not available at {BASE_URL} -- skipping integration tests. "
                f"Start the server with: cd backend && python api.py\n"
                f"Error: {exc}"
            )

    # Cache a single successful response for tests that only inspect the
    # response shape (avoids redundant LLM calls that cost money).
    _cached_response: requests.Response | None = None
    _cached_json: dict | None = None

    @classmethod
    def _get_cached_response(cls) -> tuple[requests.Response, dict]:
        """Return a cached successful optimizer response, calling the API once."""
        if cls._cached_response is None:
            cls._cached_response = _post_optimizer()
            cls._cached_json = cls._cached_response.json()
        return cls._cached_response, cls._cached_json

    # ------------------------------------------------------------------
    # 1. Health endpoint
    # ------------------------------------------------------------------

    def test_health_endpoint(self):
        """GET /health returns 200 with status 'ok'."""
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")

    # ------------------------------------------------------------------
    # 2. Basic CSV upload
    # ------------------------------------------------------------------

    def test_optimizer_basic_csv(self):
        """POST with a simple CSV returns 200, success=true, non-empty results."""
        resp, body = self._get_cached_response()
        self.assertEqual(resp.status_code, 200, f"Expected 200, got {resp.status_code}: {body}")
        self.assertTrue(body.get("success"), f"success should be True: {body}")
        self.assertIn("results", body)
        self.assertIsInstance(body["results"], list)
        self.assertGreater(len(body["results"]), 0, "results array should not be empty")

    # ------------------------------------------------------------------
    # 3. LLM output is real (not an error string)
    # ------------------------------------------------------------------

    def test_llm_output_is_real(self):
        """Result outputs should NOT be error stubs from _call_llm_tracked."""
        _, body = self._get_cached_response()
        for result in body["results"]:
            output = result.get("output", "")
            self.assertFalse(
                output.startswith("[LLM error"),
                f"Output looks like an LLM error: {output[:120]}",
            )
            self.assertFalse(
                output.startswith("[LiteLLM not installed"),
                f"LiteLLM is not installed on the backend: {output[:120]}",
            )
            # Also check original_output
            orig = result.get("original_output", "")
            self.assertFalse(
                orig.startswith("[LLM error"),
                f"Original output looks like an LLM error: {orig[:120]}",
            )
            self.assertFalse(
                orig.startswith("[LiteLLM not installed"),
                f"LiteLLM is not installed on the backend: {orig[:120]}",
            )

    # ------------------------------------------------------------------
    # 4. Result fields complete
    # ------------------------------------------------------------------

    def test_result_fields_complete(self):
        """Each result object contains every required field."""
        _, body = self._get_cached_response()
        for i, result in enumerate(body["results"]):
            missing = self.RESULT_FIELDS - set(result.keys())
            self.assertFalse(
                missing,
                f"Result[{i}] is missing fields: {missing}",
            )

    # ------------------------------------------------------------------
    # 5. Summary fields complete
    # ------------------------------------------------------------------

    def test_summary_fields_complete(self):
        """Summary object contains every required field."""
        _, body = self._get_cached_response()
        summary = body.get("summary", {})
        missing = self.SUMMARY_FIELDS - set(summary.keys())
        self.assertFalse(
            missing,
            f"Summary is missing fields: {missing}",
        )

    # ------------------------------------------------------------------
    # 6. Scores are valid floats in [0, 1]
    # ------------------------------------------------------------------

    def test_scores_are_valid(self):
        """score and original_score are floats between 0.0 and 1.0."""
        _, body = self._get_cached_response()
        for i, result in enumerate(body["results"]):
            for key in ("score", "original_score"):
                val = result[key]
                self.assertIsInstance(val, (int, float), f"Result[{i}].{key} should be numeric")
                self.assertGreaterEqual(val, 0.0, f"Result[{i}].{key}={val} should be >= 0")
                self.assertLessEqual(val, 1.0, f"Result[{i}].{key}={val} should be <= 1")

    # ------------------------------------------------------------------
    # 7. Latency is positive (proves real LLM calls)
    # ------------------------------------------------------------------

    def test_latency_is_positive(self):
        """latency_ms and original_latency_ms must be > 0 for real LLM calls."""
        _, body = self._get_cached_response()
        for i, result in enumerate(body["results"]):
            self.assertGreater(
                result["latency_ms"], 0,
                f"Result[{i}].latency_ms should be > 0 (proves real LLM call)",
            )
            self.assertGreater(
                result["original_latency_ms"], 0,
                f"Result[{i}].original_latency_ms should be > 0 (proves real LLM call)",
            )

    # ------------------------------------------------------------------
    # 8. Tokens are positive
    # ------------------------------------------------------------------

    def test_tokens_are_positive(self):
        """prompt_tokens and completion_tokens must be > 0."""
        _, body = self._get_cached_response()
        for i, result in enumerate(body["results"]):
            self.assertGreater(
                result["prompt_tokens"], 0,
                f"Result[{i}].prompt_tokens should be > 0",
            )
            self.assertGreater(
                result["completion_tokens"], 0,
                f"Result[{i}].completion_tokens should be > 0",
            )

    # ------------------------------------------------------------------
    # 9. Cost is calculated
    # ------------------------------------------------------------------

    def test_cost_is_calculated(self):
        """cost_usd must be > 0 when tokens are consumed."""
        _, body = self._get_cached_response()
        for i, result in enumerate(body["results"]):
            self.assertGreater(
                result["cost_usd"], 0,
                f"Result[{i}].cost_usd should be > 0",
            )

    # ------------------------------------------------------------------
    # 10. Works without comments_col
    # ------------------------------------------------------------------

    def test_no_comments_col(self):
        """Omitting comments_col should still succeed; comments=null in results."""
        rows = [
            {
                "User Input": "What color is the sky?",
                "Expected Output": "The sky is blue.",
            },
        ]
        resp = _post_optimizer(csv_rows=rows)
        body = resp.json()
        self.assertEqual(resp.status_code, 200, f"Expected 200: {body}")
        self.assertTrue(body.get("success"), f"success should be True: {body}")
        for result in body["results"]:
            self.assertIsNone(
                result.get("comments"),
                "comments should be null when comments_col is not provided",
            )

    # ------------------------------------------------------------------
    # 11. Works with agent_filter
    # ------------------------------------------------------------------

    def test_with_agent_filter(self):
        """Providing agent_filter filters rows to matching agent and is echoed in summary."""
        rows = [
            {
                "Agent Name": "AlphaBot",
                "User Input": "2 + 2",
                "Expected Output": "4",
            },
            {
                "Agent Name": "BetaBot",
                "User Input": "3 + 3",
                "Expected Output": "6",
            },
        ]
        resp = _post_optimizer(
            csv_rows=rows,
            form_overrides={"agent_filter": "AlphaBot"},
        )
        body = resp.json()
        self.assertEqual(resp.status_code, 200, f"Expected 200: {body}")
        self.assertTrue(body.get("success"), f"success should be True: {body}")
        # Only AlphaBot's row should appear
        self.assertEqual(
            body["summary"]["total_rows"], 1,
            "agent_filter='AlphaBot' should yield exactly 1 row",
        )
        self.assertEqual(
            body["summary"]["agent_filter"], "AlphaBot",
            "summary.agent_filter should echo the filter value",
        )

    # ------------------------------------------------------------------
    # 12. Invalid template (missing {input} placeholder)
    # ------------------------------------------------------------------

    def test_invalid_template(self):
        """Template without {input} should return an error."""
        resp = _post_optimizer(
            form_overrides={"prompt_template": "Just do something"},
        )
        body = resp.json()
        # The API returns 400 for this validation error
        self.assertIn(resp.status_code, (400, 500), f"Expected 400 or 500: {body}")
        self.assertIn("error", body, "Response should contain an error message")
        self.assertIn(
            "{input}",
            body["error"].lower(),
            "Error message should mention the missing {input} placeholder",
        )

    # ------------------------------------------------------------------
    # 13. No file uploaded
    # ------------------------------------------------------------------

    def test_no_file(self):
        """Omitting the file should return 400."""
        resp = _post_optimizer(include_file=False)
        self.assertEqual(resp.status_code, 400, f"Expected 400: {resp.text}")
        body = resp.json()
        self.assertIn("error", body)

    # ------------------------------------------------------------------
    # 14. CSV format support
    # ------------------------------------------------------------------

    def test_csv_format_support(self):
        """CSV files are accepted (the test data is already CSV, but we
        explicitly verify the .csv extension is allowed and processed)."""
        rows = [
            {
                "User Input": "Name a primary color.",
                "Expected Output": "Red, blue, or yellow.",
            },
        ]
        resp = _post_optimizer(csv_rows=rows)
        body = resp.json()
        self.assertEqual(resp.status_code, 200, f"Expected 200: {body}")
        self.assertTrue(body.get("success"), f"CSV should be accepted: {body}")
        self.assertGreater(len(body["results"]), 0, "Should return results for CSV input")


if __name__ == "__main__":
    unittest.main()
