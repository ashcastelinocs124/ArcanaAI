"""
Comprehensive test suite for the v1 Trace API validators.

63 tests across 7 test classes covering:
1. Format detection (8 tests)
2. Journey validation (12 tests)
3. Flat record normalization (14 tests)
4. End-to-end normalize_input (10 tests)
5. Agent ID generation (5 tests)
6. Edge cases (8 tests)
7. Sample data source (6 tests)

Run with:
    cd backend && python -m pytest tests/test_validators.py -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import django

# Django setup before any model imports
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "arcana.settings")
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
django.setup()

import pytest

from api.models import ExecutionTrace
from api.validators import (
    ValidationError,
    _validate_journeys,
    detect_format,
    normalize_flat_to_journeys,
    normalize_input,
)

# Path to real sample data
SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "test" / "keywords_ai_agent_output_samples.json"
FLAT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_sample(**overrides):
    """Create a minimal valid sample with required fields."""
    base = {
        "agent_id": "a1",
        "agent_name": "TestAgent",
        "input": "hello",
        "output": "world",
        "parent_id": None,
    }
    base.update(overrides)
    return base


def _minimal_journey(**overrides):
    """Create a minimal valid journey."""
    base = {
        "trace_id": "test-trace-1",
        "journey_name": "Test Journey",
        "samples": [_minimal_sample()],
    }
    base.update(overrides)
    return base


# ===========================================================================
# Test Class 1: Format Detection (8 tests)
# ===========================================================================

class TestFormatDetection:
    def test_detect_journey_grouped(self):
        result = detect_format({"journeys": [{"trace_id": "x", "samples": []}]})
        assert result == "journey_grouped"

    def test_detect_flat_records(self):
        result = detect_format({"records": [{"trace_id": "x", "agent_id": "a1"}]})
        assert result == "flat_records"

    def test_detect_single_journey(self):
        result = detect_format({"trace_id": "x", "samples": [{"agent_id": "a1"}]})
        assert result == "single_journey"

    def test_detect_empty_journeys_array(self):
        result = detect_format({"journeys": []})
        assert result == "journey_grouped"

    def test_detect_unrecognized_format(self):
        with pytest.raises(ValidationError) as exc_info:
            detect_format({"foo": "bar"})
        assert "Unrecognized format" in exc_info.value.errors[0]

    def test_detect_no_keys(self):
        with pytest.raises(ValidationError) as exc_info:
            detect_format({})
        assert "Unrecognized format" in exc_info.value.errors[0]

    def test_detect_journeys_not_array(self):
        with pytest.raises(ValidationError) as exc_info:
            detect_format({"journeys": "string"})
        assert "'journeys' must be an array" in exc_info.value.errors[0]

    def test_detect_records_not_array(self):
        with pytest.raises(ValidationError) as exc_info:
            detect_format({"records": {}})
        assert "'records' must be an array" in exc_info.value.errors[0]


# ===========================================================================
# Test Class 2: Journey Validation (12 tests)
# ===========================================================================

class TestJourneyValidation:
    def test_valid_minimal_journey(self):
        errors = _validate_journeys([_minimal_journey()])
        assert errors == []

    def test_missing_trace_id(self):
        journey = _minimal_journey()
        del journey["trace_id"]
        errors = _validate_journeys([journey])
        assert any("missing required field 'trace_id'" in e for e in errors)

    def test_empty_samples_array(self):
        errors = _validate_journeys([_minimal_journey(samples=[])])
        assert any("no samples provided" in e for e in errors)

    def test_missing_agent_id(self):
        """agent_id should be auto-generated if missing, so no error."""
        sample = _minimal_sample()
        del sample["agent_id"]
        errors = _validate_journeys([_minimal_journey(samples=[sample])])
        assert errors == []

    def test_missing_agent_name(self):
        sample = _minimal_sample()
        del sample["agent_name"]
        errors = _validate_journeys([_minimal_journey(samples=[sample])])
        assert any("'agent_name'" in e for e in errors)

    def test_missing_input(self):
        sample = _minimal_sample()
        del sample["input"]
        errors = _validate_journeys([_minimal_journey(samples=[sample])])
        assert any("'input'" in e for e in errors)

    def test_missing_output(self):
        sample = _minimal_sample()
        del sample["output"]
        errors = _validate_journeys([_minimal_journey(samples=[sample])])
        assert any("'output'" in e for e in errors)

    def test_multiple_missing_fields(self):
        sample = {"agent_id": "a1", "parent_id": None}
        errors = _validate_journeys([_minimal_journey(samples=[sample])])
        assert len(errors) >= 3  # agent_name, input, output

    def test_duplicate_agent_ids(self):
        s1 = _minimal_sample(agent_id="dup")
        s2 = _minimal_sample(agent_id="dup", agent_name="Agent2", input="hi", output="bye")
        errors = _validate_journeys([_minimal_journey(samples=[s1, s2])])
        assert any("duplicate agent_ids" in e for e in errors)

    def test_null_parent_id_valid(self):
        sample = _minimal_sample(parent_id=None)
        errors = _validate_journeys([_minimal_journey(samples=[sample])])
        assert errors == []

    def test_parent_ids_array_valid(self):
        sample = _minimal_sample(parent_ids=["a1", "a2"])
        errors = _validate_journeys([_minimal_journey(samples=[sample])])
        assert errors == []

    def test_multiple_journeys_independent_validation(self):
        valid = _minimal_journey(trace_id="good")
        invalid = _minimal_journey(trace_id="bad", samples=[{"agent_id": "x"}])
        errors = _validate_journeys([valid, invalid])
        # Only journey[1] has errors
        assert all("journey[1]" in e for e in errors)
        assert not any("journey[0]" in e for e in errors)


# ===========================================================================
# Test Class 3: Flat Record Normalization (14 tests)
# ===========================================================================

class TestFlatRecordNormalization:
    def test_groups_by_trace_id(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y"},
            {"trace_id": "t1", "agent_id": "a2", "agent_name": "B", "input": "x", "output": "y"},
            {"trace_id": "t2", "agent_id": "a3", "agent_name": "C", "input": "x", "output": "y"},
            {"trace_id": "t2", "agent_id": "a4", "agent_name": "D", "input": "x", "output": "y"},
        ]
        journeys = normalize_flat_to_journeys(records)
        assert len(journeys) == 2
        trace_ids = {j["trace_id"] for j in journeys}
        assert trace_ids == {"t1", "t2"}
        for j in journeys:
            assert len(j["samples"]) == 2

    def test_preserves_journey_name(self):
        records = [
            {"trace_id": "t1", "journey_name": "My Journey", "agent_id": "a1",
             "agent_name": "A", "input": "x", "output": "y"},
        ]
        journeys = normalize_flat_to_journeys(records)
        assert journeys[0]["journey_name"] == "My Journey"

    def test_default_journey_name(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y"},
        ]
        journeys = normalize_flat_to_journeys(records)
        assert journeys[0]["journey_name"] == "Trace t1"

    def test_latency_ms_to_seconds(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y",
             "telemetry": {"latency_ms": 1500}},
        ]
        journeys = normalize_flat_to_journeys(records)
        sample = journeys[0]["samples"][0]
        assert sample["telemetry"]["latency"] == 1.5

    def test_ttft_ms_to_seconds(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y",
             "telemetry": {"ttft_ms": 200}},
        ]
        journeys = normalize_flat_to_journeys(records)
        sample = journeys[0]["samples"][0]
        assert sample["telemetry"]["ttft"] == 0.2

    def test_tokens_in_out_to_prompt_completion(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y",
             "telemetry": {"tokens_in": 100, "tokens_out": 50}},
        ]
        journeys = normalize_flat_to_journeys(records)
        tokens = journeys[0]["samples"][0]["telemetry"]["tokens"]
        assert tokens["prompt_tokens"] == 100
        assert tokens["completion_tokens"] == 50

    def test_metadata_model_to_model_parameters(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y",
             "metadata": {"model": "gpt-4o"}},
        ]
        journeys = normalize_flat_to_journeys(records)
        sample = journeys[0]["samples"][0]
        assert sample["model_parameters"]["model"] == "gpt-4o"

    def test_preserves_already_normalized_telemetry(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y",
             "telemetry": {"latency": 1.5}},
        ]
        journeys = normalize_flat_to_journeys(records)
        assert journeys[0]["samples"][0]["telemetry"]["latency"] == 1.5

    def test_skips_records_without_trace_id(self):
        records = [
            {"agent_id": "a1", "agent_name": "A", "input": "x", "output": "y"},
            {"trace_id": "t1", "agent_id": "a2", "agent_name": "B", "input": "x", "output": "y"},
        ]
        journeys = normalize_flat_to_journeys(records)
        assert len(journeys) == 1
        assert journeys[0]["trace_id"] == "t1"

    def test_preserves_parent_id(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y",
             "parent_id": "agent-1"},
        ]
        journeys = normalize_flat_to_journeys(records)
        assert journeys[0]["samples"][0]["parent_id"] == "agent-1"

    def test_preserves_parent_ids_array(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y",
             "parent_ids": ["a1", "a2"]},
        ]
        journeys = normalize_flat_to_journeys(records)
        assert journeys[0]["samples"][0]["parent_ids"] == ["a1", "a2"]

    def test_strips_trace_level_fields_from_sample(self):
        records = [
            {"trace_id": "t1", "journey_name": "JN", "agent_id": "a1",
             "agent_name": "A", "input": "x", "output": "y"},
        ]
        journeys = normalize_flat_to_journeys(records)
        sample = journeys[0]["samples"][0]
        assert "trace_id" not in sample
        assert "journey_name" not in sample

    def test_empty_records_array(self):
        journeys = normalize_flat_to_journeys([])
        assert journeys == []

    def test_mixed_telemetry_formats(self):
        records = [
            {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y",
             "telemetry": {"latency_ms": 1000}},
            {"trace_id": "t1", "agent_id": "a2", "agent_name": "B", "input": "x", "output": "y",
             "telemetry": {"latency": 2.5}},
        ]
        journeys = normalize_flat_to_journeys(records)
        samples = journeys[0]["samples"]
        assert samples[0]["telemetry"]["latency"] == 1.0
        assert samples[1]["telemetry"]["latency"] == 2.5


# ===========================================================================
# Test Class 4: End-to-End normalize_input (10 tests)
# ===========================================================================

class TestNormalizeInput:
    def test_journey_grouped_passthrough(self):
        data = {"journeys": [_minimal_journey()]}
        journeys, fmt = normalize_input(data)
        assert fmt == "journey_grouped"
        assert len(journeys) == 1
        assert journeys[0]["trace_id"] == "test-trace-1"

    def test_flat_records_normalized(self):
        data = {
            "records": [
                {"trace_id": "t1", "agent_id": "a1", "agent_name": "A", "input": "x", "output": "y"},
                {"trace_id": "t2", "agent_id": "a2", "agent_name": "B", "input": "x", "output": "y"},
            ]
        }
        journeys, fmt = normalize_input(data)
        assert fmt == "flat_records"
        assert len(journeys) == 2

    def test_single_journey_wrapped(self):
        data = {
            "trace_id": "t1",
            "journey_name": "Single",
            "samples": [_minimal_sample()],
        }
        journeys, fmt = normalize_input(data)
        assert fmt == "single_journey"
        assert len(journeys) == 1
        assert journeys[0]["trace_id"] == "t1"

    def test_validation_errors_aggregated(self):
        data = {
            "journeys": [
                _minimal_journey(trace_id="good"),
                _minimal_journey(trace_id="bad1", samples=[{"agent_id": "x"}]),
                _minimal_journey(trace_id="bad2", samples=[{"agent_id": "y"}]),
            ]
        }
        with pytest.raises(ValidationError) as exc_info:
            normalize_input(data)
        # Errors from both bad journeys
        assert len(exc_info.value.errors) >= 2

    def test_idempotent_normalization(self):
        data = {"journeys": [_minimal_journey()]}
        j1, f1 = normalize_input(data)
        # Re-wrap and normalize again
        data2 = {"journeys": j1}
        j2, f2 = normalize_input(data2)
        assert f1 == f2
        assert len(j1) == len(j2)
        assert j1[0]["trace_id"] == j2[0]["trace_id"]

    def test_real_sample_data_validates(self):
        """Real sample data may have intentionally null outputs (e.g. PII guardrail).
        Filter those out and validate the rest passes."""
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")
        with open(SAMPLE_DATA_PATH) as f:
            data = json.load(f)
        # Filter out journeys with intentionally null outputs (guardrail blocked)
        valid_journeys = []
        for j in data["journeys"]:
            if all(s.get("output") for s in j.get("samples", [])):
                valid_journeys.append(j)
        filtered = {"journeys": valid_journeys}
        journeys, fmt = normalize_input(filtered)
        assert fmt == "journey_grouped"
        assert len(journeys) >= 9  # 10 total minus 1 guardrail-blocked

    def test_real_flat_data_validates(self):
        if not FLAT_DATA_PATH.exists():
            pytest.skip("Flat data not available")
        with open(FLAT_DATA_PATH) as f:
            data = json.load(f)
        if "records" not in data:
            pytest.skip("data.json is not flat records format")
        journeys, fmt = normalize_input(data)
        assert fmt == "flat_records"
        assert len(journeys) > 0

    def test_large_journey_50_agents(self):
        samples = [
            _minimal_sample(agent_id=f"agent-{i}", agent_name=f"Agent{i}")
            for i in range(50)
        ]
        data = {"journeys": [_minimal_journey(samples=samples)]}
        journeys, fmt = normalize_input(data)
        assert len(journeys[0]["samples"]) == 50

    def test_unicode_in_agent_fields(self):
        sample = _minimal_sample(
            input="東京からパリへのフライトを予約してください 🛫",
            output="フライトが見つかりました！ ✈️",
            agent_name="日本語エージェント",
        )
        data = {"journeys": [_minimal_journey(samples=[sample])]}
        journeys, _ = normalize_input(data)
        assert journeys[0]["samples"][0]["input"] == "東京からパリへのフライトを予約してください 🛫"

    def test_empty_string_fields_fail_validation(self):
        sample = _minimal_sample(input="", output="")
        data = {"journeys": [_minimal_journey(samples=[sample])]}
        with pytest.raises(ValidationError) as exc_info:
            normalize_input(data)
        assert any("'input' cannot be empty" in e for e in exc_info.value.errors)
        assert any("'output' cannot be empty" in e for e in exc_info.value.errors)


# ===========================================================================
# Test Class 5: Agent ID Generation (5 tests)
# ===========================================================================

class TestAgentIdGeneration:
    def test_auto_generates_missing_agent_id(self):
        sample = _minimal_sample()
        del sample["agent_id"]
        journey = _minimal_journey(samples=[sample])
        errors = _validate_journeys([journey])
        assert errors == []
        assert journey["samples"][0]["agent_id"] == "test-trace-1-agent-1"

    def test_preserves_existing_agent_id(self):
        sample = _minimal_sample(agent_id="my-custom-id")
        journey = _minimal_journey(samples=[sample])
        errors = _validate_journeys([journey])
        assert errors == []
        assert journey["samples"][0]["agent_id"] == "my-custom-id"

    def test_mixed_present_and_missing(self):
        s1 = _minimal_sample(agent_id="existing")
        s2 = _minimal_sample(agent_name="Agent2", input="a", output="b")
        del s2["agent_id"]
        s3 = _minimal_sample(agent_name="Agent3", input="c", output="d")
        del s3["agent_id"]
        journey = _minimal_journey(samples=[s1, s2, s3])
        errors = _validate_journeys([journey])
        assert errors == []
        assert journey["samples"][0]["agent_id"] == "existing"
        assert journey["samples"][1]["agent_id"] == "test-trace-1-agent-1"
        assert journey["samples"][2]["agent_id"] == "test-trace-1-agent-2"

    def test_counter_is_per_journey(self):
        s1 = _minimal_sample()
        del s1["agent_id"]
        s2 = _minimal_sample(agent_name="Agent2", input="a", output="b")
        del s2["agent_id"]
        j1 = _minimal_journey(trace_id="t1", samples=[s1])
        j2 = _minimal_journey(trace_id="t2", samples=[s2])
        errors = _validate_journeys([j1, j2])
        assert errors == []
        assert j1["samples"][0]["agent_id"] == "t1-agent-1"
        assert j2["samples"][0]["agent_id"] == "t2-agent-1"

    def test_generated_ids_are_unique(self):
        samples = []
        for i in range(5):
            s = _minimal_sample(agent_name=f"Agent{i}", input=f"in{i}", output=f"out{i}")
            del s["agent_id"]
            samples.append(s)
        journey = _minimal_journey(samples=samples)
        errors = _validate_journeys([journey])
        assert errors == []
        ids = [s["agent_id"] for s in journey["samples"]]
        assert len(ids) == len(set(ids))


# ===========================================================================
# Test Class 6: Edge Cases (8 tests)
# ===========================================================================

class TestEdgeCases:
    def test_extra_fields_preserved(self):
        """Extra fields in samples should pass through validation."""
        sample = _minimal_sample(custom_score=0.9, custom_metadata={"key": "val"})
        journey = _minimal_journey(samples=[sample])
        errors = _validate_journeys([journey])
        assert errors == []
        assert journey["samples"][0]["custom_score"] == 0.9

    def test_deeply_nested_core_payload(self):
        sample = _minimal_sample(core_payload={
            "tool_calls": [{"arguments": {"nested": {"deep": True}}}],
            "reasoning_blocks": [],
        })
        journey = _minimal_journey(samples=[sample])
        errors = _validate_journeys([journey])
        assert errors == []

    def test_very_long_input_output(self):
        long_text = "x" * 50_000
        sample = _minimal_sample(input=long_text, output=long_text)
        journey = _minimal_journey(samples=[sample])
        errors = _validate_journeys([journey])
        assert errors == []
        assert len(journey["samples"][0]["input"]) == 50_000

    def test_null_values_in_optional_fields(self):
        sample = _minimal_sample(telemetry=None, metadata=None, core_payload=None)
        journey = _minimal_journey(samples=[sample])
        errors = _validate_journeys([journey])
        assert errors == []

    def test_integer_trace_id_coerced(self):
        journey = _minimal_journey(trace_id=12345)
        errors = _validate_journeys([journey])
        assert errors == []
        assert journey["trace_id"] == "12345"

    def test_whitespace_only_fields(self):
        sample = _minimal_sample(input="   ", output="\t\n")
        journey = _minimal_journey(samples=[sample])
        errors = _validate_journeys([journey])
        assert any("'input' cannot be empty" in e for e in errors)
        assert any("'output' cannot be empty" in e for e in errors)

    def test_single_agent_journey(self):
        sample = _minimal_sample(parent_id=None)
        journey = _minimal_journey(samples=[sample])
        errors = _validate_journeys([journey])
        assert errors == []

    def test_diamond_pattern_parent_ids(self):
        root = _minimal_sample(agent_id="root", agent_name="Root", parent_id=None)
        left = _minimal_sample(agent_id="left", agent_name="Left", input="a", output="b", parent_id="root")
        right = _minimal_sample(agent_id="right", agent_name="Right", input="c", output="d", parent_id="root")
        merge = _minimal_sample(agent_id="merge", agent_name="Merge", input="e", output="f", parent_ids=["left", "right"])
        journey = _minimal_journey(samples=[root, left, right, merge])
        errors = _validate_journeys([journey])
        assert errors == []


# ===========================================================================
# Test Class 7: Sample Data Source (6 tests)
# ===========================================================================

class TestSampleDataSource:
    """Tests for the sample data loading path.

    These test the views_v1._load_sample_data function and the
    POST /api/v1/traces endpoint with source="sample".
    Uses Django test client for integration tests.
    """

    def test_source_sample_loads_all(self):
        from api.views_v1 import _load_sample_data
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")
        journeys = _load_sample_data()
        assert len(journeys) >= 10

    def test_source_sample_filter_by_trace_ids(self):
        from api.views_v1 import _load_sample_data
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")
        journeys = _load_sample_data(trace_ids=["tr-a7f2-9b3c-4e1d-8f6a"])
        assert len(journeys) == 1
        assert journeys[0]["trace_id"] == "tr-a7f2-9b3c-4e1d-8f6a"

    def test_source_sample_invalid_trace_id(self):
        from api.views_v1 import _load_sample_data
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")
        with pytest.raises(ValidationError) as exc_info:
            _load_sample_data(trace_ids=["nonexistent"])
        assert "not found in sample data" in exc_info.value.errors[0]

    def test_source_sample_with_auto_evaluate(self):
        """Test that auto_evaluate flag is accepted (doesn't actually run LLM)."""
        from django.test import RequestFactory
        from api.views_v1 import traces_list_create
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")

        factory = RequestFactory()
        # auto_evaluate without user_goal should just upload without evaluating
        request = factory.post(
            "/api/v1/traces",
            data=json.dumps({
                "source": "sample",
                "trace_ids": ["tr-a7f2-9b3c-4e1d-8f6a"],
                "auto_evaluate": True,
                # No user_goal — evaluation should be skipped
            }),
            content_type="application/json",
        )
        response = traces_list_create(request)
        data = json.loads(response.content)
        assert data["success"] is True
        assert data["data"]["source"] == "sample"
        # No evaluation since user_goal is missing
        assert "evaluation" not in data["data"]

    def test_user_data_not_mixed_with_sample(self):
        """User-provided data should be used when no source field."""
        data = {"journeys": [_minimal_journey(trace_id="user-custom-1")]}
        journeys, fmt = normalize_input(data)
        assert fmt == "journey_grouped"
        assert journeys[0]["trace_id"] == "user-custom-1"

    def test_source_sample_idempotent(self):
        """Loading sample data twice should update, not duplicate."""
        from django.test import RequestFactory
        from api.views_v1 import traces_list_create
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")

        # Clean slate — remove any existing trace from prior tests
        ExecutionTrace.objects.filter(trace_id="tr-a7f2-9b3c-4e1d-8f6a").delete()

        factory = RequestFactory()
        payload = json.dumps({
            "source": "sample",
            "trace_ids": ["tr-a7f2-9b3c-4e1d-8f6a"],
        })

        # First call — should create
        req1 = factory.post("/api/v1/traces", data=payload, content_type="application/json")
        resp1 = traces_list_create(req1)
        data1 = json.loads(resp1.content)

        # Second call — should update
        req2 = factory.post("/api/v1/traces", data=payload, content_type="application/json")
        resp2 = traces_list_create(req2)
        data2 = json.loads(resp2.content)

        assert data1["data"]["created"] == 1
        assert data2["data"]["updated"] == 1
        assert data2["data"]["created"] == 0

        # Only one trace in DB
        count = ExecutionTrace.objects.filter(trace_id="tr-a7f2-9b3c-4e1d-8f6a").count()
        assert count == 1
