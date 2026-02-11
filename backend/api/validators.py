"""
Input format detection, validation, and normalization for the v1 Trace API.

Accepts three input formats:
1. Journey-grouped: {"journeys": [{trace_id, samples: [...]}]}
2. Flat records:    {"records": [{trace_id, agent_id, input, output, ...}]}
3. Single journey:  {"trace_id": "...", "samples": [...]}

All are normalized to journey-grouped internally.
"""
from __future__ import annotations


class ValidationError(Exception):
    """Raised when input data fails validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# Format Detection
# ---------------------------------------------------------------------------

def detect_format(data: dict) -> str:
    """Detect the input format from the top-level keys.

    Returns one of: "journey_grouped", "flat_records", "single_journey"
    Raises ValidationError if the format cannot be determined.
    """
    if not isinstance(data, dict):
        raise ValidationError(["Input must be a JSON object"])

    if "journeys" in data:
        if not isinstance(data["journeys"], list):
            raise ValidationError(["'journeys' must be an array"])
        return "journey_grouped"

    if "records" in data:
        if not isinstance(data["records"], list):
            raise ValidationError(["'records' must be an array"])
        return "flat_records"

    if "trace_id" in data and "samples" in data:
        return "single_journey"

    raise ValidationError(
        ["Unrecognized format. Expected 'journeys', 'records', or 'trace_id'+'samples' keys"]
    )


# ---------------------------------------------------------------------------
# Normalization (flat records → journey-grouped)
# ---------------------------------------------------------------------------

def normalize_flat_to_journeys(records: list[dict]) -> list[dict]:
    """Convert flat records to journey-grouped format.

    Groups records by trace_id, converts telemetry units:
    - latency_ms → latency (seconds)
    - ttft_ms → ttft (seconds)
    - tokens_in/tokens_out → tokens.prompt_tokens/completion_tokens
    - metadata.model → model_parameters.model
    """
    if not records:
        return []

    # Group by trace_id
    grouped: dict[str, dict] = {}
    for record in records:
        trace_id = record.get("trace_id")
        if not trace_id:
            continue  # skip records without trace_id

        trace_id = str(trace_id)
        if trace_id not in grouped:
            grouped[trace_id] = {
                "trace_id": trace_id,
                "journey_name": record.get("journey_name", f"Trace {trace_id}"),
                "samples": [],
            }

        # Build normalized sample
        sample = {}

        # Copy core fields
        for field in ("agent_id", "agent_name", "input", "output", "parent_id", "parent_ids",
                       "core_payload", "evaluation_signals", "agent_turn"):
            if field in record:
                sample[field] = record[field]

        # Normalize telemetry
        raw_telemetry = record.get("telemetry", {}) or {}
        telemetry = {}

        # latency: prefer already-normalized (seconds), convert from ms
        if "latency" in raw_telemetry:
            telemetry["latency"] = raw_telemetry["latency"]
        elif "latency_ms" in raw_telemetry:
            telemetry["latency"] = raw_telemetry["latency_ms"] / 1000.0

        # ttft: same logic
        if "ttft" in raw_telemetry:
            telemetry["ttft"] = raw_telemetry["ttft"]
        elif "ttft_ms" in raw_telemetry:
            telemetry["ttft"] = raw_telemetry["ttft_ms"] / 1000.0

        # tokens: convert flat tokens_in/out to nested
        if "tokens" in raw_telemetry:
            telemetry["tokens"] = raw_telemetry["tokens"]
        elif "tokens_in" in raw_telemetry or "tokens_out" in raw_telemetry:
            telemetry["tokens"] = {
                "prompt_tokens": raw_telemetry.get("tokens_in", 0),
                "completion_tokens": raw_telemetry.get("tokens_out", 0),
            }

        if telemetry:
            sample["telemetry"] = telemetry

        # model: metadata.model → model_parameters.model
        metadata = record.get("metadata", {}) or {}
        if metadata.get("model"):
            sample["model_parameters"] = {"model": metadata["model"]}
        elif "model_parameters" in record:
            sample["model_parameters"] = record["model_parameters"]

        # Preserve any extra fields not already handled
        skip_keys = {
            "trace_id", "journey_name", "agent_id", "agent_name", "input", "output",
            "parent_id", "parent_ids", "telemetry", "metadata", "model_parameters",
            "core_payload", "evaluation_signals", "agent_turn",
        }
        for key, value in record.items():
            if key not in skip_keys and key not in sample:
                sample[key] = value

        grouped[trace_id]["samples"].append(sample)

    return list(grouped.values())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_SAMPLE_FIELDS = ("agent_name", "input", "output")


def _validate_journeys(journeys: list[dict]) -> list[str]:
    """Validate a list of journeys. Returns list of error strings."""
    errors = []

    for i, journey in enumerate(journeys):
        prefix = f"journey[{i}]"

        # trace_id required
        trace_id = journey.get("trace_id")
        if not trace_id:
            errors.append(f"{prefix}: missing required field 'trace_id'")

        # Coerce trace_id to string
        if trace_id is not None and not isinstance(trace_id, str):
            journey["trace_id"] = str(trace_id)

        # samples required and non-empty
        samples = journey.get("samples")
        if not isinstance(samples, list):
            errors.append(f"{prefix}: 'samples' must be an array")
            continue
        if len(samples) == 0:
            errors.append(f"{prefix}: no samples provided")
            continue

        # Per-sample validation
        seen_agent_ids = set()
        agent_id_counter = 0

        for j, sample in enumerate(samples):
            sample_prefix = f"{prefix}.samples[{j}]"

            # Auto-generate agent_id if missing
            agent_id = sample.get("agent_id")
            if not agent_id:
                agent_id_counter += 1
                tid = journey.get("trace_id", f"journey-{i}")
                sample["agent_id"] = f"{tid}-agent-{agent_id_counter}"
            else:
                if agent_id in seen_agent_ids:
                    errors.append(f"{sample_prefix}: duplicate agent_ids '{agent_id}'")
                seen_agent_ids.add(agent_id)

            # Required fields
            for field in REQUIRED_SAMPLE_FIELDS:
                value = sample.get(field)
                if value is None:
                    errors.append(f"{sample_prefix}: missing required field '{field}'")
                elif isinstance(value, str) and not value.strip():
                    errors.append(f"{sample_prefix}: '{field}' cannot be empty")

    return errors


# ---------------------------------------------------------------------------
# Top-Level Normalization
# ---------------------------------------------------------------------------

def normalize_input(data: dict) -> tuple[list[dict], str]:
    """Detect format, normalize to journey-grouped, validate.

    Returns (journeys_list, format_detected).
    Raises ValidationError on validation failure.
    """
    fmt = detect_format(data)

    if fmt == "journey_grouped":
        journeys = data["journeys"]
    elif fmt == "flat_records":
        journeys = normalize_flat_to_journeys(data["records"])
    elif fmt == "single_journey":
        journeys = [data]
    else:
        raise ValidationError([f"Unknown format: {fmt}"])

    # Validate all journeys
    errors = _validate_journeys(journeys)
    if errors:
        raise ValidationError(errors)

    return journeys, fmt
