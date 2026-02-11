"""
OpenAI-compatible proxy API.

Proxies chat completion requests to OpenAI while capturing telemetry
and storing execution traces for DAG construction and evaluation.

Usage (one-line SDK change):
    client = OpenAI(base_url="https://your-api.com/api/v1/proxy")

Endpoint:
    POST /api/v1/proxy/chat/completions

Custom headers (consumed by proxy, not forwarded):
    X-Trace-Id       — Required. Groups calls into a journey.
    X-Agent-Name     — Optional. Human-readable agent name (default: model name).
    X-Parent-Agent-Id — Optional. Parent agent for DAG edges (default: null/root).
    X-Journey-Name   — Optional. Journey display name (default: "Trace {trace_id}").
"""
from __future__ import annotations

import json
import logging
import os
import time

import httpx
from django.db import transaction
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators.http import require_http_methods

from .models import AgentExecution, ExecutionTrace
from .response import api_error

logger = logging.getLogger(__name__)

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")

# Module-level httpx client for connection pooling (thread-safe)
_http_client: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(timeout=120.0)
    return _http_client


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@require_http_methods(["POST"])
def chat_completions(request):
    """Proxy a chat completions request to OpenAI, capture telemetry, store trace."""

    # --- Extract and validate headers ---
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        return api_error("Missing Authorization header", code="missing_auth", status=401)

    trace_id = request.headers.get("X-Trace-Id", "")
    if not trace_id:
        return api_error(
            "X-Trace-Id header is required to group calls into a journey",
            code="missing_trace_id",
            status=400,
        )

    agent_name = request.headers.get("X-Agent-Name", "")
    parent_agent_id = request.headers.get("X-Parent-Agent-Id", "") or None
    journey_name = request.headers.get("X-Journey-Name", "") or f"Trace {trace_id}"

    # --- Parse request body ---
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return api_error("Invalid JSON body", code="invalid_json", status=400)

    # Default agent_name to model if not provided
    if not agent_name:
        agent_name = body.get("model", "unknown")

    # Extract input: last user message
    input_text = _extract_user_input(body.get("messages", []))

    # --- Determine streaming vs non-streaming ---
    is_streaming = body.get("stream", False)

    if is_streaming:
        return _handle_streaming(body, auth_header, trace_id, agent_name, parent_agent_id, journey_name, input_text)
    else:
        return _handle_non_streaming(body, auth_header, trace_id, agent_name, parent_agent_id, journey_name, input_text)


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------

def _handle_non_streaming(body, auth_header, trace_id, agent_name, parent_agent_id, journey_name, input_text):
    """Forward request to OpenAI, capture telemetry, store trace, return response unchanged."""
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }

    t_start = time.time()
    try:
        resp = _get_http_client().post(OPENAI_BASE_URL, json=body, headers=headers)
    except httpx.ConnectError:
        return api_error("Failed to connect to upstream provider", code="upstream_connect_error", status=502)
    except httpx.TimeoutException:
        return api_error("Upstream request timed out", code="upstream_timeout", status=504)
    except httpx.HTTPError:
        return api_error("Upstream request failed", code="upstream_error", status=502)
    t_end = time.time()

    latency_s = t_end - t_start

    # Parse OpenAI response for telemetry (best-effort)
    output_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    model_used = body.get("model", "")

    if resp.status_code == 200:
        try:
            resp_json = resp.json()
            choices = resp_json.get("choices", [])
            if choices:
                output_text = choices[0].get("message", {}).get("content", "") or ""
            usage = resp_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            model_used = resp_json.get("model", model_used)
        except Exception:
            logger.warning("Failed to parse OpenAI response for telemetry", exc_info=True)

    # Persist trace (best-effort)
    _persist_proxy_call(
        trace_id=trace_id,
        journey_name=journey_name,
        agent_name=agent_name,
        parent_id=parent_agent_id,
        input_text=input_text,
        output_text=output_text,
        model=model_used,
        latency_s=latency_s,
        ttft_s=latency_s,  # For non-streaming, TTFT ≈ total latency
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    # Return OpenAI response unchanged
    return HttpResponse(
        resp.content,
        status=resp.status_code,
        content_type=resp.headers.get("content-type", "application/json"),
    )


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------

def _handle_streaming(body, auth_header, trace_id, agent_name, parent_agent_id, journey_name, input_text):
    """Stream chunks from OpenAI through to client, accumulate output, persist after stream ends."""
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }

    # Inject stream_options to get usage in the final chunk
    if "stream_options" not in body:
        body["stream_options"] = {"include_usage": True}
    elif not body["stream_options"].get("include_usage"):
        body["stream_options"]["include_usage"] = True

    # State accumulated across chunks
    state = {
        "full_output": [],
        "t_start": time.time(),
        "first_chunk_time": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "model": body.get("model", ""),
    }

    def stream_generator():
        try:
            with _get_http_client().stream("POST", OPENAI_BASE_URL, json=body, headers=headers) as upstream:
                # If OpenAI returns an error status, yield the full body and return
                if upstream.status_code != 200:
                    for chunk in upstream.iter_bytes():
                        yield chunk
                    return

                for line in upstream.iter_lines():
                    if not line:
                        continue

                    # Yield SSE line to client immediately
                    yield line + "\n\n"

                    # Parse for telemetry accumulation
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            continue
                        try:
                            chunk_data = json.loads(data_str)

                            # Record TTFT on first content chunk
                            if state["first_chunk_time"] is None:
                                state["first_chunk_time"] = time.time()

                            # Accumulate content
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    state["full_output"].append(content)

                            # Extract usage from final chunk
                            usage = chunk_data.get("usage")
                            if usage:
                                state["prompt_tokens"] = usage.get("prompt_tokens", 0)
                                state["completion_tokens"] = usage.get("completion_tokens", 0)

                            # Extract model
                            model = chunk_data.get("model")
                            if model:
                                state["model"] = model
                        except (json.JSONDecodeError, KeyError):
                            pass
        except httpx.ConnectError:
            yield f'data: {json.dumps({"error": {"message": "Failed to connect to upstream provider", "type": "proxy_error"}})}\n\n'
            yield "data: [DONE]\n\n"
            return
        except httpx.TimeoutException:
            yield f'data: {json.dumps({"error": {"message": "Upstream request timed out", "type": "proxy_error"}})}\n\n'
            yield "data: [DONE]\n\n"
            return
        except httpx.HTTPError:
            yield f'data: {json.dumps({"error": {"message": "Upstream request failed", "type": "proxy_error"}})}\n\n'
            yield "data: [DONE]\n\n"
            return
        finally:
            # Persist after stream ends (best-effort)
            t_end = time.time()
            ttft_s = (state["first_chunk_time"] - state["t_start"]) if state["first_chunk_time"] else 0
            latency_s = t_end - state["t_start"]

            try:
                _persist_proxy_call(
                    trace_id=trace_id,
                    journey_name=journey_name,
                    agent_name=agent_name,
                    parent_id=parent_agent_id,
                    input_text=input_text,
                    output_text="".join(state["full_output"]),
                    model=state["model"],
                    latency_s=latency_s,
                    ttft_s=ttft_s,
                    prompt_tokens=state["prompt_tokens"],
                    completion_tokens=state["completion_tokens"],
                )
            except Exception:
                logger.warning("Failed to persist streaming proxy call", exc_info=True)

    response = StreamingHttpResponse(
        stream_generator(),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_user_input(messages: list[dict]) -> str:
    """Extract the last user message content from the messages array."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Handle content as string or list of content parts
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Concatenate text parts
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                return " ".join(parts)
    return ""


def _persist_proxy_call(
    *,
    trace_id: str,
    journey_name: str,
    agent_name: str,
    parent_id: str | None,
    input_text: str,
    output_text: str,
    model: str,
    latency_s: float,
    ttft_s: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Persist a proxy call as an agent execution within a trace.

    Best-effort: logs warnings on failure, never raises.
    """
    try:
        # Ensure the trace exists (create outside the lock to avoid deadlocks)
        ExecutionTrace.objects.get_or_create(
            trace_id=trace_id,
            defaults={
                "journey_name": journey_name,
                "workflow_type": "proxy",
                "status": "active",
                "raw_journey": {"trace_id": trace_id, "journey_name": journey_name, "samples": []},
            },
        )

        # Atomic block: lock the trace row so concurrent calls get sequential IDs
        with transaction.atomic():
            trace_obj = ExecutionTrace.objects.select_for_update().get(trace_id=trace_id)

            # Update journey_name if it was set explicitly (not the default)
            if journey_name != f"Trace {trace_id}" and trace_obj.journey_name != journey_name:
                trace_obj.journey_name = journey_name

            # Generate agent_id based on existing sample count (safe under lock)
            existing_count = trace_obj.agents.count()
            agent_id = f"{trace_id}-call-{existing_count + 1}"

            # Build sample dict for raw_journey
            sample = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "parent_id": parent_id,
                "input": input_text,
                "output": output_text,
                "model_parameters": {"model": model},
                "telemetry": {
                    "ttft": round(ttft_s, 4),
                    "latency": round(latency_s, 4),
                    "tokens": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                },
            }

            # Append to raw_journey samples (safe under lock)
            raw = trace_obj.raw_journey or {"trace_id": trace_id, "journey_name": journey_name, "samples": []}
            raw.setdefault("samples", []).append(sample)
            trace_obj.raw_journey = raw
            trace_obj.save(update_fields=["raw_journey", "journey_name"])

            # Create AgentExecution row
            AgentExecution.objects.create(
                trace=trace_obj,
                agent_id=agent_id,
                agent_name=agent_name,
                parent_id=parent_id or "",
                input_text=input_text,
                output_text=output_text,
                latency_ms=round(latency_s * 1000, 1),
                ttft_ms=round(ttft_s * 1000, 1),
                tokens=prompt_tokens + completion_tokens,
                model=model,
                status="completed",
            )

        logger.info("Persisted proxy call: trace=%s agent=%s (%s)", trace_id, agent_id, agent_name)

    except Exception:
        logger.warning("Failed to persist proxy call for trace %s", trace_id, exc_info=True)
