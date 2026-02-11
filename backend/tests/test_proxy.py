"""
Tests for the OpenAI-compatible proxy API.

Covers:
- Header validation (missing auth, missing trace-id)
- Non-streaming proxy with telemetry capture and trace persistence
- Streaming proxy with chunk pass-through and post-stream persistence
- Multi-agent DAG construction via X-Parent-Agent-Id
- Upstream error pass-through (4xx/5xx forwarded unchanged)
- Network error handling (502/504)
- Input extraction from messages array
- Custom headers NOT forwarded to OpenAI
- stream_options injection for token counts
"""
import json
from unittest.mock import MagicMock, patch

from django.test import TestCase, RequestFactory

from api.models import AgentExecution, ExecutionTrace
from api.views_proxy import (
    _extract_user_input,
    _persist_proxy_call,
    chat_completions,
)


# ---------------------------------------------------------------------------
# Helper: mock OpenAI response
# ---------------------------------------------------------------------------

def _openai_response(content="Hello!", model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5):
    """Build a realistic OpenAI chat completions response dict."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _make_request(factory, body, headers=None):
    """Build a POST request with JSON body and custom headers."""
    headers = headers or {}
    request = factory.post(
        "/api/v1/proxy/chat/completions",
        data=json.dumps(body),
        content_type="application/json",
    )
    # Django test RequestFactory uses META for headers
    for key, value in headers.items():
        meta_key = "HTTP_" + key.upper().replace("-", "_")
        request.META[meta_key] = value
    return request


def _mock_http_client(mock_get_fn, *, post_return=None, post_side_effect=None, stream_return=None):
    """Configure a mock _get_http_client to return a mock client with desired behavior."""
    mock_client = MagicMock()
    if post_return is not None:
        mock_client.post.return_value = post_return
    if post_side_effect is not None:
        mock_client.post.side_effect = post_side_effect
    if stream_return is not None:
        mock_client.stream.return_value = stream_return
    mock_get_fn.return_value = mock_client
    return mock_client


# ---------------------------------------------------------------------------
# Unit tests: _extract_user_input
# ---------------------------------------------------------------------------

class TestExtractUserInput(TestCase):
    def test_simple_string_content(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello world"},
        ]
        self.assertEqual(_extract_user_input(messages), "Hello world")

    def test_last_user_message_wins(self):
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up question"},
        ]
        self.assertEqual(_extract_user_input(messages), "Follow-up question")

    def test_no_user_messages(self):
        messages = [{"role": "system", "content": "You are helpful."}]
        self.assertEqual(_extract_user_input(messages), "")

    def test_empty_messages(self):
        self.assertEqual(_extract_user_input([]), "")

    def test_multipart_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        self.assertEqual(_extract_user_input(messages), "Describe this image:")


# ---------------------------------------------------------------------------
# Unit tests: _persist_proxy_call
# ---------------------------------------------------------------------------

class TestPersistProxyCall(TestCase):
    def test_creates_trace_and_agent(self):
        _persist_proxy_call(
            trace_id="test-persist-1",
            journey_name="Test Journey",
            agent_name="TestAgent",
            parent_id=None,
            input_text="hello",
            output_text="world",
            model="gpt-4o-mini",
            latency_s=0.5,
            ttft_s=0.1,
            prompt_tokens=10,
            completion_tokens=5,
        )

        trace = ExecutionTrace.objects.get(trace_id="test-persist-1")
        self.assertEqual(trace.journey_name, "Test Journey")
        self.assertEqual(trace.workflow_type, "proxy")
        self.assertEqual(len(trace.raw_journey["samples"]), 1)

        agent = trace.agents.first()
        self.assertEqual(agent.agent_name, "TestAgent")
        self.assertEqual(agent.agent_id, "test-persist-1-call-1")
        self.assertEqual(agent.input_text, "hello")
        self.assertEqual(agent.output_text, "world")
        self.assertEqual(agent.tokens, 15)
        self.assertAlmostEqual(agent.latency_ms, 500.0, places=0)
        self.assertAlmostEqual(agent.ttft_ms, 100.0, places=0)

    def test_appends_to_existing_trace(self):
        # First call
        _persist_proxy_call(
            trace_id="test-persist-2",
            journey_name="Multi-Agent",
            agent_name="Agent1",
            parent_id=None,
            input_text="q1",
            output_text="a1",
            model="gpt-4o-mini",
            latency_s=0.3,
            ttft_s=0.05,
            prompt_tokens=5,
            completion_tokens=3,
        )
        # Second call
        _persist_proxy_call(
            trace_id="test-persist-2",
            journey_name="Multi-Agent",
            agent_name="Agent2",
            parent_id="test-persist-2-call-1",
            input_text="q2",
            output_text="a2",
            model="gpt-4o",
            latency_s=0.8,
            ttft_s=0.1,
            prompt_tokens=20,
            completion_tokens=10,
        )

        trace = ExecutionTrace.objects.get(trace_id="test-persist-2")
        self.assertEqual(trace.agents.count(), 2)
        self.assertEqual(len(trace.raw_journey["samples"]), 2)

        agent2 = trace.agents.get(agent_name="Agent2")
        self.assertEqual(agent2.parent_id, "test-persist-2-call-1")
        self.assertEqual(agent2.agent_id, "test-persist-2-call-2")

    def test_does_not_raise_on_failure(self):
        """Persistence is best-effort — should log, not raise."""
        with patch("api.views_proxy.ExecutionTrace.objects") as mock_qs:
            mock_qs.get_or_create.side_effect = Exception("DB down")
            # Should not raise
            _persist_proxy_call(
                trace_id="fail-test",
                journey_name="X",
                agent_name="A",
                parent_id=None,
                input_text="",
                output_text="",
                model="gpt-4o-mini",
                latency_s=0,
                ttft_s=0,
                prompt_tokens=0,
                completion_tokens=0,
            )


# ---------------------------------------------------------------------------
# Integration tests: chat_completions endpoint
# ---------------------------------------------------------------------------

class TestChatCompletionsEndpoint(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_missing_auth_returns_401(self):
        request = _make_request(
            self.factory,
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Trace-Id": "t1"},
        )
        resp = chat_completions(request)
        self.assertEqual(resp.status_code, 401)
        data = json.loads(resp.content)
        self.assertFalse(data["success"])
        self.assertIn("Authorization", data["error"]["message"])

    def test_missing_trace_id_returns_400(self):
        request = _make_request(
            self.factory,
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-test123"},
        )
        resp = chat_completions(request)
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.content)
        self.assertIn("X-Trace-Id", data["error"]["message"])

    def test_invalid_json_returns_400(self):
        request = self.factory.post(
            "/api/v1/proxy/chat/completions",
            data="not json",
            content_type="application/json",
        )
        request.META["HTTP_AUTHORIZATION"] = "Bearer sk-test"
        request.META["HTTP_X_TRACE_ID"] = "t1"
        resp = chat_completions(request)
        self.assertEqual(resp.status_code, 400)

    @patch("api.views_proxy._get_http_client")
    def test_non_streaming_proxies_and_persists(self, mock_get_client):
        """Non-streaming: forwards to OpenAI, returns response, stores trace."""
        openai_resp = _openai_response("Hello from proxy!", prompt_tokens=12, completion_tokens=8)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(openai_resp).encode()
        mock_resp.json.return_value = openai_resp
        mock_resp.headers = {"content-type": "application/json"}

        _mock_http_client(mock_get_client, post_return=mock_resp)

        request = _make_request(
            self.factory,
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Say hello"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Trace-Id": "proxy-test-1",
                "X-Agent-Name": "Greeter",
                "X-Journey-Name": "Greeting Flow",
            },
        )
        resp = chat_completions(request)

        # Response is forwarded unchanged
        self.assertEqual(resp.status_code, 200)
        resp_data = json.loads(resp.content)
        self.assertEqual(resp_data["choices"][0]["message"]["content"], "Hello from proxy!")

        # Trace was persisted
        trace = ExecutionTrace.objects.get(trace_id="proxy-test-1")
        self.assertEqual(trace.journey_name, "Greeting Flow")
        self.assertEqual(trace.workflow_type, "proxy")

        agent = trace.agents.first()
        self.assertEqual(agent.agent_name, "Greeter")
        self.assertEqual(agent.input_text, "Say hello")
        self.assertEqual(agent.output_text, "Hello from proxy!")
        self.assertEqual(agent.tokens, 20)  # 12 + 8

    @patch("api.views_proxy._get_http_client")
    def test_non_streaming_upstream_error_forwarded(self, mock_get_client):
        """Upstream 4xx/5xx errors are forwarded unchanged."""
        error_body = {"error": {"message": "Invalid API key", "type": "invalid_request_error"}}
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.content = json.dumps(error_body).encode()
        mock_resp.json.return_value = error_body
        mock_resp.headers = {"content-type": "application/json"}

        _mock_http_client(mock_get_client, post_return=mock_resp)

        request = _make_request(
            self.factory,
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "Authorization": "Bearer sk-bad",
                "X-Trace-Id": "err-test",
            },
        )
        resp = chat_completions(request)
        self.assertEqual(resp.status_code, 401)
        data = json.loads(resp.content)
        self.assertEqual(data["error"]["message"], "Invalid API key")

    @patch("api.views_proxy._get_http_client")
    def test_network_error_returns_502(self, mock_get_client):
        """Connection failure returns 502."""
        import httpx
        _mock_http_client(mock_get_client, post_side_effect=httpx.ConnectError("Connection refused"))

        request = _make_request(
            self.factory,
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Trace-Id": "net-err-test",
            },
        )
        resp = chat_completions(request)
        self.assertEqual(resp.status_code, 502)
        data = json.loads(resp.content)
        # Error message should not leak internal details
        self.assertNotIn("Connection refused", data["error"]["message"])

    @patch("api.views_proxy._get_http_client")
    def test_timeout_returns_504(self, mock_get_client):
        """Timeout returns 504."""
        import httpx
        _mock_http_client(mock_get_client, post_side_effect=httpx.ReadTimeout("timed out"))

        request = _make_request(
            self.factory,
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Trace-Id": "timeout-test",
            },
        )
        resp = chat_completions(request)
        self.assertEqual(resp.status_code, 504)

    @patch("api.views_proxy._get_http_client")
    def test_agent_name_defaults_to_model(self, mock_get_client):
        """When X-Agent-Name is not set, agent_name defaults to the model name."""
        openai_resp = _openai_response("ok")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(openai_resp).encode()
        mock_resp.json.return_value = openai_resp
        mock_resp.headers = {"content-type": "application/json"}

        _mock_http_client(mock_get_client, post_return=mock_resp)

        request = _make_request(
            self.factory,
            body={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Trace-Id": "default-name-test",
            },
        )
        chat_completions(request)

        agent = AgentExecution.objects.filter(trace__trace_id="default-name-test").first()
        self.assertEqual(agent.agent_name, "gpt-4o")

    @patch("api.views_proxy._get_http_client")
    def test_custom_headers_not_forwarded_to_openai(self, mock_get_client):
        """X-Trace-Id, X-Agent-Name, X-Parent-Agent-Id are NOT sent to OpenAI."""
        openai_resp = _openai_response("ok")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(openai_resp).encode()
        mock_resp.json.return_value = openai_resp
        mock_resp.headers = {"content-type": "application/json"}

        mock_client = _mock_http_client(mock_get_client, post_return=mock_resp)

        request = _make_request(
            self.factory,
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "test"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Trace-Id": "header-fwd-test",
                "X-Agent-Name": "TestAgent",
                "X-Parent-Agent-Id": "parent-1",
                "X-Journey-Name": "Test Journey",
            },
        )
        chat_completions(request)

        # Check the headers that were actually forwarded
        call_args = mock_client.post.call_args
        forwarded_headers = call_args.kwargs.get("headers", {})
        self.assertIn("Authorization", forwarded_headers)
        self.assertNotIn("X-Trace-Id", forwarded_headers)
        self.assertNotIn("X-Agent-Name", forwarded_headers)
        self.assertNotIn("X-Parent-Agent-Id", forwarded_headers)
        self.assertNotIn("X-Journey-Name", forwarded_headers)

    @patch("api.views_proxy._get_http_client")
    def test_multi_agent_dag_construction(self, mock_get_client):
        """Two calls with parent relationship create a 2-node DAG."""
        openai_resp = _openai_response("response")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(openai_resp).encode()
        mock_resp.json.return_value = openai_resp
        mock_resp.headers = {"content-type": "application/json"}

        _mock_http_client(mock_get_client, post_return=mock_resp)

        # Call 1: root agent
        req1 = _make_request(
            self.factory,
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "route this"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Trace-Id": "dag-build-test",
                "X-Agent-Name": "Router",
            },
        )
        chat_completions(req1)

        # Call 2: child agent
        req2 = _make_request(
            self.factory,
            body={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "process"}]},
            headers={
                "Authorization": "Bearer sk-test",
                "X-Trace-Id": "dag-build-test",
                "X-Agent-Name": "Worker",
                "X-Parent-Agent-Id": "dag-build-test-call-1",
            },
        )
        chat_completions(req2)

        trace = ExecutionTrace.objects.get(trace_id="dag-build-test")
        self.assertEqual(trace.agents.count(), 2)
        self.assertEqual(len(trace.raw_journey["samples"]), 2)

        router = trace.agents.get(agent_name="Router")
        worker = trace.agents.get(agent_name="Worker")
        self.assertEqual(router.parent_id, "")
        self.assertEqual(worker.parent_id, "dag-build-test-call-1")
        self.assertEqual(router.agent_id, "dag-build-test-call-1")
        self.assertEqual(worker.agent_id, "dag-build-test-call-2")

    @patch("api.views_proxy._get_http_client")
    def test_streaming_returns_sse_response(self, mock_get_client):
        """Streaming request returns StreamingHttpResponse with SSE content type."""
        # Build mock streaming response
        chunks = [
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"},"index":0}],"model":"gpt-4o-mini"}',
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"},"index":0}],"model":"gpt-4o-mini"}',
            'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" world"},"index":0}],"model":"gpt-4o-mini"}',
            'data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2},"model":"gpt-4o-mini"}',
            "data: [DONE]",
        ]

        mock_upstream = MagicMock()
        mock_upstream.status_code = 200
        mock_upstream.iter_lines.return_value = iter(chunks)
        mock_upstream.__enter__ = MagicMock(return_value=mock_upstream)
        mock_upstream.__exit__ = MagicMock(return_value=False)

        _mock_http_client(mock_get_client, stream_return=mock_upstream)

        request = _make_request(
            self.factory,
            body={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Count to 5"}],
                "stream": True,
            },
            headers={
                "Authorization": "Bearer sk-test",
                "X-Trace-Id": "stream-test-1",
                "X-Agent-Name": "StreamAgent",
            },
        )
        resp = chat_completions(request)

        # Verify it's a streaming response
        from django.http import StreamingHttpResponse
        self.assertIsInstance(resp, StreamingHttpResponse)
        self.assertEqual(resp["Content-Type"], "text/event-stream")

        # Consume the stream
        output = b"".join(resp.streaming_content)
        output_str = output.decode()
        self.assertIn("Hello", output_str)
        self.assertIn(" world", output_str)
        self.assertIn("[DONE]", output_str)

        # Verify trace was persisted after streaming
        trace = ExecutionTrace.objects.get(trace_id="stream-test-1")
        agent = trace.agents.first()
        self.assertEqual(agent.agent_name, "StreamAgent")
        self.assertEqual(agent.output_text, "Hello world")
        self.assertEqual(agent.tokens, 7)  # 5 + 2

    @patch("api.views_proxy._get_http_client")
    def test_streaming_injects_stream_options(self, mock_get_client):
        """Streaming auto-injects stream_options.include_usage if not present."""
        chunks = ['data: [DONE]']

        mock_upstream = MagicMock()
        mock_upstream.status_code = 200
        mock_upstream.iter_lines.return_value = iter(chunks)
        mock_upstream.__enter__ = MagicMock(return_value=mock_upstream)
        mock_upstream.__exit__ = MagicMock(return_value=False)

        mock_client = _mock_http_client(mock_get_client, stream_return=mock_upstream)

        request = _make_request(
            self.factory,
            body={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={
                "Authorization": "Bearer sk-test",
                "X-Trace-Id": "stream-opts-test",
            },
        )
        resp = chat_completions(request)
        # Consume stream
        b"".join(resp.streaming_content)

        # Verify stream_options was injected in the forwarded body
        call_args = mock_client.stream.call_args
        forwarded_body = call_args.kwargs.get("json", {})
        self.assertTrue(forwarded_body.get("stream_options", {}).get("include_usage"))
