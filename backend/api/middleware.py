"""
Middleware to inject user-provided API keys into the environment.

The frontend sends the user's OpenAI API key via the X-API-Key header.
This middleware sets it as OPENAI_API_KEY so LiteLLM picks it up.

Note: We do NOT restore the old key after the request. With SSE streaming,
the response body is consumed after middleware returns, so restoring would
clear the key before LLM calls happen. Since there is no server-side key
(users provide their own), this is safe.
"""
import os


class APIKeyMiddleware:
    """Read X-API-Key header and set OPENAI_API_KEY env var."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        api_key = request.headers.get('X-API-Key', '').strip()
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        return self.get_response(request)
