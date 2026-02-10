"""
Middleware to inject user-provided API keys into the environment.

The frontend sends the user's OpenAI API key via the X-API-Key header.
This middleware sets it as OPENAI_API_KEY for the duration of the request
so that LiteLLM and other LLM libraries pick it up automatically.
"""
import os


class APIKeyMiddleware:
    """Read X-API-Key header and set OPENAI_API_KEY env var per-request."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        api_key = request.headers.get('X-API-Key', '').strip()
        original_key = os.environ.get('OPENAI_API_KEY', '')

        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        response = self.get_response(request)

        # Restore original key (or clear if there wasn't one)
        if api_key:
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
            else:
                os.environ.pop('OPENAI_API_KEY', None)

        return response
