"""
Consistent JSON response envelope for the v1 API.

All v1 endpoints return:
    {
        "success": true/false,
        "data": ...,        # payload on success
        "error": ...,       # error info on failure
        "meta": {...}       # pagination, timing, etc.
    }
"""
from django.http import JsonResponse


def api_response(data=None, meta=None, status=200):
    """Return a success response with standard envelope."""
    body = {"success": True}
    if data is not None:
        body["data"] = data
    if meta:
        body["meta"] = meta
    return JsonResponse(body, status=status)


def api_error(message, code=None, status=400, details=None):
    """Return an error response with standard envelope."""
    error = {"message": message}
    if code:
        error["code"] = code
    if details:
        error["details"] = details
    return JsonResponse({"success": False, "error": error}, status=status)
