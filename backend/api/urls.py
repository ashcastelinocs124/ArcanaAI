"""API URL configuration."""
from pathlib import Path

from django.http import FileResponse
from django.urls import path

from . import views

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / 'frontend'


def serve_index(request):
    """Serve frontend/index.html at root URL."""
    return FileResponse(open(FRONTEND_DIR / 'index.html', 'rb'), content_type='text/html')


urlpatterns = [
    path('', serve_index),
    path('health', views.health_check),
    path('api/traces', views.list_traces),
    path('api/traces/upload', views.upload_traces),
    path('api/optimizer/run', views.run_prompt_optimizer),
    path('api/optimizer/status/<str:task_id>', views.optimizer_status),
    path('api/optimizer/cancel/<str:task_id>', views.optimizer_cancel),
    path('api/pipeline/analyze', views.analyze_pipeline),
    path('api/evaluations/run', views.run_evaluations),
    path('api/gateway/route', views.route_gateway),
    path('api/workflow/run', views.run_workflow),
    path('api/workflow/topologies', views.get_workflow_topologies),
    path('api/workflow/stream', views.stream_workflow),
]
