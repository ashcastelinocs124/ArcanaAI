"""API URL configuration."""
from django.urls import path

from . import views

urlpatterns = [
    path('health', views.health_check),
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
