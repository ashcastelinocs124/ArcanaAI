"""URL patterns for the v1 Trace Ingestion REST API."""
from django.urls import include, path

from . import views_v1

urlpatterns = [
    # Batch must come before <str:trace_id> to avoid "batch" being captured as trace_id
    path('traces/batch', views_v1.batch_upload, name='v1-batch-upload'),
    path('traces/batch/<str:batch_id>', views_v1.batch_status, name='v1-batch-status'),

    # CRUD
    path('traces', views_v1.traces_list_create, name='v1-traces-list-create'),
    path('traces/<str:trace_id>', views_v1.trace_detail, name='v1-trace-detail'),

    # Evaluation
    path('traces/<str:trace_id>/evaluate', views_v1.trace_evaluate, name='v1-trace-evaluate'),
    path('traces/<str:trace_id>/evaluation', views_v1.trace_evaluation, name='v1-trace-evaluation'),

    # OpenAI-compatible proxy
    path('proxy/', include('api.urls_proxy')),
]
