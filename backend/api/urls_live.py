"""
URL routing for live monitoring API endpoints.
"""
from django.urls import path
from . import views_live_monitor as views

urlpatterns = [
    # Session management
    path('sessions', views.list_sessions, name='live-sessions-list'),
    path('sessions/create', views.create_session, name='live-session-create'),  # Must come before <session_id>
    path('sessions/<str:session_id>', views.get_session, name='live-session-detail'),

    # Action tracking
    path('sessions/<str:session_id>/actions', views.track_action, name='live-track-action'),

    # Drift detection
    path('sessions/<str:session_id>/drift', views.report_drift, name='live-report-drift'),

    # Session control
    path('sessions/<str:session_id>/pause', views.pause_session, name='live-pause'),
    path('sessions/<str:session_id>/resume', views.resume_session, name='live-resume'),
    path('sessions/<str:session_id>/complete', views.complete_session, name='live-complete'),

    # Alert management
    path('sessions/<str:session_id>/alerts/<int:alert_id>/override',
         views.override_drift, name='live-override-drift'),

    # File-based state (for hook integration)
    path('sessions/<str:session_id>/file-state', views.get_file_state, name='live-file-state'),
    path('sessions/<str:session_id>/drift-check', views.trigger_drift_check, name='live-drift-check'),
]
