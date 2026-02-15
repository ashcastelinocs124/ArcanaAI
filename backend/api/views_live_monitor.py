"""
API endpoints for live monitoring sessions.
"""
import json
import sys
from pathlib import Path
from django.views.decorators.http import require_http_methods
from django.utils import timezone

# Add parent dirs to path
_backend_dir = str(Path(__file__).resolve().parent.parent)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from .models_live import LiveSession, LiveAction, DriftAlert
from .response import api_response, api_error


# ============================================================================
# Live Session Management
# ============================================================================

@require_http_methods(["GET"])
def list_sessions(request):
    """
    List all live monitoring sessions.

    GET /api/live/sessions
    Query params:
        - status: Filter by status (active, paused, drifted, completed)
        - limit: Number of results (default 50)
    """
    try:
        status = request.GET.get('status')
        limit = int(request.GET.get('limit', 50))

        query = LiveSession.objects.all()

        if status:
            query = query.filter(status=status)

        sessions = query[:limit]

        return api_response({
            "sessions": [
                {
                    "trace_id": s.session_id,  # Return as trace_id
                    "user_goal": s.user_goal,
                    "status": s.status,
                    "total_actions": s.total_actions,
                    "drift_alerts": s.drift_alerts,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None
                }
                for s in sessions
            ],
            "count": len(sessions)
        })

    except Exception as e:
        return api_error(str(e), 500)


@require_http_methods(["POST"])
def create_session(request):
    """
    Create a new live monitoring session.

    POST /api/live/sessions
    Body: {
        "trace_id": "tr-123",  # Accept trace_id
        "user_goal": "Add user authentication",
        "checkpoint_interval": 5,
        "llm_model": "gpt-4o-mini"
    }
    """
    try:
        data = json.loads(request.body)

        # Accept both trace_id (new) and session_id (backwards compat)
        trace_id = data.get('trace_id') or data.get('session_id')
        user_goal = data.get('user_goal')

        if not trace_id or not user_goal:
            return api_error("trace_id and user_goal are required", 400)

        # Check if session already exists
        if LiveSession.objects.filter(session_id=trace_id).exists():
            return api_error(f"Session {trace_id} already exists", 400)

        # Create session (store as session_id internally)
        session = LiveSession.objects.create(
            session_id=trace_id,
            user_goal=user_goal,
            checkpoint_interval=data.get('checkpoint_interval', 5),
            llm_model=data.get('llm_model', 'gpt-4o-mini')
        )

        return api_response({
            "trace_id": session.session_id,  # Return as trace_id
            "user_goal": session.user_goal,
            "status": session.status,
            "created_at": session.created_at.isoformat()
        }, status=201)

    except Exception as e:
        return api_error(str(e), 500)


@require_http_methods(["GET"])
def get_session(request, session_id):
    """
    Get details of a live monitoring session.

    GET /api/live/sessions/<trace_id>
    """
    try:
        session = LiveSession.objects.get(session_id=session_id)

        # Get recent actions
        recent_actions = LiveAction.objects.filter(session=session).order_by('-action_num')[:10]

        # Get drift alerts
        alerts = DriftAlert.objects.filter(session=session)

        return api_response({
            "trace_id": session.session_id,  # Return as trace_id
            "user_goal": session.user_goal,
            "status": session.status,
            "total_actions": session.total_actions,
            "drift_alerts": session.drift_alerts,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "recent_actions": [
                {
                    "action_num": a.action_num,
                    "tool": a.tool,
                    "file_path": a.file_path,
                    "reasoning": a.reasoning,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in recent_actions
            ],
            "alerts": [
                {
                    "id": alert.id,
                    "detected_at": alert.detected_at.isoformat(),
                    "action_count": alert.action_count,
                    "confidence": alert.confidence,
                    "reason": alert.reason,
                    "recommendation": alert.recommendation,
                    "user_action": alert.user_action
                }
                for alert in alerts
            ]
        })

    except LiveSession.DoesNotExist:
        return api_error(f"Session {session_id} not found", 404)
    except Exception as e:
        return api_error(str(e), 500)


# ============================================================================
# Action Tracking
# ============================================================================

@require_http_methods(["POST"])
def track_action(request, session_id):
    """
    Track a new action for a live session.

    POST /api/live/sessions/<session_id>/actions
    Body: {
        "tool": "Edit",
        "file_path": "backend/auth.py",
        "reasoning": "Adding login endpoint",
        "input": "...",
        "output": "..."
    }
    """
    try:
        session = LiveSession.objects.get(session_id=session_id)

        data = json.loads(request.body)
        tool = data.get('tool')

        if not tool:
            return api_error("tool is required", 400)

        # Create action
        action = LiveAction.objects.create(
            session=session,
            action_num=session.total_actions + 1,
            tool=tool,
            file_path=data.get('file_path', ''),
            reasoning=data.get('reasoning', ''),
            input_text=data.get('input', ''),
            output_text=data.get('output', ''),
            metadata=data.get('metadata', {})
        )

        # Update session action count
        session.total_actions += 1
        session.save()

        return api_response({
            "action_num": action.action_num,
            "session_id": session.session_id
        }, status=201)

    except LiveSession.DoesNotExist:
        return api_error(f"Session {session_id} not found", 404)
    except Exception as e:
        return api_error(str(e), 500)


# ============================================================================
# Drift Detection
# ============================================================================

@require_http_methods(["POST"])
def report_drift(request, session_id):
    """
    Report a drift detection alert.

    POST /api/live/sessions/<session_id>/drift
    Body: {
        "action_count": 15,
        "aligned": false,
        "confidence": 0.87,
        "reason": "Agent is writing tests before implementing core logic",
        "recommendation": "Implement endpoints first",
        "recent_actions": [...]
    }
    """
    try:
        session = LiveSession.objects.get(session_id=session_id)

        data = json.loads(request.body)

        # Create drift alert
        alert = DriftAlert.objects.create(
            session=session,
            action_count=data.get('action_count', 0),
            aligned=data.get('aligned', False),
            confidence=data.get('confidence', 0.0),
            reason=data.get('reason', ''),
            recommendation=data.get('recommendation', ''),
            recent_actions=data.get('recent_actions', [])
        )

        # Update session
        session.drift_alerts += 1
        session.status = 'drifted'
        session.save()

        return api_response({
            "alert_id": alert.id,
            "session_id": session.session_id
        }, status=201)

    except LiveSession.DoesNotExist:
        return api_error(f"Session {session_id} not found", 404)
    except Exception as e:
        return api_error(str(e), 500)


# ============================================================================
# Session Control
# ============================================================================

@require_http_methods(["POST"])
def pause_session(request, session_id):
    """
    Pause a live session.

    POST /api/live/sessions/<session_id>/pause
    """
    try:
        session = LiveSession.objects.get(session_id=session_id)
        session.status = 'paused'
        session.save()

        return api_response({"status": "paused"})

    except LiveSession.DoesNotExist:
        return api_error(f"Session {session_id} not found", 404)
    except Exception as e:
        return api_error(str(e), 500)


@require_http_methods(["POST"])
def resume_session(request, session_id):
    """
    Resume a paused session.

    POST /api/live/sessions/<session_id>/resume
    """
    try:
        session = LiveSession.objects.get(session_id=session_id)
        session.status = 'active'
        session.save()

        return api_response({"status": "active"})

    except LiveSession.DoesNotExist:
        return api_error(f"Session {session_id} not found", 404)
    except Exception as e:
        return api_error(str(e), 500)


@require_http_methods(["POST"])
def complete_session(request, session_id):
    """
    Mark a session as completed.

    POST /api/live/sessions/<session_id>/complete
    """
    try:
        session = LiveSession.objects.get(session_id=session_id)
        session.status = 'completed'
        session.completed_at = timezone.now()
        session.save()

        return api_response({"status": "completed"})

    except LiveSession.DoesNotExist:
        return api_error(f"Session {session_id} not found", 404)
    except Exception as e:
        return api_error(str(e), 500)


@require_http_methods(["POST"])
def override_drift(request, session_id, alert_id):
    """
    User overrides a drift alert and continues.

    POST /api/live/sessions/<session_id>/alerts/<alert_id>/override
    """
    try:
        session = LiveSession.objects.get(session_id=session_id)
        alert = DriftAlert.objects.get(id=alert_id, session=session)

        alert.user_action = 'overridden'
        alert.save()

        # Resume session
        session.status = 'active'
        session.save()

        return api_response({"status": "overridden"})

    except LiveSession.DoesNotExist:
        return api_error(f"Session {session_id} not found", 404)
    except DriftAlert.DoesNotExist:
        return api_error(f"Alert {alert_id} not found", 404)
    except Exception as e:
        return api_error(str(e), 500)


# ============================================================================
# File-based State (Hook Integration)
# ============================================================================

@require_http_methods(["GET"])
def get_file_state(request, session_id):
    """
    Read file-based state from /tmp/arcana-monitor/ (hook integration).

    GET /api/live/sessions/<session_id>/file-state

    Returns:
    {
        "session_id": "...",
        "user_goal": "...",
        "actions": [...],
        "action_count": 10,
        "drift_alert": {...} or null,
        "has_drift_flag": true/false
    }
    """
    try:
        import json
        from pathlib import Path

        state_dir = Path("/tmp/arcana-monitor")
        state_file = state_dir / f"{session_id}.json"
        drift_file = state_dir / f"{session_id}.drift"

        if not state_file.exists():
            return api_error(f"No file-based state found for session {session_id}", 404)

        # Read state file
        state = json.loads(state_file.read_text())

        # Check if drift flag exists
        has_drift_flag = drift_file.exists()
        drift_data = None
        if has_drift_flag:
            try:
                drift_data = json.loads(drift_file.read_text())
            except:
                pass

        return api_response({
            "session_id": state.get("session_id"),
            "user_goal": state.get("user_goal"),
            "actions": state.get("actions", []),
            "action_count": state.get("action_count", 0),
            "drift_alert": state.get("drift_alert") or drift_data,
            "has_drift_flag": has_drift_flag,
            "created_at": state.get("created_at")
        })

    except json.JSONDecodeError:
        return api_error("Invalid state file format", 500)
    except Exception as e:
        return api_error(str(e), 500)


@require_http_methods(["POST"])
def trigger_drift_check(request, session_id):
    """
    Trigger interactive drift detection - creates .drift.pending file
    and waits for user response.

    POST /api/live/sessions/<session_id>/drift-check
    Body:
    {
        "reason": "Why drift was detected",
        "confidence": 0.75,
        "recommendation": "What to do",
        "action_number": 7
    }

    Returns:
    {
        "user_choice": "continue" or "stop",
        "timeout": false,
        "drift_data": {...}
    }
    """
    try:
        import json
        import time
        from pathlib import Path

        # Parse request body
        body = json.loads(request.body.decode('utf-8'))
        reason = body.get('reason', 'Workflow drift detected')
        confidence = body.get('confidence', 0.5)
        recommendation = body.get('recommendation', '')
        action_number = body.get('action_number')

        state_dir = Path("/tmp/arcana-monitor")
        state_dir.mkdir(exist_ok=True)

        # Create .drift.pending file
        drift_data = {
            "session_id": session_id,
            "reason": reason,
            "confidence": confidence,
            "recommendation": recommendation,
            "action_number": action_number,
            "timestamp": time.time()
        }

        pending_file = state_dir / f"{session_id}.drift.pending"
        pending_file.write_text(json.dumps(drift_data, indent=2))

        # Wait for user response (timeout after 60 seconds)
        response_file = state_dir / f"{session_id}.drift.response"
        timeout_seconds = 60
        start_time = time.time()

        while (time.time() - start_time) < timeout_seconds:
            if response_file.exists():
                # User responded!
                response_data = json.loads(response_file.read_text())

                # Clean up response file
                response_file.unlink()

                return api_response({
                    "user_choice": response_data.get("user_choice"),
                    "timeout": False,
                    "drift_data": drift_data,
                    "user_responded_at": response_data.get("timestamp")
                })

            # Check every 0.5 seconds
            time.sleep(0.5)

        # Timeout - no response from user
        # Clean up pending file
        if pending_file.exists():
            pending_file.unlink()

        return api_response({
            "user_choice": "timeout",
            "timeout": True,
            "drift_data": drift_data,
            "message": "User did not respond within 60 seconds"
        })

    except json.JSONDecodeError:
        return api_error("Invalid request body", 400)
    except Exception as e:
        return api_error(str(e), 500)
