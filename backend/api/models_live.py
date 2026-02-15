"""
Database models for live monitoring sessions.
"""
from django.db import models
from django.utils import timezone


class LiveSession(models.Model):
    """
    A live monitoring session for a coding agent task.
    """
    session_id = models.CharField(max_length=100, unique=True, db_index=True)
    user_goal = models.TextField()
    status = models.CharField(
        max_length=20,
        choices=[
            ('active', 'Active'),
            ('paused', 'Paused'),
            ('drifted', 'Drifted'),
            ('completed', 'Completed'),
            ('stopped', 'Stopped')
        ],
        default='active'
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Metrics
    total_actions = models.IntegerField(default=0)
    drift_alerts = models.IntegerField(default=0)

    # Configuration
    checkpoint_interval = models.IntegerField(default=5)
    llm_model = models.CharField(max_length=50, default='gpt-4o-mini')

    class Meta:
        db_table = 'live_sessions'
        ordering = ['-created_at']

    def __str__(self):
        return f"LiveSession({self.session_id}): {self.user_goal[:50]}"


class LiveAction(models.Model):
    """
    A single action taken by the agent during a live session.
    """
    session = models.ForeignKey(
        LiveSession,
        on_delete=models.CASCADE,
        related_name='actions'
    )
    action_num = models.IntegerField()

    # Action details
    tool = models.CharField(max_length=50)  # Read, Edit, Write, Bash, etc.
    file_path = models.CharField(max_length=500, blank=True)
    reasoning = models.TextField(blank=True)
    input_text = models.TextField(blank=True)
    output_text = models.TextField(blank=True)

    # Timestamp
    timestamp = models.DateTimeField(default=timezone.now)

    # Metadata
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = 'live_actions'
        ordering = ['action_num']
        unique_together = [['session', 'action_num']]

    def __str__(self):
        return f"Action {self.action_num}: {self.tool} {self.file_path}"


class DriftAlert(models.Model):
    """
    A drift detection alert for a live session.
    """
    session = models.ForeignKey(
        LiveSession,
        on_delete=models.CASCADE,
        related_name='alerts'
    )

    # When drift was detected
    detected_at = models.DateTimeField(default=timezone.now)
    action_count = models.IntegerField()  # How many actions before drift

    # LLM assessment
    aligned = models.BooleanField(default=False)
    confidence = models.FloatField()  # 0-1
    reason = models.TextField()
    recommendation = models.TextField()

    # User response
    user_action = models.CharField(
        max_length=20,
        choices=[
            ('paused', 'Paused Agent'),
            ('overridden', 'Override & Continue'),
            ('redirected', 'Changed Goal'),
            ('stopped', 'Stopped Agent'),
            ('pending', 'Pending User Decision')
        ],
        default='pending'
    )

    # Recent actions that triggered drift (JSON)
    recent_actions = models.JSONField(default=list)

    class Meta:
        db_table = 'drift_alerts'
        ordering = ['-detected_at']

    def __str__(self):
        return f"DriftAlert for {self.session.session_id}: {self.reason[:50]}"
