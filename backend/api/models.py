"""
Django models for ArcanaAI persistence layer.

Replaces file-based storage (JSONL, temp CSVs, transient JSON)
with proper database-backed persistence.
"""
from django.db import models


class RoutingDecision(models.Model):
    """Persists gateway routing decisions (replaces gateway_cache.jsonl)."""
    intent = models.TextField()
    task_type = models.CharField(max_length=100)
    latency_budget_ms = models.FloatField(null=True, blank=True)
    cost_budget = models.FloatField(null=True, blank=True)
    quality = models.CharField(max_length=50, null=True, blank=True)
    model_id = models.CharField(max_length=100)
    provider = models.CharField(max_length=100)
    score = models.FloatField()
    rationale = models.TextField(blank=True, default='')
    source = models.CharField(max_length=50, default='router')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['task_type', '-created_at']),
        ]

    def __str__(self):
        return f"{self.task_type} → {self.model_id} ({self.source})"


class OptimizationRun(models.Model):
    """Persists optimization run metadata (replaces temp CSV header)."""
    file_hash = models.CharField(max_length=64, blank=True, default='')
    prompt_template = models.TextField()
    eval_model = models.CharField(max_length=100, default='gpt-4o')
    optimizer_model = models.CharField(max_length=100, default='gpt-4o')
    target_score = models.FloatField(default=0.85)
    max_iters = models.IntegerField(default=3)
    original_avg_score = models.FloatField(default=0)
    optimized_avg_score = models.FloatField(default=0)
    improvement_pct = models.FloatField(default=0)
    total_cost_usd = models.FloatField(default=0)
    agent_filter = models.CharField(max_length=200, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Run {self.pk} ({self.eval_model}, {self.improvement_pct:+.1f}%)"


class OptimizationResult(models.Model):
    """Persists per-row optimization results (replaces temp CSV rows)."""
    run = models.ForeignKey(OptimizationRun, on_delete=models.CASCADE, related_name='results')
    row_index = models.IntegerField()
    input_text = models.TextField()
    gold_text = models.TextField()
    output_text = models.TextField()
    score = models.FloatField()
    iterations = models.IntegerField(default=0)
    latency_ms = models.FloatField(default=0)
    tokens = models.IntegerField(default=0)
    cost_usd = models.FloatField(default=0)

    class Meta:
        ordering = ['run', 'row_index']

    def __str__(self):
        return f"Run {self.run_id} row {self.row_index} (score={self.score:.2f})"


class ExecutionTrace(models.Model):
    """Persists workflow execution traces (replaces transient JSON responses)."""
    trace_id = models.CharField(max_length=100, unique=True)
    journey_name = models.CharField(max_length=300, blank=True, default='')
    workflow_type = models.CharField(max_length=100, default='custom')
    goal = models.TextField(blank=True, default='')
    status = models.CharField(max_length=50, default='completed')
    raw_journey = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['trace_id']),
        ]

    def __str__(self):
        return f"{self.trace_id} ({self.workflow_type})"


class AgentExecution(models.Model):
    """Persists per-agent execution data within a trace."""
    trace = models.ForeignKey(ExecutionTrace, on_delete=models.CASCADE, related_name='agents')
    agent_id = models.CharField(max_length=100)
    agent_name = models.CharField(max_length=200)
    parent_id = models.CharField(max_length=100, null=True, blank=True)
    input_text = models.TextField(blank=True, default='')
    output_text = models.TextField(blank=True, default='')
    latency_ms = models.FloatField(default=0)
    ttft_ms = models.FloatField(default=0)
    tokens = models.IntegerField(default=0)
    model = models.CharField(max_length=100, blank=True, default='')
    status = models.CharField(max_length=50, default='completed')

    class Meta:
        ordering = ['trace', 'agent_name']

    def __str__(self):
        return f"{self.agent_name} in {self.trace.trace_id}"


class BatchUpload(models.Model):
    """Tracks async batch upload operations via Celery."""
    batch_id = models.CharField(max_length=100, unique=True)
    total_traces = models.IntegerField(default=0)
    created_count = models.IntegerField(default=0)
    updated_count = models.IntegerField(default=0)
    failed_count = models.IntegerField(default=0)
    errors = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=50, default='pending')
    auto_evaluate = models.BooleanField(default=False)
    user_goal = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Batch {self.batch_id} ({self.status})"


class EvaluationResult(models.Model):
    """Persists cascade/checkpoint evaluation results for a trace."""
    trace = models.ForeignKey(ExecutionTrace, on_delete=models.CASCADE, related_name='evaluations')
    user_goal = models.TextField()
    evaluation_mode = models.CharField(max_length=50, default='cascade')
    result_data = models.JSONField(default=dict, blank=True)
    llm_calls_made = models.IntegerField(default=0)
    duration_ms = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Eval {self.trace.trace_id} ({self.evaluation_mode})"
