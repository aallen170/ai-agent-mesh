"""
control_plane.queue -- Task queue and routing for the AIMESH control plane.

Public API:
    TaskRequest       Input: what to run and where.
    TaskRecord        Persisted state of a task throughout its lifecycle.
    TaskStatus        Lifecycle states (pending, dispatched, completed, failed, dlq).
    TaskRouter        Submits tasks, tracks state in Redis, and marks results.
    ResultCollector   Background service that drains results, retries, and manages DLQ.
"""
from .task import TaskRequest, TaskRecord, TaskStatus
from .router import TaskRouter
from .collector import ResultCollector

__all__ = ["TaskRequest", "TaskRecord", "TaskStatus", "TaskRouter", "ResultCollector"]
