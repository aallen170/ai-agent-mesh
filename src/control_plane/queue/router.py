"""
router.py — TaskRouter: the AIMESH control-plane task dispatcher.

Responsibilities
----------------
1. Accept a TaskRequest (tier + task_type + payload).
2. Resolve the compute tier — via TaskClassifier (AIMESH-12) if tier is None.
3. Write a TaskRecord to Redis so the task's lifecycle is tracked.
4. Enqueue the task to the correct tier stream via StreamsClient.
5. Update the record's status as the task moves through its lifecycle.
6. Receive results from workers and mark tasks completed or failed.
7. List tasks by status for monitoring / the LangGraph orchestrator.

Redis key layout
----------------
aimesh:task:{task_id}   Hash  — one TaskRecord per task
aimesh:tasks:index      Set   — all task_ids ever submitted (for enumeration)

Auto-classification (AIMESH-12)
--------------------------------
Pass a TaskClassifier to the constructor and leave TaskRequest.tier as None
to have the classifier pick the appropriate tier automatically.  If the
classifier is not configured, or its model call fails, the router defaults
to FALLBACK_TIER (2) and logs a warning.

Thread safety
-------------
TaskRouter is thread-safe.  The underlying RedisClient uses a connection
pool so concurrent calls from multiple threads are safe.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Optional

import redis

from ...telemetry import get_meter, get_tracer
from ..redis.client import RedisClient
from ..redis.streams import StreamsClient
from .task import TaskRecord, TaskRequest, TaskStatus

if TYPE_CHECKING:
    from ..classifier.classifier import TaskClassifier

logger = logging.getLogger(__name__)

_tracer = get_tracer(__name__)
_meter = get_meter(__name__)

# Metrics
_tasks_submitted = _meter.create_counter(
    "aimesh.tasks.submitted",
    unit="1",
    description="Total tasks submitted to the mesh",
)
_tasks_completed = _meter.create_counter(
    "aimesh.tasks.completed",
    unit="1",
    description="Total tasks that reached a terminal state (completed or failed)",
)
_task_duration = _meter.create_histogram(
    "aimesh.task.duration",
    unit="s",
    description="End-to-end task duration from enqueue to completion in seconds",
)

# Redis key templates
_TASK_KEY = "aimesh:task:{task_id}"
_TASKS_INDEX = "aimesh:tasks:index"

# Tier used when the classifier is absent or fails
_FALLBACK_TIER = 2


class TaskRouter:
    """
    Submit tasks, track their state, and record results.

    Parameters
    ----------
    redis_client    Shared RedisClient (connection pool is thread-safe).
    streams_client  StreamsClient used to enqueue tasks to the tier streams.
    classifier      Optional TaskClassifier (AIMESH-12).  When supplied,
                    tasks submitted with ``tier=None`` are automatically
                    classified before dispatch.  If absent, tier-less tasks
                    fall back to tier 2.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        streams_client: StreamsClient,
        classifier: "TaskClassifier | None" = None,
    ) -> None:
        self._r = redis_client.r
        self._streams = streams_client
        self._classifier = classifier

    # ------------------------------------------------------------------
    # 1. Submit
    # ------------------------------------------------------------------

    def submit(self, request: TaskRequest) -> TaskRecord:
        """
        Submit a task to the mesh.

        If ``request.tier`` is None, the classifier resolves the tier first.
        Creates a TaskRecord in Redis, enqueues the task to the appropriate
        tier stream, then updates the record to DISPATCHED.

        Returns the TaskRecord in its final DISPATCHED state.

        Raises
        ------
        ValueError  If the resolved tier is invalid.
        redis.RedisError  If the Redis write fails.
        """
        with _tracer.start_as_current_span(
            "aimesh.task.submit",
            attributes={
                "task.id": request.task_id,
                "task.type": request.task_type,
            },
        ) as span:
            tier = self._resolve_tier(request)
            span.set_attribute("task.tier", tier)

            record = TaskRecord(
                task_id=request.task_id,
                task_type=request.task_type,
                tier=tier,
                status=TaskStatus.PENDING,
                enqueued_at=time.time(),
            )

            # Persist the PENDING record first so the task is visible immediately
            self._save(record)
            logger.info(
                "Task %s submitted (type=%s, tier=%d)",
                record.task_id, record.task_type, record.tier,
            )

            # Write to the Redis stream — this is what the worker consumes
            self._streams.enqueue_task(
                tier=tier,
                task_type=request.task_type,
                payload=request.payload,
                task_id=request.task_id,
            )

            # Transition to DISPATCHED
            record.status = TaskStatus.DISPATCHED
            record.dispatched_at = time.time()
            self._save(record)
            logger.debug("Task %s dispatched to tier-%d stream", record.task_id, record.tier)

            _tasks_submitted.add(
                1, {"tier": str(tier), "task_type": request.task_type}
            )

            return record

    # ------------------------------------------------------------------
    # 2. Record results (called when a worker publishes to aimesh:results)
    # ------------------------------------------------------------------

    def mark_completed(
        self,
        task_id: str,
        device_id: str,
        result: dict,
        error: str | None = None,
    ) -> Optional[TaskRecord]:
        """
        Update a task's record when its result arrives from the results stream.

        Sets status to COMPLETED or FAILED depending on whether *error* is set.
        Returns the updated TaskRecord, or None if the task_id is unknown.

        This is called by whatever component reads the ``aimesh:results``
        stream (e.g. the LangGraph orchestrator or a control-plane loop).
        """
        record = self.get_task(task_id)
        if record is None:
            logger.warning(
                "mark_completed called for unknown task %r — ignoring", task_id
            )
            return None

        record.status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
        record.completed_at = time.time()
        record.device_id = device_id
        record.result = result
        record.error = error
        self._save(record)

        status_label = "failed" if error else "completed"
        _tasks_completed.add(
            1,
            {
                "tier": str(record.tier),
                "task_type": record.task_type,
                "status": status_label,
            },
        )
        if record.duration is not None:
            _task_duration.record(
                record.duration,
                {"tier": str(record.tier), "task_type": record.task_type},
            )

        if error:
            logger.warning("Task %s FAILED on %r: %s", task_id, device_id, error)
        else:
            logger.info(
                "Task %s COMPLETED on %r (%.2fs)",
                task_id, device_id, record.duration or 0,
            )
        return record

    def process_results_stream(self, consumer_name: str = "control-plane") -> int:
        """
        Drain the results stream and update task records for each result received.

        Returns the number of results processed.

        Call this in a loop (or from the LangGraph orchestrator) to keep
        task records up to date.  Each result is ACKed after the record
        is saved.
        """
        raw_results = self._streams.read_results(
            consumer_name=consumer_name,
            count=50,
            block_ms=0,  # non-blocking
        )
        processed = 0
        for raw in raw_results:
            self.mark_completed(
                task_id=raw["task_id"],
                device_id=raw["device_id"],
                result=raw["result"],
                error=raw["error"],
            )
            self._streams.ack_result(raw["_msg_id"])
            processed += 1
        return processed

    # ------------------------------------------------------------------
    # 3. Queries
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Return the TaskRecord for a given task_id, or None if not found."""
        key = _TASK_KEY.format(task_id=task_id)
        data = self._r.hgetall(key)
        if not data:
            return None
        return TaskRecord.from_dict(data)

    def list_tasks(self, status: str | None = None) -> list[TaskRecord]:
        """
        Return all known tasks, optionally filtered by status.

        Parameters
        ----------
        status  One of the TaskStatus constants, or None for all tasks.
        """
        task_ids = self._r.smembers(_TASKS_INDEX)
        records: list[TaskRecord] = []
        for task_id in task_ids:
            record = self.get_task(task_id)
            if record is None:
                continue
            if status is not None and record.status != status:
                continue
            records.append(record)
        return records

    def list_pending(self) -> list[TaskRecord]:
        """Return tasks that have been submitted but not yet dispatched."""
        return self.list_tasks(status=TaskStatus.PENDING)

    def list_dispatched(self) -> list[TaskRecord]:
        """Return tasks that are in-flight (dispatched, awaiting a worker result)."""
        return self.list_tasks(status=TaskStatus.DISPATCHED)

    def list_completed(self) -> list[TaskRecord]:
        """Return successfully completed tasks."""
        return self.list_tasks(status=TaskStatus.COMPLETED)

    def list_failed(self) -> list[TaskRecord]:
        """Return tasks that ended in an error."""
        return self.list_tasks(status=TaskStatus.FAILED)

    def count_by_status(self) -> dict[str, int]:
        """
        Return a summary of task counts grouped by status.

        Example::

            {"pending": 0, "dispatched": 2, "completed": 14, "failed": 1}
        """
        counts: dict[str, int] = {
            TaskStatus.PENDING: 0,
            TaskStatus.DISPATCHED: 0,
            TaskStatus.COMPLETED: 0,
            TaskStatus.FAILED: 0,
        }
        for record in self.list_tasks():
            if record.status in counts:
                counts[record.status] += 1
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_tier(self, request: TaskRequest) -> int:
        """
        Return the compute tier for *request*.

        If ``request.tier`` is set, return it directly.
        Otherwise, ask the classifier.  If the classifier is not configured
        or fails, fall back to ``_FALLBACK_TIER`` with a warning.
        """
        if request.tier is not None:
            return request.tier

        if self._classifier is not None:
            # Extract a human-readable prompt from the payload for classification.
            # The "prompt" key is the canonical field; fall back to a JSON dump.
            prompt_text = request.payload.get("prompt") or json.dumps(request.payload)
            result = self._classifier.classify(
                prompt=str(prompt_text),
                task_type=request.task_type,
            )
            if result.fallback:
                logger.warning(
                    "Classifier returned fallback tier %d for task %s: %s",
                    result.tier, request.task_id, result.reasoning,
                )
            else:
                logger.info(
                    "Classifier assigned tier %d for task %s (%.0fms): %s",
                    result.tier, request.task_id, result.elapsed_ms, result.reasoning,
                )
            return result.tier

        # No classifier configured at all
        logger.warning(
            "No tier specified and no classifier configured for task %s — "
            "defaulting to tier %d",
            request.task_id, _FALLBACK_TIER,
        )
        return _FALLBACK_TIER

    def _save(self, record: TaskRecord) -> None:
        """Write (or overwrite) a TaskRecord to Redis atomically."""
        key = _TASK_KEY.format(task_id=record.task_id)
        pipe = self._r.pipeline()
        pipe.hset(key, mapping=record.to_dict())
        pipe.sadd(_TASKS_INDEX, record.task_id)
        pipe.execute()
