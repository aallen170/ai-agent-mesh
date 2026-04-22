"""
router.py — TaskRouter: the AIMESH control-plane task dispatcher.

Responsibilities
----------------
1. Accept a TaskRequest (tier + task_type + payload).
2. Write a TaskRecord to Redis so the task's lifecycle is tracked.
3. Enqueue the task to the correct tier stream via StreamsClient.
4. Update the record's status as the task moves through its lifecycle.
5. Receive results from workers and mark tasks completed or failed.
6. List tasks by status for monitoring / the LangGraph orchestrator.

Redis key layout
----------------
aimesh:task:{task_id}   Hash  — one TaskRecord per task
aimesh:tasks:index      Set   — all task_ids ever submitted (for enumeration)

Note: tier-based classification (choosing *which* tier to route to) is
handled by the caller for now.  AIMESH-12 will add a classifier that
auto-assigns tiers from task content.

Thread safety
-------------
TaskRouter is thread-safe.  The underlying RedisClient uses a connection
pool so concurrent calls from multiple threads are safe.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import redis

from ..redis.client import RedisClient
from ..redis.streams import StreamsClient
from .task import TaskRecord, TaskRequest, TaskStatus

logger = logging.getLogger(__name__)

# Redis key templates
_TASK_KEY = "aimesh:task:{task_id}"
_TASKS_INDEX = "aimesh:tasks:index"


class TaskRouter:
    """
    Submit tasks, track their state, and record results.

    Parameters
    ----------
    redis_client    Shared RedisClient (connection pool is thread-safe).
    streams_client  StreamsClient used to enqueue tasks to the tier streams.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        streams_client: StreamsClient,
    ) -> None:
        self._r = redis_client.r
        self._streams = streams_client

    # ------------------------------------------------------------------
    # 1. Submit
    # ------------------------------------------------------------------

    def submit(self, request: TaskRequest) -> TaskRecord:
        """
        Submit a task to the mesh.

        Creates a TaskRecord in Redis, enqueues the task to the appropriate
        tier stream, then updates the record to DISPATCHED.

        Returns the TaskRecord in its final DISPATCHED state.

        Raises
        ------
        ValueError  If the tier is invalid.
        redis.RedisError  If the Redis write fails.
        """
        record = TaskRecord(
            task_id=request.task_id,
            task_type=request.task_type,
            tier=request.tier,
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
            tier=request.tier,
            task_type=request.task_type,
            payload=request.payload,
            task_id=request.task_id,
        )

        # Transition to DISPATCHED
        record.status = TaskStatus.DISPATCHED
        record.dispatched_at = time.time()
        self._save(record)
        logger.debug("Task %s dispatched to tier-%d stream", record.task_id, record.tier)

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

    def _save(self, record: TaskRecord) -> None:
        """Write (or overwrite) a TaskRecord to Redis atomically."""
        key = _TASK_KEY.format(task_id=record.task_id)
        pipe = self._r.pipeline()
        pipe.hset(key, mapping=record.to_dict())
        pipe.sadd(_TASKS_INDEX, record.task_id)
        pipe.execute()
