"""
StreamsClient -- helpers for the AIMESH tier-based task queues.

Each compute tier gets its own Redis Stream:
    aimesh:tasks:tier0   iPhones / iPads / Android tablet
    aimesh:tasks:tier1   iGPU laptop
    aimesh:tasks:tier2   dGPU laptop + desktop PC
    aimesh:tasks:tier3   RunPod serverless
    aimesh:tasks:tier4   Claude Sonnet / Opus

Results are written back to a single results stream:
    aimesh:results

Tasks that exhaust retries are written to the dead-letter queue stream:
    aimesh:dlq

Consumer groups are created once per stream (MKSTREAM if the stream
does not exist yet) and are idempotent -- safe to call on every startup.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import redis

from .client import RedisClient

logger = logging.getLogger(__name__)

TASK_STREAM = "aimesh:tasks:tier{tier}"
RESULT_STREAM = "aimesh:results"
DLQ_STREAM = "aimesh:dlq"
CONSUMER_GROUP = "aimesh-workers"

TIERS = [0, 1, 2, 3, 4]


class StreamsClient:
    """High-level interface for enqueuing tasks and reading results via Redis Streams."""

    def __init__(self, client: RedisClient) -> None:
        self._r = client.r

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def ensure_streams(self) -> None:
        """Create all tier streams and their consumer group if they do not already exist."""
        streams = [TASK_STREAM.format(tier=t) for t in TIERS] + [RESULT_STREAM, DLQ_STREAM]
        for stream in streams:
            try:
                self._r.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
                logger.info("Created consumer group on %s", stream)
            except redis.ResponseError as exc:
                if "BUSYGROUP" in str(exc):
                    logger.debug("Consumer group already exists on %s", stream)
                else:
                    raise

    # ------------------------------------------------------------------
    # Enqueue (control plane -> worker)
    # ------------------------------------------------------------------

    def enqueue_task(
        self,
        tier: int,
        task_type: str,
        payload: dict[str, Any],
        task_id: str | None = None,
    ) -> str:
        """Write a task to the appropriate tier stream. Returns the task_id."""
        if tier not in TIERS:
            raise ValueError(f"Invalid tier {tier!r}. Must be one of {TIERS}")

        task_id = task_id or str(uuid.uuid4())
        stream = TASK_STREAM.format(tier=tier)

        msg_id = self._r.xadd(
            stream,
            {
                "task_id": task_id,
                "task_type": task_type,
                "payload": json.dumps(payload),
            },
        )
        logger.debug("Enqueued task %s -> %s (msg %s)", task_id, stream, msg_id)
        return task_id

    # ------------------------------------------------------------------
    # Consume (worker <- control plane)
    # ------------------------------------------------------------------

    def read_tasks(
        self,
        tier: int,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> list[dict[str, Any]]:
        """Read up to *count* undelivered tasks for a given tier."""
        stream = TASK_STREAM.format(tier=tier)
        raw = self._r.xreadgroup(
            CONSUMER_GROUP,
            consumer_name,
            {stream: ">"},
            count=count,
            block=block_ms,
        )
        if not raw:
            return []

        tasks: list[dict[str, Any]] = []
        for _stream, messages in raw:
            for msg_id, fields in messages:
                tasks.append({
                    "_msg_id": msg_id,
                    "task_id": fields["task_id"],
                    "task_type": fields["task_type"],
                    "payload": json.loads(fields["payload"]),
                })
        return tasks

    def ack_task(self, tier: int, msg_id: str) -> None:
        """Acknowledge successful processing of a task message."""
        stream = TASK_STREAM.format(tier=tier)
        self._r.xack(stream, CONSUMER_GROUP, msg_id)
        logger.debug("ACKed msg %s on %s", msg_id, stream)

    # ------------------------------------------------------------------
    # Results (worker -> control plane)
    # ------------------------------------------------------------------

    def publish_result(
        self,
        task_id: str,
        tier: int,
        device_id: str,
        result: dict[str, Any],
        error: str | None = None,
        metadata: Any | None = None,
    ) -> str:
        """Write a task result (or error) to the results stream."""
        meta_str = ""
        if metadata is not None:
            if hasattr(metadata, "to_dict"):
                meta_str = json.dumps(metadata.to_dict())
            elif isinstance(metadata, dict):
                meta_str = json.dumps(metadata)

        msg_id = self._r.xadd(
            RESULT_STREAM,
            {
                "task_id": task_id,
                "tier": str(tier),
                "device_id": device_id,
                "result": json.dumps(result),
                "error": error or "",
                "metadata": meta_str,
            },
        )
        logger.debug("Published result for task %s (msg %s)", task_id, msg_id)
        return msg_id

    def read_results(
        self,
        consumer_name: str,
        count: int = 50,
        block_ms: int = 1000,
    ) -> list[dict[str, Any]]:
        """Read pending results from the results stream."""
        raw = self._r.xreadgroup(
            CONSUMER_GROUP,
            consumer_name,
            {RESULT_STREAM: ">"},
            count=count,
            block=block_ms,
        )
        if not raw:
            return []

        results: list[dict[str, Any]] = []
        for _stream, messages in raw:
            for msg_id, fields in messages:
                meta_raw = fields.get("metadata", "")
                results.append({
                    "_msg_id": msg_id,
                    "task_id": fields["task_id"],
                    "tier": int(fields["tier"]),
                    "device_id": fields["device_id"],
                    "result": json.loads(fields["result"]),
                    "error": fields["error"] or None,
                    "metadata": json.loads(meta_raw) if meta_raw else None,
                })
        return results

    def ack_result(self, msg_id: str) -> None:
        """Acknowledge a result message."""
        self._r.xack(RESULT_STREAM, CONSUMER_GROUP, msg_id)

    # ------------------------------------------------------------------
    # Dead-letter queue
    # ------------------------------------------------------------------

    def enqueue_dlq(
        self,
        task_id: str,
        task_type: str,
        tier: int,
        error: str,
        retry_count: int,
    ) -> str:
        """Write a permanently failed task to the DLQ stream."""
        msg_id = self._r.xadd(
            DLQ_STREAM,
            {
                "task_id": task_id,
                "task_type": task_type,
                "tier": str(tier),
                "error": error,
                "retry_count": str(retry_count),
            },
        )
        logger.warning(
            "Task %s moved to DLQ after %d retries (error: %s)",
            task_id, retry_count, error,
        )
        return msg_id

    def read_dlq(self, count: int = 100) -> list[dict[str, Any]]:
        """Read entries from the DLQ stream (non-destructive XRANGE scan)."""
        raw = self._r.xrange(DLQ_STREAM, count=count)
        entries = []
        for msg_id, fields in raw:
            entries.append({
                "_msg_id": msg_id,
                "task_id": fields["task_id"],
                "task_type": fields["task_type"],
                "tier": int(fields["tier"]),
                "error": fields["error"],
                "retry_count": int(fields["retry_count"]),
            })
        return entries

    def dlq_length(self) -> int:
        """Return the current number of entries in the DLQ."""
        return self._r.xlen(DLQ_STREAM)

    # ------------------------------------------------------------------
    # Pending entry recovery (XAUTOCLAIM)
    # ------------------------------------------------------------------

    def reclaim_stale_tasks(
        self,
        tier: int,
        consumer_name: str,
        min_idle_ms: int = 600_000,
        count: int = 10,
    ) -> list[dict[str, Any]]:
        """Claim tasks delivered to a crashed consumer and not ACKed."""
        stream = TASK_STREAM.format(tier=tier)
        result = self._r.xautoclaim(
            stream, CONSUMER_GROUP, consumer_name,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
        _next_id, messages, _deleted = result
        reclaimed = []
        for msg_id, fields in messages:
            reclaimed.append({
                "_msg_id": msg_id,
                "task_id": fields["task_id"],
                "task_type": fields["task_type"],
                "payload": json.loads(fields["payload"]),
            })
        if reclaimed:
            logger.warning("Reclaimed %d stale tasks on %s", len(reclaimed), stream)
        return reclaimed
