"""
StreamsClient — helpers for the AIMESH tier-based task queues.

Each compute tier gets its own Redis Stream:
    aimesh:tasks:tier0   iPhones / iPads / Android tablet
    aimesh:tasks:tier1   iGPU laptop
    aimesh:tasks:tier2   dGPU laptop + desktop PC
    aimesh:tasks:tier3   RunPod serverless
    aimesh:tasks:tier4   Claude Sonnet / Opus

Results are written back to a single results stream:
    aimesh:results

Consumer groups are created once per stream (MKSTREAM if the stream
doesn't exist yet) and are idempotent — safe to call on every startup.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import redis

from .client import RedisClient

logger = logging.getLogger(__name__)

# Stream key templates
TASK_STREAM = "aimesh:tasks:tier{tier}"
RESULT_STREAM = "aimesh:results"
CONSUMER_GROUP = "aimesh-workers"

# All valid tiers
TIERS = [0, 1, 2, 3, 4]


class StreamsClient:
    """
    High-level interface for enqueuing tasks and reading results via
    Redis Streams.  Instantiate once per process; thread-safe.
    """

    def __init__(self, client: RedisClient) -> None:
        self._r = client.r

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def ensure_streams(self) -> None:
        """
        Create all tier streams and their consumer group if they don't
        already exist.  Safe to call on every control-plane startup.
        """
        streams = [TASK_STREAM.format(tier=t) for t in TIERS] + [RESULT_STREAM]
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
    # Enqueue (control plane → worker)
    # ------------------------------------------------------------------

    def enqueue_task(
        self,
        tier: int,
        task_type: str,
        payload: dict[str, Any],
        task_id: str | None = None,
    ) -> str:
        """
        Write a task to the appropriate tier stream.

        Returns the Redis message ID of the enqueued entry.
        """
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
        logger.debug("Enqueued task %s → %s (msg %s)", task_id, stream, msg_id)
        return task_id

    # ------------------------------------------------------------------
    # Consume (worker ← control plane)
    # ------------------------------------------------------------------

    def read_tasks(
        self,
        tier: int,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> list[dict[str, Any]]:
        """
        Read up to *count* undelivered tasks for a given tier.

        Uses XREADGROUP so each message is delivered to exactly one
        consumer.  Blocks for *block_ms* milliseconds if the stream is
        empty (set to 0 to block indefinitely).

        Returns a list of task dicts (includes ``_msg_id`` for ACK).
        """
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
                task = {
                    "_msg_id": msg_id,
                    "task_id": fields["task_id"],
                    "task_type": fields["task_type"],
                    "payload": json.loads(fields["payload"]),
                }
                tasks.append(task)
        return tasks

    def ack_task(self, tier: int, msg_id: str) -> None:
        """Acknowledge successful processing of a task message."""
        stream = TASK_STREAM.format(tier=tier)
        self._r.xack(stream, CONSUMER_GROUP, msg_id)
        logger.debug("ACKed msg %s on %s", msg_id, stream)

    # ------------------------------------------------------------------
    # Results (worker → control plane)
    # ------------------------------------------------------------------

    def publish_result(
        self,
        task_id: str,
        tier: int,
        device_id: str,
        result: dict[str, Any],
        error: str | None = None,
    ) -> str:
        """Write a task result (or error) to the results stream."""
        msg_id = self._r.xadd(
            RESULT_STREAM,
            {
                "task_id": task_id,
                "tier": str(tier),
                "device_id": device_id,
                "result": json.dumps(result),
                "error": error or "",
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
                results.append({
                    "_msg_id": msg_id,
                    "task_id": fields["task_id"],
                    "tier": int(fields["tier"]),
                    "device_id": fields["device_id"],
                    "result": json.loads(fields["result"]),
                    "error": fields["error"] or None,
                })
        return results

    def ack_result(self, msg_id: str) -> None:
        """Acknowledge a result message."""
        self._r.xack(RESULT_STREAM, CONSUMER_GROUP, msg_id)

    # ------------------------------------------------------------------
    # Pending entry recovery (XAUTOCLAIM)
    # ------------------------------------------------------------------

    def reclaim_stale_tasks(
        self,
        tier: int,
        consumer_name: str,
        min_idle_ms: int = 600_000,  # 10 minutes
        count: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Claim tasks that were delivered to a crashed consumer and not
        ACKed within *min_idle_ms* milliseconds.  Call periodically from
        the control plane watchdog.
        """
        stream = TASK_STREAM.format(tier=tier)
        result = self._r.xautoclaim(
            stream, CONSUMER_GROUP, consumer_name,
            min_idle_time=min_idle_ms,
            start_id="0-0",
            count=count,
        )
        # xautoclaim returns (next_start_id, [(msg_id, fields), ...], deleted_ids)
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
