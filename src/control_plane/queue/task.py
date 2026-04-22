"""
task.py — Task models for the AIMESH control-plane queue.

TaskRequest     What the caller submits: where to send it (tier) and what to run.
TaskStatus      The lifecycle states a task moves through.
TaskRecord      The full persisted record of a task, stored as a Redis hash.

Redis key layout
----------------
aimesh:task:{task_id}   Hash  — one per task (TaskRecord fields)
aimesh:tasks:index      Set   — all known task_ids (for enumeration / GC)
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class TaskStatus:
    """
    String constants for task lifecycle states.

    Using plain string constants (rather than an Enum) so values survive
    a Redis round-trip without any coercion — Redis stores everything as
    bytes / strings.

    Flow:  PENDING → DISPATCHED → COMPLETED
                               → FAILED
    """
    PENDING = "pending"
    DISPATCHED = "dispatched"   # Written to the tier stream, awaiting a worker
    COMPLETED = "completed"     # Worker returned a successful result
    FAILED = "failed"           # Worker returned an error, or timed out


@dataclass
class TaskRequest:
    """
    Input object passed to TaskRouter.submit().

    Fields
    ------
    task_type   Application-defined label for the kind of work
                (e.g. "llm_inference", "embedding", "summarise").
    payload     Arbitrary dict passed through to the worker unchanged.
    tier        Which compute tier should handle this task (0–4).
                The caller is responsible for choosing the right tier.
                Tier-based auto-classification will be added in AIMESH-12.
    task_id     Optional stable ID.  Auto-generated (UUID4) if not provided.
                Pass an explicit ID to make re-submissions idempotent.
    """
    task_type: str
    payload: dict[str, Any]
    tier: int
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class TaskRecord:
    """
    The full persisted record of a task.  Stored as a flat Redis hash so
    every field is a string; use to_dict() / from_dict() to convert.

    Fields
    ------
    task_id         Globally unique identifier.
    task_type       Mirrors TaskRequest.task_type.
    tier            Target compute tier.
    status          Current lifecycle state (see TaskStatus).
    enqueued_at     Unix timestamp when the task was first submitted.
    dispatched_at   Unix timestamp when the task was written to the stream.
    completed_at    Unix timestamp when the result was received (or None).
    device_id       ID of the worker that handled the task (set on completion).
    result          Dict returned by the worker on success (or None).
    error           Error string on failure (or None on success).
    """
    task_id: str
    task_type: str
    tier: int
    status: str = TaskStatus.PENDING
    enqueued_at: float = field(default_factory=time.time)
    dispatched_at: float | None = None
    completed_at: float | None = None
    device_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    # ------------------------------------------------------------------
    # Redis serialisation — everything must be a string for HSET
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, str]:
        """Flatten to a string-only dict suitable for Redis HSET."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "tier": str(self.tier),
            "status": self.status,
            "enqueued_at": str(self.enqueued_at),
            "dispatched_at": str(self.dispatched_at) if self.dispatched_at is not None else "",
            "completed_at": str(self.completed_at) if self.completed_at is not None else "",
            "device_id": self.device_id or "",
            "result": json.dumps(self.result) if self.result is not None else "",
            "error": self.error or "",
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskRecord":
        """Reconstruct a TaskRecord from a Redis HGETALL response."""
        def _opt_float(val: str) -> float | None:
            return float(val) if val else None

        def _opt_str(val: str) -> str | None:
            return val if val else None

        def _opt_json(val: str) -> dict[str, Any] | None:
            return json.loads(val) if val else None

        return cls(
            task_id=d["task_id"],
            task_type=d["task_type"],
            tier=int(d["tier"]),
            status=d.get("status", TaskStatus.PENDING),
            enqueued_at=float(d.get("enqueued_at", 0)),
            dispatched_at=_opt_float(d.get("dispatched_at", "")),
            completed_at=_opt_float(d.get("completed_at", "")),
            device_id=_opt_str(d.get("device_id", "")),
            result=_opt_json(d.get("result", "")),
            error=_opt_str(d.get("error", "")),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """True if the task is in a final state (no further state changes expected)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    @property
    def duration(self) -> float | None:
        """Wall-clock seconds from enqueue to completion, or None if not yet done."""
        if self.completed_at is None:
            return None
        return self.completed_at - self.enqueued_at

    def __repr__(self) -> str:
        return (
            f"TaskRecord(id={self.task_id!r}, type={self.task_type!r}, "
            f"tier={self.tier}, status={self.status!r})"
        )
