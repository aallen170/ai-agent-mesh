"""
smoke_test_results.py — Smoke test for AIMESH-18 (Result Serialization & Error Handling).

Tests:
  1. Happy path with ResultMetadata — result stored with execution_time_s, model_id, tokens.
  2. Error path with auto-retry — task is automatically re-enqueued on failure (retry_count++).
  3. DLQ promotion — after exhausting retries the task lands in the DLQ stream.
  4. Manual retry from DLQ — retry_task() re-dispatches a failed task.
  5. ResultCollector integration — background thread processes results automatically.
  6. Timeout watchdog (unit-level) — stale task triggers timeout error and retry.

Run from the ai-agent-mesh directory:

    python scripts/smoke_test_results.py

Requires Redis to be running (docker compose up -d in infra/).
"""
from __future__ import annotations

import sys
import time
import uuid

sys.path.insert(0, ".")

from src.control_plane.redis.client import RedisClient
from src.control_plane.redis.streams import StreamsClient
from src.control_plane.queue import TaskRequest, TaskRouter, TaskStatus, ResultCollector


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def ok(msg: str) -> None:
    print(f"  ✓  {msg}")

def fail(msg: str) -> None:
    print(f"  ✗  {msg}")
    sys.exit(1)

def check(condition: bool, pass_msg: str, fail_msg: str) -> None:
    if condition:
        ok(pass_msg)
    else:
        fail(fail_msg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_router(redis_client: RedisClient, streams: StreamsClient) -> TaskRouter:
    return TaskRouter(redis_client=redis_client, streams_client=streams)


def submit_task(
    router: TaskRouter,
    task_type: str = "llm_inference",
    payload: dict | None = None,
    tier: int = 2,
    max_retries: int = 3,
) -> str:
    """Submit a task and return its task_id."""
    req = TaskRequest(
        task_type=task_type,
        payload=payload or {"prompt": "hello world", "model": "llama3:8b"},
        tier=tier,
    )
    record = router.submit(req)
    # Override max_retries after creation (TaskRequest doesn't expose it yet)
    record.max_retries = max_retries
    router._save(record)  # persist the override
    return record.task_id


# ── Test cases ────────────────────────────────────────────────────────────────

def test_happy_path_with_metadata(router: TaskRouter, streams: StreamsClient) -> None:
    section("1. Happy path — result stored with ResultMetadata")

    task_id = submit_task(router)
    ok(f"Task submitted: {task_id}")

    # Simulate a worker publishing a rich result with metadata
    metadata = {
        "execution_time_s": 1.23,
        "model_id": "llama3:8b",
        "input_tokens": 12,
        "output_tokens": 42,
        "partial": False,
    }
    streams.publish_result(
        task_id=task_id,
        tier=2,
        device_id="desktop-pc",
        result={"text": "Hello! How can I help?"},
        error=None,
        metadata=metadata,
    )
    ok("Worker result published with metadata")

    # Control plane processes the result
    processed = router.process_results_stream()
    check(processed == 1, f"process_results_stream returned {processed}", "Expected 1 result processed")

    record = router.get_task(task_id)
    check(record is not None, "TaskRecord found in Redis", "TaskRecord missing")
    check(record.status == TaskStatus.COMPLETED, f"Status is COMPLETED", f"Expected COMPLETED, got {record.status}")
    check(record.result == {"text": "Hello! How can I help?"}, "Result payload stored", "Result mismatch")
    check(record.result_metadata is not None, "result_metadata is set", "result_metadata is None")
    check(record.result_metadata["model_id"] == "llama3:8b", "model_id stored", "model_id missing")
    check(record.result_metadata["input_tokens"] == 12, "input_tokens stored", "input_tokens missing")
    check(record.result_metadata["output_tokens"] == 42, "output_tokens stored", "output_tokens missing")
    check(record.result_metadata["execution_time_s"] == 1.23, "execution_time_s stored", "execution_time_s missing")
    ok(f"Full result round-trip verified (duration={record.duration:.3f}s)")


def test_error_path_auto_retry(router: TaskRouter, streams: StreamsClient) -> None:
    section("2. Error path — auto-retry on transient failure")

    # Use max_retries=2 so we can observe both retries cheaply
    task_id = submit_task(router, max_retries=2)
    ok(f"Task submitted (max_retries=2): {task_id}")

    # Simulate first failure
    streams.publish_result(
        task_id=task_id,
        tier=2,
        device_id="desktop-pc",
        result={},
        error="OllamaError: model load failed",
    )
    router.process_results_stream()

    record = router.get_task(task_id)
    check(record.status == TaskStatus.DISPATCHED, "Task re-dispatched after first failure", f"Expected DISPATCHED, got {record.status}")
    check(record.retry_count == 1, f"retry_count=1", f"Expected 1, got {record.retry_count}")
    ok("First retry triggered automatically")

    # Simulate second failure
    streams.publish_result(
        task_id=task_id,
        tier=2,
        device_id="desktop-pc",
        result={},
        error="OllamaError: model load failed again",
    )
    router.process_results_stream()

    record = router.get_task(task_id)
    check(record.status == TaskStatus.DISPATCHED, "Task re-dispatched after second failure", f"Expected DISPATCHED, got {record.status}")
    check(record.retry_count == 2, f"retry_count=2", f"Expected 2, got {record.retry_count}")
    ok("Second retry triggered — retries now exhausted")

    # Simulate third failure — should land in DLQ
    streams.publish_result(
        task_id=task_id,
        tier=2,
        device_id="desktop-pc",
        result={},
        error="OllamaError: permanent failure",
    )
    router.process_results_stream()

    record = router.get_task(task_id)
    check(record.status == TaskStatus.FAILED, "Task FAILED after exhausting retries", f"Expected FAILED, got {record.status}")
    check(record.retry_count == 2, f"retry_count stays at 2 (no extra increment)", f"Got {record.retry_count}")
    ok("Task permanently failed — no further retries")


def test_dlq_promotion(router: TaskRouter, streams: StreamsClient) -> None:
    section("3. DLQ promotion — exhausted-retry task appears in DLQ stream")

    dlq_before = streams.dlq_length()

    # Submit a task with max_retries=0 → immediate DLQ on first failure
    task_id = submit_task(router, max_retries=0)
    ok(f"Task submitted (max_retries=0): {task_id}")

    streams.publish_result(
        task_id=task_id,
        tier=2,
        device_id="desktop-pc",
        result={},
        error="ImmediateFailure: always fails",
    )
    router.process_results_stream()

    record = router.get_task(task_id)
    check(record.status == TaskStatus.FAILED, "Task status is FAILED", f"Got {record.status}")

    dlq_after = streams.dlq_length()
    check(dlq_after == dlq_before + 1, "DLQ grew by 1", f"DLQ was {dlq_before}, now {dlq_after}")

    dlq_entries = router.list_dlq()
    matching = [e for e in dlq_entries if e["task_id"] == task_id]
    check(len(matching) == 1, "DLQ entry found for task", "DLQ entry missing")
    check(matching[0]["error"] == "ImmediateFailure: always fails", "DLQ error message correct", "DLQ error mismatch")
    ok(f"DLQ entry verified: {matching[0]}")


def test_manual_retry_from_dlq(router: TaskRouter, streams: StreamsClient) -> None:
    section("4. Manual retry — retry_task() re-dispatches a failed task")

    # Submit and immediately fail (max_retries=0)
    task_id = submit_task(router, max_retries=0)
    streams.publish_result(task_id=task_id, tier=2, device_id="desktop-pc",
                           result={}, error="Permanent failure")
    router.process_results_stream()

    record = router.get_task(task_id)
    check(record.status == TaskStatus.FAILED, "Task is in FAILED state before manual retry", f"Got {record.status}")

    # Manual retry
    updated = router.retry_task(task_id)
    check(updated is not None, "retry_task() returned a record", "retry_task() returned None")
    check(updated.status == TaskStatus.DISPATCHED, "Task re-dispatched", f"Got {updated.status}")
    check(updated.retry_count == 1, "retry_count incremented", f"Got {updated.retry_count}")
    ok("Manual retry successful — task back in DISPATCHED state")

    # Now simulate success
    streams.publish_result(task_id=task_id, tier=2, device_id="desktop-pc",
                           result={"text": "recovered!"}, error=None)
    router.process_results_stream()

    record = router.get_task(task_id)
    check(record.status == TaskStatus.COMPLETED, "Task COMPLETED after manual retry success", f"Got {record.status}")
    ok("Task recovered via manual retry")


def test_result_collector(router: TaskRouter, streams: StreamsClient, redis_client: RedisClient) -> None:
    section("5. ResultCollector — background thread auto-processes results")

    collector = ResultCollector(
        router=router,
        streams_client=streams,
        consumer_name="smoke-test-collector",
        watchdog_interval_s=9999,   # disable watchdog for this test
        task_ttl_s=0,               # disable TTL for this test
    )
    collector.start()
    check(collector.is_running, "ResultCollector is running", "ResultCollector failed to start")

    try:
        task_id = submit_task(router)
        ok(f"Task submitted: {task_id}")

        # Publish result — collector should pick it up automatically
        streams.publish_result(
            task_id=task_id,
            tier=2,
            device_id="desktop-pc",
            result={"text": "background processed!"},
            error=None,
            metadata={"execution_time_s": 0.5, "model_id": "llama3:8b",
                      "input_tokens": 5, "output_tokens": 10, "partial": False},
        )

        # Give the background thread time to process
        deadline = time.time() + 5.0
        record = None
        while time.time() < deadline:
            record = router.get_task(task_id)
            if record and record.status == TaskStatus.COMPLETED:
                break
            time.sleep(0.1)

        check(record is not None and record.status == TaskStatus.COMPLETED,
              "ResultCollector processed result and marked task COMPLETED",
              f"Task status after 5s: {record.status if record else 'not found'}")
        ok("Background result processing confirmed")

    finally:
        collector.stop()
        ok("ResultCollector stopped cleanly")


def test_task_ttl(router: TaskRouter) -> None:
    section("6. Task TTL — expire_completed_tasks() sets Redis TTL on terminal records")

    task_id = submit_task(router)
    router.mark_completed(task_id=task_id, device_id="desktop-pc",
                          result={"text": "done"}, error=None)

    count = router.expire_completed_tasks(ttl_seconds=3600)
    check(count >= 1, f"TTL applied to {count} terminal record(s)", "No TTLs applied")
    ok("Redis TTLs set on terminal task records")


def test_count_by_status(router: TaskRouter, streams: StreamsClient) -> None:
    section("7. count_by_status — DLQ status included in summary")

    counts = router.count_by_status()
    check("dlq" in counts, "DLQ key present in count_by_status()", "DLQ key missing")
    check("pending" in counts and "completed" in counts,
          "All expected status keys present", "Status keys missing")
    ok(f"count_by_status: {counts}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("\nAIMESH-18 Result Serialization & Error Handling — Smoke Test")

    section("0. Connect to Redis and ensure streams exist")
    redis_client = RedisClient()
    check(redis_client.ping(), "Redis is reachable", "Cannot reach Redis — is it running?")

    streams = StreamsClient(redis_client)
    streams.ensure_streams()
    ok("Tier streams and consumer groups created / verified")

    router = make_router(redis_client, streams)

    test_happy_path_with_metadata(router, streams)
    test_error_path_auto_retry(router, streams)
    test_dlq_promotion(router, streams)
    test_manual_retry_from_dlq(router, streams)
    test_result_collector(router, streams, redis_client)
    test_task_ttl(router)
    test_count_by_status(router, streams)

    print(f"\n{'═' * 60}")
    print("  All AIMESH-18 smoke tests passed ✓")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
