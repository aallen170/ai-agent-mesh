"""
smoke_test_router.py — Manual end-to-end test for AIMESH-10 (Task Router).

Tests the full submit → dispatch → result → completion cycle without
needing a real worker agent — results are simulated by publishing directly
to the results stream, then calling process_results_stream().

Run from the ai-agent-mesh directory:

    python scripts/smoke_test_router.py

Requires Redis to be running (docker compose up -d in infra/).
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from src.control_plane.redis.client import RedisClient
from src.control_plane.redis.streams import StreamsClient
from src.control_plane.queue import TaskRequest, TaskRouter, TaskStatus


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nAIMESH-10 Task Router — Smoke Test")

    # ── Setup ──────────────────────────────────────────────────────────────
    section("1. Connect to Redis and ensure streams exist")
    redis_client = RedisClient()
    check(redis_client.ping(), "Redis is reachable", "Cannot reach Redis — is it running?")

    streams = StreamsClient(redis_client)
    streams.ensure_streams()
    ok("Tier streams and consumer groups created / verified")

    router = TaskRouter(redis_client, streams)

    # ── Step 2: Submit a task ─────────────────────────────────────────────
    section("2. Submit a task to Tier 2 (dGPU)")
    request = TaskRequest(
        task_type="llm_inference",
        payload={"prompt": "Explain Redis Streams in one sentence.", "model": "llama3:70b"},
        tier=2,
    )
    record = router.submit(request)

    check(record.task_id is not None, f"Task ID assigned: {record.task_id}", "No task ID assigned")
    check(record.status == TaskStatus.DISPATCHED, "Status is DISPATCHED after submit", f"Status is {record.status!r}, expected DISPATCHED")
    check(record.tier == 2, "Tier is 2", f"Tier is {record.tier}")
    check(record.dispatched_at is not None, "dispatched_at is set", "dispatched_at is None")
    ok(f"Record: {record}")

    # ── Step 3: Verify record persisted in Redis ───────────────────────────
    section("3. Verify task record was persisted to Redis")
    fetched = router.get_task(record.task_id)
    check(fetched is not None, "get_task() returned the record", "get_task() returned None")
    check(fetched.task_type == "llm_inference", "task_type round-tripped correctly", "task_type mismatch")
    check(fetched.status == TaskStatus.DISPATCHED, "Status persisted correctly", f"Status is {fetched.status!r}")
    check(fetched.tier == 2, "Tier persisted correctly", f"Tier is {fetched.tier}")

    # ── Step 4: Verify task appears in the tier-2 Redis stream ────────────
    section("4. Verify task is in the tier-2 Redis stream")
    # Read tasks as if we are a worker — consumer name must differ from any other
    # active consumer to get fresh messages; use a unique test consumer name.
    test_consumer = f"smoke-test-{int(time.time())}"
    queued_tasks = streams.read_tasks(tier=2, consumer_name=test_consumer, count=10, block_ms=500)

    our_task = next((t for t in queued_tasks if t["task_id"] == record.task_id), None)
    check(our_task is not None, "Task found in tier-2 stream", "Task NOT found in tier-2 stream")
    check(our_task["task_type"] == "llm_inference", "task_type in stream is correct", "task_type mismatch in stream")
    check(our_task["payload"]["prompt"].startswith("Explain"), "Payload round-tripped correctly", "Payload mismatch")

    # ACK the message so it doesn't clog the stream after the test
    streams.ack_task(tier=2, msg_id=our_task["_msg_id"])
    ok("Test consumer ACKed the message")

    # ── Step 5: Simulate a result from a worker ────────────────────────────
    section("5. Simulate a worker publishing a result")
    streams.publish_result(
        task_id=record.task_id,
        tier=2,
        device_id="desktop-pc",
        result={"text": "Redis Streams are an append-only log that supports consumer groups."},
        error=None,
    )
    ok("Simulated result published to aimesh:results stream")

    # ── Step 6: Process the results stream ────────────────────────────────
    section("6. Process results stream and mark task completed")
    n = router.process_results_stream(consumer_name=f"smoke-cp-{int(time.time())}")
    check(n >= 1, f"process_results_stream() processed {n} result(s)", "No results processed")

    completed = router.get_task(record.task_id)
    check(completed is not None, "Task record still exists", "Task record gone after processing")
    check(completed.status == TaskStatus.COMPLETED, "Task status is COMPLETED", f"Status is {completed.status!r}")
    check(completed.device_id == "desktop-pc", "device_id set correctly", f"device_id is {completed.device_id!r}")
    check(completed.error is None, "No error on completed task", f"Error: {completed.error}")
    check("Redis Streams" in (completed.result or {}).get("text", ""), "Result payload persisted correctly", "Result payload missing or wrong")
    check(completed.duration is not None, f"Duration recorded: {completed.duration:.3f}s", "Duration is None")

    # ── Step 7: Simulate a failing task ───────────────────────────────────
    section("7. Submit a task and simulate a worker failure")
    fail_request = TaskRequest(
        task_type="llm_inference",
        payload={"prompt": "This one will fail.", "model": "llama3:70b"},
        tier=2,
        max_retries=0,  # exhaust retries immediately so first error → FAILED
    )
    fail_record = router.submit(fail_request)

    # Drain the stream so it doesn't accumulate
    draining = streams.read_tasks(tier=2, consumer_name=test_consumer, count=10, block_ms=500)
    for t in draining:
        streams.ack_task(tier=2, msg_id=t["_msg_id"])

    streams.publish_result(
        task_id=fail_record.task_id,
        tier=2,
        device_id="desktop-pc",
        result={},
        error="OllamaError: model not loaded",
    )

    n = router.process_results_stream(consumer_name=f"smoke-cp-{int(time.time())}-2")
    failed = router.get_task(fail_record.task_id)
    check(failed.status == TaskStatus.FAILED, "Failed task status is FAILED", f"Status is {failed.status!r}")
    check(failed.error == "OllamaError: model not loaded", "Error message persisted", f"Error is {failed.error!r}")

    # ── Step 8: Listing and counting ──────────────────────────────────────
    section("8. Task listing and status counts")
    counts = router.count_by_status()
    print(f"  Status counts: {counts}")
    check(counts[TaskStatus.COMPLETED] >= 1, "At least 1 COMPLETED task", "No COMPLETED tasks")
    check(counts[TaskStatus.FAILED] >= 1, "At least 1 FAILED task", "No FAILED tasks")

    completed_list = router.list_completed()
    check(any(t.task_id == record.task_id for t in completed_list), "list_completed() includes our task", "Our task missing from list_completed()")

    failed_list = router.list_failed()
    check(any(t.task_id == fail_record.task_id for t in failed_list), "list_failed() includes failed task", "Failed task missing from list_failed()")

    # ── Step 9: is_terminal property ──────────────────────────────────────
    section("9. TaskRecord.is_terminal")
    check(completed.is_terminal, "COMPLETED record is terminal", "COMPLETED record is NOT terminal")
    check(failed.is_terminal, "FAILED record is terminal", "FAILED record is NOT terminal")

    dispatched_still = router.get_task(fail_record.task_id)   # now completed
    check(dispatched_still.is_terminal, "Post-fail record is terminal", "Expected terminal")

    # ── Cleanup ────────────────────────────────────────────────────────────
    redis_client.close()

    print(f"\n{'═' * 60}")
    print("  All checks passed — AIMESH-10 task router is working correctly.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
