#!/usr/bin/env python3
"""
AIMESH Redis smoke test — AIMESH-8
Verifies that Redis Streams + Pub/Sub are working correctly.

Usage:
    # Start Redis first:
    #   cd infra && docker compose up -d
    #
    # Then run this from the repo root:
    #   python scripts/smoke_test_redis.py

Set REDIS_URL env var to override the default (redis://localhost:6379/0).
"""
import json
import sys
import threading
import time

# Allow running from repo root without installing the package
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from control_plane.redis.client import RedisClient
from control_plane.redis.streams import StreamsClient
from control_plane.redis.pubsub import PubSubClient, REGISTRY_CHANNEL

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def check(label: str, ok: bool) -> None:
    print(f"  {PASS if ok else FAIL}  {label}")
    if not ok:
        sys.exit(1)


def main() -> None:
    print("\n=== AIMESH Redis smoke test ===\n")

    # ------------------------------------------------------------------
    # 1. Connection
    # ------------------------------------------------------------------
    print("1. Connection")
    client = RedisClient()
    check("ping", client.ping())

    # ------------------------------------------------------------------
    # 2. Streams bootstrap
    # ------------------------------------------------------------------
    print("\n2. Streams bootstrap")
    streams = StreamsClient(client)
    streams.ensure_streams()

    r = client.r
    # Verify all tier streams + results stream exist
    for tier in range(5):
        key = f"aimesh:tasks:tier{tier}"
        check(f"stream exists: {key}", r.exists(key) == 1)
    check("stream exists: aimesh:results", r.exists("aimesh:results") == 1)

    # ------------------------------------------------------------------
    # 3. Enqueue + read + ack (tier 2)
    # ------------------------------------------------------------------
    print("\n3. Enqueue → read → ack (tier 2)")
    task_id = streams.enqueue_task(
        tier=2,
        task_type="text_generation",
        payload={"prompt": "Hello from smoke test", "max_tokens": 64},
    )
    check(f"task enqueued (id={task_id})", bool(task_id))

    tasks = streams.read_tasks(tier=2, consumer_name="smoke-test-consumer", count=1)
    check("task received", len(tasks) == 1)
    check("task_id matches", tasks[0]["task_id"] == task_id)
    check("payload round-trips", tasks[0]["payload"]["prompt"] == "Hello from smoke test")

    streams.ack_task(tier=2, msg_id=tasks[0]["_msg_id"])
    check("task ACKed", True)

    # ------------------------------------------------------------------
    # 4. Publish result + read result
    # ------------------------------------------------------------------
    print("\n4. Publish result → read result")
    streams.publish_result(
        task_id=task_id,
        tier=2,
        device_id="desktop-pc",
        result={"text": "smoke test response", "tokens": 5},
    )

    results = streams.read_results(consumer_name="smoke-test-consumer", count=1)
    check("result received", len(results) == 1)
    check("result task_id matches", results[0]["task_id"] == task_id)
    check("result device_id matches", results[0]["device_id"] == "desktop-pc")
    check("result payload intact", results[0]["result"]["text"] == "smoke test response")
    streams.ack_result(results[0]["_msg_id"])
    check("result ACKed", True)

    # ------------------------------------------------------------------
    # 5. Pub/Sub
    # ------------------------------------------------------------------
    print("\n5. Pub/Sub")
    pubsub = PubSubClient(client)
    received: list[dict] = []
    event = threading.Event()

    def handler(channel: str, data: dict) -> None:
        received.append({"channel": channel, "data": data})
        event.set()

    pubsub.subscribe([REGISTRY_CHANNEL], handler)
    time.sleep(0.2)  # let subscriber thread start

    pubsub.publish_registry_event("device_joined", {"device_id": "smoke-test-device", "tier": 2})
    event.wait(timeout=3)
    pubsub.stop()

    check("event received", len(received) == 1)
    check("event type correct", received[0]["data"].get("type") == "device_joined")
    check("event device_id correct", received[0]["data"].get("device_id") == "smoke-test-device")

    # ------------------------------------------------------------------
    # 6. Cleanup
    # ------------------------------------------------------------------
    print("\n6. Cleanup")
    keys_to_delete = (
        [f"aimesh:tasks:tier{t}" for t in range(5)]
        + ["aimesh:results"]
    )
    r.delete(*keys_to_delete)
    check("test streams deleted", True)
    client.close()

    print("\n\033[92mAll checks passed — Redis is ready for AIMESH.\033[0m\n")


if __name__ == "__main__":
    main()
