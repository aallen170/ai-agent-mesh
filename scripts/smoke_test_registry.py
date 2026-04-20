"""
smoke_test_registry.py — Manual end-to-end test for AIMESH-9 (Device Registry).

Run from the repo root:

    python scripts/smoke_test_registry.py

Requires Redis to be running (docker compose up -d in infra/).
"""
from __future__ import annotations

import sys
import time

# Make sure the src package is importable from the repo root
sys.path.insert(0, ".")

from src.control_plane.redis.client import RedisClient
from src.control_plane.redis.pubsub import PubSubClient
from src.control_plane.registry.device import Capabilities, DeviceInfo, DeviceStatus, Tier
from src.control_plane.registry.registry import DeviceRegistry


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


# ── Fixtures ──────────────────────────────────────────────────────────────────

DESKTOP = DeviceInfo(
    device_id="desktop-pc",
    tier=Tier.DGPU,
    name="Desktop PC",
    capabilities=Capabilities(
        model_ids=["llama3:70b", "mistral:7b"],
        ram_gb=32.0,
        vram_gb=12.0,
        gpu_name="RTX 3060",
        cpu_cores=16,
        os="Windows 11",
    ),
)

IPAD = DeviceInfo(
    device_id="ipad-1",
    tier=Tier.MOBILE,
    name="iPad #1",
    capabilities=Capabilities(
        model_ids=["phi3:mini"],
        ram_gb=8.0,
        os="iPadOS 17",
    ),
)


# ── Test steps ────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nAIMESH-9 Device Registry — Smoke Test")

    # ── Setup ──────────────────────────────────────────────────────────────
    section("1. Connect to Redis")
    redis_client = RedisClient()
    check(redis_client.ping(), "Redis is reachable", "Cannot reach Redis — is it running?")

    pubsub_client = PubSubClient(redis_client)
    registry = DeviceRegistry(redis_client, pubsub_client, heartbeat_ttl=5.0)

    # Subscribe to registry events so we can see them fire
    events: list[dict] = []
    pubsub_client.subscribe(
        ["aimesh:registry"],
        handler=lambda _ch, msg: events.append(msg),
    )

    # ── Cleanup any leftover state from a previous run ─────────────────────
    registry.deregister(DESKTOP.device_id)
    registry.deregister(IPAD.device_id)

    # ── Step 2: Register devices ───────────────────────────────────────────
    section("2. Register two devices")
    registry.register(DESKTOP)
    registry.register(IPAD)

    desktop = registry.get_device(DESKTOP.device_id)
    check(desktop is not None, "Desktop PC found in registry", "Desktop PC NOT found")
    check(desktop.status == DeviceStatus.ONLINE, "Desktop PC is online", "Desktop PC is not online")
    check(desktop.tier == Tier.DGPU, "Desktop PC tier is DGPU", "Desktop PC tier mismatch")
    check("RTX 3060" in desktop.capabilities.gpu_name, "GPU name round-tripped correctly", "GPU name lost in serialisation")

    ipad = registry.get_device(IPAD.device_id)
    check(ipad is not None, "iPad #1 found in registry", "iPad #1 NOT found")
    check(ipad.tier == Tier.MOBILE, "iPad #1 tier is MOBILE", "iPad tier mismatch")

    # ── Step 3: List and filter ────────────────────────────────────────────
    section("3. List and filter devices")
    all_devices = registry.list_devices()
    check(len(all_devices) >= 2, f"list_devices() returned {len(all_devices)} device(s)", "list_devices() returned too few")

    dgpu_devices = registry.list_devices(tier=Tier.DGPU)
    check(len(dgpu_devices) == 1, "list_devices(tier=DGPU) returns 1 device", "Tier filter not working")
    check(dgpu_devices[0].device_id == DESKTOP.device_id, "Correct device returned for DGPU tier", "Wrong device")

    available = registry.list_available()
    check(len(available) == 2, f"list_available() returns 2 online devices", "Available count wrong")

    summary = registry.count_by_tier()
    print(f"\n  Tier summary: {summary}")
    check(summary[int(Tier.DGPU)]["total"] == 1, "count_by_tier() Tier 2 total = 1", "count_by_tier wrong")

    # ── Step 4: Heartbeat ─────────────────────────────────────────────────
    section("4. Heartbeat")
    result = registry.heartbeat(DESKTOP.device_id)
    check(result is True, "heartbeat() returned True for known device", "heartbeat() returned False unexpectedly")

    result = registry.heartbeat("ghost-device")
    check(result is False, "heartbeat() returned False for unknown device", "heartbeat() should have returned False")

    result = registry.heartbeat(DESKTOP.device_id, status=DeviceStatus.BUSY)
    desktop_busy = registry.get_device(DESKTOP.device_id)
    check(desktop_busy.status == DeviceStatus.BUSY, "Status updated to BUSY via heartbeat", "Status not updated")

    # Reset back to online
    registry.heartbeat(DESKTOP.device_id, status=DeviceStatus.ONLINE)

    # ── Step 5: Stale device expiry ────────────────────────────────────────
    section("5. Stale device expiry (heartbeat_ttl=5s)")
    print("  Waiting 6 seconds without a heartbeat from iPad #1...")
    time.sleep(6)

    # Only send a heartbeat for the desktop to keep it alive
    registry.heartbeat(DESKTOP.device_id)

    expired = registry.expire_stale_devices()
    check(IPAD.device_id in expired, "iPad #1 correctly expired as stale", "iPad #1 was NOT expired")

    ipad_after = registry.get_device(IPAD.device_id)
    check(ipad_after.status == DeviceStatus.OFFLINE, "iPad #1 status is now OFFLINE", "iPad #1 status wrong after expiry")

    desktop_after = registry.get_device(DESKTOP.device_id)
    check(desktop_after.status == DeviceStatus.ONLINE, "Desktop PC is still ONLINE", "Desktop PC was incorrectly expired")

    # ── Step 6: Re-registration ────────────────────────────────────────────
    section("6. Re-registration preserves registered_at")
    original_registered_at = ipad_after.registered_at
    time.sleep(0.1)
    registry.register(IPAD)
    ipad_reregistered = registry.get_device(IPAD.device_id)
    check(
        abs(ipad_reregistered.registered_at - original_registered_at) < 0.01,
        "registered_at preserved after re-registration",
        "registered_at changed on re-registration",
    )
    check(ipad_reregistered.status == DeviceStatus.ONLINE, "iPad #1 back online after re-registration", "Status wrong")

    # ── Step 7: Deregister ─────────────────────────────────────────────────
    section("7. Deregistration")
    result = registry.deregister(DESKTOP.device_id)
    check(result is True, "deregister() returned True for known device", "deregister() returned False")
    check(registry.get_device(DESKTOP.device_id) is None, "Desktop PC removed from registry", "Desktop PC still present")

    result = registry.deregister("ghost-device")
    check(result is False, "deregister() returned False for unknown device", "Should have been False")

    # ── Step 8: Registry events ────────────────────────────────────────────
    section("8. Registry pub/sub events")
    time.sleep(0.3)  # Give the background listener thread a moment to drain
    event_types = [e.get("type") for e in events]
    print(f"  Events received: {event_types}")
    check("device_joined" in event_types, "device_joined event fired", "device_joined event missing")
    check("device_offline" in event_types, "device_offline event fired", "device_offline event missing")
    check("device_left" in event_types, "device_left event fired", "device_left event missing")

    # ── Cleanup ────────────────────────────────────────────────────────────
    registry.deregister(IPAD.device_id)
    pubsub_client.stop()
    redis_client.close()

    print(f"\n{'═' * 60}")
    print("  All checks passed — AIMESH-9 registry is working correctly.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
