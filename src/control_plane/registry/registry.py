"""
registry.py — DeviceRegistry: the authoritative store of all mesh nodes.

Responsibilities
----------------
1. Register a device (write its DeviceInfo hash to Redis).
2. Accept heartbeats and update last_seen / status.
3. Expire devices that stop heartbeating (mark offline).
4. Query devices — by ID, tier, or availability.
5. Emit pub/sub registry events on join / status change / leave.

Redis key layout
----------------
aimesh:registry:device:{device_id}   Hash  — one per device (DeviceInfo fields)
aimesh:registry:index                Set   — all known device_ids (for fast enumeration)
"""
from __future__ import annotations

import logging
import time
import threading
from typing import Optional

from ..redis.client import RedisClient
from ..redis.pubsub import PubSubClient
from .device import DeviceInfo, DeviceStatus, Tier

logger = logging.getLogger(__name__)

# Redis key templates
_DEVICE_KEY = "aimesh:registry:device:{device_id}"
_INDEX_KEY = "aimesh:registry:index"

# How long (seconds) without a heartbeat before a device is marked offline
DEFAULT_HEARTBEAT_TTL = 30.0

# How often the watchdog thread checks for stale devices
DEFAULT_WATCHDOG_INTERVAL = 10.0


class DeviceRegistry:
    """
    Authoritative registry of all AIMESH nodes.

    Thread-safe: registration, heartbeat, and queries can be called from
    any thread.  The optional watchdog runs in its own daemon thread.

    Parameters
    ----------
    redis_client    Shared RedisClient (connection pool is thread-safe).
    pubsub_client   Used to emit join/leave/status events to other components.
    heartbeat_ttl   Seconds after which a silent device is marked offline.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        pubsub_client: PubSubClient,
        heartbeat_ttl: float = DEFAULT_HEARTBEAT_TTL,
    ) -> None:
        self._r = redis_client.r
        self._pubsub = pubsub_client
        self._heartbeat_ttl = heartbeat_ttl
        self._watchdog_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # 1. Registration
    # ------------------------------------------------------------------

    def register(self, device: DeviceInfo) -> None:
        """
        Write (or overwrite) a device record and add it to the index.

        Calling register() on an already-known device acts as a re-registration
        (e.g. after a worker restarts) — it refreshes capabilities and resets
        status to online without losing the original registered_at timestamp.
        """
        existing = self.get_device(device.device_id)
        if existing:
            # Preserve original registration timestamp on re-register
            device.registered_at = existing.registered_at
            logger.info("Re-registering device %r (tier %s)", device.device_id, device.tier.name)
        else:
            logger.info("Registering new device %r (tier %s)", device.device_id, device.tier.name)

        device.status = DeviceStatus.ONLINE
        device.last_seen = time.time()

        # Write the hash in a pipeline — single round-trip
        pipe = self._r.pipeline()
        pipe.hset(_DEVICE_KEY.format(device_id=device.device_id), mapping=device.to_dict())
        pipe.sadd(_INDEX_KEY, device.device_id)
        pipe.execute()

        event_type = "device_rejoined" if existing else "device_joined"
        self._pubsub.publish_registry_event(event_type, {
            "device_id": device.device_id,
            "tier": int(device.tier),
            "name": device.name,
        })

    # ------------------------------------------------------------------
    # 2. Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(
        self,
        device_id: str,
        status: str = DeviceStatus.ONLINE,
        metrics: dict | None = None,
    ) -> bool:
        """
        Record a heartbeat from a device.

        Updates ``last_seen`` and ``status`` in Redis.  If *metrics* is
        provided each key is stored with a ``metric_`` prefix so live system
        data (cpu_pct, mem_pct, etc.) is available alongside static capability
        fields without polluting the DeviceInfo schema.

        Returns True if the device was known, False if it was unrecognised
        (caller should trigger a full re-registration).

        Parameters
        ----------
        device_id   The device sending the heartbeat.
        status      New status string ("online" / "busy" / "offline").
        metrics     Optional dict of live metrics from HealthReporter.get_metrics().
                    Keys are stored as ``metric_<key>`` in the device hash.
                    Pass None (or omit) when psutil is unavailable.
        """
        key = _DEVICE_KEY.format(device_id=device_id)

        # Check existence before updating — unknown devices should re-register
        if not self._r.exists(key):
            logger.warning("Heartbeat from unknown device %r — re-registration needed", device_id)
            return False

        pipe = self._r.pipeline()
        pipe.hset(key, "last_seen", str(time.time()))
        pipe.hset(key, "status", status)

        if metrics:
            for k, v in metrics.items():
                pipe.hset(key, f"metric_{k}", str(v))

        pipe.execute()

        logger.debug(
            "Heartbeat from %r (status=%s, metrics=%s)",
            device_id, status, list(metrics.keys()) if metrics else None,
        )
        return True

    # ------------------------------------------------------------------
    # 3. Expiry / watchdog
    # ------------------------------------------------------------------

    def expire_stale_devices(self) -> list[str]:
        """
        Check all registered devices and mark offline any that haven't
        heartbeated within heartbeat_ttl seconds.

        Returns the list of device_ids that were marked offline this pass.
        """
        now = time.time()
        expired: list[str] = []

        for device_id in self._r.smembers(_INDEX_KEY):
            key = _DEVICE_KEY.format(device_id=device_id)
            raw = self._r.hmget(key, "status", "last_seen")
            status, last_seen_raw = raw

            if status == DeviceStatus.OFFLINE:
                continue  # Already offline, nothing to do

            try:
                last_seen = float(last_seen_raw or 0)
            except ValueError:
                last_seen = 0.0

            if now - last_seen > self._heartbeat_ttl:
                self._r.hset(key, "status", DeviceStatus.OFFLINE)
                expired.append(device_id)
                logger.warning(
                    "Device %r marked offline (last seen %.1fs ago)",
                    device_id, now - last_seen,
                )
                self._pubsub.publish_registry_event("device_offline", {
                    "device_id": device_id,
                    "last_seen": last_seen,
                })

        return expired

    def start_watchdog(self, interval: float = DEFAULT_WATCHDOG_INTERVAL) -> None:
        """
        Start a background daemon thread that calls expire_stale_devices()
        every *interval* seconds.  Safe to call once at control-plane startup.
        """
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            logger.warning("Watchdog already running")
            return

        self._stop_event.clear()

        def _loop() -> None:
            logger.info("Registry watchdog started (interval=%.1fs, ttl=%.1fs)",
                        interval, self._heartbeat_ttl)
            while not self._stop_event.wait(timeout=interval):
                try:
                    expired = self.expire_stale_devices()
                    if expired:
                        logger.info("Watchdog expired %d device(s): %s", len(expired), expired)
                except Exception:
                    logger.exception("Error in registry watchdog")
            logger.info("Registry watchdog stopped")

        self._watchdog_thread = threading.Thread(target=_loop, name="registry-watchdog", daemon=True)
        self._watchdog_thread.start()

    def stop_watchdog(self) -> None:
        """Signal the watchdog thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=5)
            self._watchdog_thread = None

    # ------------------------------------------------------------------
    # 4. Deregistration
    # ------------------------------------------------------------------

    def deregister(self, device_id: str) -> bool:
        """
        Remove a device from the registry entirely.

        Returns True if the device existed, False if it was already unknown.
        Used when a device explicitly shuts down cleanly.
        """
        key = _DEVICE_KEY.format(device_id=device_id)
        pipe = self._r.pipeline()
        pipe.delete(key)
        pipe.srem(_INDEX_KEY, device_id)
        results = pipe.execute()

        existed = bool(results[0])  # DEL returns number of keys deleted
        if existed:
            logger.info("Deregistered device %r", device_id)
            self._pubsub.publish_registry_event("device_left", {"device_id": device_id})
        return existed

    # ------------------------------------------------------------------
    # 5. Queries
    # ------------------------------------------------------------------

    def get_device(self, device_id: str) -> Optional[DeviceInfo]:
        """Return a single DeviceInfo, or None if not found."""
        key = _DEVICE_KEY.format(device_id=device_id)
        data = self._r.hgetall(key)
        if not data:
            return None
        return DeviceInfo.from_dict(data)

    def list_devices(self, tier: Optional[Tier] = None) -> list[DeviceInfo]:
        """
        Return all registered devices, optionally filtered by tier.
        Ordering is not guaranteed (Redis Set membership).
        """
        device_ids = self._r.smembers(_INDEX_KEY)
        devices: list[DeviceInfo] = []

        for device_id in device_ids:
            device = self.get_device(device_id)
            if device is None:
                continue
            if tier is not None and device.tier != tier:
                continue
            devices.append(device)

        return devices

    def list_available(self, tier: Optional[Tier] = None) -> list[DeviceInfo]:
        """
        Return only devices that are online and ready to accept tasks.
        Optionally filter by tier.
        """
        return [d for d in self.list_devices(tier=tier) if d.is_available]

    def count_by_tier(self) -> dict[int, dict[str, int]]:
        """
        Return a summary of device counts grouped by tier.

        Example return value::

            {
                0: {"total": 3, "available": 2},
                2: {"total": 2, "available": 2},
            }

        Only tiers with at least one device are included.
        """
        summary: dict[int, dict[str, int]] = {}
        for device in self.list_devices():
            t = int(device.tier)
            if t not in summary:
                summary[t] = {"total": 0, "available": 0}
            summary[t]["total"] += 1
            if device.is_available:
                summary[t]["available"] += 1
        return summary
