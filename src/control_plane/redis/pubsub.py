"""
PubSubClient — pub/sub helpers for device heartbeats and registry events.

Channel layout:
    aimesh:heartbeat:{device_id}   worker → control plane, periodic ping
    aimesh:registry                control plane → all workers, broadcast events
    aimesh:events                  any component → any component, general events

All messages are JSON-encoded dicts with a mandatory ``type`` field.
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from typing import Any

import redis

from .client import RedisClient

logger = logging.getLogger(__name__)

# Channel constants
HEARTBEAT_CHANNEL = "aimesh:heartbeat:{device_id}"
REGISTRY_CHANNEL = "aimesh:registry"
EVENTS_CHANNEL = "aimesh:events"


class PubSubClient:
    """
    Thin wrapper around redis-py PubSub for AIMESH event channels.

    Publishing is synchronous.  Subscribing starts a background daemon
    thread that dispatches incoming messages to registered handlers.
    """

    def __init__(self, client: RedisClient) -> None:
        self._r = client.r
        self._pubsub: redis.client.PubSub | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish_heartbeat(self, device_id: str, payload: dict[str, Any]) -> int:
        """
        Publish a heartbeat from a device to the control plane.
        Returns the number of subscribers that received the message.
        """
        channel = HEARTBEAT_CHANNEL.format(device_id=device_id)
        msg = json.dumps({"type": "heartbeat", "device_id": device_id, **payload})
        n = self._r.publish(channel, msg)
        logger.debug("Heartbeat published on %s (%d receivers)", channel, n)
        return n

    def publish_registry_event(self, event_type: str, data: dict[str, Any]) -> int:
        """
        Broadcast a registry event to all subscribers (e.g. device joined/left).
        """
        msg = json.dumps({"type": event_type, **data})
        n = self._r.publish(REGISTRY_CHANNEL, msg)
        logger.debug("Registry event %r → %d receivers", event_type, n)
        return n

    def publish_event(self, event_type: str, data: dict[str, Any]) -> int:
        """Publish a general event to the events channel."""
        msg = json.dumps({"type": event_type, **data})
        return self._r.publish(EVENTS_CHANNEL, msg)

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        channels: list[str],
        handler: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """
        Subscribe to *channels* and dispatch each message to *handler*.

        handler(channel: str, message: dict) is called in a background thread.
        Call ``stop()`` to unsubscribe and join the thread.
        """
        self._pubsub = self._r.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(**{ch: self._make_dispatch(handler) for ch in channels})
        self._thread = self._pubsub.run_in_thread(sleep_time=0.1, daemon=True)
        logger.info("Subscribed to channels: %s", channels)

    def subscribe_heartbeats(
        self,
        handler: Callable[[str, dict[str, Any]], None],
    ) -> None:
        """Subscribe to heartbeats from all devices (pattern match)."""
        if self._pubsub is None:
            self._pubsub = self._r.pubsub(ignore_subscribe_messages=True)

        pattern = "aimesh:heartbeat:*"
        self._pubsub.psubscribe(**{pattern: self._make_dispatch(handler)})

        if self._thread is None:
            self._thread = self._pubsub.run_in_thread(sleep_time=0.1, daemon=True)
        logger.info("Subscribed to heartbeat pattern: %s", pattern)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Unsubscribe and stop the background listener thread."""
        if self._thread:
            self._thread.stop()
            self._thread = None
        if self._pubsub:
            self._pubsub.close()
            self._pubsub = None
        logger.info("PubSubClient stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_dispatch(
        handler: Callable[[str, dict[str, Any]], None],
    ) -> Callable[[dict], None]:
        """Wrap a user handler to parse the raw redis-py message dict."""
        def dispatch(raw: dict) -> None:
            try:
                channel = raw.get("channel", "")
                data = json.loads(raw.get("data", "{}"))
                handler(channel, data)
            except Exception:
                logger.exception("Error in PubSub handler for channel %r", raw.get("channel"))
        return dispatch
