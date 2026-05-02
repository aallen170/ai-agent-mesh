"""
collector.py — ResultCollector: background control-plane result processor.

Responsibilities
----------------
1. Continuously drain the ``aimesh:results`` Redis stream.
2. For each result, call TaskRouter.mark_completed() which handles:
   - Success: mark COMPLETED, store metadata.
   - Transient failure (retries remain): re-enqueue automatically.
   - Permanent failure (retries exhausted): mark FAILED, write to DLQ.
3. Periodically run the timeout watchdog: reclaim stale stream messages
   (worker crashed without ACKing) and update the corresponding TaskRecords.
4. Optionally apply a TTL to terminal task records to prevent Redis bloat.

Usage
-----
    collector = ResultCollector(router, streams_client)
    collector.start()        # launches background threads; returns immediately
    ...
    collector.stop()         # signals threads to stop; join()s them

The collector is designed to run on the control-plane host (e.g. alongside
the LangGraph orchestrator or as a standalone sidecar process).  Only one
collector should run per deployment to avoid double-processing results.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from ..redis.streams import StreamsClient, TIERS
from .task import TaskStatus

if TYPE_CHECKING:
    from .router import TaskRouter

logger = logging.getLogger(__name__)

# Default intervals
_RESULT_POLL_INTERVAL_S = 1.0       # How often to poll if no results (non-blocking fallback)
_WATCHDOG_INTERVAL_S = 60.0         # How often to run the stale-task watchdog
_STALE_TASK_IDLE_MS = 600_000       # 10 min: treat un-ACKed tasks as stale
_TTL_INTERVAL_S = 3600.0            # How often to apply TTLs (once per hour)
_DEFAULT_TASK_TTL_S = 86400         # 24 h TTL on completed/failed/dlq records


class ResultCollector:
    """
    Background service that drains the results stream and keeps TaskRecords
    up to date.

    Parameters
    ----------
    router              TaskRouter used to mark completions and handle retries.
    streams_client      StreamsClient for direct stream operations (watchdog).
    consumer_name       Redis consumer group member name for this process.
                        Must be unique if multiple control-plane instances run.
    watchdog_interval_s How often (seconds) to scan for stale dispatched tasks.
    stale_task_idle_ms  Min idle time (ms) before a task is reclaimed as stale.
    task_ttl_s          TTL (seconds) applied to terminal records.
                        Set to 0 to disable automatic expiry.
    """

    def __init__(
        self,
        router: "TaskRouter",
        streams_client: StreamsClient,
        consumer_name: str = "control-plane",
        watchdog_interval_s: float = _WATCHDOG_INTERVAL_S,
        stale_task_idle_ms: int = _STALE_TASK_IDLE_MS,
        task_ttl_s: int = _DEFAULT_TASK_TTL_S,
    ) -> None:
        self._router = router
        self._streams = streams_client
        self._consumer_name = consumer_name
        self._watchdog_interval_s = watchdog_interval_s
        self._stale_task_idle_ms = stale_task_idle_ms
        self._task_ttl_s = task_ttl_s

        self._stop_event = threading.Event()
        self._result_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._ttl_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Launch the background threads.  Returns immediately.

        Starts:
          - result thread: continuously drains aimesh:results
          - watchdog thread: periodically reclaims stale dispatched tasks
          - TTL thread (if task_ttl_s > 0): periodically expires old records
        """
        if self._result_thread and self._result_thread.is_alive():
            logger.warning("ResultCollector already running — ignoring start()")
            return

        self._stop_event.clear()

        self._result_thread = threading.Thread(
            target=self._result_loop,
            name="result-collector",
            daemon=True,
        )
        self._result_thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="result-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

        if self._task_ttl_s > 0:
            self._ttl_thread = threading.Thread(
                target=self._ttl_loop,
                name="result-ttl",
                daemon=True,
            )
            self._ttl_thread.start()

        logger.info(
            "ResultCollector started (consumer=%r, watchdog_interval=%.0fs, ttl=%ds)",
            self._consumer_name, self._watchdog_interval_s, self._task_ttl_s,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """
        Signal threads to stop and wait for them to finish.

        Parameters
        ----------
        timeout     Maximum seconds to wait for each thread to join.
        """
        logger.info("ResultCollector stopping...")
        self._stop_event.set()
        for thread in (self._result_thread, self._watchdog_thread, self._ttl_thread):
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
        logger.info("ResultCollector stopped")

    @property
    def is_running(self) -> bool:
        """True if the collector threads are alive."""
        return bool(self._result_thread and self._result_thread.is_alive())

    # ------------------------------------------------------------------
    # Result loop — drains aimesh:results continuously
    # ------------------------------------------------------------------

    def _result_loop(self) -> None:
        """
        Runs in the result thread.  Blocks on XREADGROUP with a short
        timeout so the stop event is checked promptly.
        """
        logger.debug("Result loop started (consumer=%r)", self._consumer_name)
        while not self._stop_event.is_set():
            try:
                processed = self._router.process_results_stream(
                    consumer_name=self._consumer_name,
                )
                if processed:
                    logger.debug("Result loop: processed %d result(s)", processed)
            except Exception:
                logger.exception("Error in result loop — sleeping 2s before retry")
                time.sleep(2)

        logger.debug("Result loop exited")

    # ------------------------------------------------------------------
    # Watchdog loop — reclaims stale dispatched tasks
    # ------------------------------------------------------------------

    def _watchdog_loop(self) -> None:
        """
        Runs in the watchdog thread.  Periodically calls reclaim_stale_tasks()
        for every tier and updates the corresponding TaskRecord to FAILED
        (or retries it if retries remain).
        """
        logger.debug(
            "Watchdog loop started (interval=%.0fs, idle_threshold=%dms)",
            self._watchdog_interval_s, self._stale_task_idle_ms,
        )
        while not self._stop_event.wait(timeout=self._watchdog_interval_s):
            for tier in TIERS:
                try:
                    self._process_stale_tier(tier)
                except Exception:
                    logger.exception("Watchdog error processing tier %d", tier)
        logger.debug("Watchdog loop exited")

    def _process_stale_tier(self, tier: int) -> None:
        """
        Reclaim stale messages for one tier and update TaskRecord state.

        A stale message is one that was delivered to a worker but never ACKed
        within _stale_task_idle_ms.  This typically means the worker crashed.

        Strategy:
          - If the task still has retries: ACK the stale message and re-enqueue
            (mark_completed with a timeout error triggers the retry logic).
          - If retries are exhausted: same — mark_completed sends it to DLQ.
        """
        stale = self._streams.reclaim_stale_tasks(
            tier=tier,
            consumer_name=self._consumer_name,
            min_idle_ms=self._stale_task_idle_ms,
            count=10,
        )
        for entry in stale:
            task_id = entry["task_id"]
            msg_id = entry["_msg_id"]
            logger.warning(
                "Watchdog: stale task %s on tier-%d (msg %s) — treating as timeout",
                task_id, tier, msg_id,
            )
            # Synthesize a timeout error result; mark_completed handles retry/DLQ
            self._router.mark_completed(
                task_id=task_id,
                device_id="watchdog",
                result={},
                error="TaskTimeout: worker did not ACK within the idle threshold",
            )
            # ACK the reclaimed stream message so it doesn't keep getting reclaimed
            try:
                self._streams.ack_task(tier, msg_id)
            except Exception:
                logger.exception("Watchdog: failed to ACK stale msg %s", msg_id)

    # ------------------------------------------------------------------
    # TTL loop — expires old terminal records
    # ------------------------------------------------------------------

    def _ttl_loop(self) -> None:
        """
        Runs in the TTL thread.  Applies a Redis EXPIRE to all terminal
        task records once per _TTL_INTERVAL_S seconds.
        """
        logger.debug("TTL loop started (ttl=%ds)", self._task_ttl_s)
        # Initial sleep so startup isn't immediately dominated by a full scan
        if self._stop_event.wait(timeout=_TTL_INTERVAL_S):
            return
        while not self._stop_event.is_set():
            try:
                count = self._router.expire_completed_tasks(ttl_seconds=self._task_ttl_s)
                if count:
                    logger.debug("TTL loop: applied TTL to %d task records", count)
            except Exception:
                logger.exception("Error in TTL loop")
            self._stop_event.wait(timeout=_TTL_INTERVAL_S)
        logger.debug("TTL loop exited")
