"""
health.py — HealthReporter: live system metrics and HTTP health check endpoint.

Two responsibilities
--------------------
1. get_metrics()
   Samples CPU utilisation and memory usage via psutil and returns a plain
   dict.  Called on every heartbeat so the control plane always has a fresh
   view of each worker's load.  If psutil is not installed the method returns
   an empty dict — no crash, just no metrics.

2. HTTP health endpoint  (optional, started via start_server())
   A tiny stdlib HTTP server runs in a daemon thread on a configurable port
   (default 8080).  Any monitoring tool, load balancer, or Docker HEALTHCHECK
   that can reach the worker machine can probe it without touching Redis.

   GET /        → same as /health
   GET /health  → 200 JSON body (see schema below)
   anything else → 404

   Response schema::

       {
           "status":        "online",         # live value from BaseWorker
           "device_id":     "gaming-laptop",
           "tier":          2,
           "uptime_s":      142.3,            # seconds since worker started
           "cpu_pct":       12.4,             # % across all cores (psutil)
           "mem_used_gb":   5.12,
           "mem_total_gb":  31.9,
           "mem_pct":       16.0
       }

   cpu_pct / mem_* keys are omitted when psutil is unavailable.

Usage (called automatically by BaseWorker)
------------------------------------------
    reporter = HealthReporter()
    reporter.start_server(
        port=8080,
        device_id="gaming-laptop",
        tier=2,
        status_fn=lambda: worker_status,   # callable so value stays live
    )
    metrics = reporter.get_metrics()       # {"cpu_pct": ..., "mem_pct": ...}
    reporter.stop_server()
"""
from __future__ import annotations

import json
import logging
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler
from typing import Callable

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthReporter:
    """
    Provides live system metrics and an optional HTTP health check server.

    Parameters
    ----------
    None — configure at call time via start_server().
    """

    def __init__(self) -> None:
        self._server: socketserver.TCPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # 1. System metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, float]:
        """
        Return a snapshot of live system metrics.

        Returns an empty dict when psutil is not installed so callers
        never need to handle None or conditionals.

        Keys (when available)
        ---------------------
        cpu_pct        CPU utilisation across all cores (0–100 %).
                       Uses the non-blocking variant (interval=None); psutil
                       returns the delta since the last call, which is accurate
                       enough for periodic heartbeats.
        mem_used_gb    RAM currently in use (GiB, 2 decimal places).
        mem_total_gb   Total physical RAM (GiB, 2 decimal places).
        mem_pct        Memory utilisation (0–100 %).
        """
        if not _PSUTIL_AVAILABLE:
            logger.debug("psutil not available — health metrics skipped")
            return {}

        mem = psutil.virtual_memory()
        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "mem_used_gb": round(mem.used / 1024 ** 3, 2),
            "mem_total_gb": round(mem.total / 1024 ** 3, 2),
            "mem_pct": round(mem.percent, 1),
        }

    # ------------------------------------------------------------------
    # 2. HTTP health server
    # ------------------------------------------------------------------

    def start_server(
        self,
        port: int,
        device_id: str,
        tier: int,
        status_fn: Callable[[], str],
    ) -> None:
        """
        Start the HTTP health check server in a daemon thread.

        Parameters
        ----------
        port        TCP port to listen on (e.g. 8080).
        device_id   Included verbatim in the JSON response.
        tier        Included verbatim in the JSON response.
        status_fn   Zero-argument callable that returns the current device
                    status string ("online" / "busy" / "offline").  Called on
                    every request so the value is always live.

        Calling start_server() when a server is already running is a no-op
        (logs a warning).  OSError on bind failure is caught and logged; the
        worker continues without a health endpoint rather than crashing.
        """
        if self._server is not None:
            logger.warning("Health server already running — ignoring duplicate start_server() call")
            return

        reporter = self

        class _HealthHandler(BaseHTTPRequestHandler):
            """Minimal HTTP handler — serves /health only."""

            def do_GET(self) -> None:  # noqa: N802  (stdlib naming convention)
                if self.path not in ("/", "/health"):
                    self.send_response(404)
                    self.end_headers()
                    return

                payload: dict = {
                    "status": status_fn(),
                    "device_id": device_id,
                    "tier": tier,
                    "uptime_s": round(time.time() - reporter._start_time, 1),
                }
                payload.update(reporter.get_metrics())

                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args: object) -> None:  # noqa: N802
                """Route access logs through Python logging instead of stderr."""
                logger.debug("health-server %s - " + fmt, self.address_string(), *args)

        class _ReuseAddrServer(socketserver.TCPServer):
            # Must be a class variable — TCPServer reads it before server_bind()
            allow_reuse_address = True

        try:
            server = _ReuseAddrServer(("0.0.0.0", port), _HealthHandler)
        except OSError as exc:
            logger.error(
                "Could not bind health check server to port %d: %s — "
                "worker will run without HTTP health endpoint",
                port, exc,
            )
            return

        self._server = server
        self._server_thread = threading.Thread(
            target=server.serve_forever,
            name=f"health-server:{port}",
            daemon=True,
        )
        self._server_thread.start()
        logger.info("Health check server listening on 0.0.0.0:%d (GET /health)", port)

    def stop_server(self) -> None:
        """
        Shut down the HTTP server gracefully.

        Safe to call even if start_server() was never called or failed to bind.
        Blocks for up to 3 seconds waiting for the server thread to exit.
        """
        if self._server is None:
            return
        self._server.shutdown()
        self._server = None
        if self._server_thread:
            self._server_thread.join(timeout=3)
            self._server_thread = None
        logger.info("Health check server stopped")
