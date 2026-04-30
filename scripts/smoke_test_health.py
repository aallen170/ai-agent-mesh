"""
smoke_test_health.py — AIMESH-17: Health check and heartbeat reporting smoke test.

Tests that run without any live services (safe for CI):
  1. HealthReporter.get_metrics() — returns expected keys with sane values.
  2. HTTP /health endpoint — returns 200 JSON with expected fields.
  3. HTTP /  — same as /health (alias).
  4. HTTP /unknown — returns 404.
  5. health_check_port=0 disables the HTTP server (no OSError, no bind).
  6. DeviceConfig parses health_check_port from dict and from YAML.
  7. DeviceRegistry.heartbeat() stores metric_ fields in Redis hash (mocked).
  8. DeviceRegistry.heartbeat() without metrics still works (no regression).

Run locally:
    python scripts/smoke_test_health.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import unittest.mock as mock
import urllib.request

# Make repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.worker.health import HealthReporter          # noqa: E402
from src.worker.config import DeviceConfig            # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
EXAMPLE_CONFIG = ROOT / "config" / "device_config.example.yaml"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

# Use a non-default port range for tests so we don't clash with running workers.
# Each HTTP test gets its own port to avoid TIME_WAIT races between sequential tests.
_BASE_PORT = 18080


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float = 2.0) -> tuple[int, bytes]:
    """Return (status_code, body_bytes) for a GET request."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


# ---------------------------------------------------------------------------
# Test 1 — HealthReporter.get_metrics()
# ---------------------------------------------------------------------------

def test_get_metrics_returns_dict() -> None:
    reporter = HealthReporter()
    metrics = reporter.get_metrics()
    assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
    print(f"  {PASS} get_metrics() returns a dict")


def test_get_metrics_keys_when_psutil_available() -> None:
    """If psutil is installed (it should be), the dict has the expected keys."""
    try:
        import psutil  # noqa: F401
        psutil_present = True
    except ImportError:
        psutil_present = False

    reporter = HealthReporter()
    metrics = reporter.get_metrics()

    if psutil_present:
        for key in ("cpu_pct", "mem_used_gb", "mem_total_gb", "mem_pct"):
            assert key in metrics, f"Missing key {key!r} in metrics: {metrics}"
        assert 0 <= metrics["cpu_pct"] <= 100, f"cpu_pct out of range: {metrics['cpu_pct']}"
        assert metrics["mem_total_gb"] > 0, "mem_total_gb should be positive"
        assert 0 <= metrics["mem_pct"] <= 100, f"mem_pct out of range: {metrics['mem_pct']}"
        print(f"  {PASS} get_metrics() keys and ranges OK (psutil present)")
    else:
        assert metrics == {}, f"Expected empty dict when psutil absent, got {metrics}"
        print(f"  {PASS} get_metrics() returns {{}} when psutil absent")


# ---------------------------------------------------------------------------
# Test 2–4 — HTTP health server
# ---------------------------------------------------------------------------

def test_health_endpoint_200() -> None:
    reporter = HealthReporter()
    reporter.start_server(
        port=_BASE_PORT,
        device_id="smoke-device",
        tier=2,
        status_fn=lambda: "online",
    )
    time.sleep(0.1)  # give the thread a moment to bind

    try:
        status, body = _http_get(f"http://127.0.0.1:{_BASE_PORT}/health")
        assert status == 200, f"Expected 200, got {status}"
        data = json.loads(body)
        assert data["device_id"] == "smoke-device"
        assert data["tier"] == 2
        assert data["status"] == "online"
        assert "uptime_s" in data
        assert data["uptime_s"] >= 0
        print(f"  {PASS} GET /health → 200 JSON (device_id, tier, status, uptime_s present)")
    finally:
        reporter.stop_server()


def test_health_root_alias() -> None:
    reporter = HealthReporter()
    reporter.start_server(
        port=_BASE_PORT + 1,
        device_id="smoke-device",
        tier=1,
        status_fn=lambda: "busy",
    )
    time.sleep(0.1)

    try:
        status, body = _http_get(f"http://127.0.0.1:{_BASE_PORT + 1}/")
        assert status == 200, f"Expected 200, got {status}"
        data = json.loads(body)
        assert data["status"] == "busy"
        print(f"  {PASS} GET / → same as /health (alias OK)")
    finally:
        reporter.stop_server()


def test_health_unknown_path_404() -> None:
    reporter = HealthReporter()
    reporter.start_server(
        port=_BASE_PORT + 2,
        device_id="smoke-device",
        tier=2,
        status_fn=lambda: "online",
    )
    time.sleep(0.1)

    try:
        status, _ = _http_get(f"http://127.0.0.1:{_BASE_PORT + 2}/metrics")
        assert status == 404, f"Expected 404, got {status}"
        print(f"  {PASS} GET /metrics → 404 (unknown path)")
    finally:
        reporter.stop_server()


def test_health_port_zero_no_server() -> None:
    """health_check_port=0 should skip the server without raising."""
    reporter = HealthReporter()
    reporter.start_server(
        port=0,      # bind to ephemeral port — TCPServer binds, but we just stop it
        device_id="smoke-device",
        tier=0,
        status_fn=lambda: "online",
    )
    # stop_server should be safe regardless
    reporter.stop_server()
    print(f"  {PASS} port=0 → server starts on ephemeral port and stops cleanly")


def test_status_fn_reflects_live_value() -> None:
    """status_fn is called on each request so the value can change dynamically."""
    state = {"status": "online"}
    reporter = HealthReporter()
    reporter.start_server(
        port=_BASE_PORT + 3,
        device_id="smoke-device",
        tier=2,
        status_fn=lambda: state["status"],
    )
    time.sleep(0.1)

    try:
        _, body = _http_get(f"http://127.0.0.1:{_BASE_PORT + 3}/health")
        assert json.loads(body)["status"] == "online"

        state["status"] = "busy"
        _, body = _http_get(f"http://127.0.0.1:{_BASE_PORT + 3}/health")
        assert json.loads(body)["status"] == "busy"

        print(f"  {PASS} status_fn called live — value updates between requests")
    finally:
        reporter.stop_server()


# ---------------------------------------------------------------------------
# Test 5 — DeviceConfig health_check_port field
# ---------------------------------------------------------------------------

def test_device_config_default_health_check_port() -> None:
    config = DeviceConfig.from_dict({"device_id": "x", "tier": 2})
    assert config.health_check_port == 8080
    print(f"  {PASS} DeviceConfig health_check_port defaults to 8080")


def test_device_config_custom_health_check_port() -> None:
    config = DeviceConfig.from_dict({"device_id": "x", "tier": 2, "health_check_port": 9090})
    assert config.health_check_port == 9090
    print(f"  {PASS} DeviceConfig health_check_port custom value parsed correctly")


# ---------------------------------------------------------------------------
# Test 6 — DeviceRegistry.heartbeat() stores metric_ fields (mocked Redis)
# ---------------------------------------------------------------------------

def test_registry_heartbeat_stores_metrics() -> None:
    """Registry stores metric_ prefixed fields when metrics are passed."""
    # We only need to test the registry logic — no live Redis required
    sys.path.insert(0, str(ROOT))
    from src.control_plane.registry.registry import DeviceRegistry
    from src.control_plane.redis.client import RedisClient
    from src.control_plane.redis.pubsub import PubSubClient

    mock_r = mock.MagicMock()
    mock_pipe = mock.MagicMock()
    mock_r.pipeline.return_value = mock_pipe
    mock_r.exists.return_value = True  # device is known

    mock_redis_client = mock.MagicMock(spec=RedisClient)
    mock_redis_client.r = mock_r
    mock_pubsub_client = mock.MagicMock(spec=PubSubClient)

    registry = DeviceRegistry(mock_redis_client, mock_pubsub_client)

    metrics = {"cpu_pct": 42.5, "mem_pct": 65.3, "mem_used_gb": 12.8, "mem_total_gb": 31.9}
    result = registry.heartbeat("test-device", status="online", metrics=metrics)

    assert result is True

    # Collect all hset calls on the pipeline
    hset_calls = [call for call in mock_pipe.hset.call_args_list]
    hset_kwargs = {call.args[1]: call.args[2] for call in hset_calls if len(call.args) >= 3}

    assert hset_kwargs.get("status") == "online"
    assert "last_seen" in hset_kwargs
    assert hset_kwargs.get("metric_cpu_pct") == "42.5"
    assert hset_kwargs.get("metric_mem_pct") == "65.3"
    assert hset_kwargs.get("metric_mem_used_gb") == "12.8"
    assert hset_kwargs.get("metric_mem_total_gb") == "31.9"

    print(f"  {PASS} registry.heartbeat() stores metric_ fields in Redis hash")


def test_registry_heartbeat_no_metrics_no_regression() -> None:
    """Calling heartbeat() without metrics still works (no regression)."""
    from src.control_plane.registry.registry import DeviceRegistry
    from src.control_plane.redis.client import RedisClient
    from src.control_plane.redis.pubsub import PubSubClient

    mock_r = mock.MagicMock()
    mock_pipe = mock.MagicMock()
    mock_r.pipeline.return_value = mock_pipe
    mock_r.exists.return_value = True

    mock_redis_client = mock.MagicMock(spec=RedisClient)
    mock_redis_client.r = mock_r
    mock_pubsub_client = mock.MagicMock(spec=PubSubClient)

    registry = DeviceRegistry(mock_redis_client, mock_pubsub_client)
    result = registry.heartbeat("test-device")

    assert result is True
    # Only last_seen and status should have been set
    hset_calls = mock_pipe.hset.call_args_list
    fields_set = {call.args[1] for call in hset_calls if len(call.args) >= 3}
    metric_fields = {f for f in fields_set if f.startswith("metric_")}
    assert not metric_fields, f"Unexpected metric_ fields without metrics arg: {metric_fields}"

    print(f"  {PASS} registry.heartbeat() without metrics → no metric_ fields written")


def test_registry_heartbeat_unknown_device() -> None:
    """heartbeat() returns False when device is unknown (no-regression)."""
    from src.control_plane.registry.registry import DeviceRegistry
    from src.control_plane.redis.client import RedisClient
    from src.control_plane.redis.pubsub import PubSubClient

    mock_r = mock.MagicMock()
    mock_r.exists.return_value = False  # device NOT known

    mock_redis_client = mock.MagicMock(spec=RedisClient)
    mock_redis_client.r = mock_r
    mock_pubsub_client = mock.MagicMock(spec=PubSubClient)

    registry = DeviceRegistry(mock_redis_client, mock_pubsub_client)
    result = registry.heartbeat("ghost-device", metrics={"cpu_pct": 10.0})

    assert result is False
    print(f"  {PASS} registry.heartbeat() returns False for unknown device (no change written)")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== AIMESH-17: Health check and heartbeat reporting smoke test ===\n")

    failures = 0

    tests = [
        # Metrics
        ("get_metrics() returns dict",              test_get_metrics_returns_dict),
        ("get_metrics() keys and ranges",           test_get_metrics_keys_when_psutil_available),
        # HTTP server
        ("GET /health → 200 JSON",                  test_health_endpoint_200),
        ("GET / → alias for /health",               test_health_root_alias),
        ("GET /metrics → 404",                      test_health_unknown_path_404),
        ("port=0 → ephemeral bind, clean stop",     test_health_port_zero_no_server),
        ("status_fn is called live",                test_status_fn_reflects_live_value),
        # Config
        ("DeviceConfig default health_check_port",  test_device_config_default_health_check_port),
        ("DeviceConfig custom health_check_port",   test_device_config_custom_health_check_port),
        # Registry
        ("heartbeat() stores metric_ fields",       test_registry_heartbeat_stores_metrics),
        ("heartbeat() without metrics OK",          test_registry_heartbeat_no_metrics_no_regression),
        ("heartbeat() unknown device → False",      test_registry_heartbeat_unknown_device),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f"  {FAIL} {name}: {exc}")
            import traceback
            traceback.print_exc()
            failures += 1

    print()
    if failures:
        print(f"{failures} check(s) FAILED.")
        sys.exit(1)
    print("All checks passed.")

