"""
smoke_test_worker.py — AIMESH-15: GatewayWorker smoke test.

Tests that run without any live services (safe for CI):
  1. DeviceConfig loads correctly from a dict and from the example YAML.
  2. litellm_url field is populated (new in AIMESH-15).
  3. GatewayWorker can be constructed from a config (no Redis connection made yet).
  4. process_task() returns the right ResultEnvelope shapes:
       - missing 'messages' field → error result
       - unknown task_type        → error result
       - valid llm_inference task → GatewayClient called, result returned
     (The gateway call is mocked so no network connection is needed.)
  5. _default_model() falls back correctly when no model in payload.

Run locally:
    python scripts/smoke_test_worker.py
"""
from __future__ import annotations

import pathlib
import sys
import types
import unittest.mock as mock

# Make repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.worker.config import DeviceConfig              # noqa: E402
from src.worker.contract import ResultEnvelope, TaskEnvelope  # noqa: E402
from src.worker.gateway_worker import GatewayWorker     # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
EXAMPLE_CONFIG = ROOT / "config" / "device_config.example.yaml"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> DeviceConfig:
    """Build a minimal DeviceConfig for testing."""
    base = {
        "device_id": "test-device",
        "name": "Test Device",
        "tier": 2,
        "models": ["tier2/llama3.1-70b-desktop"],
    }
    base.update(overrides)
    return DeviceConfig.from_dict(base)


def _make_envelope(task_type: str = "llm_inference", payload: dict | None = None) -> TaskEnvelope:
    return TaskEnvelope(
        task_id="test-task-001",
        task_type=task_type,
        payload=payload or {},
        msg_id="0-1",
    )


def _mock_gateway_response(text: str = "Hello from the model!") -> mock.MagicMock:
    """Build a mock that looks like an openai ChatCompletion response."""
    usage = mock.MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    usage.total_tokens = 30

    choice = mock.MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"

    response = mock.MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Test 1 — DeviceConfig loading
# ---------------------------------------------------------------------------

def test_device_config_from_dict() -> None:
    config = _make_config()
    assert config.device_id == "test-device"
    assert config.tier == 2
    assert config.model_ids == ["tier2/llama3.1-70b-desktop"]
    assert config.litellm_url == "http://localhost:4000/v1"  # default
    print(f"  {PASS} DeviceConfig.from_dict OK (litellm_url defaults correctly)")


def test_device_config_from_yaml() -> None:
    assert EXAMPLE_CONFIG.exists(), f"Example config missing: {EXAMPLE_CONFIG}"
    config = DeviceConfig.from_yaml(str(EXAMPLE_CONFIG))
    assert config.device_id == "gaming-laptop"
    assert config.tier == 2
    assert config.litellm_url == "http://desktop.local:4000/v1"
    assert len(config.model_ids) > 0
    print(f"  {PASS} DeviceConfig.from_yaml OK (parsed {EXAMPLE_CONFIG.name})")


def test_device_config_litellm_url_override() -> None:
    config = _make_config(**{"litellm_url": "http://custom-host:4000/v1"})
    assert config.litellm_url == "http://custom-host:4000/v1"
    print(f"  {PASS} DeviceConfig litellm_url override OK")


# ---------------------------------------------------------------------------
# Test 2 — GatewayWorker construction
# ---------------------------------------------------------------------------

def test_worker_constructs() -> None:
    config = _make_config()
    # Patch BaseWorker.__init__ to avoid touching Redis
    with mock.patch("src.worker.contract.BaseWorker.__init__", return_value=None):
        worker = GatewayWorker.__new__(GatewayWorker)
        worker.config = config
        worker._gateway = mock.MagicMock()
    assert worker.config.device_id == "test-device"
    print(f"  {PASS} GatewayWorker construction OK")


# ---------------------------------------------------------------------------
# Test 3 — process_task: error paths
# ---------------------------------------------------------------------------

def test_process_task_missing_messages() -> None:
    config = _make_config()
    with mock.patch("src.worker.contract.BaseWorker.__init__", return_value=None):
        worker = GatewayWorker.__new__(GatewayWorker)
        worker.config = config
        worker._gateway = mock.MagicMock()

    task = _make_envelope(payload={"model": "tier2/llama3.1-70b-desktop"})  # no messages
    result = worker.process_task(task)

    assert isinstance(result, ResultEnvelope)
    assert result.error is not None
    assert "messages" in result.error
    assert not result.success
    print(f"  {PASS} Missing 'messages' → error result")


def test_process_task_unknown_type() -> None:
    config = _make_config()
    with mock.patch("src.worker.contract.BaseWorker.__init__", return_value=None):
        worker = GatewayWorker.__new__(GatewayWorker)
        worker.config = config
        worker._gateway = mock.MagicMock()

    task = _make_envelope(task_type="unsupported_type", payload={})
    result = worker.process_task(task)

    assert result.error is not None
    assert "unsupported_type" in result.error
    print(f"  {PASS} Unknown task_type → error result")


def test_process_task_no_model_and_no_config_default() -> None:
    config = _make_config(**{"models": []})  # no models configured
    with mock.patch("src.worker.contract.BaseWorker.__init__", return_value=None):
        worker = GatewayWorker.__new__(GatewayWorker)
        worker.config = config
        worker._gateway = mock.MagicMock()

    task = _make_envelope(payload={"messages": [{"role": "user", "content": "hi"}]})
    result = worker.process_task(task)

    assert result.error is not None
    assert "model" in result.error.lower()
    print(f"  {PASS} No model in payload or config → error result")


# ---------------------------------------------------------------------------
# Test 4 — process_task: success path (mocked gateway)
# ---------------------------------------------------------------------------

def test_process_task_success() -> None:
    config = _make_config()
    gateway_mock = mock.MagicMock()
    gateway_mock.complete.return_value = _mock_gateway_response("Hello from the model!")

    with mock.patch("src.worker.contract.BaseWorker.__init__", return_value=None):
        worker = GatewayWorker.__new__(GatewayWorker)
        worker.config = config
        worker._gateway = gateway_mock

    task = _make_envelope(payload={
        "model": "tier2/llama3.1-70b-desktop",
        "messages": [{"role": "user", "content": "Say hi"}],
        "temperature": 0.5,
        "max_tokens": 64,
    })
    result = worker.process_task(task)

    assert result.success, f"Expected success, got error: {result.error}"
    assert result.result["text"] == "Hello from the model!"
    assert result.result["model"] == "tier2/llama3.1-70b-desktop"
    assert result.result["usage"]["total_tokens"] == 30
    assert result.result["finish_reason"] == "stop"

    # Confirm gateway was called with the right args
    gateway_mock.complete.assert_called_once_with(
        model="tier2/llama3.1-70b-desktop",
        messages=[{"role": "user", "content": "Say hi"}],
        temperature=0.5,
        max_tokens=64,
    )
    print(f"  {PASS} Successful inference task → correct ResultEnvelope")


def test_process_task_uses_config_default_model() -> None:
    """When payload has no 'model' key, falls back to config.model_ids[0]."""
    config = _make_config()
    gateway_mock = mock.MagicMock()
    gateway_mock.complete.return_value = _mock_gateway_response("response")

    with mock.patch("src.worker.contract.BaseWorker.__init__", return_value=None):
        worker = GatewayWorker.__new__(GatewayWorker)
        worker.config = config
        worker._gateway = gateway_mock

    task = _make_envelope(payload={
        "messages": [{"role": "user", "content": "hi"}],
        # no 'model' key
    })
    result = worker.process_task(task)

    assert result.success
    called_model = gateway_mock.complete.call_args[1]["model"]
    assert called_model == "tier2/llama3.1-70b-desktop"
    print(f"  {PASS} Default model from config used when payload has no 'model'")


def test_process_task_gateway_error() -> None:
    """Gateway exceptions are caught and returned as error results."""
    config = _make_config()
    gateway_mock = mock.MagicMock()
    gateway_mock.complete.side_effect = ConnectionError("Gateway unreachable")

    with mock.patch("src.worker.contract.BaseWorker.__init__", return_value=None):
        worker = GatewayWorker.__new__(GatewayWorker)
        worker.config = config
        worker._gateway = gateway_mock

    task = _make_envelope(payload={
        "model": "tier2/llama3.1-70b-desktop",
        "messages": [{"role": "user", "content": "hi"}],
    })
    result = worker.process_task(task)

    assert not result.success
    assert "Gateway error" in result.error
    assert "Gateway unreachable" in result.error
    print(f"  {PASS} Gateway exception → error ResultEnvelope (worker keeps running)")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== AIMESH-15: GatewayWorker smoke test ===\n")

    failures = 0

    tests = [
        ("DeviceConfig.from_dict",           test_device_config_from_dict),
        ("DeviceConfig.from_yaml",            test_device_config_from_yaml),
        ("DeviceConfig litellm_url override", test_device_config_litellm_url_override),
        ("GatewayWorker construction",        test_worker_constructs),
        ("Missing 'messages' → error",        test_process_task_missing_messages),
        ("Unknown task_type → error",         test_process_task_unknown_type),
        ("No model anywhere → error",         test_process_task_no_model_and_no_config_default),
        ("Successful inference (mocked)",     test_process_task_success),
        ("Default model from config",         test_process_task_uses_config_default_model),
        ("Gateway exception → error result",  test_process_task_gateway_error),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f"  {FAIL} {name}: {exc}")
            failures += 1

    print()
    if failures:
        print(f"{failures} check(s) FAILED.")
        sys.exit(1)
    print("All checks passed.")
