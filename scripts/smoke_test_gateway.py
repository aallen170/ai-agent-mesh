"""
smoke_test_gateway.py — AIMESH-11: LiteLLM gateway smoke test.

Checks that run without a live gateway (safe for CI):
  1. infra/litellm_config.yaml is valid YAML and contains all five tiers.
  2. GatewayClient instantiates correctly and reads env vars / defaults.

Live check (skipped in CI unless LITELLM_BASE_URL is set and reachable):
  3. Gateway /health/liveliness endpoint responds 200.
  4. /models endpoint returns at least one model per tier.

Run locally against a live gateway:
    docker compose -f infra/docker-compose.yml up -d
    LITELLM_BASE_URL=http://localhost:4000/v1 python scripts/smoke_test_gateway.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import urllib.error
import urllib.request

import yaml

# Make the repo root importable
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.control_plane.gateway import GatewayClient  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
CONFIG_PATH = ROOT / "infra" / "litellm_config.yaml"

EXPECTED_TIERS = {"tier0", "tier1", "tier2", "tier3", "tier4"}

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m–\033[0m"


# ---------------------------------------------------------------------------
# Test 1 — config file
# ---------------------------------------------------------------------------

def test_config_parses() -> None:
    assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "Config root must be a YAML mapping"
    assert "model_list" in config, "Config missing 'model_list' key"
    assert "general_settings" in config, "Config missing 'general_settings' key"

    model_list = config["model_list"]
    assert isinstance(model_list, list) and model_list, "model_list must be a non-empty list"

    model_names: list[str] = [m["model_name"] for m in model_list]
    present_tiers = {name.split("/")[0] for name in model_names}
    missing = EXPECTED_TIERS - present_tiers

    assert not missing, (
        f"litellm_config.yaml is missing entries for tiers: {sorted(missing)}\n"
        f"  Present: {sorted(present_tiers)}\n"
        f"  All model_names: {model_names}"
    )

    print(
        f"  {PASS} Config OK — {len(model_names)} models across tiers "
        f"{sorted(present_tiers)}"
    )


# ---------------------------------------------------------------------------
# Test 2 — client construction
# ---------------------------------------------------------------------------

def test_client_instantiates() -> None:
    # Explicit args
    c1 = GatewayClient(base_url="http://localhost:4000/v1", api_key="sk-test")
    assert c1.base_url == "http://localhost:4000/v1"
    assert c1.api_key == "sk-test"

    # Defaults from env / fallback
    os.environ.pop("LITELLM_BASE_URL", None)
    os.environ.pop("LITELLM_MASTER_KEY", None)
    c2 = GatewayClient()
    assert c2.base_url == "http://localhost:4000/v1"
    assert c2.api_key == "sk-aimesh-local"

    # Env var override
    os.environ["LITELLM_BASE_URL"] = "http://remote-host:4000/v1"
    os.environ["LITELLM_MASTER_KEY"] = "sk-env-key"
    c3 = GatewayClient()
    assert c3.base_url == "http://remote-host:4000/v1"
    assert c3.api_key == "sk-env-key"

    # Restore
    os.environ.pop("LITELLM_BASE_URL", None)
    os.environ.pop("LITELLM_MASTER_KEY", None)

    print(f"  {PASS} GatewayClient instantiation and env-var resolution OK")


# ---------------------------------------------------------------------------
# Test 3 — live health check (skipped if gateway unreachable)
# ---------------------------------------------------------------------------

def test_live_health() -> bool:
    """Returns True if gateway is reachable, False if skipped."""
    base = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    health_url = base.replace("/v1", "") + "/health/liveliness"

    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            body = resp.read().decode()
            print(f"  {PASS} Live health OK → {health_url}  ({body[:80].strip()})")
            return True
    except (urllib.error.URLError, OSError) as exc:
        print(
            f"  {SKIP} Live health SKIPPED — gateway not reachable\n"
            f"     ({exc})\n"
            "     Start with: docker compose -f infra/docker-compose.yml up litellm"
        )
        return False


# ---------------------------------------------------------------------------
# Test 4 — live model list (only if gateway is reachable)
# ---------------------------------------------------------------------------

def test_live_models() -> None:
    client = GatewayClient()
    models = client.list_models()

    present_tiers = {m.split("/")[0] for m in models if "/" in m}
    missing = EXPECTED_TIERS - present_tiers

    if missing:
        print(
            f"  {FAIL} Model list incomplete — missing tiers: {sorted(missing)}\n"
            f"     Registered: {sorted(models)}"
        )
        sys.exit(1)

    print(
        f"  {PASS} Model list OK — {len(models)} models, "
        f"tiers present: {sorted(present_tiers)}"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== AIMESH-11: LiteLLM gateway smoke test ===\n")

    failures = 0

    print("[1] Config parsing ...")
    try:
        test_config_parses()
    except (AssertionError, Exception) as exc:
        print(f"  {FAIL} {exc}")
        failures += 1

    print("\n[2] GatewayClient instantiation ...")
    try:
        test_client_instantiates()
    except (AssertionError, Exception) as exc:
        print(f"  {FAIL} {exc}")
        failures += 1

    print("\n[3] Live health check ...")
    gateway_up = test_live_health()

    if gateway_up:
        print("\n[4] Live model list ...")
        try:
            test_live_models()
        except (AssertionError, Exception) as exc:
            print(f"  {FAIL} {exc}")
            failures += 1
    else:
        print(f"\n[4] Live model list ... {SKIP} (gateway not running)")

    if failures:
        print(f"\n{failures} check(s) FAILED.")
        sys.exit(1)

    print("\nAll checks passed.")
