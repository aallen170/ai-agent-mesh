"""
run_worker.py — AIMESH worker entrypoint.

Loads a device config file, instantiates a GatewayWorker, and runs it.
This is the script you start on each device to bring it into the mesh.

Usage
-----
    python scripts/run_worker.py --config config/my_device.yaml

    # Override Redis URL without editing the config file:
    REDIS_URL=redis://desktop.local:6379/0 python scripts/run_worker.py \\
        --config config/my_device.yaml

    # Override the LiteLLM gateway URL:
    LITELLM_BASE_URL=http://desktop.local:4000/v1 python scripts/run_worker.py \\
        --config config/gaming_laptop.yaml

Signals
-------
Send SIGINT (Ctrl+C) or SIGTERM to stop the worker gracefully after the
current task finishes.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import signal
import sys

# Make the repo root importable when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.worker.config import DeviceConfig          # noqa: E402
from src.worker.gateway_worker import GatewayWorker  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start an AIMESH GatewayWorker on this device.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the device_config.yaml file for this device.",
    )
    parser.add_argument(
        "--redis-url",
        metavar="URL",
        default=None,
        help="Override the Redis URL from the config (also: REDIS_URL env var).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)

    log = logging.getLogger("run_worker")

    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    log.info("Loading device config from %s", config_path)
    config = DeviceConfig.from_yaml(str(config_path))
    log.info("Device: %r  tier=%d  models=%s", config.name, config.tier, config.model_ids)

    worker = GatewayWorker(config, redis_url=args.redis_url)

    # Graceful shutdown on SIGTERM (e.g. systemd / Docker stop)
    def _handle_sigterm(signum, frame):  # noqa: ANN001
        log.info("SIGTERM received — stopping worker after current task...")
        worker.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        worker.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping worker after current task...")
        worker.stop()


if __name__ == "__main__":
    main()
