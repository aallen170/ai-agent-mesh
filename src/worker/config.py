"""
config.py — DeviceConfig: parsed from device_config.yaml on each worker node.

Every device that joins the mesh carries a YAML file that declares its
hardware tier, available models, and connection settings.  DeviceConfig
is the typed Python representation of that file.

Example device_config.yaml
---------------------------
device_id: gaming-laptop
name: Gaming Laptop
tier: 2
redis_url: redis://desktop.local:6379/0
heartbeat_interval: 10.0

models:
  - llama3:70b
  - codellama:34b

hardware:
  ram_gb: 32.0
  vram_gb: 12.0
  gpu_name: "NVIDIA RTX 3080"
  cpu_cores: 16
  os: "Windows 11"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

try:
    import yaml  # PyYAML — optional dep; only needed on worker nodes
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


_DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


@dataclass
class DeviceConfig:
    """
    Typed representation of a device's ``device_config.yaml``.

    Fields
    ------
    device_id           Stable unique identifier (e.g. hostname or UUID).
    name                Human-readable label shown in the registry.
    tier                Compute tier (0–4).
    redis_url           Connection string for the control-plane Redis instance.
    heartbeat_interval  Seconds between heartbeat publishes (default 10).
    model_ids           Models available on this device (e.g. ["llama3:8b"]).
    ram_gb              Total system RAM in gigabytes.
    vram_gb             GPU VRAM in gigabytes (0 if no discrete GPU).
    gpu_name            GPU model string, e.g. "NVIDIA RTX 3080".
    cpu_cores           Number of logical CPU cores.
    os                  Operating system string, e.g. "macOS 14.4".
    """

    device_id: str
    name: str
    tier: int
    redis_url: str = _DEFAULT_REDIS_URL
    heartbeat_interval: float = 10.0
    model_ids: list[str] = field(default_factory=list)
    ram_gb: float = 0.0
    vram_gb: float = 0.0
    gpu_name: str = ""
    cpu_cores: int = 0
    os: str = ""

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "DeviceConfig":
        """
        Load a DeviceConfig from a YAML file.

        Requires PyYAML (``pip install pyyaml``).
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to load device_config.yaml. "
                "Install it with: pip install pyyaml"
            )
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls._from_raw(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DeviceConfig":
        """Load a DeviceConfig from a plain dict (e.g. parsed from env or tests)."""
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> "DeviceConfig":
        hardware: dict[str, Any] = raw.get("hardware", {})
        models: list[str] = raw.get("models", [])
        return cls(
            device_id=raw["device_id"],
            name=raw.get("name", raw["device_id"]),
            tier=int(raw["tier"]),
            redis_url=raw.get("redis_url", _DEFAULT_REDIS_URL),
            heartbeat_interval=float(raw.get("heartbeat_interval", 10.0)),
            model_ids=models,
            ram_gb=float(hardware.get("ram_gb", 0)),
            vram_gb=float(hardware.get("vram_gb", 0)),
            gpu_name=hardware.get("gpu_name", ""),
            cpu_cores=int(hardware.get("cpu_cores", 0)),
            os=hardware.get("os", ""),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_yaml_template(self) -> str:
        """
        Emit a YAML template string that can be used as a starting point
        for a new device's config file.
        """
        return (
            f"device_id: {self.device_id}\n"
            f"name: {self.name!r}\n"
            f"tier: {self.tier}\n"
            f"redis_url: {self.redis_url}\n"
            f"heartbeat_interval: {self.heartbeat_interval}\n"
            f"\nmodels:\n"
            + "".join(f"  - {m}\n" for m in self.model_ids)
            + f"\nhardware:\n"
            f"  ram_gb: {self.ram_gb}\n"
            f"  vram_gb: {self.vram_gb}\n"
            f"  gpu_name: {self.gpu_name!r}\n"
            f"  cpu_cores: {self.cpu_cores}\n"
            f"  os: {self.os!r}\n"
        )

    def __repr__(self) -> str:
        return (
            f"DeviceConfig(id={self.device_id!r}, name={self.name!r}, "
            f"tier={self.tier}, models={self.model_ids})"
        )
