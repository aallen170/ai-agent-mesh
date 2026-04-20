"""
device.py — DeviceInfo dataclass and supporting enums for the AIMESH registry.

A DeviceInfo represents one physical node in the mesh (phone, laptop, desktop, etc.).
It is stored as a Redis hash and round-tripped via to_dict() / from_dict().
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any


class Tier(IntEnum):
    """
    Compute tier, ordered by inference capability (ascending).

    0  — phones / tablets        (MLX / MLC)
    1  — iGPU laptop             (Ollama, small models)
    2  — dGPU laptop / desktop   (Ollama, larger models)
    3  — RunPod serverless        (cloud GPU burst)
    4  — Claude Sonnet / Opus    (Anthropic API)
    """
    MOBILE = 0
    IGPU = 1
    DGPU = 2
    RUNPOD = 3
    CLAUDE = 4


class DeviceStatus(str):
    """
    Plain string constants rather than an Enum so they survive a Redis
    round-trip without any coercion — Redis stores everything as strings.
    """
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"        # connected but currently processing a task


@dataclass
class Capabilities:
    """
    Hardware capabilities reported by a device at registration time.

    All fields are optional — a Tier-0 phone won't have a GPU model,
    a cloud RunPod node won't have a battery level, etc.
    """
    model_ids: list[str] = field(default_factory=list)  # e.g. ["llama3:8b", "phi3:mini"]
    ram_gb: float = 0.0
    vram_gb: float = 0.0
    gpu_name: str = ""
    cpu_cores: int = 0
    os: str = ""                                          # e.g. "macOS 14.4", "Windows 11"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_ids": ",".join(self.model_ids),       # Redis hash values must be strings
            "ram_gb": str(self.ram_gb),
            "vram_gb": str(self.vram_gb),
            "gpu_name": self.gpu_name,
            "cpu_cores": str(self.cpu_cores),
            "os": self.os,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Capabilities":
        raw_models = d.get("model_ids", "")
        return cls(
            model_ids=[m for m in raw_models.split(",") if m],
            ram_gb=float(d.get("ram_gb", 0)),
            vram_gb=float(d.get("vram_gb", 0)),
            gpu_name=d.get("gpu_name", ""),
            cpu_cores=int(d.get("cpu_cores", 0)),
            os=d.get("os", ""),
        )


@dataclass
class DeviceInfo:
    """
    A single node in the AIMESH mesh.

    Fields
    ------
    device_id   Stable unique identifier chosen by the device (e.g. hostname or UUID).
    tier        Compute tier (0–4).
    name        Human-readable label, e.g. "Gaming Laptop" or "iPad #1".
    capabilities  Hardware details reported at registration.
    status      "online" | "offline" | "busy"
    last_seen   Unix timestamp (float) of the most recent heartbeat.
    registered_at  Unix timestamp of first registration.
    """
    device_id: str
    tier: Tier
    name: str
    capabilities: Capabilities = field(default_factory=Capabilities)
    status: str = DeviceStatus.OFFLINE
    last_seen: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Redis serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, str]:
        """
        Flatten to a string-only dict suitable for HSET.
        Capabilities are inlined with a 'cap_' prefix so everything
        lives in one Redis hash — no nested structures needed.
        """
        base = {
            "device_id": self.device_id,
            "tier": str(int(self.tier)),
            "name": self.name,
            "status": self.status,
            "last_seen": str(self.last_seen),
            "registered_at": str(self.registered_at),
        }
        caps = {f"cap_{k}": v for k, v in self.capabilities.to_dict().items()}
        return {**base, **caps}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeviceInfo":
        """Reconstruct a DeviceInfo from a Redis HGETALL response."""
        cap_raw = {k[4:]: v for k, v in d.items() if k.startswith("cap_")}
        return cls(
            device_id=d["device_id"],
            tier=Tier(int(d["tier"])),
            name=d.get("name", d["device_id"]),
            capabilities=Capabilities.from_dict(cap_raw),
            status=d.get("status", DeviceStatus.OFFLINE),
            last_seen=float(d.get("last_seen", 0)),
            registered_at=float(d.get("registered_at", 0)),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if the device can accept new tasks right now."""
        return self.status == DeviceStatus.ONLINE

    def __repr__(self) -> str:
        return (
            f"DeviceInfo(id={self.device_id!r}, tier={self.tier.name}, "
            f"name={self.name!r}, status={self.status!r})"
        )
