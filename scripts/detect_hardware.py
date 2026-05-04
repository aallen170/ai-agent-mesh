#!/usr/bin/env python3
"""
detect_hardware.py — AIMESH cross-platform hardware detection tool.

Detects this device's hardware specs and writes a ready-to-use device config
YAML to config/<device_id>.yaml, with all fields populated.

Supported platforms:
  Windows               — psutil + nvidia-smi + wmic
  Linux                 — psutil + nvidia-smi + lspci/rocm-smi
  macOS (Intel/Apple)   — psutil + nvidia-smi + system_profiler/sysctl
  iOS / iPadOS (a-Shell)— sysctl + Apple model string lookup table
  Android (Termux)      — /proc/cpuinfo + /sys/ filesystem

Usage:
    python scripts/detect_hardware.py
    python scripts/detect_hardware.py --output config/my_device.yaml
    python scripts/detect_hardware.py --control-plane 192.168.1.10
    python scripts/detect_hardware.py --is-control-plane
    python scripts/detect_hardware.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import pathlib
import platform
import re
import shutil
import socket
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Optional dependency: psutil — auto-installed if missing.
# psutil is listed in pyproject.toml, so `pip install -e .` covers it.
# On a fresh device that hasn't been set up yet, we install it on the fly
# so onboarding works without any prior setup step.
# ---------------------------------------------------------------------------
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    import subprocess as _sp
    print("psutil not found — installing automatically...")
    try:
        _sp.check_call(
            [sys.executable, "-m", "pip", "install", "psutil", "--quiet"],
            timeout=60,
        )
        import psutil
        HAS_PSUTIL = True
        print("psutil installed.\n")
    except Exception as _e:
        print(f"Warning: could not auto-install psutil ({_e}).")
        print("RAM and CPU detection will use fallback methods.\n")
        HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Apple hardware model identifier → (display_name, chip, ram_gb, cpu_cores, os_prefix)
#
# Source: https://www.theiphonewiki.com/wiki/Models
# ram_gb reflects the *base* configuration; actual RAM is read at runtime and
# takes precedence — the table is only used when direct measurement fails.
# ---------------------------------------------------------------------------
APPLE_MODEL_TABLE: dict[str, tuple[str, str, float, int, str]] = {
    # --- iPad Pro (M4) -------------------------------------------------
    "iPad16,3": ("iPad Pro 11-inch (M4)",        "Apple M4",       8.0,  10, "iPadOS"),
    "iPad16,4": ("iPad Pro 11-inch (M4)",        "Apple M4",      16.0,  10, "iPadOS"),
    "iPad16,5": ("iPad Pro 13-inch (M4)",        "Apple M4",      16.0,  10, "iPadOS"),
    "iPad16,6": ("iPad Pro 13-inch (M4)",        "Apple M4",      16.0,  10, "iPadOS"),
    # --- iPad Pro (M2) -------------------------------------------------
    "iPad14,3": ("iPad Pro 11-inch (M2)",        "Apple M2",       8.0,   8, "iPadOS"),
    "iPad14,4": ("iPad Pro 11-inch (M2)",        "Apple M2",       8.0,   8, "iPadOS"),
    "iPad14,5": ("iPad Pro 12.9-inch (M2)",      "Apple M2",       8.0,   8, "iPadOS"),
    "iPad14,6": ("iPad Pro 12.9-inch (M2)",      "Apple M2",      16.0,   8, "iPadOS"),
    # --- iPad Pro (M1) -------------------------------------------------
    "iPad13,4": ("iPad Pro 11-inch (M1)",        "Apple M1",       8.0,   8, "iPadOS"),
    "iPad13,5": ("iPad Pro 11-inch (M1)",        "Apple M1",       8.0,   8, "iPadOS"),
    "iPad13,6": ("iPad Pro 11-inch (M1)",        "Apple M1",       8.0,   8, "iPadOS"),
    "iPad13,7": ("iPad Pro 11-inch (M1)",        "Apple M1",       8.0,   8, "iPadOS"),
    "iPad13,8": ("iPad Pro 12.9-inch (M1)",      "Apple M1",      16.0,   8, "iPadOS"),
    "iPad13,9": ("iPad Pro 12.9-inch (M1)",      "Apple M1",      16.0,   8, "iPadOS"),
    "iPad13,10":("iPad Pro 12.9-inch (M1)",      "Apple M1",      16.0,   8, "iPadOS"),
    "iPad13,11":("iPad Pro 12.9-inch (M1)",      "Apple M1",      16.0,   8, "iPadOS"),
    # --- iPad Air (M2) -------------------------------------------------
    "iPad14,8": ("iPad Air 11-inch (M2)",        "Apple M2",       8.0,   8, "iPadOS"),
    "iPad14,9": ("iPad Air 11-inch (M2)",        "Apple M2",       8.0,   8, "iPadOS"),
    "iPad14,10":("iPad Air 13-inch (M2)",        "Apple M2",       8.0,   8, "iPadOS"),
    "iPad14,11":("iPad Air 13-inch (M2)",        "Apple M2",       8.0,   8, "iPadOS"),
    # --- iPad Air (M1) -------------------------------------------------
    "iPad13,16":("iPad Air (M1, 5th gen)",       "Apple M1",       8.0,   8, "iPadOS"),
    "iPad13,17":("iPad Air (M1, 5th gen)",       "Apple M1",       8.0,   8, "iPadOS"),
    # --- iPad mini (A15) -----------------------------------------------
    "iPad14,1": ("iPad mini (6th gen)",          "Apple A15 Bionic", 4.0, 6, "iPadOS"),
    "iPad14,2": ("iPad mini (6th gen)",          "Apple A15 Bionic", 4.0, 6, "iPadOS"),
    # --- iPhone 15 Pro -------------------------------------------------
    "iPhone16,1":("iPhone 15 Pro",              "Apple A17 Pro",   8.0,   6, "iOS"),
    "iPhone16,2":("iPhone 15 Pro Max",          "Apple A17 Pro",   8.0,   6, "iOS"),
    # --- iPhone 15 ----------------------------------------------------
    "iPhone15,4":("iPhone 15",                  "Apple A16 Bionic", 6.0,  6, "iOS"),
    "iPhone15,5":("iPhone 15 Plus",             "Apple A16 Bionic", 6.0,  6, "iOS"),
    # --- iPhone 14 Pro ------------------------------------------------
    "iPhone15,2":("iPhone 14 Pro",              "Apple A16 Bionic", 6.0,  6, "iOS"),
    "iPhone15,3":("iPhone 14 Pro Max",          "Apple A16 Bionic", 6.0,  6, "iOS"),
    # --- iPhone 14 ----------------------------------------------------
    "iPhone14,7":("iPhone 14",                  "Apple A15 Bionic", 6.0,  6, "iOS"),
    "iPhone14,8":("iPhone 14 Plus",             "Apple A15 Bionic", 6.0,  6, "iOS"),
    # --- iPhone 13 ----------------------------------------------------
    "iPhone14,2":("iPhone 13 Pro",              "Apple A15 Bionic", 6.0,  6, "iOS"),
    "iPhone14,3":("iPhone 13 Pro Max",          "Apple A15 Bionic", 6.0,  6, "iOS"),
    "iPhone14,4":("iPhone 13 mini",             "Apple A15 Bionic", 4.0,  6, "iOS"),
    "iPhone14,5":("iPhone 13",                  "Apple A15 Bionic", 4.0,  6, "iOS"),
    # --- macOS Apple Silicon ------------------------------------------
    "Mac16,1":   ("MacBook Pro 14-inch (M4)",   "Apple M4",        16.0, 10, "macOS"),
    "Mac16,2":   ("MacBook Pro 16-inch (M4)",   "Apple M4",        24.0, 12, "macOS"),
    "Mac16,3":   ("MacBook Air 13-inch (M4)",   "Apple M4",        16.0, 10, "macOS"),
    "Mac16,4":   ("MacBook Air 15-inch (M4)",   "Apple M4",        16.0, 10, "macOS"),
    "Mac15,3":   ("MacBook Pro 14-inch (M3)",   "Apple M3",         8.0,  8, "macOS"),
    "Mac15,6":   ("MacBook Pro 14-inch (M3 Pro)","Apple M3 Pro",   18.0, 11, "macOS"),
    "Mac15,7":   ("MacBook Pro 16-inch (M3 Pro)","Apple M3 Pro",   18.0, 11, "macOS"),
    "Mac15,8":   ("MacBook Pro 14-inch (M3 Max)","Apple M3 Max",   36.0, 14, "macOS"),
    "Mac15,9":   ("MacBook Pro 14-inch (M3 Max)","Apple M3 Max",   48.0, 14, "macOS"),
    "Mac15,10":  ("MacBook Pro 16-inch (M3 Max)","Apple M3 Max",   48.0, 16, "macOS"),
    "Mac15,11":  ("MacBook Pro 16-inch (M3 Max)","Apple M3 Max",   64.0, 16, "macOS"),
    "Mac15,12":  ("MacBook Air 13-inch (M3)",   "Apple M3",         8.0,  8, "macOS"),
    "Mac15,13":  ("MacBook Air 15-inch (M3)",   "Apple M3",         8.0,  8, "macOS"),
    "Mac14,2":   ("MacBook Air (M2)",           "Apple M2",         8.0,  8, "macOS"),
    "Mac14,5":   ("MacBook Pro 14-inch (M2 Pro)","Apple M2 Pro",   16.0, 10, "macOS"),
    "Mac14,6":   ("MacBook Pro 16-inch (M2 Pro)","Apple M2 Pro",   16.0, 12, "macOS"),
    "Mac14,9":   ("MacBook Pro 14-inch (M2 Max)","Apple M2 Max",   32.0, 12, "macOS"),
    "Mac14,10":  ("MacBook Pro 16-inch (M2 Max)","Apple M2 Max",   32.0, 12, "macOS"),
    "Mac14,15":  ("MacBook Air 15-inch (M2)",   "Apple M2",         8.0,  8, "macOS"),
    "MacBookPro18,1":("MacBook Pro 16-inch (M1 Pro)","Apple M1 Pro",16.0,10, "macOS"),
    "MacBookPro18,2":("MacBook Pro 16-inch (M1 Max)","Apple M1 Max",32.0,10, "macOS"),
    "MacBookPro18,3":("MacBook Pro 14-inch (M1 Pro)","Apple M1 Pro",16.0,10, "macOS"),
    "MacBookPro18,4":("MacBook Pro 14-inch (M1 Max)","Apple M1 Max",32.0,10, "macOS"),
    "MacBookAir10,1":("MacBook Air (M1)",        "Apple M1",         8.0,  8, "macOS"),
}

# Model names registered in infra/litellm_config.yaml, keyed by tier and role.
LITELLM_MODELS: dict[str, list[str]] = {
    "tier0/ipad1":   ["tier0/llama3.2-3b-ipad1"],
    "tier0/ipad2":   ["tier0/llama3.2-3b-ipad2"],
    "tier0/iphone":  ["tier0/llama3.2-3b-iphone"],
    "tier0/android": ["tier0/llama3.2-3b-android"],
    "tier1":         ["tier1/llama3.1-8b"],
    "tier2/desktop": ["tier2/llama3.2-3b-desktop", "tier2/llama3.1-70b-desktop"],
    "tier2/laptop":  ["tier2/mixtral-8x7b-laptop"],
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class HardwareInfo:
    hostname: str = "unknown"
    local_ip: str = "127.0.0.1"
    os_name: str = "unknown"
    ram_gb: float = 0.0
    cpu_cores: int = 0
    gpu_name: str = "unknown"
    vram_gb: float = 0.0
    is_discrete_gpu: bool = False
    is_apple_silicon: bool = False
    is_ios: bool = False
    is_android: bool = False
    apple_model_id: Optional[str] = None
    apple_display_name: Optional[str] = None
    tier: int = 1


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def _is_ios() -> bool:
    """True when running on iPadOS / iOS (e.g. inside a-Shell)."""
    return os.path.exists("/private/var/mobile") or os.path.exists("/var/mobile")


def _is_android() -> bool:
    """True when running on Android (e.g. inside Termux)."""
    return (
        "ANDROID_ROOT" in os.environ
        or "TERMUX_VERSION" in os.environ
        or os.path.exists("/data/data/com.termux")
        or os.path.exists("/system/build.prop")
    )


def _run(cmd: list[str], timeout: int = 10) -> Optional[str]:
    """Run a subprocess, return stdout as string or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def detect_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def detect_local_ip() -> str:
    """
    Get the LAN IP address by opening a UDP socket toward a public address.
    No data is actually sent — this is purely a routing trick.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def detect_ram_gb() -> float:
    """Total system RAM in GB (or unified memory on Apple Silicon)."""
    if HAS_PSUTIL:
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    # Linux / Android fallback
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 ** 2), 1)
    except Exception:
        pass
    # macOS / iOS fallback via sysctl
    out = _run(["sysctl", "-n", "hw.memsize"])
    if out:
        try:
            return round(int(out) / (1024 ** 3), 1)
        except ValueError:
            pass
    return 0.0


def detect_cpu_cores() -> int:
    """Logical (hyper-threaded) CPU core count."""
    if HAS_PSUTIL:
        return psutil.cpu_count(logical=True) or 0
    try:
        return os.cpu_count() or 0
    except Exception:
        return 0


def detect_os_name() -> str:
    """Human-readable OS name and version."""
    system = platform.system()

    if system == "Windows":
        _, ver, _, _ = platform.win32_ver()
        build = ver.split(".")[-1] if ver else ""
        # Distinguish Windows 10 vs 11 by build number
        try:
            if int(build) >= 22000:
                return "Windows 11"
        except ValueError:
            pass
        return "Windows 10"

    if system == "Darwin":
        if _is_ios():
            # Try sysctl for iOS version
            ver = _run(["sysctl", "-n", "kern.osproductversion"]) or platform.mac_ver()[0]
            return f"iPadOS {ver}" if "iPad" in (detect_apple_model_id() or "") else f"iOS {ver}"
        ver = platform.mac_ver()[0]
        return f"macOS {ver}"

    if system == "Linux":
        if _is_android():
            out = _run(["getprop", "ro.build.version.release"])
            return f"Android {out}" if out else "Android"
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except Exception:
            pass
        return f"Linux {platform.release()}"

    return system


def detect_apple_model_id() -> Optional[str]:
    """
    Returns the Apple hardware model identifier, e.g. 'iPad14,5' or 'Mac14,2'.
    Works on macOS and iOS (a-Shell).
    """
    # hw.machine is the primary source on iOS/iPadOS
    for key in ("hw.machine", "hw.model"):
        out = _run(["sysctl", "-n", key])
        if out and out not in ("arm64", "x86_64", "i386", "arm64e"):
            return out
    return None


def _detect_nvidia() -> tuple[Optional[str], float]:
    """Returns (gpu_name, vram_gb) for the primary NVIDIA GPU, or (None, 0.0)."""
    if not shutil.which("nvidia-smi"):
        return None, 0.0
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if out:
        line = out.splitlines()[0]
        parts = line.split(",")
        if len(parts) >= 2:
            name = parts[0].strip()
            try:
                vram_gb = round(float(parts[1].strip()) / 1024, 1)
                return name, vram_gb
            except ValueError:
                return parts[0].strip(), 0.0
    return None, 0.0


def _detect_gpu_windows() -> tuple[Optional[str], float, bool]:
    """
    Returns (gpu_name, vram_gb, is_discrete) using wmic on Windows.
    Prefers discrete GPUs over integrated ones.
    """
    out = _run([
        "wmic", "path", "win32_videocontroller",
        "get", "Name,AdapterRAM",
        "/format:csv",
    ], timeout=15)
    if not out:
        return None, 0.0, False

    DISCRETE_KEYWORDS = ("NVIDIA", "GeForce", "RTX", "GTX", "Quadro",
                         "AMD Radeon RX", "AMD Radeon Pro", "Radeon RX")
    IGPU_KEYWORDS = ("Intel", "AMD Radeon Graphics", "Radeon Vega", "AMD Radeon(TM)")

    best_discrete: tuple[Optional[str], float] = (None, 0.0)
    best_igpu: tuple[Optional[str], float] = (None, 0.0)

    for line in out.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("node"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            adapter_ram = int(parts[1].strip())
            vram_gb = round(adapter_ram / (1024 ** 3), 1)
        except (ValueError, IndexError):
            vram_gb = 0.0
        name = parts[2].strip() if len(parts) > 2 else ""
        if any(k in name for k in DISCRETE_KEYWORDS):
            if vram_gb > best_discrete[1]:
                best_discrete = (name, vram_gb)
        elif any(k in name for k in IGPU_KEYWORDS):
            if best_igpu[0] is None:
                best_igpu = (name, vram_gb)

    if best_discrete[0]:
        return best_discrete[0], best_discrete[1], True
    if best_igpu[0]:
        return best_igpu[0], best_igpu[1], False
    return None, 0.0, False


def _detect_gpu_linux() -> tuple[Optional[str], float, bool]:
    """Returns (gpu_name, vram_gb, is_discrete) on Linux."""
    # Try lspci first (broadly available)
    if shutil.which("lspci"):
        out = _run(["lspci"])
        if out:
            discrete_re = re.compile(r"(NVIDIA|AMD|ATI|Radeon RX)", re.IGNORECASE)
            igpu_re = re.compile(r"Intel.*Graphics|AMD.*Radeon.*Vega", re.IGNORECASE)
            for line in out.splitlines():
                if not any(tag in line for tag in ("VGA", "Display", "3D")):
                    continue
                name = line.split(": ", 1)[-1].strip()
                if discrete_re.search(name):
                    # Try rocm-smi for VRAM
                    vram_gb = 0.0
                    if shutil.which("rocm-smi"):
                        rocm = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
                        if rocm:
                            for rline in rocm.splitlines():
                                if "VRAM Total Memory" in rline:
                                    try:
                                        mb = float(rline.split(",")[-1].strip())
                                        vram_gb = round(mb / 1024, 1)
                                    except ValueError:
                                        pass
                    return name, vram_gb, True
                if igpu_re.search(name):
                    return name, 0.0, False
    return None, 0.0, False


def _detect_apple_silicon() -> tuple[Optional[str], bool]:
    """Returns (chip_name, is_apple_silicon) on macOS / iOS."""
    # sysctl machdep.cpu.brand_string works on macOS (not iOS)
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if out and "Apple" in out:
        return out, True

    # system_profiler on macOS
    out = _run(["system_profiler", "SPHardwareDataType"], timeout=20)
    if out:
        for line in out.splitlines():
            if "Chip" in line or "Processor Name" in line:
                chip = line.split(":", 1)[-1].strip()
                if any(x in chip for x in ("Apple", "M1", "M2", "M3", "M4")):
                    return chip, True

    # iOS: look up from model table
    model_id = detect_apple_model_id()
    if model_id and model_id in APPLE_MODEL_TABLE:
        _, chip, *_ = APPLE_MODEL_TABLE[model_id]
        return chip, True

    return None, False


def _detect_android_gpu() -> tuple[Optional[str], float]:
    """Best-effort GPU name on Android from /sys/ and getprop."""
    # Adreno
    out = _run(["getprop", "ro.board.platform"])
    if out:
        return f"Qualcomm Adreno ({out})", 0.0
    # Mali
    mali = pathlib.Path("/sys/class/misc/mali0")
    if mali.exists():
        return "ARM Mali GPU", 0.0
    return "unknown", 0.0


# ---------------------------------------------------------------------------
# Master detection routine
# ---------------------------------------------------------------------------

def detect_all() -> HardwareInfo:
    info = HardwareInfo()

    info.is_ios = _is_ios()
    info.is_android = _is_android()
    info.hostname = detect_hostname()
    info.local_ip = detect_local_ip()
    info.ram_gb = detect_ram_gb()
    info.cpu_cores = detect_cpu_cores()
    info.os_name = detect_os_name()

    system = platform.system()

    # --- GPU / chip detection ---
    if info.is_ios:
        info.apple_model_id = detect_apple_model_id()
        chip, is_as = _detect_apple_silicon()
        info.is_apple_silicon = is_as
        info.gpu_name = chip or "Apple Silicon (unified)"
        # vram == unified memory for Apple devices
        info.vram_gb = info.ram_gb
        if info.apple_model_id and info.apple_model_id in APPLE_MODEL_TABLE:
            entry = APPLE_MODEL_TABLE[info.apple_model_id]
            info.apple_display_name = entry[0]
            info.gpu_name = entry[1]
            # Trust measured RAM; only fall back to table if we couldn't measure
            if info.ram_gb == 0.0:
                info.ram_gb = entry[2]
            if info.cpu_cores == 0:
                info.cpu_cores = entry[3]

    elif info.is_android:
        gpu, vram = _detect_android_gpu()
        info.gpu_name = gpu
        info.vram_gb = vram

    elif system == "Windows":
        # Try NVIDIA first (most common for Tier 2 Windows machines)
        nvidia_name, nvidia_vram = _detect_nvidia()
        if nvidia_name:
            info.gpu_name = nvidia_name
            info.vram_gb = nvidia_vram
            info.is_discrete_gpu = True
        else:
            gpu_name, vram_gb, is_discrete = _detect_gpu_windows()
            info.gpu_name = gpu_name or "unknown"
            info.vram_gb = vram_gb
            info.is_discrete_gpu = is_discrete

    elif system == "Darwin":
        chip, is_as = _detect_apple_silicon()
        info.is_apple_silicon = is_as
        info.apple_model_id = detect_apple_model_id()
        if is_as:
            info.gpu_name = chip or "Apple Silicon (unified)"
            info.vram_gb = info.ram_gb  # unified memory
            if info.apple_model_id and info.apple_model_id in APPLE_MODEL_TABLE:
                entry = APPLE_MODEL_TABLE[info.apple_model_id]
                info.apple_display_name = entry[0]
                info.gpu_name = entry[1]
                if info.ram_gb == 0.0:
                    info.ram_gb = entry[2]
                if info.cpu_cores == 0:
                    info.cpu_cores = entry[3]
        else:
            # Intel Mac — try nvidia-smi for eGPU, else report iGPU
            nvidia_name, nvidia_vram = _detect_nvidia()
            if nvidia_name:
                info.gpu_name = nvidia_name
                info.vram_gb = nvidia_vram
                info.is_discrete_gpu = True
            else:
                info.gpu_name = "Intel integrated graphics"
                info.vram_gb = 0.0

    elif system == "Linux":
        nvidia_name, nvidia_vram = _detect_nvidia()
        if nvidia_name:
            info.gpu_name = nvidia_name
            info.vram_gb = nvidia_vram
            info.is_discrete_gpu = True
        else:
            gpu_name, vram_gb, is_discrete = _detect_gpu_linux()
            info.gpu_name = gpu_name or "unknown"
            info.vram_gb = vram_gb
            info.is_discrete_gpu = is_discrete

    # --- Tier inference ---
    info.tier = _infer_tier(info)

    return info


def _infer_tier(info: HardwareInfo) -> int:
    """Infer the AIMESH tier from detected hardware."""
    if info.is_ios or info.is_android:
        return 0
    if info.is_discrete_gpu and info.vram_gb >= 4.0:
        return 2
    if info.is_discrete_gpu:
        # Discrete GPU but low VRAM — still Tier 2 (will just run smaller models)
        return 2
    if info.is_apple_silicon:
        # Apple Silicon Macs are capable but use unified memory — Tier 1
        return 1
    # iGPU or no GPU
    return 1


# ---------------------------------------------------------------------------
# Tier 0 device slot detection
# ---------------------------------------------------------------------------

def _infer_tier0_slot(device_id: str, model_id: Optional[str]) -> str:
    """Map device_id / Apple model string to a tier0 slot key for LITELLM_MODELS."""
    lower = device_id.lower()
    if "ipad" in lower:
        if "2" in lower:
            return "tier0/ipad2"
        return "tier0/ipad1"
    if "iphone" in lower:
        return "tier0/iphone"
    if model_id:
        if model_id.startswith("iPad"):
            if any(x in lower for x in ("2", "second", "b")):
                return "tier0/ipad2"
            return "tier0/ipad1"
        if model_id.startswith("iPhone"):
            return "tier0/iphone"
    return "tier0/android"


# ---------------------------------------------------------------------------
# Device-id helper
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Convert a hostname to a lowercase slug suitable for device_id."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

def build_config_yaml(
    info: HardwareInfo,
    device_id: str,
    display_name: str,
    is_control_plane: bool,
    control_plane_host: str,
) -> str:
    """Return a device config YAML string with all fields populated."""

    if is_control_plane:
        redis_url = "redis://localhost:6379/0"
        litellm_url = "http://localhost:4000/v1"
        cp_note = "  # this device IS the control plane"
    else:
        redis_url = f"redis://{control_plane_host}:6379/0"
        litellm_url = f"http://{control_plane_host}:4000/v1"
        cp_note = f"  # desktop PC running control plane"

    # Models
    if info.tier == 0:
        slot = _infer_tier0_slot(device_id, info.apple_model_id)
        models = LITELLM_MODELS.get(slot, LITELLM_MODELS["tier0/android"])
    elif info.tier == 1:
        models = LITELLM_MODELS["tier1"]
    else:
        models = LITELLM_MODELS["tier2/desktop"] if is_control_plane else LITELLM_MODELS["tier2/laptop"]

    models_yaml = "\n".join(f"  - {m}" for m in models)

    heartbeat = 15.0 if info.tier == 0 else 10.0

    # Hardware block
    vram_comment = ""
    if info.is_apple_silicon:
        vram_comment = "  # unified memory — GPU shares system RAM"
    elif info.tier in (0, 1) and not info.is_discrete_gpu:
        vram_comment = "  # no discrete GPU; iGPU shares system RAM"

    # Tier-specific header comment
    if info.tier == 0:
        tier_comment = "# Tier 0: mobile / edge device (MLX or MLC inference)"
    elif info.tier == 1:
        tier_comment = "# Tier 1: iGPU laptop (Ollama)"
    else:
        role = "desktop PC (control plane + worker)" if is_control_plane else "dGPU laptop (worker only)"
        tier_comment = f"# Tier 2: {role} (Ollama)"

    return textwrap.dedent(f"""\
        # {device_id}.yaml — AIMESH device config
        # Generated by scripts/detect_hardware.py
        {tier_comment}

        # ---------------------------------------------------------------------------
        # Identity
        # ---------------------------------------------------------------------------
        device_id: {device_id}
        name: "{display_name}"

        # ---------------------------------------------------------------------------
        # Tier
        # ---------------------------------------------------------------------------
        tier: {info.tier}

        # ---------------------------------------------------------------------------
        # Connections
        # ---------------------------------------------------------------------------
        redis_url: {redis_url}{cp_note}
        litellm_url: {litellm_url}{cp_note}

        # ---------------------------------------------------------------------------
        # Models
        # ---------------------------------------------------------------------------
        models:
        {models_yaml}

        # ---------------------------------------------------------------------------
        # Heartbeat
        # ---------------------------------------------------------------------------
        heartbeat_interval: {heartbeat}

        # ---------------------------------------------------------------------------
        # Hardware
        # ---------------------------------------------------------------------------
        hardware:
          ram_gb: {info.ram_gb}
          vram_gb: {info.vram_gb}{vram_comment}
          gpu_name: "{info.gpu_name}"
          cpu_cores: {info.cpu_cores}
          os: "{info.os_name}"
        """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect hardware and generate an AIMESH device config YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help="Path to write the config YAML (default: config/<device_id>.yaml).",
    )
    parser.add_argument(
        "--control-plane",
        metavar="HOST",
        default=None,
        help=(
            "Hostname or IP of the desktop PC running Redis and LiteLLM "
            "(default: 'desktop-pc.local'). Ignored if --is-control-plane is set."
        ),
    )
    parser.add_argument(
        "--is-control-plane",
        action="store_true",
        default=False,
        help=(
            "Set this flag when running on the desktop PC that hosts Redis and "
            "LiteLLM. Sets redis_url and litellm_url to localhost."
        ),
    )
    parser.add_argument(
        "--device-id",
        metavar="ID",
        default=None,
        help="Override the auto-detected device_id slug (e.g. 'ipad-1', 'gaming-laptop').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the detected values and generated YAML without writing any file.",
    )
    return parser.parse_args()


def _print_detection_summary(info: HardwareInfo) -> None:
    print("\n── Hardware Detection Summary ──────────────────────────────")
    print(f"  Hostname   : {info.hostname}")
    print(f"  Local IP   : {info.local_ip}")
    print(f"  OS         : {info.os_name}")
    print(f"  RAM        : {info.ram_gb} GB")
    print(f"  CPU cores  : {info.cpu_cores}")
    if info.apple_model_id:
        print(f"  Apple ID   : {info.apple_model_id}", end="")
        if info.apple_display_name:
            print(f"  ({info.apple_display_name})", end="")
        print()
    print(f"  GPU        : {info.gpu_name}")
    if info.is_discrete_gpu:
        print(f"  VRAM       : {info.vram_gb} GB (discrete)")
    elif info.is_apple_silicon:
        print(f"  VRAM       : {info.vram_gb} GB (unified)")
    else:
        print(f"  VRAM       : {info.vram_gb} GB (integrated / shared)")
    print(f"  Inferred tier: {info.tier}")
    print("─────────────────────────────────────────────────────────────\n")


def main() -> None:
    args = _parse_args()

    print("Detecting hardware...", flush=True)
    info = detect_all()
    _print_detection_summary(info)

    # Device ID
    device_id = args.device_id or _slugify(info.hostname)
    if not args.device_id and not args.dry_run:
        suggested = device_id
        response = input(f"Device ID [{suggested}]: ").strip()
        if response:
            device_id = _slugify(response)

    # Display name
    if info.apple_display_name:
        default_name = info.apple_display_name
    else:
        default_name = info.hostname
    if not args.dry_run:
        response = input(f"Display name [{default_name}]: ").strip()
        display_name = response if response else default_name
    else:
        display_name = default_name

    # Control plane host
    is_cp = args.is_control_plane
    if not is_cp and not args.dry_run:
        response = input("Is this the desktop PC running the control plane? [y/N]: ").strip().lower()
        is_cp = response in ("y", "yes")

    control_plane_host = args.control_plane or "desktop-pc.local"
    if not is_cp and not args.control_plane and not args.dry_run:
        response = input(f"Control plane hostname or IP [{control_plane_host}]: ").strip()
        if response:
            control_plane_host = response

    # Generate YAML
    yaml_content = build_config_yaml(
        info=info,
        device_id=device_id,
        display_name=display_name,
        is_control_plane=is_cp,
        control_plane_host=control_plane_host,
    )

    if args.dry_run:
        print("── Generated config (dry run) ──────────────────────────────")
        print(yaml_content)
        return

    # Determine output path
    if args.output:
        out_path = pathlib.Path(args.output)
    else:
        # Walk up from this script's location to find the repo root
        script_dir = pathlib.Path(__file__).parent
        repo_root = script_dir.parent
        config_dir = repo_root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        out_path = config_dir / f"{device_id}.yaml"

    out_path.write_text(yaml_content, encoding="utf-8")
    print(f"✓ Config written to: {out_path}")
    print(f"\nNext step: review the file, then start the worker with:")
    print(f"  python scripts/run_worker.py --config {out_path}\n")


if __name__ == "__main__":
    main()
