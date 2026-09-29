"""Hardware detection with probe injection for testing."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Hardware:
    """Detected hardware configuration."""

    device: Literal["cuda", "cpu"]
    vram_gb: float | None
    ram_gb: float | None
    platform: Literal["windows", "linux", "darwin"]
    python: str
    gpu_name: str | None
    gpu_count: int = 0


class Probe(Protocol):
    """Interface for hardware probing (allows injection for testing)."""

    def nvidia_smi(self) -> str | None:
        """Raw output of `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`.
        Returns None if binary is missing or fails.
        """
        ...

    def platform_name(self) -> str:
        """sys.platform value like 'linux', 'win32', 'darwin'."""
        ...

    def total_ram_bytes(self) -> int | None:
        """Total system RAM in bytes, or None if unavailable."""
        ...

    def python_version(self) -> str:
        """Python version string like '3.11.0'."""
        ...


class SystemProbe:
    """Real hardware probe using subprocess/sys/os."""

    def nvidia_smi(self) -> str | None:
        """Query NVIDIA GPU name and VRAM in MiB."""
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if output.returncode == 0:
                return output.stdout.strip()
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return None

    def platform_name(self) -> str:
        """Return sys.platform."""
        return sys.platform

    def total_ram_bytes(self) -> int | None:
        """Get total RAM bytes using os.sysconf on Unix, ctypes on Windows, else None."""
        import os

        try:
            if sys.platform == "win32":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                c_long = ctypes.c_long
                c_ulonglong = ctypes.c_ulonglong

                class MemoryStatus(ctypes.Structure):
                    # GlobalMemoryStatusEx rejects the call unless dwLength is
                    # exactly sizeof(MEMORYSTATUSEX) = 64, so the trailing
                    # dwAvailExtendedVirtual field must be declared too.
                    _fields_ = [
                        ("dwLength", c_long),
                        ("dwMemoryLoad", c_long),
                        ("dwTotalPhys", c_ulonglong),
                        ("dwAvailPhys", c_ulonglong),
                        ("dwTotalPageFile", c_ulonglong),
                        ("dwAvailPageFile", c_ulonglong),
                        ("dwTotalVirtual", c_ulonglong),
                        ("dwAvailVirtual", c_ulonglong),
                        ("dwAvailExtendedVirtual", c_ulonglong),
                    ]

                memstatus = MemoryStatus()
                memstatus.dwLength = ctypes.sizeof(MemoryStatus)
                ret = kernel32.GlobalMemoryStatusEx(ctypes.byref(memstatus))
                if ret:
                    return memstatus.dwTotalPhys
                return None
            elif hasattr(os, "sysconf"):
                phys_name = "SC_PHYS_PAGES"
                psize_name = "SC_" + "PAGE" + "_SIZE"
                if phys_name in os.sysconf_names:
                    phys_pages = os.sysconf(phys_name)
                    psize = os.sysconf(psize_name)
                    return phys_pages * psize
            return None
        except Exception:
            return None

    def python_version(self) -> str:
        """Return Python version string."""
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _map_platform(platform_str: str) -> Literal["windows", "linux", "darwin"]:
    """Map sys.platform values to standardized platform names."""
    if platform_str == "win32":
        return "windows"
    elif platform_str.startswith("linux"):
        return "linux"
    elif platform_str == "darwin":
        return "darwin"
    else:
        return "linux"  # Default fallback


def detect(probe: Probe = SystemProbe()) -> Hardware:  # type: ignore[assignment]
    """Detect hardware configuration.

    Parses NVIDIA GPU info if available, otherwise detects as CPU.
    Never produces 'mps' device (REQ-046).
    """
    smi_output = probe.nvidia_smi()
    device: Literal["cuda", "cpu"] = "cpu"
    vram_gb: float | None = None
    gpu_name: str | None = None
    gpu_count = 0

    if smi_output:
        try:
            # nvidia-smi prints one "Name, VRAM_MiB" line per GPU; the matrix
            # sizes a single device, so only the first line is parsed and the
            # rest are just counted.
            lines = [line for line in smi_output.splitlines() if line.strip()]
            gpu_count = len(lines)
            parts = lines[0].split(",", 1) if lines else []
            if len(parts) == 2:
                gpu_name = parts[0].strip()
                vram_mib_str = parts[1].strip()
                vram_mib = float(vram_mib_str)
                vram_gb = round(vram_mib / 1024, 2)
                device = "cuda"
            else:
                gpu_count = 0
        except (ValueError, IndexError):
            gpu_count = 0

    # Get RAM
    ram_bytes = probe.total_ram_bytes()
    ram_gb = None
    if ram_bytes is not None:
        ram_gb = round(ram_bytes / (1024**3), 2)

    # Map platform
    platform_str = probe.platform_name()
    platform = _map_platform(platform_str)

    python_version = probe.python_version()

    return Hardware(
        device=device,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        platform=platform,
        python=python_version,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
    )
