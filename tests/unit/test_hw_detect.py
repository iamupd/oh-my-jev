"""Test hardware detection with fake probes. REQ-001, REQ-046."""
# REQ-001
# REQ-046

from __future__ import annotations

import sys

import pytest

from omj.hw.detect import Hardware, Probe, SystemProbe, detect


class FakeProbe(Probe):
    """Fake probe for testing."""

    def __init__(
        self,
        nvidia_smi_output: str | None = None,
        platform: str = "linux",
        total_ram_bytes: int | None = None,
        python_version: str = "3.11.0",
    ) -> None:
        self._nvidia_smi_output = nvidia_smi_output
        self._platform = platform
        self._total_ram_bytes = total_ram_bytes
        self._python_version = python_version

    def nvidia_smi(self) -> str | None:
        return self._nvidia_smi_output

    def platform_name(self) -> str:
        return self._platform

    def total_ram_bytes(self) -> int | None:
        return self._total_ram_bytes

    def python_version(self) -> str:
        return self._python_version


class TestHardwareDetectBasic:
    """REQ-001: Basic hardware detection."""

    def test_detect_cuda_with_gpu_name_and_vram(self) -> None:
        """Parse NVIDIA GPU info from nvidia-smi output."""
        probe = FakeProbe(
            nvidia_smi_output="NVIDIA GeForce RTX 4060 Laptop GPU, 8188",
            platform="linux",
            total_ram_bytes=16 * 1024**3,
            python_version="3.11.0",
        )
        hw = detect(probe)
        assert hw.device == "cuda"
        assert hw.vram_gb == 8.0  # 8188 MiB → 8.0 GB (rounded to 2 decimals)
        assert hw.gpu_name == "NVIDIA GeForce RTX 4060 Laptop GPU"
        assert hw.ram_gb is not None
        assert abs(hw.ram_gb - 16.0) < 0.01
        assert hw.platform == "linux"
        assert hw.python == "3.11.0"

    def test_detect_cpu_no_gpu(self) -> None:
        """Detect CPU when nvidia-smi returns None."""
        probe = FakeProbe(
            nvidia_smi_output=None,
            platform="linux",
            total_ram_bytes=8 * 1024**3,
            python_version="3.11.0",
        )
        hw = detect(probe)
        assert hw.device == "cpu"
        assert hw.vram_gb is None
        assert hw.gpu_name is None

    def test_platform_mapping_windows(self) -> None:
        """Map win32 platform to windows."""
        probe = FakeProbe(
            nvidia_smi_output=None,
            platform="win32",
            total_ram_bytes=4 * 1024**3,
            python_version="3.11.0",
        )
        hw = detect(probe)
        assert hw.platform == "windows"

    def test_platform_mapping_linux(self) -> None:
        """Map linux platform to linux."""
        probe = FakeProbe(
            nvidia_smi_output=None,
            platform="linux",
            total_ram_bytes=4 * 1024**3,
            python_version="3.11.0",
        )
        hw = detect(probe)
        assert hw.platform == "linux"

    def test_platform_mapping_linux_variant(self) -> None:
        """Map linux2 variant platform to linux."""
        probe = FakeProbe(
            nvidia_smi_output=None,
            platform="linux2",
            total_ram_bytes=4 * 1024**3,
            python_version="3.11.0",
        )
        hw = detect(probe)
        assert hw.platform == "linux"

    def test_platform_mapping_darwin(self) -> None:
        """Map darwin platform to darwin."""
        probe = FakeProbe(
            nvidia_smi_output=None,
            platform="darwin",
            total_ram_bytes=4 * 1024**3,
            python_version="3.11.0",
        )
        hw = detect(probe)
        assert hw.platform == "darwin"

    def test_vram_rounding_to_2_decimals(self) -> None:
        """VRAM should be MiB/1024 rounded to 2 decimals."""
        probe = FakeProbe(
            nvidia_smi_output="GPU Name, 12345",
            platform="linux",
            total_ram_bytes=16 * 1024**3,
            python_version="3.11.0",
        )
        hw = detect(probe)
        assert hw.device == "cuda"
        assert hw.vram_gb == round(12345 / 1024, 2)

    def test_never_detect_mps(self) -> None:
        """REQ-046: Never produce 'mps' as device (Windows constraint)."""
        # Test that mps is never a possible device value
        probe = FakeProbe(
            nvidia_smi_output=None,
            platform="win32",
            total_ram_bytes=4 * 1024**3,
            python_version="3.11.0",
        )
        hw = detect(probe)
        assert hw.device in ("cuda", "cpu")
        assert hw.device != "mps"

    def test_ram_none_when_not_available(self) -> None:
        """RAM should be None when total_ram_bytes() returns None."""
        probe = FakeProbe(
            nvidia_smi_output=None,
            platform="linux",
            total_ram_bytes=None,
            python_version="3.11.0",
        )
        hw = detect(probe)
        assert hw.ram_gb is None


class TestMultiGpuAndRealProbe:
    """REQ-001: multi-GPU nvidia-smi output and the real Windows RAM probe."""

    def test_multi_gpu_output_uses_first_line_and_counts_gpus(self) -> None:
        # REQ-001: nvidia-smi prints one line per GPU; parsing the whole blob
        # makes float() fail and the machine look like a CPU box.
        probe = FakeProbe(
            nvidia_smi_output=(
                "NVIDIA GeForce RTX 4090, 24564\nNVIDIA GeForce RTX 4090, 24564"
            ),
            platform="linux",
            total_ram_bytes=64 * 1024**3,
            python_version="3.11.9",
        )
        hw = detect(probe)
        assert hw.device == "cuda"
        assert hw.gpu_name == "NVIDIA GeForce RTX 4090"
        assert hw.vram_gb == round(24564 / 1024, 2)
        assert hw.gpu_count == 2

    def test_single_gpu_reports_gpu_count_one(self) -> None:
        # REQ-001
        probe = FakeProbe(
            nvidia_smi_output="NVIDIA GeForce RTX 4060 Laptop GPU, 8188",
            platform="win32",
            total_ram_bytes=16 * 1024**3,
            python_version="3.11.9",
        )
        hw = detect(probe)
        assert hw.gpu_count == 1

    def test_cpu_reports_gpu_count_zero(self) -> None:
        # REQ-001
        hw = detect(FakeProbe(nvidia_smi_output=None))
        assert hw.gpu_count == 0

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only RAM probe")
    def test_system_probe_reads_total_ram_on_windows(self) -> None:
        # REQ-001: MEMORYSTATUSEX must be 64 bytes or GlobalMemoryStatusEx fails
        # and RAM silently reads as unknown.
        total = SystemProbe().total_ram_bytes()
        assert total is not None
        assert total > 0
