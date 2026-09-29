"""Test backend selection matrix. REQ-002, REQ-003."""
# REQ-002
# REQ-003

from __future__ import annotations

import pytest

from omj.errors import ErrorCode, OmjError
from omj.hw.detect import Hardware
from omj.hw.matrix import Overrides, Selection, select_backend


class TestBackendMatrix:
    """REQ-002: Backend selection matrix boundaries."""

    def test_cuda_vram_24gb_and_above(self) -> None:
        """cuda with vram ≥ 24 selects kev backend."""
        hw = Hardware(
            device="cuda",
            vram_gb=24.0,
            ram_gb=32.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 4090",
        )
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False)
        assert sel.backend == "kev"
        assert sel.model == "jaredpalmer/kev-9b"
        assert sel.quant == "bf16"
        assert sel.revision == ""

    def test_cuda_vram_24_5gb(self) -> None:
        """cuda with vram > 24 also selects kev."""
        hw = Hardware(
            device="cuda",
            vram_gb=24.5,
            ram_gb=32.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 4090",
        )
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False)
        assert sel.backend == "kev"

    def test_cuda_vram_23_9gb(self) -> None:
        """cuda with vram < 24 selects semif Qwen3.5-4B."""
        hw = Hardware(
            device="cuda",
            vram_gb=23.9,
            ram_gb=32.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 4080",
        )
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False)
        assert sel.backend == "semif"
        assert sel.model == "Qwen/Qwen3.5-4B"
        assert sel.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        assert sel.quant == "bf16"

    def test_cuda_vram_12gb_exactly(self) -> None:
        """cuda with vram == 12 selects semif Qwen3.5-4B."""
        hw = Hardware(
            device="cuda",
            vram_gb=12.0,
            ram_gb=16.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 4060 Ti",
        )
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False)
        assert sel.backend == "semif"
        assert sel.model == "Qwen/Qwen3.5-4B"

    def test_cuda_vram_11_9gb(self) -> None:
        """cuda with vram < 12 selects semif Qwen3.5-2B."""
        hw = Hardware(
            device="cuda",
            vram_gb=11.9,
            ram_gb=16.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 4060",
        )
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False)
        assert sel.backend == "semif"
        assert sel.model == "Qwen/Qwen3.5-2B"
        assert sel.revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"
        assert sel.quant == "bf16"

    def test_cuda_vram_6gb_exactly(self) -> None:
        """cuda with vram == 6 selects semif Qwen3.5-2B."""
        hw = Hardware(
            device="cuda",
            vram_gb=6.0,
            ram_gb=8.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 2080",
        )
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False)
        assert sel.backend == "semif"
        assert sel.model == "Qwen/Qwen3.5-2B"

    def test_cuda_vram_5_9gb_no_key_raises_e_no_gpu(self) -> None:
        """cuda with vram < 6 and no key raises E_NO_GPU."""
        hw = Hardware(
            device="cuda",
            vram_gb=5.9,
            ram_gb=8.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 2060",
        )
        with pytest.raises(OmjError) as exc_info:
            select_backend(hw, has_typesafe_key=False, has_openrouter_key=False)
        assert exc_info.value.code == ErrorCode.E_NO_GPU

    def test_cpu_no_key_raises_e_no_gpu(self) -> None:
        """cpu with no key raises E_NO_GPU."""
        hw = Hardware(
            device="cpu",
            vram_gb=None,
            ram_gb=8.0,
            platform="linux",
            python="3.11.0",
            gpu_name=None,
        )
        with pytest.raises(OmjError) as exc_info:
            select_backend(hw, has_typesafe_key=False, has_openrouter_key=False)
        assert exc_info.value.code == ErrorCode.E_NO_GPU

    def test_cpu_with_typesafe_key(self) -> None:
        """cpu with typesafe key selects typesafe backend."""
        hw = Hardware(
            device="cpu",
            vram_gb=None,
            ram_gb=8.0,
            platform="linux",
            python="3.11.0",
            gpu_name=None,
        )
        sel = select_backend(hw, has_typesafe_key=True, has_openrouter_key=False)
        assert sel.backend == "typesafe"
        assert sel.model == "jev-latest"
        assert sel.quant == "bf16"
        assert "typesafe" in sel.reason.lower()

    def test_cpu_with_openrouter_key(self) -> None:
        """cpu with openrouter key (no typesafe) selects typesafe backend."""
        hw = Hardware(
            device="cpu",
            vram_gb=None,
            ram_gb=8.0,
            platform="linux",
            python="3.11.0",
            gpu_name=None,
        )
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=True)
        assert sel.backend == "typesafe"
        assert sel.model == "jev-latest"
        assert "openrouter" in sel.reason.lower()

    def test_vram_less_than_6_with_typesafe_key(self) -> None:
        """vram < 6 with typesafe key selects typesafe."""
        hw = Hardware(
            device="cuda",
            vram_gb=5.9,
            ram_gb=8.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 2060",
        )
        sel = select_backend(hw, has_typesafe_key=True, has_openrouter_key=False)
        assert sel.backend == "typesafe"
        assert sel.model == "jev-latest"


class TestOverrides:
    """REQ-003: Overrides replace matrix results."""

    def test_backend_override(self) -> None:
        """Override backend in matrix result."""
        hw = Hardware(
            device="cuda",
            vram_gb=24.0,
            ram_gb=32.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 4090",
        )
        overrides = Overrides(backend="mock")
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False, overrides=overrides)
        assert sel.backend == "mock"

    def test_model_override(self) -> None:
        """Override model in matrix result."""
        hw = Hardware(
            device="cuda",
            vram_gb=12.0,
            ram_gb=16.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 4060 Ti",
        )
        overrides = Overrides(model="custom/model")
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False, overrides=overrides)
        assert sel.model == "custom/model"

    def test_quant_override(self) -> None:
        """Override quant in matrix result."""
        hw = Hardware(
            device="cuda",
            vram_gb=12.0,
            ram_gb=16.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 4060 Ti",
        )
        overrides = Overrides(quant="4bit-prequant")
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False, overrides=overrides)
        assert sel.quant == "4bit-prequant"

    def test_backend_override_prevents_no_gpu_raise(self) -> None:
        """When backend is overridden, E_NO_GPU should not be raised."""
        hw = Hardware(
            device="cuda",
            vram_gb=5.9,
            ram_gb=8.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 2060",
        )
        overrides = Overrides(backend="mock")
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False, overrides=overrides)
        assert sel.backend == "mock"

    def test_combined_overrides(self) -> None:
        """Multiple overrides work together."""
        hw = Hardware(
            device="cuda",
            vram_gb=24.0,
            ram_gb=32.0,
            platform="linux",
            python="3.11.0",
            gpu_name="RTX 4090",
        )
        overrides = Overrides(backend="semif", model="different/model", quant="4bit-prequant")
        sel = select_backend(hw, has_typesafe_key=False, has_openrouter_key=False, overrides=overrides)
        assert sel.backend == "semif"
        assert sel.model == "different/model"
        assert sel.quant == "4bit-prequant"


class TestOverrideConsistency:
    """REQ-003: an override must not leave another backend's model/revision behind."""

    def _cuda_24gb(self) -> Hardware:
        return Hardware(
            device="cuda",
            vram_gb=24.0,
            ram_gb=32.0,
            platform="linux",
            python="3.11.9",
            gpu_name="RTX 4090",
        )

    def _cuda_8gb(self) -> Hardware:
        return Hardware(
            device="cuda",
            vram_gb=8.0,
            ram_gb=16.0,
            platform="windows",
            python="3.11.9",
            gpu_name="RTX 4060",
        )

    def test_backend_override_to_typesafe_resets_model_and_revision(self) -> None:
        # REQ-003: --backend typesafe must not keep the matrix's Qwen model.
        sel = select_backend(
            self._cuda_8gb(),
            has_typesafe_key=False,
            has_openrouter_key=False,
            overrides=Overrides(backend="typesafe"),
        )
        assert sel.backend == "typesafe"
        assert sel.model == "jev-latest"
        assert sel.revision == ""

    def test_backend_override_to_kev_resets_model_and_revision(self) -> None:
        # REQ-003
        sel = select_backend(
            self._cuda_8gb(),
            has_typesafe_key=False,
            has_openrouter_key=False,
            overrides=Overrides(backend="kev"),
        )
        assert sel.backend == "kev"
        assert sel.model == "jaredpalmer/kev-4b"
        assert sel.revision == ""

    def test_backend_override_to_semif_uses_semif_defaults(self) -> None:
        # REQ-003: kev matrix result overridden to semif gets semif's pinned model.
        sel = select_backend(
            self._cuda_24gb(),
            has_typesafe_key=False,
            has_openrouter_key=False,
            overrides=Overrides(backend="semif"),
        )
        assert sel.backend == "semif"
        assert sel.model == "Qwen/Qwen3.5-2B"
        assert sel.revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"

    def test_backend_override_to_mock_clears_model_and_revision(self) -> None:
        # REQ-003
        sel = select_backend(
            self._cuda_8gb(),
            has_typesafe_key=False,
            has_openrouter_key=False,
            overrides=Overrides(backend="mock"),
        )
        assert sel.backend == "mock"
        assert sel.model == ""
        assert sel.revision == ""

    def test_backend_override_matching_matrix_keeps_pinned_revision(self) -> None:
        # REQ-003: --backend semif on a semif machine changes nothing.
        sel = select_backend(
            self._cuda_8gb(),
            has_typesafe_key=False,
            has_openrouter_key=False,
            overrides=Overrides(backend="semif"),
        )
        assert sel.model == "Qwen/Qwen3.5-2B"
        assert sel.revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"

    def test_model_override_clears_the_pinned_revision(self) -> None:
        # REQ-003: the pinned revision belongs to the replaced model.
        sel = select_backend(
            self._cuda_8gb(),
            has_typesafe_key=False,
            has_openrouter_key=False,
            overrides=Overrides(model="custom/model"),
        )
        assert sel.model == "custom/model"
        assert sel.revision == ""

    def test_model_override_with_explicit_revision_is_honored(self) -> None:
        # REQ-003
        sel = select_backend(
            self._cuda_8gb(),
            has_typesafe_key=False,
            has_openrouter_key=False,
            overrides=Overrides(model="custom/model", revision="deadbeef"),
        )
        assert sel.model == "custom/model"
        assert sel.revision == "deadbeef"

    def test_model_override_equal_to_matrix_model_keeps_revision(self) -> None:
        # REQ-003
        sel = select_backend(
            self._cuda_8gb(),
            has_typesafe_key=False,
            has_openrouter_key=False,
            overrides=Overrides(model="Qwen/Qwen3.5-2B"),
        )
        assert sel.revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"


# REQ-003
class TestResidualOverrides:
    """Review residuals: bare --model on a small GPU, kev default sized to the tier."""

    @staticmethod
    def _hw(device: str, vram: float | None) -> Hardware:
        return Hardware(device=device, vram_gb=vram, ram_gb=16.0, platform="windows", python="3.11.9", gpu_name=None)

    def test_model_override_alone_on_small_gpu_uses_semif(self) -> None:
        # REQ-003: --model alone must override the matrix instead of raising E_NO_GPU
        sel = select_backend(self._hw("cuda", 4.0), False, False, Overrides(model="Qwen/Qwen3.5-0.8B"))
        assert sel.backend == "semif"
        assert sel.model == "Qwen/Qwen3.5-0.8B"
        assert sel.revision == ""

    def test_model_override_alone_on_cpu_still_raises(self) -> None:
        # REQ-003: semif needs a GPU, so a bare --model on CPU keeps E_NO_GPU
        with pytest.raises(OmjError) as info:
            select_backend(self._hw("cpu", None), False, False, Overrides(model="Qwen/Qwen3.5-0.8B"))
        assert info.value.code is ErrorCode.E_NO_GPU

    def test_kev_override_default_follows_gpu_tier(self) -> None:
        # REQ-003: kev default model matches the matrix tier (9b at >=24GB, 4b below)
        big = select_backend(self._hw("cuda", 16.0), False, False, Overrides(backend="kev"))
        assert big.model == "jaredpalmer/kev-4b"
        cpu = select_backend(self._hw("cpu", None), True, False, Overrides(backend="kev"))
        assert cpu.model == "jaredpalmer/kev-4b"
