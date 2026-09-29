"""Pick bf16 or 4-bit (nf4) loading for a Hugging Face model id from its size tag and the GPU memory.

Used wherever a user names a model instead of writing a config file (`init --model`,
`bench --model`, `ui --target name=org/model`), so they never have to set `quant` by hand.
"""

from __future__ import annotations

import re

# "Qwen3.5-4B", "Llama-3.2-3B-Instruct", "gemma-3-1b-it", "Qwen3.5-0.8B": digits followed by b/B, then a
# separator or the end. The version number ("3.5", "3.2") never matches because no "b" follows it.
_SIZE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)[bB](?![a-zA-Z0-9])")
BF16_GB_PER_BILLION = 2.0
OVERHEAD_GB = 1.5  # activations, KV cache at a 4,096-token context, CUDA context


def param_billions(model_id: str) -> float | None:
    """Parameter count in billions read from the model name, or None when the name has no size tag."""
    matches = _SIZE.findall(model_id.rsplit("/", 1)[-1].replace("_", "-"))
    return float(matches[-1]) if matches else None


def auto_quant(model_id: str, vram_gb: float | None) -> str:
    """"nf4" when the model would not fit in bf16 on this GPU, else "bf16" (also when either is unknown)."""
    billions = param_billions(model_id)
    if billions is None or vram_gb is None:
        return "bf16"
    return "nf4" if billions * BF16_GB_PER_BILLION + OVERHEAD_GB > vram_gb else "bf16"


_HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def is_hf_id(text: str) -> bool:
    """True for "org/name" Hugging Face model ids (not paths, not backend names)."""
    return bool(_HF_ID.match(text))


def named_model_backend(backend: dict, model_id: str, vram_gb: float | None) -> dict:
    """A semif [backend] section (as a dict) for a model given by name, starting from `backend`.

    The revision is a copy already in the omj cache, else the default branch; quant follows `auto_quant`; the adapter and the
    calibration file of the configured model do not carry over to a different model.
    """
    from omj.models.download import cached_revision

    cached = cached_revision(model_id)
    out = dict(backend)
    out.update(name="semif", model=model_id, revision="" if cached in (None, "main") else cached,
               quant=auto_quant(model_id, vram_gb), adapter="", calibration="", offline=False)
    return out


def _is_adapter_repo(model_id: str) -> bool:
    """True when the Hub repo (or its cached copy) is a LoRA adapter, not a full model."""
    import os

    from omj.models.download import cached_revision

    cached = cached_revision(model_id)
    if cached is not None:
        from omj.config import omj_home

        return (omj_home() / "models" / model_id.replace("/", "__") / cached / "adapter_config.json").is_file()
    if os.environ.get("OMJ_NO_NETWORK") == "1" or os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    try:
        from huggingface_hub import file_exists

        return file_exists(model_id, "adapter_config.json")
    except Exception:
        return False  # unknown or unreachable: let the normal load report the real error


def resolve_named_model(backend: dict, model_id: str, vram_gb: float | None) -> tuple[dict, str | None]:
    """The semif section for `--model org/name`, which may be a full model or a LoRA adapter repo.

    An adapter repo is downloaded (it is small) and loaded on the base model it names, so
    `omj bench --model corners-ai/CoCo-Decision-4B-Ko` works like any other model id.
    """
    if not _is_adapter_repo(model_id):
        return named_model_backend(backend, model_id, vram_gb), None
    from omj.backends.adapter_info import follow_adapter_base
    from omj.models.download import cached_revision, download_model

    cached = cached_revision(model_id)
    path = download_model(model_id, "" if cached in (None, "main") else cached)
    out = dict(backend)
    out.update(name="semif", adapter=str(path), calibration="", offline=False)
    follow_adapter_base(out, str(path), vram_gb)
    return out, f"{model_id} is a LoRA adapter; loading it on {out.get('model')} ({out.get('quant')})."


def configured_vram_gb(config: object) -> float | None:
    """GPU memory recorded by `omj init`, else detected now; None without a CUDA GPU."""
    recorded = getattr(getattr(config, "hardware", None), "vram_gb", None)
    if recorded is not None:
        return recorded
    from omj.hw.detect import SystemProbe, detect

    return detect(SystemProbe()).vram_gb
