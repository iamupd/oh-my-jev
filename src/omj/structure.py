"""Structure options for the decision readout: noncausal full attention and a dedicated readout head.

Qwen3.5 mixes Gated DeltaNet (linear) layers with full-attention layers. ``enable_noncausal`` makes the
full-attention layers bidirectional over the non-padding tokens; the linear layers stay recurrent.
``ReadoutHead`` replaces the LM head with a ``Linear(hidden, R)`` over the option-letter tokens only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

CONFIG_FILE = "decision_config.json"
READOUT_FILE = "readout.safetensors"
STRUCTURES = ("causal", "noncausal_readout")


def text_model(model: nn.Module) -> nn.Module:
    """The module that owns ``embed_tokens`` and ``layers`` (works through PEFT wrappers)."""
    for module in model.modules():
        if hasattr(module, "layers") and hasattr(module, "embed_tokens") and hasattr(module.config, "layer_types"):
            return module
    raise ValueError("no decoder with embed_tokens/layers/layer_types found")


class NoncausalHandle:
    """Hook handles plus an ``enabled`` switch (False = layers run with their stock causal mask)."""

    def __init__(self) -> None:
        self._handles: list = []
        self.enabled = True

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def enable_noncausal(model: nn.Module) -> NoncausalHandle:
    """Make full-attention layers bidirectional; padding keys stay masked. Needs SDPA and no KV cache."""
    inner = text_model(model)
    holder: dict[str, Tensor | None] = {"padding": None}
    state = NoncausalHandle()

    def capture(_module, args, kwargs):
        holder["padding"] = kwargs.get("attention_mask")
        return None

    def full_mask(_module, args, kwargs):
        if not state.enabled:
            return None
        hidden = args[0] if args else kwargs["hidden_states"]
        padding = holder["padding"]
        if padding is None:
            padding = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        kwargs["attention_mask"] = padding[:, None, None, :].bool()
        return args, kwargs

    state._handles.append(inner.register_forward_pre_hook(capture, with_kwargs=True))
    for layer_type, layer in zip(inner.config.layer_types, inner.layers):
        if layer_type == "full_attention":
            state._handles.append(layer.register_forward_pre_hook(full_mask, with_kwargs=True))
    return state


def hidden_last(model: nn.Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Final-norm hidden state at the last position (inputs are left-padded)."""
    out = text_model(model)(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.last_hidden_state[:, -1, :]


class ReadoutHead(nn.Module):
    def __init__(self, hidden_size: int, n_options: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, n_options, bias=False)

    @classmethod
    def from_lm_head(cls, model: nn.Module, token_ids: Sequence[int]) -> "ReadoutHead":
        weight = model.get_output_embeddings().weight
        head = cls(weight.shape[1], len(token_ids))
        with torch.no_grad():
            head.proj.weight.copy_(weight[list(token_ids)].detach().to(head.proj.weight.dtype))
        return head.to(weight.device)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.proj(hidden.to(self.proj.weight.dtype))


def remap_label_ids(label_token_ids: Tensor, label_mask: Tensor, table: Sequence[int]) -> Tensor:
    """Map vocabulary token ids [B,K] to readout rows; masked (padding) slots become row 0."""
    lookup = {int(token): row for row, token in enumerate(table)}
    rows = torch.zeros_like(label_token_ids)
    for b in range(label_token_ids.shape[0]):
        for k in range(label_token_ids.shape[1]):
            if bool(label_mask[b, k]):
                token = int(label_token_ids[b, k])
                if token not in lookup:
                    raise ValueError(f"token id {token} not in readout table")
                rows[b, k] = lookup[token]
    return rows


def save_readout(directory: Path | str, head: ReadoutHead, table: Sequence[int], *, structure: str) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    save_file({"proj.weight": head.proj.weight.detach().cpu().contiguous()}, str(target / READOUT_FILE))
    (target / CONFIG_FILE).write_text(
        json.dumps({"structure": structure, "readout_token_ids": [int(t) for t in table]}), encoding="utf-8"
    )


def load_readout(directory: Path | str) -> tuple[ReadoutHead, list[int], str]:
    target = Path(directory)
    config = json.loads((target / CONFIG_FILE).read_text(encoding="utf-8"))
    weight = load_file(str(target / READOUT_FILE))["proj.weight"]
    head = ReadoutHead(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        head.proj.weight.copy_(weight)
    return head, list(config["readout_token_ids"]), config["structure"]
