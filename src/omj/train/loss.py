"""Label-restricted cross-entropy loss for omj train.

The base model's full vocab logits at the last sequence position are narrowed
to the option-letter label tokens before computing the loss, matching the
restricted-logit readout the semif backend uses at inference time.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Reduction = Literal["mean", "none"]


def _restricted_logp(
    last_logits: Tensor,
    label_token_ids: Tensor,
    label_mask: Tensor | None,
) -> Tensor:
    """Gather label-token logits and return their log-softmax, computed in float32.

    Gathering keeps the loss's gradient wired only to the label-token vocab
    columns; every other vocab entry gets zero gradient. Casting to float32
    after the gather (rather than before) keeps this stable under bf16
    autocast without changing which entries are differentiable.
    """
    z = torch.gather(last_logits, dim=-1, index=label_token_ids)
    z = z.to(torch.float32)
    if label_mask is not None:
        z = z.masked_fill(~label_mask.bool(), float("-inf"))
    return torch.log_softmax(z, dim=-1)


def restricted_cross_entropy(
    last_logits: Tensor,
    label_token_ids: Tensor,
    target_index: Tensor,
    *,
    label_smoothing: float = 0.0,
    label_mask: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    """Cross-entropy over the label-token subset of the vocab, with label smoothing.

    ``last_logits`` is [B, V] (the last-position logits for the batch),
    ``label_token_ids`` is [B, K] (long, the candidate label token ids per
    example), and ``target_index`` is [B] (long, the index into K of the
    correct label). ``label_mask`` is [B, K] of bool (True = valid slot); pass
    it when K is padded out to a fixed width across examples with fewer
    options, so padded slots are excluded from both the softmax and the
    smoothing mean.
    """
    logp = _restricted_logp(last_logits, label_token_ids, label_mask)
    batch_index = torch.arange(logp.shape[0], device=logp.device)
    nll = -logp[batch_index, target_index]

    if label_mask is None:
        smooth = -logp.mean(dim=-1)
    else:
        mask = label_mask.bool()
        valid_count = mask.sum(dim=-1).clamp(min=1).to(logp.dtype)
        # -inf * 0 is nan, so zero out masked slots in logp (not logp * mask) before summing.
        logp_valid = logp.masked_fill(~mask, 0.0)
        smooth = -(logp_valid.sum(dim=-1) / valid_count)

    loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth

    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction: {reduction!r}")


def restricted_brier(
    last_logits: Tensor,
    label_token_ids: Tensor,
    target_index: Tensor,
    *,
    label_mask: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    """Multiclass Brier score over the label-token subset: sum_k (p_k - y_k)^2.

    A proper scoring rule like NLL but bounded, so it penalizes confident
    mistakes without letting them dominate; adding it to the cross-entropy
    makes probability honesty part of the objective rather than a post-hoc fix.
    """
    probs = torch.exp(_restricted_logp(last_logits, label_token_ids, label_mask))
    onehot = torch.zeros_like(probs)
    onehot[torch.arange(probs.shape[0], device=probs.device), target_index] = 1.0
    sq = (probs - onehot) ** 2
    if label_mask is not None:
        sq = sq.masked_fill(~label_mask.bool(), 0.0)
    loss = sq.sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError(f"unknown reduction: {reduction!r}")


def restricted_probs(
    last_logits: Tensor,
    label_token_ids: Tensor,
    label_mask: Tensor | None = None,
) -> Tensor:
    """Softmax probabilities over the same restricted label-token logits, for eval/calibration."""
    logp = _restricted_logp(last_logits, label_token_ids, label_mask)
    return torch.exp(logp)
