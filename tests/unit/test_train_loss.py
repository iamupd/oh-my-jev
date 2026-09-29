"""Label-restricted cross-entropy loss: hand-computed values, grad routing, masking, autocast.

# REQ-006
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from omj.train.loss import restricted_cross_entropy, restricted_probs  # noqa: E402

# Full-vocab last-position logits (V=5) for a batch of 2 examples.
LAST_LOGITS = [
    [1.0, 2.0, 0.5, -1.0, 3.0],
    [0.2, -0.3, 1.5, 2.2, 0.1],
]
# K=3 label-token columns per example, gathered from LAST_LOGITS above.
LABEL_TOKEN_IDS = [
    [0, 1, 4],
    [2, 3, 4],
]
# Restricted logits (LAST_LOGITS gathered at LABEL_TOKEN_IDS): [[1.0, 2.0, 3.0], [1.5, 2.2, 0.1]]
TARGET_INDEX = [2, 0]


def _reference_loss(z_rows: list[list[float]], target_idx: list[int], eps: float) -> list[float]:
    """Independent, non-torch reference for (1-eps)*NLL + eps*mean(-logp) over the K restricted logits."""
    losses = []
    for z, t in zip(z_rows, target_idx, strict=True):
        m = max(z)
        logsumexp = m + math.log(sum(math.exp(zi - m) for zi in z))
        nll = logsumexp - z[t]
        mean_z = sum(z) / len(z)
        smooth = logsumexp - mean_z
        losses.append((1.0 - eps) * nll + eps * smooth)
    return losses


def _restricted_z() -> list[list[float]]:
    return [
        [LAST_LOGITS[row][col] for col in LABEL_TOKEN_IDS[row]]
        for row in range(len(LAST_LOGITS))
    ]


@pytest.mark.parametrize("eps", [0.0, 0.1])
def test_matches_hand_computed_values(eps: float) -> None:
    last_logits = torch.tensor(LAST_LOGITS, dtype=torch.float32)
    label_token_ids = torch.tensor(LABEL_TOKEN_IDS, dtype=torch.long)
    target_index = torch.tensor(TARGET_INDEX, dtype=torch.long)

    expected = torch.tensor(_reference_loss(_restricted_z(), TARGET_INDEX, eps), dtype=torch.float32)

    per_example = restricted_cross_entropy(
        last_logits, label_token_ids, target_index, label_smoothing=eps, reduction="none"
    )
    torch.testing.assert_close(per_example, expected, atol=1e-6, rtol=0.0)

    mean_loss = restricted_cross_entropy(
        last_logits, label_token_ids, target_index, label_smoothing=eps, reduction="mean"
    )
    torch.testing.assert_close(mean_loss, expected.mean(), atol=1e-6, rtol=0.0)


def test_reduction_none_shape_is_batch() -> None:
    last_logits = torch.tensor(LAST_LOGITS, dtype=torch.float32)
    label_token_ids = torch.tensor(LABEL_TOKEN_IDS, dtype=torch.long)
    target_index = torch.tensor(TARGET_INDEX, dtype=torch.long)

    per_example = restricted_cross_entropy(
        last_logits, label_token_ids, target_index, label_smoothing=0.05, reduction="none"
    )
    assert per_example.shape == (2,)


def test_grad_flows_only_through_selected_label_logits() -> None:
    last_logits = torch.tensor(LAST_LOGITS, dtype=torch.float32, requires_grad=True)
    label_token_ids = torch.tensor(LABEL_TOKEN_IDS, dtype=torch.long)
    target_index = torch.tensor(TARGET_INDEX, dtype=torch.long)

    loss = restricted_cross_entropy(last_logits, label_token_ids, target_index, label_smoothing=0.1)
    loss.backward()

    grad = last_logits.grad
    assert grad is not None

    selected = torch.zeros_like(grad, dtype=torch.bool)
    for row, cols in enumerate(LABEL_TOKEN_IDS):
        selected[row, cols] = True

    assert torch.all(grad[~selected] == 0)
    assert torch.any(grad[selected] != 0)


def test_label_mask_excludes_padded_slots() -> None:
    # Pad K=3 -> K=4 with a 6th vocab column holding an extreme logit that would
    # dominate the softmax if it leaked in; label_mask must exclude it entirely.
    padded_logits = torch.tensor(
        [row + [1000.0] for row in LAST_LOGITS], dtype=torch.float32
    )
    padded_label_ids = torch.tensor(
        [cols + [5] for cols in LABEL_TOKEN_IDS], dtype=torch.long
    )
    label_mask = torch.tensor([[True, True, True, False]] * 2)
    target_index = torch.tensor(TARGET_INDEX, dtype=torch.long)

    unpadded_logits = torch.tensor(LAST_LOGITS, dtype=torch.float32)
    unpadded_label_ids = torch.tensor(LABEL_TOKEN_IDS, dtype=torch.long)

    for eps in (0.0, 0.1):
        expected = restricted_cross_entropy(
            unpadded_logits, unpadded_label_ids, target_index, label_smoothing=eps, reduction="none"
        )
        got = restricted_cross_entropy(
            padded_logits,
            padded_label_ids,
            target_index,
            label_smoothing=eps,
            label_mask=label_mask,
            reduction="none",
        )
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=0.0)

    probs = restricted_probs(padded_logits, padded_label_ids, label_mask=label_mask)
    torch.testing.assert_close(probs[:, 3], torch.zeros(2), atol=0.0, rtol=0.0)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0.0)


def test_autocast_bf16_matches_fp32_within_tolerance() -> None:
    last_logits_fp32 = torch.tensor(LAST_LOGITS, dtype=torch.float32)
    label_token_ids = torch.tensor(LABEL_TOKEN_IDS, dtype=torch.long)
    target_index = torch.tensor(TARGET_INDEX, dtype=torch.long)

    loss_fp32 = restricted_cross_entropy(
        last_logits_fp32, label_token_ids, target_index, label_smoothing=0.1
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        last_logits_bf16 = last_logits_fp32.to(torch.bfloat16)
        loss_bf16 = restricted_cross_entropy(
            last_logits_bf16, label_token_ids, target_index, label_smoothing=0.1
        )

    assert torch.isfinite(loss_bf16)
    assert abs(float(loss_bf16) - float(loss_fp32)) < 1e-2
