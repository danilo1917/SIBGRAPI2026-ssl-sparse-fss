import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


def _mascara_valida(target: Tensor) -> Tensor:
    """Return a boolean mask that is True wherever target != -1."""
    return target != -1.0


def dice_loss(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Soft Dice loss from raw logits, ignoring pixels where target == -1."""
    valid    = _mascara_valida(target)          # (B, 1, H, W) bool
    pred_sig = torch.sigmoid(pred)

    # Flatten spatial dims; keep batch.
    B = pred_sig.shape[0]
    pred_flat   = pred_sig.view(B, -1)
    target_flat = target.view(B, -1)
    valid_flat  = valid.view(B, -1)

    dices = []
    for b in range(B):
        pv = pred_flat[b][valid_flat[b]]
        tv = target_flat[b][valid_flat[b]]
        if pv.numel() == 0:
            continue
        inter = (pv * tv).sum()
        soma  = pv.sum() + tv.sum()
        dices.append(1.0 - (2.0 * inter + eps) / (soma + eps))

    if not dices:
        return pred.sum() * 0.0
    return torch.stack(dices).mean()


def bce_loss(pred: Tensor, target: Tensor, max_pos_weight: float = 30.0) -> Tensor:
    """BCE with logits, ignoring pixels where target == -1."""
    valid = _mascara_valida(target)

    if not valid.any():
        return pred.sum() * 0.0

    pred_v   = pred[valid]
    target_v = target[valid]

    n_pos = target_v.sum().clamp(min=1.0)
    n_neg = (1.0 - target_v).sum().clamp(min=1.0)
    pos_w = (n_neg / n_pos).clamp(max=max_pos_weight)

    return F.binary_cross_entropy_with_logits(pred_v, target_v, pos_weight=pos_w)


def loss_combinada(
    pred: Tensor,
    target: Tensor,
    bce_weight: float = 0.4,
    dice_weight: float = 0.6,
) -> Tensor:
    """Combined loss: ``bce_weight × BCE + dice_weight × Dice``."""
    return bce_weight * bce_loss(pred, target) + dice_weight * dice_loss(pred, target)
    