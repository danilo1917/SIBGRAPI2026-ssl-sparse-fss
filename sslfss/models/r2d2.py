# Ridge regression head following R2-D2 (Bertinetto et al., ICLR 2019).
# Reference implementation: https://github.com/bertinetto/r2d2 (MIT License).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class R2D2Encoder(nn.Module):
    """Lightweight VGG-style encoder for greyscale medical images."""

    def __init__(self, in_channels: int = 1, feat_channels: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            self._conv_block(in_channels, 64),
            self._conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),      # H/2

            # Block 2
            self._conv_block(64, 128),
            self._conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),      # H/4

            # Block 3 — dilated to maintain receptive field
            self._conv_block(128, feat_channels),
            self._conv_block(feat_channels, feat_channels, dilation=2),
            self._conv_block(feat_channels, feat_channels, dilation=2),
        )
        self._init_weights()

    @staticmethod
    def _conv_block(
        in_ch: int, out_ch: int, dilation: int = 1
    ) -> nn.Sequential:
        padding = dilation
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


class R2D2LambdaLayer(nn.Module):
    """Learnable ridge-regression regularisation parameter λ > 0."""

    def __init__(self, init_lambda: float = 1.0) -> None:
        super().__init__()
        self.log_lambda = nn.Parameter(
            torch.tensor(init_lambda).log()
        )

    @property
    def lam(self) -> Tensor:
        return self.log_lambda.exp()


class R2D2AdjustLayer(nn.Module):
    """Learnable per-class scale and bias applied to the ridge-regression logits before the loss."""

    def __init__(self, n_classes: int = 2, init_scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((n_classes,), init_scale))
        self.bias  = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits: Tensor) -> Tensor:
        """Apply per-class scale + bias."""
        return logits * self.scale + self.bias


def _rr_woodbury(Z: Tensor, Y: Tensor, lam: Tensor) -> Tensor:
    """Woodbury (dual) form of ridge regression."""
    n = Z.shape[0]
    # Z Zᵀ + λ I  →  (n, n)
    A = Z @ Z.T + lam * torch.eye(n, device=Z.device, dtype=Z.dtype)
    # Solve A x = Y  →  x = (Z Zᵀ + λ I)⁻¹ Y  →  shape (n, n_classes)
    x = torch.linalg.solve(A, Y)
    # W* = Zᵀ x  →  (d+1, n_classes)
    return Z.T @ x


def _rr_standard(Z: Tensor, Y: Tensor, lam: Tensor) -> Tensor:
    """Standard (primal) form of ridge regression."""
    d_plus1 = Z.shape[1]
    A = Z.T @ Z + lam * torch.eye(d_plus1, device=Z.device, dtype=Z.dtype)
    return torch.linalg.solve(A, Z.T @ Y)


def solve_ridge(Z: Tensor, Y: Tensor, lam: Tensor) -> Tensor:
    """Auto-select Woodbury vs. standard form based on matrix dimensions."""
    n, d_plus1 = Z.shape
    if n <= d_plus1:
        # Woodbury: cheaper when n small (typical in few-shot)
        return _rr_woodbury(Z, Y, lam)
    else:
        return _rr_standard(Z, Y, lam)


def _sample_pixels(
    fts:     Tensor,   # (C, H', W')
    mask:    Tensor,   # (H, W)  — binary {0,1}
    max_px:  int,
) -> Tensor:
    """Collect feature vectors at mask-positive pixel positions."""
    C = fts.shape[0]
    H, W = mask.shape

    # Upsample feature map to mask resolution
    fts_up = F.interpolate(
        fts.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(0)                            # (C, H, W)

    pos = mask.bool().flatten()             # (H*W,)
    if not pos.any():
        return fts_up.new_empty(0, C)

    # fts_up: (C, H*W) → transpose → (H*W, C)
    all_vecs = fts_up.reshape(C, -1).T      # (H*W, C)
    vecs     = all_vecs[pos]                # (n_pos, C)

    if vecs.shape[0] > max_px:
        idx  = torch.randperm(vecs.shape[0], device=vecs.device)[:max_px]
        vecs = vecs[idx]

    return vecs


class R2D2Seg(nn.Module):
    """R2D2 few-shot segmentation model for binary medical image segmentation."""

    def __init__(
        self,
        in_channels:    int   = 1,
        feat_channels:  int   = 256,
        init_lambda:    float = 1.0,
        learn_lambda:   bool  = True,
        init_scale:     float = 1.0,
        max_support_px: int   = 512,
    ) -> None:
        super().__init__()
        self.feat_channels  = feat_channels
        self.max_support_px = max_support_px

        self.encoder     = R2D2Encoder(in_channels, feat_channels)
        self.lambda_layer = R2D2LambdaLayer(init_lambda)
        self.adjust      = R2D2AdjustLayer(n_classes=2, init_scale=init_scale)

        if not learn_lambda:
            for p in self.lambda_layer.parameters():
                p.requires_grad_(False)


    def _collect_support_pixels(
        self,
        fts_s:   Tensor,   # (C, H', W')
        fg_mask: Tensor,   # (H, W)
        bg_mask: Tensor,   # (H, W)
    ) -> tuple[Tensor, Tensor]:
        """Sample fg and bg pixel features from one support image."""
        fg_vecs = _sample_pixels(fts_s, fg_mask, self.max_support_px)
        bg_vecs = _sample_pixels(fts_s, bg_mask, self.max_support_px)
        return fg_vecs, bg_vecs

    def _build_regression_system(
        self,
        fg_vecs_list: list[Tensor],   # one entry per support image
        bg_vecs_list: list[Tensor],
    ) -> tuple[Tensor, Tensor] | None:
        """Stack all labelled pixel features into (Z, Y) for ridge regression."""
        all_fg = [v for v in fg_vecs_list if v.shape[0] > 0]
        all_bg = [v for v in bg_vecs_list if v.shape[0] > 0]

        if not all_fg or not all_bg:
            return None

        fg_cat = torch.cat(all_fg, dim=0)   # (n_fg_total, C)
        bg_cat = torch.cat(all_bg, dim=0)   # (n_bg_total, C)

        # Build feature matrix with bias column
        ones_fg = fg_cat.new_ones(fg_cat.shape[0], 1)
        ones_bg = bg_cat.new_ones(bg_cat.shape[0], 1)
        Z_fg = torch.cat([fg_cat, ones_fg], dim=1)   # (n_fg, C+1)
        Z_bg = torch.cat([bg_cat, ones_bg], dim=1)   # (n_bg, C+1)
        Z    = torch.cat([Z_fg, Z_bg], dim=0)         # (n,   C+1)

        # One-hot labels: bg=0, fg=1
        n_fg, n_bg = fg_cat.shape[0], bg_cat.shape[0]
        Y_fg = torch.zeros(n_fg, 2, device=Z.device, dtype=Z.dtype)
        Y_fg[:, 1] = 1.0
        Y_bg = torch.zeros(n_bg, 2, device=Z.device, dtype=Z.dtype)
        Y_bg[:, 0] = 1.0
        Y = torch.cat([Y_fg, Y_bg], dim=0)            # (n, 2)

        return Z, Y

    def _classify_query(
        self,
        fts_q:   Tensor,    # (C, H', W')   — feature map of one query image
        W:       Tensor,    # (C+1, 2)      — ridge regression weight matrix
        img_size: tuple[int, int],
    ) -> Tensor:
        """Apply W* to every pixel of the query feature map."""
        C, Hp, Wp = fts_q.shape
        # Flatten spatial dims: (C, H'*W') → (H'*W', C)
        pix = fts_q.reshape(C, -1).T          # (H'*W', C)
        # Append bias column
        ones = pix.new_ones(pix.shape[0], 1)
        pix_aug = torch.cat([pix, ones], dim=1)   # (H'*W', C+1)
        # W: (C+1, 2)  →  logits: (H'*W', 2)
        logits = pix_aug @ W                       # (H'*W', 2)
        # Apply AdjustLayer (per-class scale + bias)
        logits = self.adjust(logits)               # (H'*W', 2)
        # Reshape to (1, 2, H', W')
        logits = logits.T.reshape(1, 2, Hp, Wp)
        # Upsample to input resolution
        return F.interpolate(
            logits, size=img_size, mode="bilinear", align_corners=False
        )


    def forward(
        self,
        supp_img:  Tensor,      # (B, 1, H, W)
        fg_mask:   Tensor,      # (B, 1, H, W) — fg pixels of sparse annotation
        bg_mask:   Tensor,      # (B, 1, H, W) — bg pixels of sparse annotation
        qry_img:   Tensor,      # (B, 1, H, W)
    ) -> tuple[Tensor, Tensor]:
        """Run R2D2 forward pass."""
        B        = supp_img.shape[0]
        img_size = supp_img.shape[-2:]
        lam      = self.lambda_layer.lam

        # Encode support + query together (single forward pass)
        imgs_all = torch.cat([supp_img, qry_img], dim=0)   # (2B, 1, H, W)
        fts_all  = self.encoder(imgs_all)                   # (2B, C, H', W')
        fts_sup  = fts_all[:B]                              # (B, C, H', W')
        fts_qry  = fts_all[B:]                              # (B, C, H', W')

        outputs = []
        for epi in range(B):
            fg_epi = fg_mask[epi, 0]    # (H, W)
            bg_epi = bg_mask[epi, 0]    # (H, W)
            fs     = fts_sup[epi]       # (C, H', W')
            fq     = fts_qry[epi]       # (C, H', W')

            # Collect labelled pixel feature vectors
            fg_vecs, bg_vecs = self._collect_support_pixels(fs, fg_epi, bg_epi)

            # Build and solve the ridge regression system
            system = self._build_regression_system([fg_vecs], [bg_vecs])
            if system is None:
                # Degenerate episode (no labelled pixels): return uniform logits
                dummy = fq.new_zeros(1, 2, *img_size)
                outputs.append(dummy)
                continue

            Z, Y = system
            W    = solve_ridge(Z, Y, lam)      # (C+1, 2)

            # Classify all query pixels
            pred = self._classify_query(fq, W, img_size)    # (1, 2, H, W)
            outputs.append(pred)

        pred_batch = torch.cat(outputs, dim=0)              # (B, 2, H, W)
        aux_loss   = pred_batch.new_tensor(0.0)
        return pred_batch, aux_loss


    def forward_kshot(
        self,
        sup_imgs_t: list[Tensor],      # K × (1, 1, H, W)
        fg_masks_t: list[Tensor],      # K × (1, 1, H, W)
        bg_masks_t: list[Tensor],      # K × (1, 1, H, W)
        qry_img:    Tensor,            # (1, 1, H, W)
    ) -> Tensor:
        """K-shot R2D2 forward pass (K >= 1)."""
        img_size = qry_img.shape[-2:]
        lam      = self.lambda_layer.lam

        fg_vecs_list, bg_vecs_list = [], []
        for si, fg, bg in zip(sup_imgs_t, fg_masks_t, bg_masks_t):
            # si / fg / bg may be (1,1,H,W) tensors from a list, or (1,H,W)
            # slices from iterating a stacked batch — normalise to 4D.
            if si.dim() == 3:
                si = si.unsqueeze(0)   # (1,H,W) → (1,1,H,W)
            if fg.dim() == 3:
                fg = fg.unsqueeze(0)
            if bg.dim() == 3:
                bg = bg.unsqueeze(0)
            fs = self.encoder(si)[0]   # (C, H', W')
            fg_epi = fg[0, 0]          # (H, W)
            bg_epi = bg[0, 0]          # (H, W)
            fv, bv = self._collect_support_pixels(fs, fg_epi, bg_epi)
            fg_vecs_list.append(fv)
            bg_vecs_list.append(bv)

        system = self._build_regression_system(fg_vecs_list, bg_vecs_list)
        if system is None:
            return qry_img.new_zeros(1, 2, *img_size), 0.0

        Z, Y  = system
        W     = solve_ridge(Z, Y, lam)

        fq    = self.encoder(qry_img)[0]    # (C, H', W')
        return self._classify_query(fq, W, img_size), 0.0
