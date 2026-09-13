# Adapted from PANet (Wang et al., ICCV 2019):
# https://github.com/kaixin96/PANet

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PANetEncoder(nn.Module):
    """Lightweight VGG-style encoder for grayscale medical images."""

    def __init__(self, in_channels: int = 1, feat_channels: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — no pool yet
            self._conv_block(in_channels, 64),
            self._conv_block(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),          # H/2

            # Block 2
            self._conv_block(64, 128),
            self._conv_block(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),          # H/4

            # Block 3 — dilated to maintain receptive field without more pooling
            self._conv_block(128, feat_channels),
            self._conv_block(feat_channels, feat_channels, dilation=2),
            self._conv_block(feat_channels, feat_channels, dilation=2),
        )
        self._init_weights()

    @staticmethod
    def _conv_block(
        in_ch: int, out_ch: int, dilation: int = 1
    ) -> nn.Sequential:
        padding = dilation  # = same padding for 3×3 kernel
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


class PANetSeg(nn.Module):
    """PANet few-shot segmentation model for binary medical image segmentation."""

    def __init__(
        self,
        in_channels:    int   = 1,
        feat_channels:  int   = 256,
        cosine_scaler:  float = 20.0,
        use_align_loss: bool  = True,
    ) -> None:
        super().__init__()
        self.scaler         = cosine_scaler
        self.use_align_loss = use_align_loss
        self.encoder        = PANetEncoder(in_channels, feat_channels)


    def getFeatures(self, fts: Tensor, mask: Tensor) -> Tensor:
        """Masked Average Pooling: pool feature map into a single prototype vector."""
        # Upsample features to match mask spatial size
        fts = F.interpolate(fts, size=mask.shape[-2:], mode="bilinear", align_corners=False)

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (1, 1, H, W)

        # Masked average pooling
        masked = (fts * mask).sum(dim=(2, 3))              # (1, C)
        denom  = mask.sum(dim=(2, 3)) + 1e-5               # (1, 1)
        return masked / denom                               # (1, C)

    def getPrototype(
        self,
        fg_fts: list[list[Tensor]],
        bg_fts: list[list[Tensor]],
    ) -> tuple[list[Tensor], Tensor]:
        """Average over shots to get one fg prototype per way + one bg prototype."""
        n_ways  = len(fg_fts)
        n_shots = len(fg_fts[0])
        fg_prototypes = [sum(fg_fts[w]) / n_shots for w in range(n_ways)]
        bg_prototype  = sum(sum(bg_fts[w]) / n_shots for w in range(n_ways)) / n_ways
        return fg_prototypes, bg_prototype

    def calDist(self, fts: Tensor, prototype: Tensor) -> Tensor:
        """Scaled cosine similarity between every feature vector and a prototype."""
        return F.cosine_similarity(
            fts, prototype[..., None, None], dim=1
        ) * self.scaler


    def alignLoss(
        self,
        qry_fts:   Tensor,         # (N, C, H', W')
        pred:      Tensor,         # (N, 2, H_orig, W_orig) — 2-class logits
        supp_fts:  Tensor,         # (n_ways, n_shots, C, H', W')
        fore_mask: Tensor,         # (n_ways, n_shots, H, W)
        back_mask: Tensor,         # (n_ways, n_shots, H, W)
    ) -> Tensor:
        """Prototype Alignment regularisation loss."""
        n_ways, n_shots = fore_mask.shape[:2]

        pred_mask = pred.argmax(dim=1, keepdim=True)           # (N, 1, H, W)
        # Resize to feature resolution
        pred_mask_fts = F.interpolate(
            pred_mask.float(), size=qry_fts.shape[-2:], mode="nearest"
        )                                                       # (N, 1, H', W')

        binary_masks = [pred_mask_fts == i for i in range(2)]  # bg=0, fg=1
        skip_ways    = []
        pred_stack   = torch.stack(binary_masks, dim=1).float()  # (N, 2, 1, H', W')

        # Query prototypes: (2, C)  — [bg_proto, fg_proto]
        qry_prototypes = torch.sum(
            qry_fts.unsqueeze(1) * pred_stack.squeeze(2).unsqueeze(2), dim=(0, 3, 4)
        )
        norm = pred_stack.squeeze(2).sum(dim=(0, 2, 3), keepdim=False).unsqueeze(-1) + 1e-5
        qry_prototypes = qry_prototypes / norm                  # (2, C)

        # Skip ways where the query prediction is entirely empty (no fg pixels).
        # binary_masks has exactly 2 entries [bg, fg]; fg is always index 1 in
        # the 1-way binary prediction used by SSL-Sparse-FSS.
        for w in range(n_ways):
            if binary_masks[1].sum() == 0:
                skip_ways.append(w)

        loss: Tensor = torch.tensor(0.0, device=qry_fts.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts   = supp_fts[way, [shot]]               # (1, C, H', W')
                supp_dist = [self.calDist(img_fts, p) for p in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)        # (1, 2, H', W')
                supp_pred = F.interpolate(
                    supp_pred, size=fore_mask.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                # Build support label: 1=fg, 0=bg, 255=ignore
                supp_label = torch.full_like(
                    fore_mask[way, shot], 255, dtype=torch.long
                )
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0

                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255
                ) / (n_shots * n_ways)

        return loss


    def forward(
        self,
        supp_img:        Tensor,            # (B, 1, H, W)
        supp_fg_mask:    Tensor,            # (B, 1, H, W) — foreground (pseudo-sparse)
        supp_bg_mask:    Tensor,            # (B, 1, H, W) — background (pseudo-sparse)
        qry_img:         Tensor,            # (B, 1, H, W)
        align_fg_mask:   Tensor | None = None,  # (B, 1, H, W) — dense fg for alignLoss
        align_bg_mask:   Tensor | None = None,  # (B, 1, H, W) — dense bg for alignLoss
    ) -> tuple[Tensor, Tensor]:
        """Run PANet forward pass."""
        B = supp_img.shape[0]
        img_size = supp_img.shape[-2:]

        imgs_concat = torch.cat([supp_img, qry_img], dim=0)     # (2B, 1, H, W)
        fts_all     = self.encoder(imgs_concat)                  # (2B, C, H', W')
        supp_fts    = fts_all[:B].unsqueeze(0).unsqueeze(0)     # (1, 1, B, C, H', W') → way×shot×B
        qry_fts     = fts_all[B:]                                # (B, C, H', W')

        align_loss = torch.tensor(0.0, device=supp_img.device)
        outputs    = []

        for epi in range(B):
            # 1-way, 1-shot: supp_fts[0, 0, epi] — shape (C, H', W')
            sf = supp_fts[0, 0, [epi]]  # (1, C, H', W')

            # Masked average pooling to build fg / bg prototypes
            fg_proto = self.getFeatures(sf, supp_fg_mask[[epi], 0])  # (1, C)
            bg_proto = self.getFeatures(sf, supp_bg_mask[[epi], 0])  # (1, C)

            # Cosine distance maps for query
            qf    = qry_fts[[epi]]                               # (1, C, H', W')
            dist  = [self.calDist(qf, bg_proto), self.calDist(qf, fg_proto)]
            pred  = torch.stack(dist, dim=1)                     # (1, 2, H', W')
            pred  = F.interpolate(pred, size=img_size, mode="bilinear", align_corners=False)
            outputs.append(pred)

            # Prototype alignment loss (training only)
            if self.use_align_loss and self.training:
                # Use dense masks when provided so alignLoss sees all pixels,
                # not just the sparse fraction — critical for same-image protocol.
                _align_fg = align_fg_mask[[epi], 0:1] if align_fg_mask is not None else supp_fg_mask[[epi], 0:1]
                _align_bg = align_bg_mask[[epi], 0:1] if align_bg_mask is not None else supp_bg_mask[[epi], 0:1]
                align_loss = align_loss + self.alignLoss(
                    qry_fts=qry_fts[[epi]],
                    pred=pred,
                    supp_fts=supp_fts[:, :, epi],          # (1, 1, C, H', W')
                    fore_mask=_align_fg,                    # (1, 1, H, W) → way×shot
                    back_mask=_align_bg,
                )

        pred_batch = torch.cat(outputs, dim=0)                   # (B, 2, H, W)
        return pred_batch, align_loss / B
