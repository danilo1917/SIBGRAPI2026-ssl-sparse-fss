from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sslfss.methods.base import FewShotMethod
from sslfss.data.dataset import _zscore
from sslfss.metrics import dice_score, miou_score


@dataclass
class ALPNetConfig:
    use_coco_init:   bool  = True
    proto_grid_size: int   = 8
    feature_hw:      list  = field(default_factory=lambda: [32, 32])
    lr:              float = 1e-3
    momentum:        float = 0.9
    weight_decay:    float = 5e-4
    lr_step_every:   int   = 1000 
    lr_step_gamma:   float = 0.95
    val_wsize:       int   = 2
    align:           bool  = True


class ALPNetMethod(FewShotMethod):
    def __init__(self, method_cfg: ALPNetConfig | None = None):
        self.model      = None
        self.cfg        = None
        self.method_cfg = method_cfg or ALPNetConfig()
        self.device     = None
        self.optimizer  = None
        self.scheduler  = None
        self.criterion  = None
        self._step_count = 0

    def name(self) -> str:
        return "alpnet"

    def run_name(self) -> str:
        return "sslfss_alpnet"


    def build_model(self, cfg, device, n_steps: int) -> None:
        from sslfss.models.alpnet import FewShotSeg

        self.cfg    = cfg
        self.device = device
        mc = self.method_cfg

        model_cfg = {
            'align':           mc.align,
            'use_coco_init':   mc.use_coco_init,
            'which_model':     'dlfcn_res101',
            'cls_name':        'grid_proto',
            'proto_grid_size': mc.proto_grid_size,
            'feature_hw':      list(mc.feature_hw),
        }
        self.model = FewShotSeg(in_channels=3, cfg=model_cfg).to(device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=mc.lr,
            momentum=mc.momentum,
            weight_decay=mc.weight_decay,
        )

        # Milestones: every lr_step_every iterations up to n_steps
        milestones = list(range(mc.lr_step_every, n_steps, mc.lr_step_every))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=mc.lr_step_gamma,
        )

        # CrossEntropyLoss with WCE weights [bg=0.05, fg=1.0], ignore augmented borders
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=255,
            weight=torch.FloatTensor([0.05, 1.0]).to(device),
        )

        self._step_count = 0

    def optimizer_step(self, batch: dict, device) -> float:
        """batch: dict from ALPNetDataset (after DataLoader collation, batch_size=1) Keys: sup_img   : Tensor(B, 3, H, W)  float32 sup_fg    : Tensor(B, H, W)      float32 sup_bg    : Tensor(B, H, W)      float32 qry_img   : Tensor(B, 3, H, W)  float32 qry_label : Tensor(B, H, W)     int64"""
        self.model.train()
        self.optimizer.zero_grad()

        sup_img   = batch['sup_img'].to(device)    # (B, 3, H, W)
        sup_fg    = batch['sup_fg'].to(device)     # (B, H, W)
        sup_bg    = batch['sup_bg'].to(device)     # (B, H, W)
        qry_img   = batch['qry_img'].to(device)    # (B, 3, H, W)
        qry_label = batch['qry_label'].to(device)  # (B, H, W) long

        # Reconstruct nested-list format expected by FewShotSeg.forward
        # way=1, shot=1, each element is (B, H, W)
        supp_imgs = [[sup_img]]
        fore_mask = [[sup_fg]]
        back_mask = [[sup_bg]]

        query_pred, align_loss, _, _ = self.model(
            supp_imgs, fore_mask, back_mask, [qry_img],
            isval=False, val_wsize=None,
        )
        # query_pred: (B, 2, H, W), qry_label: (B, H, W)
        query_loss = self.criterion(query_pred, qry_label)
        loss = query_loss + align_loss

        loss.backward()
        self.optimizer.step()

        self._step_count += 1
        return loss.item()

    def scheduler_step(self) -> float:
        """Called once per optimizer step. MultiStepLR steps per iteration."""
        self.scheduler.step()
        return self.scheduler.get_last_lr()[0]


    def val_episode(self, episode: tuple, val_source: list, device) -> tuple[float, float, float]:
        mc = self.method_cfg
        ds_idx, ds_name, qry_idx, sup_idxs, sup_sparses, gt_dense_np = episode
        imgs, *_ = val_source[ds_idx]

        supp_img_list = []
        fg_list       = []
        bg_list       = []
        for s_idx, s_sparse in zip(sup_idxs, sup_sparses):
            s_img = _tile3_np(_zscore(imgs[s_idx])).unsqueeze(0).to(device)   # (1,3,H,W)
            fg = torch.tensor((s_sparse == 1).astype(np.float32)).unsqueeze(0).to(device)
            bg = torch.tensor((s_sparse != 1).astype(np.float32)).unsqueeze(0).to(device)
            supp_img_list.append(s_img)
            fg_list.append(fg)
            bg_list.append(bg)
        supp_imgs = [supp_img_list]   # shape: [1][K]
        fore_mask = [fg_list]
        back_mask = [bg_list]

        qry_img_t    = _tile3_np(_zscore(imgs[qry_idx])).unsqueeze(0).to(device)
        gt_label_np  = (gt_dense_np == 1).astype(np.int64)       # {0,1}
        qry_label_t  = torch.from_numpy(gt_label_np).unsqueeze(0).to(device)

        self.model.eval()
        with torch.no_grad():
            query_pred, align_loss, _, _ = self.model(
                supp_imgs, fore_mask, back_mask, [qry_img_t],
                isval=True, val_wsize=mc.val_wsize,
            )
            query_loss = self.criterion(query_pred, qry_label_t)
            loss = (query_loss + align_loss).item()
            pred_np = query_pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.float32)

        return loss, dice_score(pred_np, gt_label_np), miou_score(pred_np, gt_label_np)

    def checkpoint_state(self) -> dict:
        return self.model.state_dict()

    def restore_checkpoint(self, path, device) -> None:
        self.model.load_state_dict(torch.load(path, map_location=device))

    def batch_size(self) -> int:
        return 1  # SSL_ALPNet always uses batch_size=1

    def load_model(self, cfg) -> None:
        from sslfss.models.alpnet import FewShotSeg

        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg.run_name = self.run_name()

        if cfg.checkpoint_path.exists():
            ckpt_path = cfg.checkpoint_path
        elif cfg.last_checkpoint_path.exists():
            ckpt_path = cfg.last_checkpoint_path
            print("[WARN] Best checkpoint not found — using last.")
        else:
            raise FileNotFoundError(
                "No ALPNet checkpoint found. Run 'python scripts/train_alpnet.py' first."
            )

        mc = self.method_cfg
        model_cfg = {
            'align':           False,   # no align loss at eval
            'use_coco_init':   mc.use_coco_init,
            'which_model':     'dlfcn_res101',
            'cls_name':        'grid_proto',
            'proto_grid_size': mc.proto_grid_size,
            'feature_hw':      list(mc.feature_hw),
        }
        self.model = FewShotSeg(in_channels=3, cfg=model_cfg).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded ALPNet: {ckpt_path}")

        # criterion needed for val_episode calls during eval
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=255,
            weight=torch.FloatTensor([0.05, 1.0]).to(self.device),
        )

    def predict(
        self,
        query_img: np.ndarray,
        support_imgs: list[np.ndarray],
        support_sparse_masks: list[np.ndarray],
    ) -> np.ndarray:
        """support_sparse_masks must be dense binary {0,1} real GT masks."""
        device = self.device
        mc = self.method_cfg

        supp_ts, fg_ts, bg_ts = [], [], []
        for s_img, s_mask in zip(support_imgs, support_sparse_masks):
            assert not np.isin(s_mask, [-1]).any(), \
                "ALPNet received a sparse mask. Study scripts must pass dense GT masks for alpnet."
            dense_mask = (s_mask > 0).astype(np.float32)
            supp_ts.append(_tile3_np(_zscore(s_img)).unsqueeze(0).to(device))
            fg_ts.append(torch.tensor(dense_mask).unsqueeze(0).to(device))
            bg_ts.append(torch.tensor(1.0 - dense_mask).unsqueeze(0).to(device))

        qry_t = _tile3_np(_zscore(query_img)).unsqueeze(0).to(device)

        # Pass all K shots at once — FewShotSeg averages prototypes across shots
        # internally (n_shots=K), which is the architecturally correct K-shot path.
        self.model.eval()
        with torch.no_grad():
            pred, _, _, _ = self.model(
                [supp_ts], [fg_ts], [bg_ts], [qry_t],
                isval=True, val_wsize=mc.val_wsize,
            )
        return pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.float32)


def _tile3_np(img: np.ndarray) -> torch.Tensor:
    """Convert (H, W) float32 to (3, H, W) Tensor by tiling the channel."""
    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)  # (1, H, W)
    return t.repeat(3, 1, 1)                                    # (3, H, W)
