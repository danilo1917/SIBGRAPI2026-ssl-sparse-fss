import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from torch.optim.lr_scheduler import CosineAnnealingLR

from sslfss.methods.base import FewShotMethod
from sslfss.data.dataset import _zscore
from sslfss.losses import loss_combinada
from sslfss.metrics import dice_score, miou_score


@dataclass
class PANetConfig:
    feat_channels:      int   = 256
    cosine_scaler:      float = 20.0
    use_align_loss:     bool  = False
    align_loss_scaler:  float = 1.0
    lr:                 float = 1e-3
    batch_size:         int   = 4


class PANetMethod(FewShotMethod):
    def __init__(self, method_cfg: PANetConfig = None):
        self.model      = None
        self.cfg        = None
        self.method_cfg = method_cfg or PANetConfig()
        self.device     = None
        self.optimizer  = None
        self.scheduler  = None

    def name(self) -> str:
        return "panet"

    def run_name(self) -> str:
        return "sslfss_panet"


    def build_model(self, cfg, device, n_steps: int) -> None:
        from sslfss.models.panet import PANetSeg
        self.cfg    = cfg
        self.device = device
        mc = self.method_cfg
        self.model  = PANetSeg(
            in_channels=cfg.in_channels,
            feat_channels=mc.feat_channels,
            cosine_scaler=mc.cosine_scaler,
            use_align_loss=mc.use_align_loss,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=mc.lr)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=n_steps, eta_min=mc.lr * 0.01
        )

    def optimizer_step(self, batch: tuple, device) -> float:
        cfg = self.cfg
        sup_img, sup_msk, q_img, q_msk = batch
        sup_img = sup_img.to(device)
        sup_msk = sup_msk.to(device)
        q_img   = q_img.to(device)
        q_msk   = q_msk.to(device)

        fg_mask  = (sup_msk == 1).float()
        bg_mask  = (sup_msk == 0).float()
        align_fg = (q_msk == 1).float()
        align_bg = (q_msk == 0).float()

        self.model.train()
        self.optimizer.zero_grad()
        pred, align_loss = self.model(sup_img, fg_mask, bg_mask, q_img, align_fg, align_bg)
        binary_logit = pred[:, 1:] - pred[:, :1]
        loss = (
            loss_combinada(binary_logit, q_msk, bce_weight=cfg.loss_bce_weight, dice_weight=cfg.loss_dice_weight)
            + self.method_cfg.align_loss_scaler * align_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip)
        self.optimizer.step()
        return loss.item()

    def scheduler_step(self) -> float:
        self.scheduler.step()
        return self.scheduler.get_last_lr()[0]

    def val_episode(self, episode: tuple, val_source: list, device) -> tuple[float, float, float]:
        cfg = self.cfg
        ds_idx, ds_name, qry_idx, sup_idxs, sup_sparses, gt_dense_np = episode
        imgs, *_ = val_source[ds_idx]

        sup_imgs_t = [
            torch.tensor(_zscore(imgs[s]), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            for s in sup_idxs
        ]
        fg_t = [
            torch.tensor((sp == 1).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            for sp in sup_sparses
        ]
        bg_t = [
            torch.tensor((sp == 0).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            for sp in sup_sparses
        ]
        qry_t  = torch.tensor(_zscore(imgs[qry_idx]), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        gt_t   = torch.tensor(gt_dense_np,            dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        k = len(sup_imgs_t)
        self.model.eval()
        with torch.no_grad():
            if k == 1:
                pred, align_loss = self.model(sup_imgs_t[0], fg_t[0], bg_t[0], qry_t)
            else:
                img_size = qry_t.shape[-2:]
                fg_protos, bg_protos = [], []
                for si, fg, bg in zip(sup_imgs_t, fg_t, bg_t):
                    fts = self.model.encoder(si)
                    fg_protos.append(self.model.getFeatures(fts, fg))
                    bg_protos.append(self.model.getFeatures(fts, bg))
                fg_proto = sum(fg_protos) / k
                bg_proto = sum(bg_protos) / k
                fts_q = self.model.encoder(qry_t)
                dist = torch.stack(
                    [self.model.calDist(fts_q, bg_proto), self.model.calDist(fts_q, fg_proto)],
                    dim=1,
                )
                pred = F.interpolate(dist, size=img_size, mode="bilinear", align_corners=False)
                align_loss = torch.tensor(0.0, device=device)
            binary_logit = pred[:, 1:] - pred[:, :1]
            loss = (
                loss_combinada(binary_logit, gt_t, bce_weight=cfg.loss_bce_weight, dice_weight=cfg.loss_dice_weight)
                + self.method_cfg.align_loss_scaler * align_loss
            ).item()
            pred_np = pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.float32)

        return loss, dice_score(pred_np, gt_dense_np), miou_score(pred_np, gt_dense_np)

    def checkpoint_state(self) -> dict:
        return self.model.state_dict()

    def restore_checkpoint(self, path, device) -> None:
        self.model.load_state_dict(torch.load(path, map_location=device))

    def batch_size(self) -> int:
        return self.method_cfg.batch_size


    def load_model(self, cfg) -> None:
        from sslfss.models.panet import PANetSeg
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg.run_name = "sslfss_panet"
        if cfg.checkpoint_path.exists():
            ckpt_path = cfg.checkpoint_path
        elif cfg.last_checkpoint_path.exists():
            ckpt_path = cfg.last_checkpoint_path
            print("[WARN] Best checkpoint not found — using last.")
        else:
            raise FileNotFoundError("No PANet checkpoint found. Run train_panet.py first.")

        mc = self.method_cfg
        self.model = PANetSeg(
            in_channels=cfg.in_channels,
            feat_channels=mc.feat_channels,
            cosine_scaler=mc.cosine_scaler,
            use_align_loss=False,
        ).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded PANet: {ckpt_path}")

    def predict(
        self,
        query_img: np.ndarray,
        support_imgs: list[np.ndarray],
        support_sparse_masks: list[np.ndarray],
    ) -> np.ndarray:
        device = self.device
        model  = self.model

        sup_imgs_t, fg_t, bg_t = [], [], []
        for img, sparse in zip(support_imgs, support_sparse_masks):
            sup_imgs_t.append(
                torch.tensor(_zscore(img), dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0).to(device)
            )
            fg_t.append(
                torch.tensor((sparse == 1).astype(np.float32), dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0).to(device)
            )
            bg_t.append(
                torch.tensor((sparse == 0).astype(np.float32), dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0).to(device)
            )

        qry_t = (
            torch.tensor(_zscore(query_img), dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0).to(device)
        )

        model.eval()
        with torch.no_grad():
            if len(sup_imgs_t) == 1:
                pred, _ = model(sup_imgs_t[0], fg_t[0], bg_t[0], qry_t)
            else:
                img_size   = qry_t.shape[-2:]
                fg_protos, bg_protos = [], []
                for si, fg, bg in zip(sup_imgs_t, fg_t, bg_t):
                    fts = model.encoder(si)
                    fg_protos.append(model.getFeatures(fts, fg))
                    bg_protos.append(model.getFeatures(fts, bg))
                fg_proto = sum(fg_protos) / len(fg_protos)
                bg_proto = sum(bg_protos) / len(bg_protos)
                fts_q = model.encoder(qry_t)
                dist  = torch.stack(
                    [model.calDist(fts_q, bg_proto), model.calDist(fts_q, fg_proto)],
                    dim=1,
                )
                pred = F.interpolate(dist, size=img_size, mode="bilinear", align_corners=False)

        return pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.float32)
