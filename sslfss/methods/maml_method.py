import torch
import numpy as np
from dataclasses import dataclass
from torch.optim.lr_scheduler import CosineAnnealingLR

from sslfss.methods.base import FewShotMethod
from sslfss.data.dataset import _zscore
from sslfss.losses import loss_combinada
from sslfss.metrics import dice_score, miou_score


@dataclass
class MAMLConfig:
    inner_lr:        float = 0.05
    meta_lr:         float = 1e-3
    inner_steps:     int   = 5
    first_order:     bool  = True
    meta_batch_size: int   = 4
    use_meta_sgd:    bool  = True


class MAMLMethod(FewShotMethod):
    def __init__(self, method_cfg: MAMLConfig = None):
        self.maml        = None
        self.cfg         = None
        self.method_cfg  = method_cfg or MAMLConfig()
        self.device      = None
        self.optimizer   = None
        self.scheduler   = None

    def name(self) -> str:
        return "maml"

    def run_name(self) -> str:
        return "sslfss_maml"

    def supports_steps_study(self) -> bool:
        return True


    def _build_maml(self, cfg, device):
        import learn2learn as l2l
        from sslfss.models.segmentation import UNetSegmenter
        mc = self.method_cfg
        base = UNetSegmenter(in_channels=cfg.in_channels, out_channels=cfg.out_channels).to(device)
        if mc.use_meta_sgd:
            return l2l.algorithms.MetaSGD(base, lr=mc.inner_lr, first_order=mc.first_order)
        return l2l.algorithms.MAML(base, lr=mc.inner_lr, first_order=mc.first_order)

    def build_model(self, cfg, device, n_steps: int) -> None:
        self.cfg    = cfg
        self.device = device
        self.maml   = self._build_maml(cfg, device)
        mc = self.method_cfg
        self.optimizer = torch.optim.Adam(self.maml.parameters(), lr=mc.meta_lr)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=n_steps, eta_min=mc.meta_lr * 0.01
        )

    def optimizer_step(self, batch: tuple, device) -> float:
        cfg = self.cfg
        mc  = self.method_cfg
        sup_img, sup_msk, q_img, q_msk = batch
        sup_img = sup_img.to(device)
        sup_msk = sup_msk.to(device)
        q_img   = q_img.to(device)
        q_msk   = q_msk.to(device)

        k = sup_img.shape[0]
        self.maml.train()
        self.optimizer.zero_grad()
        meta_loss_sum = 0.0
        for i in range(k):
            si, sm, qi, qm = sup_img[i:i+1], sup_msk[i:i+1], q_img[i:i+1], q_msk[i:i+1]
            task_model = self.maml.clone()
            for _ in range(mc.inner_steps):
                task_model.adapt(loss_combinada(
                    task_model(si), sm,
                    bce_weight=cfg.loss_bce_weight,
                    dice_weight=cfg.loss_dice_weight,
                ))
            query_loss = loss_combinada(
                task_model(qi), qm,
                bce_weight=cfg.loss_bce_weight,
                dice_weight=cfg.loss_dice_weight,
            )
            (query_loss / k).backward()
            meta_loss_sum += query_loss.item()

        torch.nn.utils.clip_grad_norm_(self.maml.parameters(), max_norm=cfg.grad_clip)
        self.optimizer.step()
        return meta_loss_sum / k

    def scheduler_step(self) -> float:
        self.scheduler.step()
        return self.scheduler.get_last_lr()[0]

    def val_episode(self, episode: tuple, val_source: list, device) -> tuple[float, float, float]:
        cfg = self.cfg
        mc  = self.method_cfg
        ds_idx, ds_name, qry_idx, sup_idxs, sup_sparses, gt_dense_np = episode
        imgs, *_ = val_source[ds_idx]

        sup_imgs_t = [
            torch.tensor(_zscore(imgs[s]), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            for s in sup_idxs
        ]
        sup_msks_t = [
            torch.tensor(sp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            for sp in sup_sparses
        ]
        qv_img = torch.tensor(_zscore(imgs[qry_idx]), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        qv_msk = torch.tensor(gt_dense_np,            dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        k = len(sup_imgs_t)
        val_model = self.maml.clone()
        val_model.train()
        for _ in range(mc.inner_steps):
            val_model.adapt(
                sum(
                    loss_combinada(val_model(si), sm,
                                   bce_weight=cfg.loss_bce_weight,
                                   dice_weight=cfg.loss_dice_weight)
                    for si, sm in zip(sup_imgs_t, sup_msks_t)
                ) / k
            )

        val_model.eval()
        with torch.no_grad():
            pred_v = val_model(qv_img)
            loss = loss_combinada(pred_v, qv_msk, bce_weight=cfg.loss_bce_weight, dice_weight=cfg.loss_dice_weight).item()
            pred_np = (torch.sigmoid(pred_v) > 0.5).float().squeeze().cpu().numpy()

        return loss, dice_score(pred_np, gt_dense_np), miou_score(pred_np, gt_dense_np)

    def checkpoint_state(self) -> dict:
        return self.maml.state_dict()

    def restore_checkpoint(self, path, device) -> None:
        self.maml.load_state_dict(torch.load(path, map_location=device))

    def batch_size(self) -> int:
        return self.method_cfg.meta_batch_size


    def load_model(self, cfg) -> None:
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cfg.checkpoint_path.exists():
            ckpt_path = cfg.checkpoint_path
        elif cfg.last_checkpoint_path.exists():
            ckpt_path = cfg.last_checkpoint_path
            print("[WARN] Best checkpoint not found — using last.")
        else:
            raise FileNotFoundError("No MAML checkpoint found. Run train.py first.")

        self.maml = self._build_maml(cfg, self.device)
        self.maml.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.maml.eval()
        print(f"Loaded MAML: {ckpt_path}")

    def predict(
        self,
        query_img: np.ndarray,
        support_imgs: list[np.ndarray],
        support_sparse_masks: list[np.ndarray],
    ) -> np.ndarray:
        return self.predict_with_steps(
            query_img, support_imgs, support_sparse_masks,
            n_steps=self.method_cfg.inner_steps,
        )

    def predict_with_steps(
        self,
        query_img: np.ndarray,
        support_imgs: list[np.ndarray],
        support_sparse_masks: list[np.ndarray],
        n_steps: int,
    ) -> np.ndarray:
        import gc
        cfg    = self.cfg
        device = self.device

        sup_imgs_t = [
            torch.tensor(_zscore(img), dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0).to(device)
            for img in support_imgs
        ]
        sup_msks_t = [
            torch.tensor(msk, dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0).to(device)
            for msk in support_sparse_masks
        ]
        q_t = (
            torch.tensor(_zscore(query_img), dtype=torch.float32)
            .unsqueeze(0).unsqueeze(0).to(device)
        )

        k = len(sup_imgs_t)
        # first_order is already set at construction time (MAMLConfig.first_order=True)
        m = self.maml.clone()
        for _ in range(n_steps):
            loss = sum(
                loss_combinada(
                    m(si), sm,
                    bce_weight=self.cfg.loss_bce_weight,
                    dice_weight=self.cfg.loss_dice_weight,
                )
                for si, sm in zip(sup_imgs_t, sup_msks_t)
            ) / k
            m.adapt(loss)

        with torch.no_grad():
            pred = (torch.sigmoid(m(q_t)) > 0.5).float().squeeze().cpu().numpy()

        # Explicitly free GPU tensors and cached memory
        del m, sup_imgs_t, sup_msks_t, q_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return pred.astype(np.float32)
