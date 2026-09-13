from abc import ABC, abstractmethod

import numpy as np


class FewShotMethod(ABC):
    @abstractmethod
    def load_model(self, cfg) -> None:
        """Load checkpoint, move to device, set eval mode."""
        ...

    @abstractmethod
    def predict(
        self,
        query_img: np.ndarray,
        support_imgs: list[np.ndarray],
        support_sparse_masks: list[np.ndarray],
    ) -> np.ndarray:
        """Return binary prediction mask (H, W) float32."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'maml', 'panet', 'r2d2'."""
        ...

    def supports_steps_study(self) -> bool:
        return False

    def predict_with_steps(
        self,
        query_img: np.ndarray,
        support_imgs: list[np.ndarray],
        support_sparse_masks: list[np.ndarray],
        n_steps: int,
    ) -> np.ndarray:
        raise NotImplementedError(
            f"Method '{self.name()}' does not support the steps study "
            "(not a gradient-based model)."
        )


    def run_name(self) -> str:
        """Checkpoint file stem, e.g. 'sslfss_maml'."""
        raise NotImplementedError

    def build_model(self, cfg, device, n_steps: int) -> None:
        """Construct model, optimizer and scheduler. Called once before training."""
        raise NotImplementedError

    def optimizer_step(self, batch: tuple, device) -> float:
        """Run one training step on a batch. Return scalar loss value."""
        raise NotImplementedError

    def scheduler_step(self) -> float:
        """Advance LR scheduler. Return current LR."""
        raise NotImplementedError

    def val_episode(
        self,
        episode: tuple,
        val_source: list,
        device,
    ) -> tuple[float, float, float]:
        """Run inference on one val episode. Return (loss, dice, miou)."""
        raise NotImplementedError

    def checkpoint_state(self) -> dict:
        """Return state dict to save."""
        raise NotImplementedError

    def restore_checkpoint(self, path, device) -> None:
        """Load state dict from path."""
        raise NotImplementedError

    def batch_size(self, cfg) -> int:
        """Training batch size for this method."""
        raise NotImplementedError

    def extra_log(self) -> str:
        """Optional extra string appended to the per-step log line."""
        return ""
