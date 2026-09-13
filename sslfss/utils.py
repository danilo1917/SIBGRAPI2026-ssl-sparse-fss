import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all RNGs for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    """Return a ``worker_init_fn`` for ``DataLoader(num_workers>0)``."""

    def _worker_init(worker_id: int) -> None:
        np.random.seed(base_seed + worker_id)
        random.seed(base_seed + worker_id)

    return _worker_init


