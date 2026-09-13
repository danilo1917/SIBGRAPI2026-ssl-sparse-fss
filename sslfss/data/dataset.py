import numpy as np
import torch
from torch.utils.data import Dataset

from sslfss.data.sparsify import aplicar_esparsidade


def _zscore(img: np.ndarray) -> np.ndarray:
    """Z-score normalise a slice before model input."""
    mu, sigma = img.mean(), img.std()
    if sigma < 1e-6:
        return img - mu
    return (img - mu) / sigma


def _spatial_params() -> dict:
    """Sample a random spatial augmentation configuration."""
    return {
        "fliplr": np.random.random() > 0.5,
        "flipud": np.random.random() > 0.5,
        "rot_k":  np.random.randint(4),
    }


def _apply_spatial(
    img: np.ndarray,
    msk: np.ndarray,
    params: dict,
    extra: np.ndarray | None = None,
) -> tuple:
    """Apply a previously-sampled spatial transform to an img/mask pair."""
    if params["fliplr"]:
        img  = np.fliplr(img).copy()
        msk  = np.fliplr(msk).copy()
        if extra is not None:
            extra = np.fliplr(extra).copy()
    if params["flipud"]:
        img  = np.flipud(img).copy()
        msk  = np.flipud(msk).copy()
        if extra is not None:
            extra = np.flipud(extra).copy()
    k = params["rot_k"]
    if k > 0:
        img  = np.rot90(img, k).copy()
        msk  = np.rot90(msk, k).copy()
        if extra is not None:
            extra = np.rot90(extra, k).copy()
    if extra is not None:
        return img, msk, extra
    return img, msk


def _para_tensor(img: np.ndarray, msk: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (H,W) numpy arrays to (1,H,W) float tensors."""
    img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    msk_t = torch.tensor(msk, dtype=torch.float32).unsqueeze(0)
    return img_t, msk_t


class MetaDatasetMulti(Dataset):
    """Multi-dataset episodic sampler for self-supervised MAML (same-image protocol)."""

    def __init__(
        self,
        datasets: list[tuple],
        pseudo_labels: dict | None = None,
        dense_label_mode: str = "pseudo",
        names: list[str] | None = None,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("datasets list must not be empty")
        if dense_label_mode not in ("pseudo", "real"):
            raise ValueError(f"dense_label_mode must be 'pseudo' or 'real', got {dense_label_mode!r}")
        if dense_label_mode == "pseudo" and (names is None or len(names) != len(datasets)):
            raise ValueError("pseudo mode requires one dataset name per entry in `datasets`")

        self.datasets         = datasets
        self.names            = list(names) if names is not None else None
        self.pseudo_labels    = pseudo_labels or {}
        self.dense_label_mode = dense_label_mode

        ds_sizes = [len(self._indices(i)) for i in range(len(datasets))]
        sqrt_sizes = np.array([np.sqrt(s) for s in ds_sizes], dtype=np.float64)
        self._ds_weights = sqrt_sizes / sqrt_sizes.sum()
        self._length = sum(ds_sizes)

        # No vol_map needed — same-image protocol uses a single slice per episode.

    def _indices(self, ds_idx: int) -> np.ndarray:
        return self.datasets[ds_idx][2]

    def _get_label(self, ds_idx: int, imgs: np.ndarray, msks: np.ndarray, idx: int) -> np.ndarray:
        if self.dense_label_mode == "real":
            return msks[idx].astype(np.float32)
        # pseudo mode — real GT must never be used as training signal
        ds_cache = self.pseudo_labels.get(self.names[ds_idx], {})
        if idx in ds_cache:
            return ds_cache[idx]
        raise RuntimeError(
            f"Slice {idx} of dataset '{self.names[ds_idx]}' not found in pseudo-label cache "
            "during pseudo-mode training. "
            "This means _filter_valid_indices() was not called or failed. "
            "Real GT masks are never used as a fallback in pseudo mode."
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, _) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        # 1. Pick a dataset (square-root balancing).
        ds_idx = int(np.random.choice(len(self.datasets), p=self._ds_weights))
        imgs, msks_reais, indices, _ = self.datasets[ds_idx]

        # 2. Pick a single slice — same slice for both support and query.
        idx = int(np.random.choice(indices))

        # 3. Fetch dense label (guaranteed non-empty by _filter_valid_indices).
        label = self._get_label(ds_idx, imgs, msks_reais, idx)
        assert label.max() > 0, (
            f"Slice {idx} has an empty label. Ensure _filter_valid_indices() was "
            "applied before constructing MetaDatasetMulti."
        )

        # 4. Sparsify for the inner loop.
        sup_msk_sparse = aplicar_esparsidade(label)

        # 5. Shared spatial augmentation (flip + rot90) applied to both views.
        params = _spatial_params()
        img_aug, sup_msk_aug = _apply_spatial(imgs[idx].copy(), sup_msk_sparse, params)
        _,       dense_aug   = _apply_spatial(imgs[idx].copy(), label.copy(), params)

        # 6. Support image alias (shared augmentation above is sufficient).
        img_sup_aug = img_aug

        sup_img_t = torch.tensor(_zscore(img_sup_aug), dtype=torch.float32).unsqueeze(0)
        qry_img_t = torch.tensor(_zscore(img_aug),     dtype=torch.float32).unsqueeze(0)
        sup_t     = torch.tensor(sup_msk_aug,          dtype=torch.float32).unsqueeze(0)
        q_msk_t   = torch.tensor(dense_aug,            dtype=torch.float32).unsqueeze(0)

        return sup_img_t, sup_t, qry_img_t, q_msk_t