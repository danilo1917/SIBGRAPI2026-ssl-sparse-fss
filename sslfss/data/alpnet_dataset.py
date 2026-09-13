# Adapted from the superpixel dataset of SSL-ALPNet (Ouyang et al., ECCV 2020):
# https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation
#
# MIT License
#
# Copyright (c) 2020 Cheng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import segmentation

from sslfss.data.alpnet_augutils import sabs_aug, transform_with_label


def _fg_mask2d(img: np.ndarray, thresh: float = 1e-4) -> np.ndarray:
    """Return binary foreground mask: pixels above threshold."""
    return (img > thresh).astype(np.float32)


def _superpix_masking(raw_seg: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Zero out superpixel labels that lie outside the foreground mask."""
    masked = raw_seg.copy()
    masked[fg_mask == 0] = 0
    return masked


def _supcls_pick_binarize(super_map: np.ndarray) -> np.ndarray:
    """Pick a random superpixel label (>=1) and return its binary mask."""
    labels = np.unique(super_map)
    fg_labels = labels[labels >= 1]
    if len(fg_labels) == 0:
        return np.zeros_like(super_map, dtype=np.float32)
    chosen = np.random.randint(len(fg_labels))
    return (super_map == fg_labels[chosen]).astype(np.float32)


def _get_mask_med_img(binary_mask: np.ndarray):
    """Return fg/bg masks as float32 tensors."""
    fg = torch.from_numpy(binary_mask.astype(np.float32))
    bg = torch.from_numpy((1.0 - binary_mask).astype(np.float32))
    return {'fg_mask': fg, 'bg_mask': bg}


def _tile3(img_2d: np.ndarray) -> torch.Tensor:
    """Convert (H,W) float32 to (3,H,W) by repeating the single channel."""
    t = torch.from_numpy(img_2d).unsqueeze(0)  # (1,H,W)
    return t.repeat(3, 1, 1)                    # (3,H,W)


class ALPNetDataset(Dataset):
    """Wraps a subset of SSL-Sparse-FSS's .npy slice array for ALPNet training."""

    def __init__(
        self,
        imgs_np: np.ndarray,
        indices: list[int],
        cache_file: Path | str | None = None,
    ):
        self.imgs    = imgs_np
        self.indices = list(indices)
        self.aug_cfg = {'aug': sabs_aug}
        self._transform = transform_with_label(self.aug_cfg)
        self._null_indices: set[int] = set()
        self._cache_path: Path | None = None  # stored for lazy per-worker re-open
        self._npz        = None               # opened lazily inside each worker
        self._in_memory  = None               # fallback when no cache file

        cache_path = Path(cache_file) if cache_file is not None else None

        if cache_path is not None and cache_path.exists():
            print(f"[ALPNetDataset] Verifying superpixel cache {cache_path} …", flush=True)
            npz = np.load(cache_path, allow_pickle=True)
            null_set    = set(npz['_nulls'].tolist())
            cached_keys = set(npz.files) - {'_nulls'}
            if all(str(i) in cached_keys or i in null_set for i in self.indices):
                self._null_indices = null_set
                self._cache_path   = cache_path   # workers will open lazily
                npz.close()                       # don't keep open in parent
                n_valid = sum(1 for i in self.indices if i not in null_set)
                print(f"[ALPNetDataset] Cache OK — {n_valid} valid, "
                      f"{len(null_set)} blank slices.", flush=True)
                return
            else:
                print("[ALPNetDataset] Cache incomplete — recomputing.", flush=True)
                npz.close()

        print(f"[ALPNetDataset] Pre-computing superpixel maps for "
              f"{len(self.indices)} slices …", flush=True)
        null_list: list[int] = []

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(cache_path, 'w', compression=zipfile.ZIP_STORED) as zf:
                for done, idx in enumerate(self.indices, 1):
                    sp = self._compute_superpix(idx)
                    if sp is None:
                        null_list.append(idx)
                    else:
                        buf = io.BytesIO()
                        np.save(buf, sp)
                        zf.writestr(f"{idx}.npy", buf.getvalue())
                    if done % 500 == 0 or done == len(self.indices):
                        print(f"  {done}/{len(self.indices)}", flush=True)
                buf = io.BytesIO()
                np.save(buf, np.array(null_list, dtype=np.int64))
                zf.writestr("_nulls.npy", buf.getvalue())

            self._null_indices = set(null_list)
            self._cache_path   = cache_path   # workers open lazily
            print(f"[ALPNetDataset] Cache saved to {cache_path} — "
                  f"{len(self.indices) - len(null_list)} valid, "
                  f"{len(null_list)} blank.", flush=True)
        else:
            # No cache — keep in memory (not recommended for large sets)
            self._in_memory = {}
            for done, idx in enumerate(self.indices, 1):
                sp = self._compute_superpix(idx)
                self._in_memory[idx] = sp
                if sp is None:
                    null_list.append(idx)
                if done % 500 == 0 or done == len(self.indices):
                    print(f"  {done}/{len(self.indices)}", flush=True)
            self._null_indices = set(null_list)
            print(f"[ALPNetDataset] Done — "
                  f"{len(self.indices) - len(null_list)} valid, "
                  f"{len(null_list)} blank.", flush=True)

    # ── Pickle support: drop open file handle so DataLoader workers re-open ──

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_npz'] = None   # NpzFile is not fork-safe; each worker re-opens
        return state


    def _compute_superpix(self, idx: int) -> np.ndarray | None:
        """Felzenszwalb MIDDLE scale, same as SSL_ALPNet notebook (min_size=400, sigma=1)."""
        img = self.imgs[idx]
        fg_mask = _fg_mask2d(img, thresh=1e-4)
        if fg_mask.sum() < 16:
            return None
        raw_seg = segmentation.felzenszwalb(img, min_size=400, sigma=1)
        # Labels start at 0 — shift by 1 so 0 = background
        raw_seg = raw_seg + 1
        masked_seg = _superpix_masking(raw_seg, fg_mask)
        return masked_seg

    def _get_superpix(self, idx: int) -> np.ndarray | None:
        """Load superpixel map lazily — opens NpzFile per-worker on first access."""
        if idx in self._null_indices:
            return None
        if self._cache_path is not None:
            if self._npz is None:
                # First access in this worker — open the file (safe after fork)
                self._npz = np.load(self._cache_path, allow_pickle=True)
            return self._npz[str(idx)]
        if self._in_memory is not None:
            if idx not in self._in_memory:
                self._in_memory[idx] = self._compute_superpix(idx)
            return self._in_memory[idx]
        return self._compute_superpix(idx)


    def _augment_once(self, img: np.ndarray, label: np.ndarray):
        """Apply sabs_aug to (img, label) and return (img_aug, label_aug)."""
        H, W = img.shape
        # Stack as [H, W, 2]: channel 0 = image, channel 1 = label
        comp = np.stack([img, label], axis=-1)          # (H, W, 2)
        t_img, t_label = self._transform(
            comp, c_label=1, c_img=1, use_onehot=False, nclass=2,
        )
        # t_img: (H, W, 1), t_label: (H, W, 1)
        img_out   = t_img[..., 0].astype(np.float32)
        label_out = t_label[..., 0].astype(np.float32)
        return img_out, label_out


    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        idx = self.indices[item]
        img = self.imgs[idx]

        # Get superpixel map and binarize one random region → pseudo-label
        super_map = self._get_superpix(idx)
        if super_map is None or super_map.max() == 0:
            # Blank slice: use all-zero label (model will see empty support)
            binary_label = np.zeros_like(img, dtype=np.float32)
        else:
            binary_label = _supcls_pick_binarize(super_map)

        # Augment twice (num_rep=2): support (aug_1), query (aug_2)
        img_sup, lbl_sup = self._augment_once(img, binary_label)
        img_qry, lbl_qry = self._augment_once(img, binary_label)

        # Convert query label to long tensor with ignore_index=255 on borders
        # (label is 0 or 1 after rint; treat as-is since rounding is done in transform)
        query_label_long = torch.from_numpy(lbl_qry).long()  # (H, W)

        # Tile to 3 channels (grayscale → RGB-like)
        sup_img_t = _tile3(img_sup)   # (3, H, W)
        qry_img_t = _tile3(img_qry)   # (3, H, W)

        # FG / BG masks for support
        sup_masks = _get_mask_med_img(lbl_sup)

        # Flat tensors — easier for DataLoader collation (batch_size=1 in training)
        return {
            'sup_img':    sup_img_t,                # (3, H, W)  float32
            'sup_fg':     torch.from_numpy(lbl_sup.astype(np.float32)),  # (H, W)
            'sup_bg':     torch.from_numpy((1.0 - lbl_sup).astype(np.float32)),  # (H, W)
            'qry_img':    qry_img_t,                # (3, H, W)  float32
            'qry_label':  query_label_long,          # (H, W)     int64
        }
