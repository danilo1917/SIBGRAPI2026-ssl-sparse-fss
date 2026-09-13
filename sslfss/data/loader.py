from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from PIL import Image

from sslfss.data.resize import redimensionar_corte


_SLICE_INDEXERS = {
    0: lambda v, k: v[k, :, :],
    1: lambda v, k: v[:, k, :],
    2: lambda v, k: v[:, :, k],
}


def _detect_format(imagesTr_dir: Path) -> str:
    """Return ``'nifti'`` or ``'png'`` based on files inside ``imagesTr_dir``."""
    if any(imagesTr_dir.glob("*.nii.gz")):
        return "nifti"
    if any(imagesTr_dir.glob("*.png")):
        return "png"
    raise FileNotFoundError(
        f"No supported files (*.nii.gz or *.png) found in '{imagesTr_dir}'. "
        "Expected NIfTI or PNG images."
    )


def _load_png_volume(
    img_path: Path,
    lbl_path: Path,
    binarize: bool = True,
    min_fg_fraction: float = 0.001,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load a single PNG image/mask pair as a 2-D 'slice'."""
    img_arr = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
    lbl_arr = np.array(Image.open(lbl_path).convert("L"), dtype=np.float32)

    msk = (lbl_arr > 127).astype(np.float32) if binarize else lbl_arr

    if msk.mean() < min_fg_fraction:
        return [], []

    p1, p99 = np.percentile(img_arr, (1, 99))
    img_arr = np.clip(img_arr, p1, p99)

    i_min, i_max = img_arr.min(), img_arr.max()
    if i_max > i_min:
        img_arr = (img_arr - i_min) / (i_max - i_min)
    else:
        img_arr = np.zeros_like(img_arr)

    return [img_arr], [msk]


def _vol_id_from_stem(stem: str, vol_map: dict[str, int]) -> int:
    """Assign an integer volume ID based on the filename stem group."""
    if "_" in stem:
        group = stem.rsplit("_", 1)[0]
    else:
        group = stem          # unique per file
    if group not in vol_map:
        vol_map[group] = len(vol_map)
    return vol_map[group]


def carregar_volume(
    caminho_img: Path | str,
    caminho_lbl: Path | str,
    slice_axis: int = 2,
    binarize: bool = True,
    min_fg_fraction: float = 0.001,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load one NIfTI image/label pair and return valid slices."""
    if slice_axis not in _SLICE_INDEXERS:
        raise ValueError(f"slice_axis must be 0, 1 or 2; got {slice_axis}")

    indexer = _SLICE_INDEXERS[slice_axis]

    img_nib = nib.load(str(caminho_img))
    lbl_nib = nib.load(str(caminho_lbl))

    volume_img = img_nib.get_fdata().astype(np.float32)
    volume_lbl = lbl_nib.get_fdata().astype(np.float32)

    # Percentile clip per volume.
    p1, p99 = np.percentile(volume_img, (1, 99))
    volume_img = np.clip(volume_img, p1, p99)

    imagens:  list[np.ndarray] = []
    mascaras: list[np.ndarray] = []

    n_slices = volume_img.shape[slice_axis]
    for k in range(n_slices):
        corte_img = indexer(volume_img, k)
        corte_lbl = indexer(volume_lbl, k)

        if binarize:
            corte_msk = (corte_lbl > 0).astype(np.float32)
        else:
            corte_msk = corte_lbl.astype(np.float32)

        fg_frac = corte_msk.mean()
        if fg_frac >= min_fg_fraction:
            # Per-slice min-max → [0, 1].
            s_min, s_max = corte_img.min(), corte_img.max()
            if s_max > s_min:
                corte_img = (corte_img - s_min) / (s_max - s_min)
            else:
                corte_img = np.zeros_like(corte_img)
            imagens.append(corte_img.astype(np.float32))
            mascaras.append(corte_msk)

    return imagens, mascaras


def carregar_dataset(
    imagesTr_dir: Path | str,
    labelsTr_dir: Path | str,
    n_volumes: Optional[int] = None,
    slice_axis: int = 2,
    binarize: bool = True,
    min_fg_fraction: float = 0.001,
    target_size: Optional[tuple[int, int]] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Load multiple volumes/images from an MSD-style folder pair."""
    imagesTr_dir = Path(imagesTr_dir)
    labelsTr_dir = Path(labelsTr_dir)

    fmt = _detect_format(imagesTr_dir)

    if fmt == "nifti":
        img_paths = sorted(imagesTr_dir.glob("*.nii.gz"))
        lbl_paths = sorted(labelsTr_dir.glob("*.nii.gz"))

        if n_volumes is not None:
            img_paths = img_paths[:n_volumes]
            lbl_paths = lbl_paths[:n_volumes]

        todas_imgs: list[np.ndarray] = []
        todas_msks: list[np.ndarray] = []
        volume_ids: list[int] = []

        for vol_idx, (img_path, lbl_path) in enumerate(zip(img_paths, lbl_paths)):
            imgs, msks = carregar_volume(
                img_path, lbl_path,
                slice_axis=slice_axis,
                binarize=binarize,
                min_fg_fraction=min_fg_fraction,
            )
            if target_size is not None:
                imgs = [redimensionar_corte(x, target_size) for x in imgs]
                msks = [redimensionar_corte(x, target_size, mode="nearest") for x in msks]
            todas_imgs.extend(imgs)
            todas_msks.extend(msks)
            volume_ids.extend([vol_idx] * len(imgs))

        print(f"  [nifti] {len(img_paths)} volumes → {len(todas_imgs)} valid slices")

    else:  # png
        img_paths = sorted(imagesTr_dir.glob("*.png"))
        lbl_paths_by_stem = {p.stem: p for p in labelsTr_dir.glob("*.png")}

        if n_volumes is not None:
            img_paths = img_paths[:n_volumes]

        todas_imgs = []
        todas_msks = []
        volume_ids = []
        vol_map: dict[str, int] = {}

        for img_path in img_paths:
            lbl_path = lbl_paths_by_stem.get(img_path.stem)
            if lbl_path is None:
                print(f"  [WARN] No matching label for '{img_path.name}'. Skipping.")
                continue
            imgs, msks = _load_png_volume(
                img_path, lbl_path,
                binarize=binarize,
                min_fg_fraction=min_fg_fraction,
            )
            if imgs:
                if target_size is not None:
                    imgs = [redimensionar_corte(x, target_size) for x in imgs]
                    msks = [redimensionar_corte(x, target_size, mode="nearest") for x in msks]
                vol_id = _vol_id_from_stem(img_path.stem, vol_map)
                todas_imgs.extend(imgs)
                todas_msks.extend(msks)
                volume_ids.append(vol_id)

        n_vols = len(set(volume_ids)) if volume_ids else 0
        print(f"  [png] {len(img_paths)} images ({n_vols} groups) → {len(todas_imgs)} valid slices")

    return todas_imgs, todas_msks, np.array(volume_ids, dtype=np.int64)


def iter_dataset(
    imagesTr_dir: Path | str,
    labelsTr_dir: Path | str,
    n_volumes: Optional[int] = None,
    slice_axis: int = 2,
    binarize: bool = True,
    min_fg_fraction: float = 0.001,
    target_size: Optional[tuple[int, int]] = None,
):
    """Generator version of ``carregar_dataset``."""
    imagesTr_dir = Path(imagesTr_dir)
    labelsTr_dir = Path(labelsTr_dir)
    fmt = _detect_format(imagesTr_dir)

    if fmt == "nifti":
        img_paths = sorted(imagesTr_dir.glob("*.nii.gz"))
        lbl_paths = sorted(labelsTr_dir.glob("*.nii.gz"))
        if n_volumes is not None:
            img_paths = img_paths[:n_volumes]
            lbl_paths = lbl_paths[:n_volumes]
        for vol_idx, (ip, lp) in enumerate(zip(img_paths, lbl_paths)):
            imgs, msks = carregar_volume(
                ip, lp, slice_axis=slice_axis,
                binarize=binarize, min_fg_fraction=min_fg_fraction,
            )
            if target_size is not None:
                imgs = [redimensionar_corte(x, target_size) for x in imgs]
                msks = [redimensionar_corte(x, target_size, mode="nearest") for x in msks]
            for img, msk in zip(imgs, msks):
                yield img, msk, vol_idx
    else:  # png
        img_paths = sorted(imagesTr_dir.glob("*.png"))
        lbl_paths_by_stem = {p.stem: p for p in labelsTr_dir.glob("*.png")}
        if n_volumes is not None:
            img_paths = img_paths[:n_volumes]
        vol_map: dict[str, int] = {}
        for ip in img_paths:
            lp = lbl_paths_by_stem.get(ip.stem)
            if lp is None:
                continue
            imgs, msks = _load_png_volume(
                ip, lp, binarize=binarize, min_fg_fraction=min_fg_fraction,
            )
            if imgs:
                if target_size is not None:
                    imgs = [redimensionar_corte(x, target_size) for x in imgs]
                    msks = [redimensionar_corte(x, target_size, mode="nearest") for x in msks]
                vol_id = _vol_id_from_stem(ip.stem, vol_map)
                for img, msk in zip(imgs, msks):
                    yield img, msk, vol_id


def iter_imagens(
    images_dir: Path | str,
    n_volumes: Optional[int] = None,
    slice_axis: int = 2,
    target_size: Optional[tuple[int, int]] = None,
):
    """Generator version of ``carregar_imagens`` (unlabelled imagesTs/)."""
    images_dir = Path(images_dir)
    indexer    = _SLICE_INDEXERS[slice_axis]
    img_files  = sorted(images_dir.glob("*.nii.gz"))
    if n_volumes is not None:
        img_files = img_files[:n_volumes]
    for vol_idx, img_path in enumerate(img_files):
        img_nib    = nib.load(str(img_path))
        volume_img = img_nib.get_fdata().astype(np.float32)
        p1, p99    = np.percentile(volume_img, (1, 99))
        volume_img = np.clip(volume_img, p1, p99)
        n_slices   = volume_img.shape[slice_axis]
        for k in range(n_slices):
            corte        = indexer(volume_img, k)
            s_min, s_max = corte.min(), corte.max()
            if s_max <= s_min:
                continue
            corte = ((corte - s_min) / (s_max - s_min)).astype(np.float32)
            if target_size is not None:
                corte = redimensionar_corte(corte, target_size)
            yield corte, np.zeros(corte.shape, dtype=np.float32), vol_idx


def carregar_imagens(
    images_dir: Path | str,
    n_volumes: Optional[int] = None,
    slice_axis: int = 2,
    target_size: Optional[tuple[int, int]] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Load NIfTI images with **no labels** (e.g. MSD imagesTs/)."""
    images_dir = Path(images_dir)
    indexer    = _SLICE_INDEXERS[slice_axis]

    img_files = sorted(images_dir.glob("*.nii.gz"))
    if n_volumes is not None:
        img_files = img_files[:n_volumes]

    todas_imgs: list[np.ndarray] = []
    todas_msks: list[np.ndarray] = []
    volume_ids: list[int]        = []

    for vol_idx, img_path in enumerate(img_files):
        img_nib    = nib.load(str(img_path))
        volume_img = img_nib.get_fdata().astype(np.float32)

        p1, p99    = np.percentile(volume_img, (1, 99))
        volume_img = np.clip(volume_img, p1, p99)

        n_slices = volume_img.shape[slice_axis]
        for k in range(n_slices):
            corte        = indexer(volume_img, k)
            s_min, s_max = corte.min(), corte.max()
            if s_max <= s_min:
                continue                          # skip blank / degenerate slices
            corte = (corte - s_min) / (s_max - s_min)
            corte = corte.astype(np.float32)
            if target_size is not None:
                corte = redimensionar_corte(corte, target_size)
            todas_imgs.append(corte)
            todas_msks.append(np.zeros(corte.shape, dtype=np.float32))
            volume_ids.append(vol_idx)

    print(f"  {len(img_files)} volumes (unlabelled) → {len(todas_imgs)} slices")
    return todas_imgs, todas_msks, np.array(volume_ids, dtype=np.int64)
