import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from config import Config
from sslfss.data.loader import iter_dataset, iter_imagens, _detect_format


def _count_slices(gen) -> int:
    """Drain a generator counting elements without storing them."""
    return sum(1 for _ in gen)


def _stream_to_memmap(
    gen,
    n: int,
    h: int,
    w: int,
    imgs_path: Path,
    msks_path: Path,
) -> np.ndarray:
    """Write ``n`` slices from ``gen`` into two memory-mapped .npy files."""
    imgs_mm = np.lib.format.open_memmap(
        imgs_path, mode="w+", dtype=np.float32, shape=(n, h, w)
    )
    msks_mm = np.lib.format.open_memmap(
        msks_path, mode="w+", dtype=np.float32, shape=(n, h, w)
    )
    vol_ids = np.empty(n, dtype=np.int64)

    for i, (img, msk, vol_id) in enumerate(gen):
        imgs_mm[i] = img
        msks_mm[i] = (msk > 0.5).astype(np.float32)
        vol_ids[i] = vol_id

    # Flush to disk
    del imgs_mm, msks_mm
    return vol_ids


def _process_dataset(
    dataset_dir: Path,
    role: str,
    cfg: Config,
    out: Path,
) -> None:
    """Stream, resize, split and save one dataset — constant-RAM design."""
    nome         = dataset_dir.name
    imagesTr_dir = dataset_dir / "imagesTr"
    labelsTr_dir = dataset_dir / "labelsTr"
    imagesTs_dir = dataset_dir / "imagesTs"

    if not imagesTr_dir.exists() or not labelsTr_dir.exists():
        print(f"[WARN] Skipping '{nome}': imagesTr/ or labelsTr/ not found.")
        return

    print(f"── Dataset: {nome}  (role={role})")

    try:
        fmt = _detect_format(imagesTr_dir)
    except FileNotFoundError as exc:
        print(f"   [WARN] {exc} Skipping.")
        return
    print(f"   format : {fmt}")

    print("   Counting labelled slices...", end=" ", flush=True)
    n_labelled = _count_slices(
        iter_dataset(
            imagesTr_dir, labelsTr_dir,
            n_volumes=cfg.n_volumes,
            slice_axis=cfg.slice_axis,
            binarize=True,
            min_fg_fraction=cfg.min_fg_fraction,
            target_size=cfg.target_size,
        )
    )
    print(f"{n_labelled}")

    if n_labelled == 0:
        print(f"   [WARN] No valid slices found. Skipping.\n")
        return

    n_ts = 0
    if role == "train" and imagesTs_dir.exists():
        print("   Counting unlabelled Ts slices...", end=" ", flush=True)
        n_ts = _count_slices(
            iter_imagens(
                imagesTs_dir,
                n_volumes=cfg.n_volumes,
                slice_axis=cfg.slice_axis,
                target_size=cfg.target_size,
            )
        )
        print(f"{n_ts}")

    n_total = n_labelled + n_ts
    h, w    = cfg.target_size

    imgs_path = out / f"{nome}_imgs.npy"
    msks_path = out / f"{nome}_msks_reais.npy"

    print(f"   Writing {n_total} slices ({h}×{w}) to disk (memmap)...")

    # Labelled slices first
    gen_labelled = iter_dataset(
        imagesTr_dir, labelsTr_dir,
        n_volumes=cfg.n_volumes,
        slice_axis=cfg.slice_axis,
        binarize=True,
        min_fg_fraction=cfg.min_fg_fraction,
        target_size=cfg.target_size,
    )

    if n_ts > 0:
        # Offset Ts vol_ids so they don't collide with Tr vol_ids.
        # We don't know the max Tr vol_id until after pass 2, so we use
        # a safe large offset (n_labelled is an upper bound on n_tr_vols).
        vol_id_offset = n_labelled
        gen_ts = (
            (img, msk, vol_id + vol_id_offset)
            for img, msk, vol_id in iter_imagens(
                imagesTs_dir,
                n_volumes=cfg.n_volumes,
                slice_axis=cfg.slice_axis,
                target_size=cfg.target_size,
            )
        )
        import itertools
        gen_combined = itertools.chain(gen_labelled, gen_ts)
    else:
        gen_combined = gen_labelled

    vol_ids = _stream_to_memmap(gen_combined, n_total, h, w, imgs_path, msks_path)

    np.save(out / f"{nome}_vol_ids.npy", vol_ids)
    (out / f"{nome}_role.txt").write_text(role)

    print(f"   images : ({n_total}, {h}, {w})  |  masks : same")

    if role == "train":
        _split_and_save_train(n_labelled, n_total, vol_ids, nome, cfg, out)
    else:
        idx_all = np.arange(n_total)
        np.save(out / f"{nome}_idx_all.npy", idx_all)
        print(f"   Test — {n_total} slices (no split)")

    print(f"   Saved to: {out.resolve()}\n")


def _split_and_save_train(
    n_labelled: int,
    n_total: int,
    vol_ids: np.ndarray,
    nome: str,
    cfg: Config,
    out: Path,
) -> None:
    """Volume-level treino/val split."""
    labelled_indices = np.arange(n_labelled)
    labelled_vol_ids = vol_ids[:n_labelled]

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg.val_frac,
        random_state=cfg.seed,
    )
    idx_tr_lab, idx_val = next(gss.split(labelled_indices, groups=labelled_vol_ids))

    ts_indices = np.arange(n_labelled, n_total)
    idx_treino = np.concatenate([idx_tr_lab, ts_indices])

    n_v_tr = len(np.unique(labelled_vol_ids[idx_tr_lab]))
    n_v_va = len(np.unique(labelled_vol_ids[idx_val]))
    print(
        f"   Split — treino: {len(idx_treino)} slices ({n_v_tr} labelled vols"
        f" + {len(ts_indices)} Ts) | val: {len(idx_val)} slices ({n_v_va} vols)"
    )

    np.save(out / f"{nome}_idx_treino.npy", idx_treino)
    np.save(out / f"{nome}_idx_val.npy",    idx_val)


def main() -> None:
    cfg = Config()
    cfg.ensure_dirs()

    out = Path(cfg.processed_dir)

    train_dirs = Config.discover_datasets(cfg.train_datasets_dir)
    test_dirs  = Config.discover_datasets(cfg.test_datasets_dir)

    if not train_dirs and not test_dirs:
        print(
            "[ERROR] No datasets found.\n"
            f"  Checked: {cfg.train_datasets_dir.resolve()}\n"
            f"           {cfg.test_datasets_dir.resolve()}\n"
            "  Each dataset must contain an imagesTr/ subfolder."
        )
        sys.exit(1)

    total = len(train_dirs) + len(test_dirs)
    print(f"Discovered {total} dataset(s): {len(train_dirs)} train, {len(test_dirs)} test\n")

    for d in train_dirs:
        _process_dataset(d, "train", cfg, out)

    for d in test_dirs:
        _process_dataset(d, "test", cfg, out)

    print("All datasets processed.")


if __name__ == "__main__":
    main()
