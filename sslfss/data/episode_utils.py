from pathlib import Path

import numpy as np


def discover_eval_datasets(cfg) -> list[tuple]:
    """Return (name, imgs, msks_reais, indices) for every role=test dataset."""
    proc = Path(cfg.processed_dir)
    result = []
    for rf in sorted(proc.glob("*_role.txt")):
        if rf.read_text().strip() != "test":
            continue
        nome = rf.name.replace("_role.txt", "")
        imgs_path = proc / f"{nome}_imgs.npy"
        msks_path = proc / f"{nome}_msks_reais.npy"
        idx_path  = proc / f"{nome}_idx_all.npy"
        if not all(p.exists() for p in (imgs_path, msks_path, idx_path)):
            print(f"[WARN] Missing files for '{nome}'. Skipping.")
            continue
        result.append((
            nome,
            np.load(imgs_path, mmap_mode="r"),
            np.load(msks_path, mmap_mode="r"),
            np.sort(np.load(idx_path)),
        ))
    return result


def load_manifest(path) -> tuple[dict, dict]:
    """Load a .npz manifest produced by generate_episodes.py."""
    data = np.load(path, allow_pickle=False)
    meta = {}
    manifest = {}

    for key in data.files:
        if key.startswith("_meta_"):
            val = data[key]
            meta[key[6:]] = val.item() if val.ndim == 0 else val
            continue
        parts = key.split("||")
        if len(parts) != 4:
            continue
        ds, q_part, t_part, field = parts
        q = int(q_part[1:])
        t = int(t_part[1:])
        manifest.setdefault(ds, {}).setdefault(q, {}).setdefault(t, {})[field] = data[key]

    return manifest, meta


def load_manifest_lazy(path) -> tuple[dict, dict, object]:
    """Lazy variant of load_manifest."""
    data = np.load(path, allow_pickle=False)  # stays open
    meta = {}
    manifest = {}

    for key in data.files:
        if key.startswith("_meta_"):
            val = data[key]
            meta[key[6:]] = val.item() if val.ndim == 0 else val
            continue
        parts = key.split("||")
        if len(parts) != 4:
            continue
        ds, q_part, t_part, field = parts
        q = int(q_part[1:])
        t = int(t_part[1:])
        # Sparse mask fields are large — store key string for on-demand resolution.
        # Small index/scalar fields (e.g. support_idxs) are loaded eagerly.
        if field.startswith("mode_"):
            value = key  # lazy: caller resolves via npzfile[key]
        else:
            value = data[key]
        manifest.setdefault(ds, {}).setdefault(q, {}).setdefault(t, {})[field] = value

    return manifest, meta, data
