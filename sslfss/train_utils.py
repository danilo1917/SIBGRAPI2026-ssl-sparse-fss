from __future__ import annotations

from pathlib import Path

import numpy as np

from sslfss.data.propose_msk import propor_mascara


def _build_pseudo_label_cache(
    datasets: list,
    names: list[str],
    split_name: str,
    seed: int = 0,
    cache_dir: Path | str | None = None,
) -> dict:
    """Pre-compute one pseudo-label per slice and return a {dataset_name: {index: mask}} dict."""
    cache_file: Path | None = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"pseudo_cache_{split_name}_seed{seed}.npz"
        if cache_file.exists():
            print(f"  Loading pseudo-label cache from disk: {cache_file}", flush=True)
            data  = np.load(cache_file, allow_pickle=False)
            cache: dict = {}
            for key in data.files:
                name, i = key.rsplit("||", 1)
                cache.setdefault(name, {})[int(i)] = data[key]
            n_valid = sum(1 for d in cache.values() for m in d.values() if m.max() > 0)
            n_total = sum(len(d) for d in cache.values())
            print(f"  Loaded — {n_valid}/{n_total} slices have a non-empty pseudo-label.\n")
            return cache

    rng_state = np.random.get_state()
    np.random.seed(seed)

    cache = {}
    total = sum(len(idx) for _, _, idx, _ in datasets)
    done  = 0

    print(f"  Pre-computing pseudo-labels for {split_name} ({total} slices)...", flush=True)

    for (imgs, _, indices, _), name in zip(datasets, names):
        ds_cache = cache.setdefault(name, {})
        for i in indices:
            i = int(i)
            ds_cache[i] = propor_mascara(imgs[i])
            done += 1
            if done % 500 == 0 or done == total:
                print(f"    {done}/{total}", flush=True)

    np.random.set_state(rng_state)
    n_valid = sum(1 for d in cache.values() for m in d.values() if m.max() > 0)
    print(f"  Done — {n_valid}/{total} slices have a non-empty pseudo-label.\n")

    if cache_file is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **{f"{name}||{i}": m for name, d in cache.items() for i, m in d.items()})
        print(f"  Cache saved to: {cache_file}")

    return cache


def _carregar_datasets(cfg: object, split: str) -> tuple[list, list[str]]:
    """Load processed .npy files for all train-role datasets."""
    proc = Path(cfg.processed_dir)
    datasets: list     = []
    names:    list[str] = []

    role_files = sorted(proc.glob("*_role.txt"))
    for rf in role_files:
        role = rf.read_text().strip()
        if role != "train":
            continue
        nome = rf.name.replace("_role.txt", "")

        imgs_path    = proc / f"{nome}_imgs.npy"
        msks_path    = proc / f"{nome}_msks_reais.npy"
        vol_ids_path = proc / f"{nome}_vol_ids.npy"
        idx_path     = proc / f"{nome}_idx_{split}.npy"

        if not all(p.exists() for p in (imgs_path, msks_path, idx_path)):
            print(f"  [WARN] Missing files for '{nome}' ({split}). Skipping.")
            continue

        imgs    = np.load(imgs_path, mmap_mode='r')
        msks    = np.load(msks_path, mmap_mode='r')
        indices = np.load(idx_path)
        vol_ids = np.load(vol_ids_path) if vol_ids_path.exists() else None

        datasets.append((imgs, msks, indices, vol_ids))
        names.append(nome)
        print(f"  {nome} ({split}): {len(indices)} slices")

    return datasets, names


class _LazyValEpisode:
    """Lazy episode handle — arrays are read from the .npz only when iterated."""

    __slots__ = ("ds_idx", "ds_name", "qry_idx", "sup_idxs", "_npz", "_prefix", "_k")

    def __init__(
        self,
        ds_idx:  int,
        ds_name: str,
        qry_idx: int,
        sup_idxs: list[int],
        npz,          # open NpzFile (shared reference, not closed here)
        prefix:  str,
    ) -> None:
        self.ds_idx  = ds_idx
        self.ds_name = ds_name
        self.qry_idx = qry_idx
        self.sup_idxs = sup_idxs
        self._npz    = npz
        self._prefix = prefix
        self._k      = len(sup_idxs)

    def __iter__(self):
        sup_sparses = [self._npz[f"{self._prefix}||sparse_{i}"] for i in range(self._k)]
        gt_dense_np = self._npz[f"{self._prefix}||gt_dense"]
        yield self.ds_idx
        yield self.ds_name
        yield self.qry_idx
        yield self.sup_idxs
        yield sup_sparses
        yield gt_dense_np


def _build_fixed_val_episodes(
    ds_val_list: list,
    val_names:   list[str],
    cfg:         object,
) -> list[_LazyValEpisode]:
    """Load pre-generated val episode handles from disk (lazy — no mask arrays in RAM)."""
    proc = Path(cfg.processed_dir)
    path = proc / f"episodes_val_s{cfg.val_seed}_k{cfg.k_shot}.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Val episode manifest not found: {path}\n"
            "Run 'python scripts/generate_episodes.py' first."
        )

    # Keep NpzFile open for the lifetime of training — reads individual arrays on demand.
    data = np.load(path, allow_pickle=False)

    # Build name → ds_idx mapping from the provided val list
    name_to_idx = {name: i for i, name in enumerate(val_names)}

    # Discover all (nome, qry_idx) pairs present in the file
    seen: list[tuple[str, int]] = []
    for key in data.files:
        if key.startswith("_meta"):
            continue
        parts = key.split("||")
        if len(parts) == 3 and parts[1].startswith("q") and parts[2] == "support_idxs":
            nome    = parts[0]
            qry_idx = int(parts[1][1:])
            seen.append((nome, qry_idx))

    episodes: list[_LazyValEpisode] = []
    for nome, qry_idx in seen:
        if nome not in name_to_idx:
            continue
        ds_idx   = name_to_idx[nome]
        prefix   = f"{nome}||q{qry_idx}"
        sup_idxs = [int(x) for x in data[f"{prefix}||support_idxs"]]
        episodes.append(_LazyValEpisode(ds_idx, nome, qry_idx, sup_idxs, data, prefix))

    by_ds: dict[str, int] = {}
    for ep in episodes:
        by_ds[ep.ds_name] = by_ds.get(ep.ds_name, 0) + 1
    for nome, cnt in by_ds.items():
        print(f"  {nome} (val episodes): {cnt} loaded")
    print(f"  Total val episodes: {len(episodes)}  (from {path})")

    return episodes


def _filter_valid_indices(
    datasets:         list,
    names:            list[str],
    cache:            dict,
    dense_label_mode: str,
    split_label:      str,
) -> list:
    """Remove slices with empty or invalid labels from every dataset's index array."""
    filtered: list = []
    for (imgs, msks, indices, vol_ids), name in zip(datasets, names):
        ds_cache = cache.get(name, {})
        valid = []
        for i in [int(x) for x in indices]:
            if dense_label_mode == "real":
                ok = msks[i].max() > 0
            else:
                # pseudo mode: only accept slices with a non-empty pseudo-label
                # real GT (msks) is intentionally never checked here
                ok = i in ds_cache and ds_cache[i].max() > 0
            if ok:
                valid.append(i)
        n_total = len(indices)
        n_valid = len(valid)
        print(f"  {name} ({split_label}): kept {n_valid}/{n_total} slices with valid labels")
        if n_valid == 0:
            raise RuntimeError(
                f"Dataset '{name}' has no valid slices in split '{split_label}'. "
                "Check pseudo-label generation or min_fg_fraction."
            )
        filtered.append((imgs, msks, np.array(valid, dtype=np.int64), vol_ids))
    return filtered


def run_val(
    method,
    episodes: list,
    val_source: list,
    device,
) -> tuple[float, float, float, float, str]:
    from collections import defaultdict
    method_model = getattr(method, "maml", None) or getattr(method, "model", None)
    if method_model is not None:
        method_model.eval()

    dices_by_ds: dict[str, list[float]] = defaultdict(list)
    mious_by_ds: dict[str, list[float]] = defaultdict(list)
    losses: list[float] = []

    for episode in episodes:
        ds_name = episode.ds_name
        loss, dice, miou = method.val_episode(episode, val_source, device)
        losses.append(loss)
        dices_by_ds[ds_name].append(dice)
        mious_by_ds[ds_name].append(miou)

    per_ds_dice = [float(np.mean(v)) for v in dices_by_ds.values()]
    per_ds_miou = [float(np.mean(v)) for v in mious_by_ds.values()]
    dice_macro  = float(np.mean(per_ds_dice))
    dice_std    = float(np.std(per_ds_dice))
    miou_macro  = float(np.mean(per_ds_miou))
    loss_mean   = float(np.mean(losses))
    breakdown   = "  ".join(
        f"{n.split('_')[-1]}=dice{np.mean(dices_by_ds[n]):.3f}/miou{np.mean(mious_by_ds[n]):.3f}"
        for n in dices_by_ds
    )

    if method_model is not None:
        method_model.train()

    return dice_macro, dice_std, miou_macro, loss_mean, breakdown
