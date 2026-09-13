import argparse
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import Config
from sslfss.data.episode_utils import discover_eval_datasets
from sslfss.data.sparsify import aplicar_esparsidade, aplicar_esparsidade_modo, SPARSE_MODES_ALL
from sslfss.train_utils import _carregar_datasets
from sslfss.utils import seed_everything


class _NpzWriter:
    """Write arrays one at a time into a .npz (ZIP) file without buffering all in RAM."""

    def __init__(self, path: Path) -> None:
        self._zf = zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True)

    def write(self, name: str, arr: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, arr)
        self._zf.writestr(f"{name}.npy", buf.getvalue())

    def close(self) -> None:
        self._zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _save_meta(writer: _NpzWriter, seed: int, k: int, n_trials: int) -> None:
    writer.write("_meta_seed",     np.array(seed,     dtype=np.int64))
    writer.write("_meta_k",        np.array(k,        dtype=np.int64))
    writer.write("_meta_n_trials", np.array(n_trials, dtype=np.int64))


def _sample_supports(indices_list: list[int], q_idx: int, k: int) -> list[int]:
    other = [i for i in indices_list if i != q_idx]
    return [int(x) for x in np.random.choice(other, size=k, replace=len(other) < k)]


def _build_manifest_standard(
    datasets: list[tuple],
    k: int,
    n_trials: int,
    seed: int,
    label: str,
    path: Path,
) -> None:
    """Build eval / shots / steps manifest, writing directly to disk."""
    total_datasets = len(datasets)
    with _NpzWriter(path) as w:
        _save_meta(w, seed, k, n_trials)
        for di, (nome, imgs, msks, indices) in enumerate(datasets):
            indices_list = [int(i) for i in indices]
            n_q = len(indices_list)
            print(f"  [{label}] {nome} ({di+1}/{total_datasets}): {n_q} queries × {n_trials} trials")
            for q_idx in indices_list:
                for trial in range(n_trials):
                    prefix   = f"{nome}||q{q_idx}||t{trial}"
                    sup_idxs = _sample_supports(indices_list, q_idx, k)
                    w.write(f"{prefix}||support_idxs", np.array(sup_idxs, dtype=np.int64))
                    for s_pos, s_idx in enumerate(sup_idxs):
                        sparse = aplicar_esparsidade(msks[s_idx].astype(np.float32))
                        w.write(f"{prefix}||sparse_{s_pos}", sparse)


def _build_manifest_sparsification(
    datasets: list[tuple],
    k: int,
    n_trials: int,
    seed: int,
    path: Path,
) -> None:
    """Build sparsification manifest, writing directly to disk."""
    total_datasets = len(datasets)
    with _NpzWriter(path) as w:
        _save_meta(w, seed, k, n_trials)
        for di, (nome, imgs, msks, indices) in enumerate(datasets):
            indices_list = [int(i) for i in indices]
            n_q = len(indices_list)
            print(f"  [sparsification] {nome} ({di+1}/{total_datasets}): "
                  f"{n_q} queries × {n_trials} trials × {len(SPARSE_MODES_ALL)} modes")
            for q_idx in indices_list:
                for trial in range(n_trials):
                    prefix   = f"{nome}||q{q_idx}||t{trial}"
                    sup_idxs = _sample_supports(indices_list, q_idx, k)
                    w.write(f"{prefix}||support_idxs", np.array(sup_idxs, dtype=np.int64))
                    for s_pos, s_idx in enumerate(sup_idxs):
                        msk = msks[s_idx].astype(np.float32)
                        for mode in SPARSE_MODES_ALL:
                            sub_seed = int(seed) ^ (q_idx * 10007) ^ (trial * 997) ^ (s_pos * 97) ^ hash(mode)
                            sub_seed &= 0x7FFFFFFF
                            saved = np.random.get_state()
                            np.random.seed(sub_seed)
                            sparse = aplicar_esparsidade_modo(msk, mode)
                            np.random.set_state(saved)
                            w.write(f"{prefix}||mode_{mode}_{s_pos}", sparse)


def _build_manifest_val(cfg, k: int, seed: int, path: Path) -> None:
    """Build val episode manifest using real GT, K-shot with distinct support/query slices."""
    datasets, names = _carregar_datasets(cfg, "val")
    if not datasets:
        print("[WARN] No val datasets found. Skipping val manifest.")
        return

    with _NpzWriter(path) as w:
        _save_meta(w, seed, k, n_trials=1)
        for di, ((imgs, msks, indices, vol_ids), nome) in enumerate(zip(datasets, names)):
            indices_list = [int(i) for i in indices]
            valid_queries = [i for i in indices_list if msks[i].max() > 0]
            print(f"  [val] {nome} ({di+1}/{len(datasets)}): {len(valid_queries)} queries")
            for qry_idx in valid_queries:
                prefix = f"{nome}||q{qry_idx}"
                # Prefer supports from different volumes than the query
                if vol_ids is not None:
                    q_vid    = int(vol_ids[qry_idx])
                    diff_vol = [i for i in indices_list if i != qry_idx and int(vol_ids[i]) != q_vid]
                    same_vol = [i for i in indices_list if i != qry_idx and int(vol_ids[i]) == q_vid]
                    pool = diff_vol if len(diff_vol) >= k else diff_vol + same_vol
                else:
                    pool = [i for i in indices_list if i != qry_idx]
                if not pool:
                    continue
                sup_idxs = [int(x) for x in np.random.choice(pool, size=k, replace=len(pool) < k)]
                w.write(f"{prefix}||support_idxs", np.array(sup_idxs, dtype=np.int64))
                for s_pos, s_idx in enumerate(sup_idxs):
                    sparse = aplicar_esparsidade(msks[s_idx].astype(np.float32))
                    w.write(f"{prefix}||sparse_{s_pos}", sparse)
                w.write(f"{prefix}||gt_dense", msks[qry_idx].astype(np.float32))


def _print_size(path: Path) -> None:
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  Saved: {path}  ({size_mb:.1f} MB)\n")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--val-seed", type=int, default=None,
                   help="Seed for val manifest (default: cfg.val_seed = 123)")
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--k_eval",   type=int, default=5,
                   help="k for eval / sparsification / steps manifests")
    p.add_argument("--k_max",    type=int, default=10,
                   help="k for shots manifest (must be >= k_eval)")
    p.add_argument("--k_val",    type=int, default=None,
                   help="k for val manifest (default: cfg.k_shot = 5)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.k_max < args.k_eval:
        print(f"[ERROR] k_max ({args.k_max}) must be >= k_eval ({args.k_eval})")
        sys.exit(1)

    cfg = Config()
    proc = Path(cfg.processed_dir)

    val_seed = args.val_seed if args.val_seed is not None else cfg.val_seed
    k_val    = args.k_val   if args.k_val   is not None else cfg.k_shot

    print(f"Seed      : {args.seed}")
    print(f"Val seed  : {val_seed}")
    print(f"n_trials  : {args.n_trials}")
    print(f"k_eval    : {args.k_eval}")
    print(f"k_max     : {args.k_max}")
    print(f"k_val     : {k_val}")
    print()

    # Single seed for the full generation — deterministic and reproducible.
    seed_everything(args.seed)

    datasets = discover_eval_datasets(cfg)
    if not datasets:
        print("[ERROR] No role=test datasets found. Run prepare_data.py first.")
        sys.exit(1)
    print(f"Found {len(datasets)} test dataset(s): {[d[0] for d in datasets]}\n")

    s, k_e, k_m, t = args.seed, args.k_eval, args.k_max, args.n_trials

    def _maybe_skip(path: Path) -> bool:
        if path.exists():
            print(f"  [SKIP] {path} already exists.\n")
            return True
        return False

    print("Generating eval manifest ...")
    p = proc / f"episodes_eval_s{s}_k{k_e}_t{t}.npz"
    if not _maybe_skip(p):
        _build_manifest_standard(datasets, k_e, t, s, label="eval", path=p)
        _print_size(p)

    print("Generating shots manifest ...")
    p = proc / f"episodes_shots_s{s}_k{k_m}_t{t}.npz"
    if not _maybe_skip(p):
        _build_manifest_standard(datasets, k_m, t, s, label="shots", path=p)
        _print_size(p)

    print("Generating sparsification manifest ...")
    p = proc / f"episodes_sparsification_s{s}_k{k_e}_t{t}.npz"
    if not _maybe_skip(p):
        _build_manifest_sparsification(datasets, k_e, t, s, path=p)
        _print_size(p)

    print("Generating steps manifest ...")
    p = proc / f"episodes_steps_s{s}_k{k_e}_t{t}.npz"
    if not _maybe_skip(p):
        _build_manifest_standard(datasets, k_e, t, s, label="steps", path=p)
        _print_size(p)

    print("Generating val manifest ...")
    seed_everything(val_seed)
    p = proc / f"episodes_val_s{val_seed}_k{k_val}.npz"
    if not _maybe_skip(p):
        _build_manifest_val(cfg, k_val, val_seed, p)
        _print_size(p)

    print("All manifests generated.")


if __name__ == "__main__":
    main()
