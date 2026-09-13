import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import Config
from sslfss.data.episode_utils import discover_eval_datasets, load_manifest
from sslfss.data.propose_msk import propor_mascara
from sslfss.metrics import dice_score, miou_score
from sslfss.methods.factory import get_method
from sslfss.utils import seed_everything
from scripts.utils.shots_utils import (
    plot_shots_line,
    plot_shots_boxes,
    plot_kshot_query_grid,
    save_csv,
    save_summary,
)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",   required=True,
                   choices=["maml", "panet", "r2d2", "alpnet"])
    p.add_argument("--episodes", required=True, help="Path to shots manifest .npz")
    p.add_argument("--k_values", required=True, nargs="+", type=int)
    p.add_argument("--mode",     default=None, help="Override dense_label_mode")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = Config()
    if args.mode:
        cfg.dense_label_mode = args.mode
    cfg.ensure_dirs()
    seed_everything(cfg.seed)

    manifest, meta = load_manifest(args.episodes)

    k_manifest = int(meta["k"])
    for k in args.k_values:
        if k > k_manifest:
            print(f"[ERROR] k={k} requested but manifest was built with k_max={k_manifest}.")
            sys.exit(1)
    k_values = sorted(args.k_values)

    method = get_method(args.method)
    cfg.run_name = method.run_name()
    method.load_model(cfg)

    datasets_by_name = {d[0]: d for d in discover_eval_datasets(cfg)}

    for nome, ds_manifest in manifest.items():
        if nome not in datasets_by_name:
            print(f"[WARN] Dataset '{nome}' not found on disk. Skipping.")
            continue

        _, imgs, msks, _ = datasets_by_name[nome]
        out_dir = Path(cfg.run_results_dir) / args.method / "shots" / nome
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}\n  {nome}\n{'='*60}")

        rows: list[dict]        = []
        k_dices: dict           = {k: [] for k in k_values}
        k_mious: dict           = {k: [] for k in k_values}
        ss_dices: list[float]   = []
        k_sample_episodes: dict = {}  # k → (sup_idxs, sup_sparses, q_idx)

        for q_idx, trials in ds_manifest.items():
            ss_pred = propor_mascara(imgs[q_idx])
            ss_dices.append(dice_score(ss_pred, msks[q_idx]))

            for trial, episode in trials.items():
                all_sup_idxs    = [int(i) for i in episode["support_idxs"]]
                all_sup_sparses = [episode[f"sparse_{s}"] for s in range(len(all_sup_idxs))]

                for k in k_values:
                    sup_idxs    = all_sup_idxs[:k]
                    sup_imgs    = [imgs[i] for i in sup_idxs]
                    sup_sparses = all_sup_sparses[:k]
                    # ALPNet is the dense-support baseline: it receives the real GT
                    # masks, while MAML/PANet/R2D2 adapt from sparse annotations only.
                    sup_masks_for_method = (
                        [msks[i] for i in sup_idxs]
                        if args.method == "alpnet"
                        else sup_sparses
                    )

                    pred = method.predict(imgs[q_idx], sup_imgs, sup_masks_for_method)
                    d    = dice_score(pred, msks[q_idx])
                    m    = miou_score(pred, msks[q_idx])
                    # Free GPU cache after every inference call
                    del pred, sup_imgs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    k_dices[k].append(d)
                    k_mious[k].append(m)
                    rows.append({"k": k, "query_idx": q_idx, "trial": trial,
                                 "dice": f"{d:.6f}", "miou": f"{m:.6f}"})
                    if k not in k_sample_episodes:
                        k_sample_episodes[k] = (sup_idxs, sup_sparses, q_idx)

        ss_mean = float(np.mean(ss_dices)) if ss_dices else 0.0
        save_csv(rows, out_dir)
        save_summary(k_values, k_dices, k_mious, ss_mean, args.method, nome, out_dir)
        plot_shots_line(k_values, k_dices, ss_mean, args.method, nome, out_dir)
        plot_shots_boxes(k_values, k_dices, ss_mean, args.method, nome, out_dir)
        plot_kshot_query_grid(method, imgs, msks, k_values, k_sample_episodes,
                         args.method, nome, out_dir)
        del rows, k_sample_episodes
        gc.collect()

    print(f"\nDone. Results under: {Path(cfg.run_results_dir) / args.method / 'shots'}")


if __name__ == "__main__":
    main()
