import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import learn2learn as l2l
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from sslfss.data.dataset import MetaDatasetMulti
from sslfss.methods import get_method
from sslfss.train_utils import (
    _build_pseudo_label_cache,
    _carregar_datasets,
    _build_fixed_val_episodes,
    _filter_valid_indices,
    run_val,
)
from sslfss.utils import seed_everything, make_worker_init_fn
from sslfss.visualization import plot_training_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["maml", "panet", "r2d2"])
    args = parser.parse_args()

    method = get_method(args.method)

    cfg          = Config()
    cfg.run_name = method.run_name()
    cfg.ensure_dirs()

    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading train split:")
    ds_treino_list, ds_treino_names = _carregar_datasets(cfg, "treino")
    print("Loading val split:")
    ds_val_list, ds_val_names       = _carregar_datasets(cfg, "val")

    if not ds_treino_list:
        print("\n[ERROR] No training data. Run 'python scripts/prepare_data.py' first.")
        sys.exit(1)

    if cfg.dense_label_mode == "pseudo":
        cache_treino = _build_pseudo_label_cache(
            ds_treino_list, ds_treino_names, "treino", seed=cfg.seed, cache_dir=cfg.processed_dir
        )
    else:
        print("[dense_label_mode=real] Using real GT masks.\n")
        cache_treino = {}

    print("\nFiltering train slices:")
    ds_treino_list = _filter_valid_indices(
        ds_treino_list, ds_treino_names, cache_treino, cfg.dense_label_mode, "treino"
    )

    # Val source: raw arrays (no filtering needed — generate_episodes.py already skips
    # empty-mask slices; methods access imgs[idx] via val_source[ds_idx])
    val_source = ds_val_list
    val_names  = ds_val_names

    print("\nLoading val episodes from pre-generated manifest:")
    val_episodes = _build_fixed_val_episodes(val_source, val_names, cfg)

    total_valid_train = sum(len(ds[2]) for ds in ds_treino_list)
    batch_sz    = method.batch_size()
    n_episodes  = cfg.n_episodes_target
    n_steps     = n_episodes // batch_sz
    epoch_steps = max(1, total_valid_train // batch_sz)
    print(
        f"\nTraining: {n_episodes} episodes ({n_steps} steps), "
        f"{total_valid_train} valid slices available\n"
    )

    rng_fast = np.random.default_rng(cfg.val_seed + 2)
    n_fast   = min(cfg.val_log_episodes, len(val_episodes))
    fast_idx = rng_fast.choice(len(val_episodes), size=n_fast, replace=False)
    val_episodes_fast = [val_episodes[i] for i in sorted(fast_idx)]

    print(
        f"Fast val : {len(val_episodes_fast)} episodes (every {cfg.log_interval} steps)\n"
        f"Full val : {len(val_episodes)} episodes (every epoch = {epoch_steps} steps)\n"
    )
    seed_everything(cfg.seed)

    ds_treino = MetaDatasetMulti(
        ds_treino_list,
        pseudo_labels=cache_treino,
        dense_label_mode=cfg.dense_label_mode,
        names=ds_treino_names,
    )
    loader_treino = DataLoader(
        ds_treino, batch_size=batch_sz, shuffle=True, num_workers=0,
        worker_init_fn=make_worker_init_fn(cfg.seed),
        drop_last=True,
    )

    is_maml = args.method == "maml"
    if is_maml:
        import learn2learn as l2l
        iter_treino = l2l.data.utils.InfiniteIterator(loader_treino)
    else:
        iter_treino = iter(loader_treino)

    method.build_model(cfg, device, n_steps)

    historico   = {"loss_treino": [], "loss_val": [], "dice_val": []}
    melhor_dice = 0.0
    melhor_step = 0

    print(f"Starting {args.method.upper()} training ({n_episodes} episodes, {n_steps} steps)...\n")

    for step in range(1, n_steps + 1):
        episodio = step * batch_sz

        if not is_maml:
            try:
                batch = next(iter_treino)
            except StopIteration:
                iter_treino = iter(loader_treino)
                batch = next(iter_treino)
        else:
            batch = next(iter_treino)

        loss = method.optimizer_step(batch, device)
        lr   = method.scheduler_step()
        historico["loss_treino"].append(loss)

        if step % cfg.log_interval == 0:
            dice_v, dice_std, miou_v, loss_v, breakdown = run_val(
                method, val_episodes_fast, val_source, device
            )
            historico["loss_val"].append(loss_v)
            historico["dice_val"].append(dice_v)

            torch.save(method.checkpoint_state(), cfg.last_checkpoint_path)
            if dice_v > melhor_dice:
                melhor_dice = dice_v
                melhor_step = episodio
                torch.save(method.checkpoint_state(), cfg.checkpoint_path)

            loss_t = float(np.mean(historico["loss_treino"][-cfg.log_interval:]))
            extra  = method.extra_log()
            print(
                f"Ep {episodio:6d}/{n_episodes} | "
                f"loss_train={loss_t:.4f} | loss_val={loss_v:.4f} | "
                f"dice={dice_v:.4f}±{dice_std:.4f} | miou={miou_v:.4f} | "
                f"best={melhor_dice:.4f} (ep {melhor_step}) | "
                + (f"{extra} | " if extra else "")
                + f"lr={lr:.2e}  [fast-val]"
            )
            print(f"  [{breakdown}]")

        if step % epoch_steps == 0:
            epoch_num = step // epoch_steps
            dice_v, dice_std, miou_v, loss_v, breakdown = run_val(
                method, val_episodes, val_source, device
            )
            torch.save(method.checkpoint_state(), cfg.last_checkpoint_path)
            if dice_v > melhor_dice:
                melhor_dice = dice_v
                melhor_step = episodio
                torch.save(method.checkpoint_state(), cfg.checkpoint_path)

            extra = method.extra_log()
            print(
                f"\n{'='*70}\n"
                f"EPOCH {epoch_num} | "
                f"loss_val={loss_v:.4f} | dice={dice_v:.4f}±{dice_std:.4f} | miou={miou_v:.4f} | "
                f"best={melhor_dice:.4f} (ep {melhor_step})"
                + (f" | {extra}" if extra else "")
                + f"  [FULL VAL]\n"
                f"  [{breakdown}]\n"
                f"{'='*70}\n"
            )

    print(f"\nTraining complete.")
    print(f"Best checkpoint : {cfg.checkpoint_path}  (ep {melhor_step}, Dice={melhor_dice:.4f})")
    print(f"Last checkpoint : {cfg.last_checkpoint_path}")

    plot_training_curves(
        historico,
        melhor_dice=melhor_dice,
        log_intervalo=cfg.log_interval,
        results_dir=cfg.run_results_dir,
        meta_batch_size=batch_sz,
    )


if __name__ == "__main__":
    main()
