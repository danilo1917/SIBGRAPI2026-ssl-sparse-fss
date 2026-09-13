from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader

from config import Config
from sslfss.data.alpnet_dataset import ALPNetDataset
from sslfss.methods.alpnet_method import ALPNetMethod, ALPNetConfig
from sslfss.train_utils import _carregar_datasets
from sslfss.utils import seed_everything

# Match reference: single thread per process
torch.set_num_threads(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ALPNet on SSL-Sparse-FSS data")
    parser.add_argument("--n_steps",      type=int,  default=Config().n_episodes_target,
                        help="Total number of training iterations (default: 36000)")
    parser.add_argument("--no_coco_init", action="store_true",
                        help="Train from scratch instead of COCO-pretrained backbone")
    args = parser.parse_args()

    SEED = 1234
    seed_everything(SEED)
    cudnn.benchmark = True

    cfg          = Config()
    cfg.run_name = "sslfss_alpnet"
    cfg.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading train split:")
    ds_treino_list, _ = _carregar_datasets(cfg, "treino")

    if not ds_treino_list:
        print("\n[ERROR] No training data. Run 'python scripts/prepare_data.py' first.")
        sys.exit(1)

    all_datasets = []
    for ds_idx, (imgs, _, indices, _) in enumerate(ds_treino_list):
        if len(indices) > 0:
            cache_file = Path(cfg.processed_dir) / f"alpnet_superpix_cache_ds{ds_idx}.npz"
            all_datasets.append(ALPNetDataset(imgs, list(indices), cache_file=cache_file))

    if not all_datasets:
        print("[ERROR] No valid training slices after dataset loading.")
        sys.exit(1)

    train_dataset = ConcatDataset(all_datasets)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,       # match reference
        pin_memory=True,     # match reference
        drop_last=True,
    )

    print(f"\nTraining: {len(train_dataset)} slices, {args.n_steps} steps\n")

    method = ALPNetMethod(ALPNetConfig(use_coco_init=not args.no_coco_init))
    method.build_model(cfg, device, n_steps=args.n_steps)

    # No validation during training — mirrors SSL_ALPNet training.py exactly.
    # Checkpoints saved every 25 000 steps + final _last.pt.
    SAVE_INTERVAL  = 25_000
    PRINT_INTERVAL = 100    # match reference print_interval

    historico = {"loss_treino": []}  # no val — only training loss tracked

    data_iter = iter(train_loader)
    print(f"Starting ALPNet training ({args.n_steps} steps)...\n")

    for step in range(1, args.n_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        try:
            loss = method.optimizer_step(batch, device)
        except Exception:
            print("Faulty batch detected, skip")
            continue

        lr = method.scheduler_step()   # MultiStepLR stepped per iteration
        historico["loss_treino"].append(loss)

        if step % PRINT_INTERVAL == 0:
            print(
                f"[step {step:>7d}/{args.n_steps}]  "
                f"loss={loss:.4f}  lr={lr:.2e}"
            )

        if step % SAVE_INTERVAL == 0:
            ckpt = cfg.run_checkpoints_dir / f"sslfss_alpnet_step{step}.pt"
            torch.save(method.checkpoint_state(), ckpt)
            print(f"  → checkpoint saved: {ckpt}")

    torch.save(method.checkpoint_state(), cfg.last_checkpoint_path)

    import numpy as np
    hist_path = cfg.run_results_dir / "sslfss_alpnet_historico.npz"
    cfg.run_results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(hist_path, loss_treino=np.array(historico["loss_treino"], dtype=np.float32))

    print(f"\nTraining complete.")
    print(f"Last checkpoint : {cfg.last_checkpoint_path}")
    print(f"Loss history    : {hist_path}")

    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    losses = np.array(historico["loss_treino"], dtype=np.float32)
    # Smooth with a 500-step rolling mean for readability
    window = min(500, len(losses))
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    steps_raw    = np.arange(1, len(losses) + 1)
    steps_smooth = np.arange(window, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps_raw,    losses, alpha=0.2, color="steelblue", linewidth=0.6, label="loss (raw)")
    ax.plot(steps_smooth, smooth, color="steelblue", linewidth=1.8, label=f"loss (smooth {window}-step)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("ALPNet training loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    chart_path = cfg.run_results_dir / "sslfss_alpnet_loss.png"
    plt.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Loss chart      : {chart_path}")


if __name__ == "__main__":
    main()
