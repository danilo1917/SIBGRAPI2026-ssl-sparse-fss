import csv
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sslfss.metrics import dice_score, miou_score
from sslfss.plot_style import (
    FIG_WIDTH, PALETTE, MARKERS,
    apply_style, wong_colors, save_fig, style_boxplot,
    gt_rgba, pred_rgba, error_rgba, legend_patches,
)

_gt_rgba    = gt_rgba
_pred_rgba  = pred_rgba
_error_rgba = error_rgba


def plot_shots_line(
    k_values: list[int],
    k_dices: dict[int, list[float]],
    ss_mean: float,
    method_name: str,
    dataset_name: str,
    out_dir: Path,
) -> None:
    apply_style()
    color  = PALETTE.get(method_name, PALETTE["maml"])
    marker = MARKERS.get(method_name, "o")
    xs     = list(range(len(k_values)))
    means  = [np.mean(k_dices[k]) for k in k_values]
    stds   = [np.std(k_dices[k])  for k in k_values]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH * 0.62))
    ax.plot(xs, means, f"{marker}-", color=color, linewidth=1.8, markersize=6,
            label=method_name.upper())
    ax.fill_between(xs,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=color, alpha=0.15)
    ax.axhline(ss_mean, color="#777777", linestyle="--", linewidth=1.2,
               label=f"SS baseline ({ss_mean:.3f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel("Number of support images ($K$)")
    ax.set_ylabel("Dice score (vs. real GT)")
    ax.set_ylim(-0.05, 1.05)
    for x, mean in zip(xs, means):
        ax.annotate(f"{mean:.3f}", (x, mean),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7, color=color)
    ax.set_title(f"Few-shot study — {method_name.upper()}  [{dataset_name}]")
    ax.legend(loc="lower right", framealpha=0.85, edgecolor="#cccccc")
    out = out_dir / "shots_line.svg"
    save_fig(fig, out)


def plot_shots_boxes(
    k_values: list[int],
    k_dices: dict[int, list[float]],
    ss_mean: float,
    method_name: str,
    dataset_name: str,
    out_dir: Path,
) -> None:
    apply_style()
    n      = len(k_values)
    colors = wong_colors(n)

    fig, ax = plt.subplots(figsize=(max(FIG_WIDTH, n * 0.85), FIG_WIDTH * 0.62))
    bp = ax.boxplot(
        [k_dices[k] for k in k_values],
        labels=[str(k) for k in k_values],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
    )
    style_boxplot(bp, colors)
    for xi, k in enumerate(k_values, start=1):
        ax.plot(xi, np.mean(k_dices[k]), "D", color="white",
                markeredgecolor=colors[xi - 1], markeredgewidth=1.6,
                markersize=7, zorder=5)

    ax.axhline(ss_mean, color="#777777", linestyle="--", linewidth=1.2,
               label=f"SS baseline ({ss_mean:.3f})")
    ax.set_xlabel("Number of support images ($K$)")
    ax.set_ylabel("Dice score (vs. real GT)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", framealpha=0.85, edgecolor="#cccccc")
    ax.set_title(f"Few-shot study — {method_name.upper()}  [{dataset_name}]")
    out = out_dir / "shots_boxes.svg"
    save_fig(fig, out)


def plot_kshot_query_grid(
    method,
    imgs: np.ndarray,
    msks: np.ndarray,
    k_values: list[int],
    k_sample_episodes: dict,        # k → (sup_idxs, sup_sparses, q_idx)
    method_name: str,
    dataset_name: str,
    out_dir: Path,
) -> None:
    """One row per k: query image | GT | prediction | error map."""
    rows_to_plot = [k for k in k_values if k in k_sample_episodes]
    if not rows_to_plot:
        return

    n_rows = len(rows_to_plot)
    n_cols = 4
    col_w  = 2.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * col_w, n_rows * col_w),
        gridspec_kw={"wspace": 0.06, "hspace": 0.50},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    apply_style()
    fig.suptitle(
        f"{method_name.upper()} — Few-shot study predictions  [{dataset_name}]",
        y=1.01,
    )
    for col, lbl in enumerate(["Query image", "Ground truth",
                                f"{method_name.upper()} prediction", "Error map"]):
        axes[0, col].set_title(lbl, fontsize=8, fontweight="bold", pad=4)

    for row_i, k in enumerate(rows_to_plot):
        sup_idxs, sup_sparses, q_idx = k_sample_episodes[k]
        img_np   = imgs[q_idx]
        real     = msks[q_idx]
        sup_imgs = [imgs[i] for i in sup_idxs]
        sup_masks = [msks[i] for i in sup_idxs] if method_name == "alpnet" else sup_sparses
        pred     = method.predict(img_np, sup_imgs, sup_masks)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        d        = dice_score(pred, real)
        m        = miou_score(pred, real)

        axes[row_i, 0].set_ylabel(f"k={k}  q#{q_idx}", fontsize=7, labelpad=4)
        axes[row_i, 0].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        axes[row_i, 0].set_title(f"GT fg: {int(real.sum())} px", fontsize=7)
        axes[row_i, 0].axis("off")

        axes[row_i, 1].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        axes[row_i, 1].imshow(_gt_rgba(real), interpolation="nearest")
        axes[row_i, 1].axis("off")

        axes[row_i, 2].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        axes[row_i, 2].imshow(_pred_rgba(pred), interpolation="nearest")
        axes[row_i, 2].set_title(f"Dice {d:.3f}  mIoU {m:.3f}", fontsize=7)
        axes[row_i, 2].axis("off")

        axes[row_i, 3].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        axes[row_i, 3].imshow(_error_rgba(pred, real), interpolation="nearest")
        axes[row_i, 3].axis("off")

    fig.legend(handles=legend_patches(f"{method_name.upper()} prediction"),
               loc="lower center", ncol=5, fontsize=8,
               framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.04))
    out = out_dir / "kshot_query_grid.svg"
    save_fig(fig, out)


def save_csv(rows: list[dict], out_dir: Path) -> None:
    out = out_dir / "results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "query_idx", "trial", "dice", "miou"])
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out}")


def save_summary(
    k_values: list[int],
    k_dices: dict[int, list[float]],
    k_mious: dict[int, list[float]],
    ss_mean: float,
    method_name: str,
    dataset_name: str,
    out_dir: Path,
) -> None:
    W = 68
    lines = [
        "=" * W,
        f"  SHOTS STUDY — {method_name.upper()}  [{dataset_name}]",
        "=" * W,
        f"  {'K':>4}  {'N':>6}  {'Dice mean±std':>16}  {'mIoU mean±std':>16}  {'Δ vs SS':>9}",
        "  " + "-" * (W - 2),
    ]
    for k in k_values:
        d = np.array(k_dices[k])
        m = np.array(k_mious[k])
        lines.append(
            f"  {k:>4}  {len(d):>6}  "
            f"{d.mean():.3f} ± {d.std():.3f}     "
            f"{m.mean():.3f} ± {m.std():.3f}     "
            f"{d.mean()-ss_mean:>+.3f}"
        )
    lines += ["  " + "-" * (W - 2), f"  {'SS':>4}             {ss_mean:.3f}", "=" * W]
    text = "\n".join(lines)
    print(text)
    (out_dir / "summary.txt").write_text(text)
