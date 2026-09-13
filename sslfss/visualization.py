from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from sslfss.plot_style import (
    apply_style, FIG_WIDTH,
    gt_rgba as _gt_rgba_mod,
    pred_rgba as _pred_rgba_mod,
    error_rgba as _error_rgba_mod,
    legend_patches as _legend_patches_mod,
    save_fig,
)


def plot_training_curves(
    historico: dict,
    melhor_dice: float,
    log_intervalo: int,
    results_dir: Path | str,
    slic_baseline: float | None = None,
    meta_batch_size: int = 1,
) -> None:
    """Save a two-panel figure: loss curves + validation Dice vs SLIC baseline."""
    apply_style()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    x_train    = [i * meta_batch_size for i in range(len(historico["loss_treino"]))]
    epocas_val = [i * log_intervalo * meta_batch_size for i in range(1, len(historico["loss_val"]) + 1)]

    from sslfss.plot_style import PALETTE
    c_train = PALETTE["maml"]
    c_val   = PALETTE["anil"]
    c_dice  = PALETTE["panet"]
    c_base  = "#777777"

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 0.52))

    axes[0].plot(x_train, historico["loss_treino"], alpha=0.4, color=c_train, label="Training — pseudo-label")
    axes[0].plot(epocas_val, historico["loss_val"], color=c_val, linewidth=1.8, label="Validation — real GT")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves\n(train\u202f=\u202fpseudo-sparse | val\u202f=\u202freal GT)")
    axes[0].legend(loc="upper right", framealpha=0.85, edgecolor="#cccccc")

    axes[1].plot(epocas_val, historico["dice_val"], color=c_dice, linewidth=1.8)
    if slic_baseline is not None:
        axes[1].axhline(y=slic_baseline, color=c_base, linestyle="--",
                        label=f"SS baseline ({slic_baseline:.3f})")
    axes[1].axhline(y=melhor_dice, color=c_dice, linestyle="--",
                    label=f"Best SSL-Sparse-FSS ({melhor_dice:.3f})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Dice score")
    axes[1].set_title("Validation Dice vs. SS baseline\n(query\u202f=\u202freal GT, support\u202f=\u202fsparse real GT)")
    axes[1].legend(loc="lower right", framealpha=0.85, edgecolor="#cccccc")

    out = results_dir / "training_curves.svg"
    save_fig(fig, out)
    print(f"Training curves saved: {out}")


def plot_predictions(
    maml: Any,
    todas_imgs_np: np.ndarray,
    msks_reais_np: np.ndarray,
    idx_teste: np.ndarray,
    cfg: Any,
    device: Any,
    n_rows: int = 6,
    dataset_name: str = "",
    results_dir=None,
) -> None:
    """Generate a grid showing K-shot predictions evaluated on unseen query slices."""
    from sslfss.data.dataset import _zscore
    from sslfss.data.sparsify import aplicar_esparsidade
    from sslfss.losses import loss_combinada
    from sslfss.metrics import dice_score, miou_score

    results_dir = Path(results_dir) if results_dir is not None else Path(cfg.run_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    _gt_rgba    = _gt_rgba_mod
    _pred_rgba  = _pred_rgba_mod
    _error_rgba = _error_rgba_mod

    k      = getattr(cfg, "k_shot", 1)
    n_cols = 4
    col_w  = 2.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * col_w, n_rows * col_w),
        gridspec_kw={"wspace": 0.06, "hspace": 0.45},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title = f"M.I.R.A. — {k}-shot test predictions  [{dataset_name}]" if dataset_name \
        else f"M.I.R.A. — {k}-shot test predictions"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

    col_titles = ["Query image", "Ground truth", "SSL-Sparse-FSS prediction", "Error map"]
    for col, lbl in enumerate(col_titles):
        axes[0, col].set_title(lbl, fontsize=8, fontweight="bold", pad=4)

    idx_list = list(idx_teste)

    maml.eval()
    for row in range(n_rows):
        # Sample query + K support slices (disjoint).
        q_idx    = int(np.random.choice(idx_list))
        other    = [i for i in idx_list if i != q_idx]
        sup_idxs = [int(x) for x in np.random.choice(other, size=k, replace=len(other) < k)]

        img_np = todas_imgs_np[q_idx]
        real   = msks_reais_np[q_idx]

        # Build K support tensors (not displayed; used only for adaptation).
        sup_imgs_t, sup_msks_t = [], []
        for s_idx in sup_idxs:
            s_img = todas_imgs_np[s_idx]
            s_msk = msks_reais_np[s_idx]
            if cfg.eval_mode == "few_shot":
                sp = aplicar_esparsidade(s_msk)
            else:
                from sslfss.data.propose_msk import propor_mascara
                pseudo = propor_mascara(s_img)
                sp = aplicar_esparsidade(pseudo) if pseudo.max() > 0 else pseudo
            sup_imgs_t.append(
                torch.tensor(_zscore(s_img), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            )
            sup_msks_t.append(
                torch.tensor(sp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            )

        q_img_t = torch.tensor(_zscore(img_np), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # K-shot inner-loop adaptation on support slices only.
        viz_model = maml.clone()
        for _ in range(cfg.inner_steps):
            sup_loss = sum(
                loss_combinada(viz_model(si), sm) for si, sm in zip(sup_imgs_t, sup_msks_t)
            ) / k
            viz_model.adapt(sup_loss)

        with torch.no_grad():
            pred_bin = (torch.sigmoid(viz_model(q_img_t)) > 0.5).float().squeeze().cpu().numpy()

        d_ours    = dice_score(pred_bin, real)
        miou_ours = miou_score(pred_bin, real)

        axes[row, 0].set_ylabel(f"q#{q_idx}  sup{sup_idxs}", fontsize=6, labelpad=4)

        # Col 0 — Query image
        axes[row, 0].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        axes[row, 0].set_title(f"GT fg: {int(real.sum())} px", fontsize=7)
        axes[row, 0].axis("off")

        # Col 1 — Ground truth overlay
        axes[row, 1].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        axes[row, 1].imshow(_gt_rgba(real), interpolation="nearest")
        axes[row, 1].axis("off")

        # Col 2 — SSL-Sparse-FSS prediction overlay
        axes[row, 2].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        axes[row, 2].imshow(_pred_rgba(pred_bin), interpolation="nearest")
        axes[row, 2].set_title(f"Dice {d_ours:.3f}  mIoU {miou_ours:.3f}", fontsize=7)
        axes[row, 2].axis("off")

        # Col 3 — Error map
        axes[row, 3].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        axes[row, 3].imshow(_error_rgba(pred_bin, real), interpolation="nearest")
        axes[row, 3].axis("off")

    import matplotlib.patches as mpatches
    fig.legend(
        handles=_legend_patches_mod("SSL-Sparse-FSS prediction"),
        loc="lower center", ncol=5, fontsize=8, framealpha=0.9,
        edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.04),
    )

    safe_name = dataset_name.replace("/", "_").replace(" ", "_") if dataset_name else "test"
    out = results_dir / f"predictions_{safe_name}.svg"
    save_fig(fig, out)
    print(f"  Prediction grid saved: {out}")


def plot_metric_boxplots(
    per_dataset_results: list[dict],
    agg_ours_dice: list[float],
    agg_ss_dice: list[float],
    agg_ours_miou: list[float],
    agg_ss_miou: list[float],
    results_dir,
) -> None:
    """Save grouped box plots comparing SSL-Sparse-FSS vs SS baseline for Dice and mIoU."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    from sslfss.plot_style import PALETTE, style_boxplot
    import matplotlib.patches as mpatches

    groups = [(r["name"].replace("_", " "), r["dices_ours"], r["dices_ss"],
               r["mious_ours"], r["mious_ss"]) for r in per_dataset_results]
    if len(groups) > 1:
        groups.append(("Aggregate", agg_ours_dice, agg_ss_dice,
                        agg_ours_miou, agg_ss_miou))

    n_groups   = len(groups)
    color_ours = PALETTE["maml"]
    color_ss   = "#777777"
    width      = 0.35
    x          = np.arange(n_groups)

    fig, axes = plt.subplots(2, 1, figsize=(max(FIG_WIDTH, n_groups * 1.6), FIG_WIDTH * 1.1),
                             gridspec_kw={"hspace": 0.50})

    for ax_idx, (metric_label, ours_key, ss_key) in enumerate([
        ("Dice  (vs real GT)", 1, 2),
        ("mIoU  (vs real GT)", 3, 4),
    ]):
        ax = axes[ax_idx]
        bp_ours = ax.boxplot(
            [g[ours_key] for g in groups],
            positions=x - width / 2,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=color_ours, alpha=0.75),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=color_ours),
            capprops=dict(color=color_ours),
            flierprops=dict(marker="o", markerfacecolor=color_ours,
                            markersize=3, alpha=0.5, linestyle="none"),
        )
        bp_ss = ax.boxplot(
            [g[ss_key] for g in groups],
            positions=x + width / 2,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=color_ss, alpha=0.75),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=color_ss),
            capprops=dict(color=color_ss),
            flierprops=dict(marker="o", markerfacecolor=color_ss,
                            markersize=3, alpha=0.5, linestyle="none"),
        )

        # Overlay mean markers
        for i, g in enumerate(groups):
            ours_vals = g[ours_key]
            ss_vals   = g[ss_key]
            ax.plot(i - width / 2, float(np.mean(ours_vals)),
                    "D", color="white", markeredgecolor=color_ours,
                    markeredgewidth=1.5, markersize=6, zorder=5)
            ax.plot(i + width / 2, float(np.mean(ss_vals)),
                    "D", color="white", markeredgecolor=color_ss,
                    markeredgewidth=1.5, markersize=6, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="#cccccc", linestyle=":", linewidth=0.7)

        # Shade aggregate column
        if len(groups) > 1:
            ax.axvspan(n_groups - 1 - 0.5, n_groups - 0.5,
                       color="lightyellow", alpha=0.5, zorder=0)

        ax.legend(
            handles=[
                mpatches.Patch(color=color_ours, alpha=0.75, label="SSL-Sparse-FSS"),
                mpatches.Patch(color=color_ss,   alpha=0.75, label="SS baseline"),
            ],
            fontsize=8, loc="lower right", framealpha=0.85, edgecolor="#cccccc",
        )

    fig.suptitle(
        "SSL-Sparse-FSS vs. SS baseline  —  Dice & mIoU against real GT\n"
        "(diamond\u202f=\u202fmean, coloured line\u202f=\u202fmedian, \u2009$K$-shot evaluation)",
    )

    out = results_dir / "boxplots_metrics.svg"
    save_fig(fig, out)
    print(f"  Box plots saved: {out}")


def plot_best_predictions(
    maml: Any,
    todas_imgs_np: np.ndarray,
    msks_reais_np: np.ndarray,
    episode_records: list[int],
    dices_ours: list[float],
    mious_ours: list[float],
    cfg: Any,
    device: Any,
    dataset_name: str = "",
    top_n: int = 6,
    results_dir=None,
) -> None:
    """Save a grid of the best query-only episodes ranked by Dice and mIoU."""
    from sslfss.data.dataset import _zscore
    from sslfss.data.sparsify import aplicar_esparsidade
    from sslfss.losses import loss_combinada
    from sslfss.metrics import dice_score, miou_score

    results_dir = Path(results_dir) if results_dir is not None else Path(cfg.run_results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    _gt_rgba    = _gt_rgba_mod
    _pred_rgba  = _pred_rgba_mod
    _error_rgba = _error_rgba_mod

    def _run_episode(record):
        """Re-run one K-shot episode; return (pred_bin, dice, miou)."""
        k_viz = getattr(cfg, "k_shot", 1)
        support_idxs, q_idx = record

        sup_imgs_t, sup_msks_t = [], []
        for s_idx in support_idxs:
            s_img = todas_imgs_np[int(s_idx)]
            s_msk = msks_reais_np[int(s_idx)]
            if cfg.eval_mode == "few_shot":
                sp = aplicar_esparsidade(s_msk)
            else:
                from sslfss.data.propose_msk import propor_mascara
                pseudo = propor_mascara(s_img)
                sp = aplicar_esparsidade(pseudo) if pseudo.max() > 0 else pseudo
            sup_imgs_t.append(
                torch.tensor(_zscore(s_img), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            )
            sup_msks_t.append(
                torch.tensor(sp, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            )

        q_img_t = torch.tensor(
            _zscore(todas_imgs_np[q_idx]), dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(device)
        real = msks_reais_np[q_idx]

        m = maml.clone()
        for _ in range(cfg.inner_steps):
            sup_loss = sum(
                loss_combinada(m(si), sm) for si, sm in zip(sup_imgs_t, sup_msks_t)
            ) / k_viz
            m.adapt(sup_loss)
        with torch.no_grad():
            pred_bin = (torch.sigmoid(m(q_img_t)) > 0.5).float().squeeze().cpu().numpy()
        return pred_bin, dice_score(pred_bin, real), miou_score(pred_bin, real)

    def _top_indices(scores, n):
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    top_dice_idx = _top_indices(dices_ours, top_n)
    top_miou_idx = _top_indices(mious_ours, top_n)

    n_rows = 2 * top_n
    n_cols = 4
    col_w  = 2.8
    k      = getattr(cfg, "k_shot", 1)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * col_w, n_rows * col_w),
        gridspec_kw={"wspace": 0.06, "hspace": 0.50},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title = f"M.I.R.A. — Best {k}-shot predictions  [{dataset_name}]" if dataset_name \
        else f"M.I.R.A. — Best {k}-shot predictions"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.005)

    col_titles     = ["Query image", "Ground truth", "SSL-Sparse-FSS prediction", "Error map"]
    section_labels = {0: f"▲ Top-{top_n} by Dice", top_n: f"▲ Top-{top_n} by mIoU"}

    maml.eval()
    for section, ranked_idx in [(0, top_dice_idx), (top_n, top_miou_idx)]:
        hdr_color = "#1a5a8a" if section == 0 else "#6b2d8b"
        for local_row, ep_i in enumerate(ranked_idx):
            row = section + local_row

            if local_row == 0:
                for col, lbl in enumerate(col_titles):
                    prefix = section_labels[section] + "\n" if col == 0 else ""
                    axes[row, col].set_title(
                        prefix + lbl, fontsize=8, fontweight="bold", pad=4, color=hdr_color,
                    )
            else:
                for col, lbl in enumerate(col_titles):
                    axes[row, col].set_title(lbl, fontsize=7, pad=3)

            support_idxs, q_idx = episode_records[ep_i]
            pred_bin, d_val, m_val = _run_episode((support_idxs, q_idx))
            img_np = todas_imgs_np[q_idx]
            real   = msks_reais_np[q_idx]

            axes[row, 0].set_ylabel(f"q#{q_idx}  sup{support_idxs}", fontsize=6, labelpad=3)

            axes[row, 0].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 0].set_title(f"GT fg: {int(real.sum())} px", fontsize=7)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 1].imshow(_gt_rgba(real), interpolation="nearest")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 2].imshow(_pred_rgba(pred_bin), interpolation="nearest")
            axes[row, 2].set_title(f"Dice {d_val:.3f}  mIoU {m_val:.3f}", fontsize=7)
            axes[row, 2].axis("off")

            axes[row, 3].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 3].imshow(_error_rgba(pred_bin, real), interpolation="nearest")
            axes[row, 3].axis("off")

    # Separator line between sections
    if top_n > 0 and n_rows > top_n:
        y_sep = 1.0 - top_n / n_rows
        fig.add_artist(plt.Line2D([0.01, 0.99], [y_sep, y_sep],
                                  transform=fig.transFigure,
                                  color="#999999", linewidth=1.5, linestyle="--"))

    import matplotlib.patches as mpatches
    fig.legend(handles=_legend_patches_mod("SSL-Sparse-FSS prediction"), loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.02))

    safe_name = dataset_name.replace("/", "_").replace(" ", "_") if dataset_name else "test"
    out = results_dir / f"best_predictions_{safe_name}.svg"
    save_fig(fig, out)
    print(f"  Best-predictions grid saved: {out}")


def plot_predictions_panet(
    model: Any,
    todas_imgs_np: np.ndarray,
    msks_reais_np: np.ndarray,
    idx_teste: np.ndarray,
    cfg: Any,
    device: Any,
    n_rows: int = 6,
    dataset_name: str = "",
    results_dir: Path | str | None = None,
) -> None:
    """Generate a grid showing K-shot PANet predictions on unseen query slices."""
    import torch.nn.functional as F
    from sslfss.data.dataset import _zscore
    from sslfss.data.propose_msk import propor_mascara
    from sslfss.data.sparsify import aplicar_esparsidade
    from sslfss.metrics import dice_score, miou_score

    out_dir = Path(results_dir) if results_dir is not None else Path(cfg.run_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    _gt_rgba    = _gt_rgba_mod
    _pred_rgba  = _pred_rgba_mod
    _error_rgba = _error_rgba_mod

    def _build_sup(s_idx):
        s_img = todas_imgs_np[int(s_idx)]
        if cfg.eval_mode == "few_shot":
            sp = aplicar_esparsidade(msks_reais_np[int(s_idx)])
        else:
            pseudo = propor_mascara(s_img)
            sp = aplicar_esparsidade(pseudo) if pseudo.max() > 0 else pseudo
        si  = torch.tensor(_zscore(s_img), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        fg  = torch.tensor((sp == 1).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        bg  = torch.tensor((sp == 0).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return si, fg, bg

    def _panet_pred(sup_idxs, q_img_t):
        """Run PANet forward for one episode; return binary numpy prediction."""
        img_size = q_img_t.shape[-2:]
        fts_q = model.encoder(q_img_t)
        if len(sup_idxs) == 1:
            si, fg, bg = _build_sup(sup_idxs[0])
            pred, _ = model(si, fg, bg, q_img_t)
        else:
            fg_protos, bg_protos = [], []
            for s_idx in sup_idxs:
                si, fg, bg = _build_sup(s_idx)
                fts_s = model.encoder(si)
                fg_protos.append(model.getFeatures(fts_s, fg))
                bg_protos.append(model.getFeatures(fts_s, bg))
            fg_proto = sum(fg_protos) / len(fg_protos)
            bg_proto = sum(bg_protos) / len(bg_protos)
            dist = torch.stack([model.calDist(fts_q, bg_proto),
                                 model.calDist(fts_q, fg_proto)], dim=1)
            pred = F.interpolate(dist, size=img_size, mode="bilinear", align_corners=False)
        return pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.float32)

    k      = getattr(cfg, "k_shot", 1)
    n_cols = 4
    col_w  = 2.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * col_w, n_rows * col_w),
        gridspec_kw={"wspace": 0.06, "hspace": 0.45},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title = f"PANet — {k}-shot test predictions  [{dataset_name}]" if dataset_name \
        else f"PANet — {k}-shot test predictions"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

    col_titles = ["Query image", "Ground truth", "PANet prediction", "Error map"]
    for col, lbl in enumerate(col_titles):
        axes[0, col].set_title(lbl, fontsize=8, fontweight="bold", pad=4)

    idx_list = list(idx_teste)
    model.eval()
    with torch.no_grad():
        for row in range(n_rows):
            q_idx    = int(np.random.choice(idx_list))
            other    = [i for i in idx_list if i != q_idx]
            sup_idxs = [int(x) for x in np.random.choice(other, size=k, replace=len(other) < k)]

            img_np  = todas_imgs_np[q_idx]
            real    = msks_reais_np[q_idx]
            q_img_t = torch.tensor(_zscore(img_np), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            pred_bin  = _panet_pred(sup_idxs, q_img_t)
            d_ours    = dice_score(pred_bin, real)
            miou_ours = miou_score(pred_bin, real)

            axes[row, 0].set_ylabel(f"q#{q_idx}  sup{sup_idxs}", fontsize=6, labelpad=4)
            axes[row, 0].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 0].set_title(f"GT fg: {int(real.sum())} px", fontsize=7)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 1].imshow(_gt_rgba(real), interpolation="nearest")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 2].imshow(_pred_rgba(pred_bin), interpolation="nearest")
            axes[row, 2].set_title(f"Dice {d_ours:.3f}  mIoU {miou_ours:.3f}", fontsize=7)
            axes[row, 2].axis("off")

            axes[row, 3].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 3].imshow(_error_rgba(pred_bin, real), interpolation="nearest")
            axes[row, 3].axis("off")

    import matplotlib.patches as mpatches
    fig.legend(handles=_legend_patches_mod("PANet prediction"), loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.04))

    safe_name = dataset_name.replace("/", "_").replace(" ", "_") if dataset_name else "test"
    out = out_dir / f"predictions_panet_{safe_name}.svg"
    save_fig(fig, out)
    print(f"  PANet prediction grid saved: {out}")


def plot_best_predictions_panet(
    model: Any,
    todas_imgs_np: np.ndarray,
    msks_reais_np: np.ndarray,
    episode_records: list,
    dices_panet: list[float],
    mious_panet: list[float],
    cfg: Any,
    device: Any,
    dataset_name: str = "",
    top_n: int = 6,
    results_dir: Path | str | None = None,
) -> None:
    """Save a grid of the best PANet episodes ranked by Dice and mIoU."""
    import torch.nn.functional as F
    from sslfss.data.dataset import _zscore
    from sslfss.data.propose_msk import propor_mascara
    from sslfss.data.sparsify import aplicar_esparsidade
    from sslfss.metrics import dice_score, miou_score

    out_dir = Path(results_dir) if results_dir is not None else Path(cfg.run_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    _gt_rgba    = _gt_rgba_mod
    _pred_rgba  = _pred_rgba_mod
    _error_rgba = _error_rgba_mod

    def _build_sup(s_idx):
        s_img = todas_imgs_np[int(s_idx)]
        if cfg.eval_mode == "few_shot":
            sp = aplicar_esparsidade(msks_reais_np[int(s_idx)])
        else:
            pseudo = propor_mascara(s_img)
            sp = aplicar_esparsidade(pseudo) if pseudo.max() > 0 else pseudo
        si = torch.tensor(_zscore(s_img), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        fg = torch.tensor((sp == 1).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        bg = torch.tensor((sp == 0).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return si, fg, bg

    def _run_episode_panet(record):
        support_idxs, q_idx = record
        img_np  = todas_imgs_np[int(q_idx)]
        real    = msks_reais_np[int(q_idx)]
        q_img_t = torch.tensor(_zscore(img_np), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        img_size = q_img_t.shape[-2:]
        if len(support_idxs) == 1:
            si, fg, bg = _build_sup(support_idxs[0])
            pred, _ = model(si, fg, bg, q_img_t)
        else:
            fg_protos, bg_protos = [], []
            for s_idx in support_idxs:
                si, fg, bg = _build_sup(s_idx)
                fts_s = model.encoder(si)
                fg_protos.append(model.getFeatures(fts_s, fg))
                bg_protos.append(model.getFeatures(fts_s, bg))
            fg_proto = sum(fg_protos) / len(fg_protos)
            bg_proto = sum(bg_protos) / len(bg_protos)
            fts_q = model.encoder(q_img_t)
            dist  = torch.stack([model.calDist(fts_q, bg_proto),
                                  model.calDist(fts_q, fg_proto)], dim=1)
            pred  = F.interpolate(dist, size=img_size, mode="bilinear", align_corners=False)
        pred_bin = pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.float32)
        return pred_bin, dice_score(pred_bin, real), miou_score(pred_bin, real)

    def _top_indices(scores, n):
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    top_dice_idx = _top_indices(dices_panet, top_n)
    top_miou_idx = _top_indices(mious_panet, top_n)

    n_rows = 2 * top_n
    n_cols = 4
    col_w  = 2.8
    k      = getattr(cfg, "k_shot", 1)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * col_w, n_rows * col_w),
        gridspec_kw={"wspace": 0.06, "hspace": 0.50},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title = f"PANet — Best {k}-shot predictions  [{dataset_name}]" if dataset_name \
        else f"PANet — Best {k}-shot predictions"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.005)

    col_titles     = ["Query image", "Ground truth", "PANet prediction", "Error map"]
    section_labels = {0: f"▲ Top-{top_n} by Dice", top_n: f"▲ Top-{top_n} by mIoU"}

    model.eval()
    with torch.no_grad():
        for section, ranked_idx in [(0, top_dice_idx), (top_n, top_miou_idx)]:
            hdr_color = "#1a5a8a" if section == 0 else "#6b2d8b"
            for local_row, ep_i in enumerate(ranked_idx):
                row = section + local_row

                if local_row == 0:
                    for col, lbl in enumerate(col_titles):
                        prefix = section_labels[section] + "\n" if col == 0 else ""
                        axes[row, col].set_title(
                            prefix + lbl, fontsize=8, fontweight="bold", pad=4, color=hdr_color,
                        )
                else:
                    for col, lbl in enumerate(col_titles):
                        axes[row, col].set_title(lbl, fontsize=7, pad=3)

                support_idxs, q_idx = episode_records[ep_i]
                pred_bin, d_val, m_val = _run_episode_panet((support_idxs, q_idx))
                img_np = todas_imgs_np[int(q_idx)]
                real   = msks_reais_np[int(q_idx)]

                axes[row, 0].set_ylabel(f"q#{q_idx}  sup{support_idxs}", fontsize=6, labelpad=3)

                axes[row, 0].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
                axes[row, 0].set_title(f"GT fg: {int(real.sum())} px", fontsize=7)
                axes[row, 0].axis("off")

                axes[row, 1].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
                axes[row, 1].imshow(_gt_rgba(real), interpolation="nearest")
                axes[row, 1].axis("off")

                axes[row, 2].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
                axes[row, 2].imshow(_pred_rgba(pred_bin), interpolation="nearest")
                axes[row, 2].set_title(f"Dice {d_val:.3f}  mIoU {m_val:.3f}", fontsize=7)
                axes[row, 2].axis("off")

                axes[row, 3].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
                axes[row, 3].imshow(_error_rgba(pred_bin, real), interpolation="nearest")
                axes[row, 3].axis("off")

    if top_n > 0 and n_rows > top_n:
        y_sep = 1.0 - top_n / n_rows
        fig.add_artist(plt.Line2D([0.01, 0.99], [y_sep, y_sep],
                                  transform=fig.transFigure,
                                  color="#999999", linewidth=1.5, linestyle="--"))

    import matplotlib.patches as mpatches
    fig.legend(handles=_legend_patches_mod("PANet prediction"), loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.02))

    safe_name = dataset_name.replace("/", "_").replace(" ", "_") if dataset_name else "test"
    out = out_dir / f"best_predictions_panet_{safe_name}.svg"
    save_fig(fig, out)
    print(f"  PANet best-predictions grid saved: {out}")


def plot_predictions_r2d2(
    model: Any,
    todas_imgs_np: np.ndarray,
    msks_reais_np: np.ndarray,
    idx_teste: np.ndarray,
    cfg: Any,
    device: Any,
    n_rows: int = 6,
    dataset_name: str = "",
    results_dir: Path | str | None = None,
) -> None:
    """Generate a grid showing K-shot R2D2 predictions on unseen query slices."""
    import torch.nn.functional as F
    from sslfss.data.dataset import _zscore
    from sslfss.data.propose_msk import propor_mascara
    from sslfss.data.sparsify import aplicar_esparsidade
    from sslfss.metrics import dice_score, miou_score

    out_dir = Path(results_dir) if results_dir is not None else Path(cfg.run_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    _gt_rgba    = _gt_rgba_mod
    _pred_rgba  = _pred_rgba_mod
    _error_rgba = _error_rgba_mod

    def _build_sup(s_idx):
        s_img = todas_imgs_np[int(s_idx)]
        if cfg.eval_mode == "few_shot":
            sp = aplicar_esparsidade(msks_reais_np[int(s_idx)])
        else:
            pseudo = propor_mascara(s_img)
            sp = aplicar_esparsidade(pseudo) if pseudo.max() > 0 else pseudo
        si  = torch.tensor(_zscore(s_img), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        fg  = torch.tensor((sp == 1).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        bg  = torch.tensor((sp == 0).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return si, fg, bg

    def _r2d2_pred(sup_idxs, q_img_t):
        if len(sup_idxs) == 1:
            si, fg, bg = _build_sup(sup_idxs[0])
            pred, _ = model(si, fg, bg, q_img_t)
        else:
            sup_imgs = []; fg_masks = []; bg_masks = []
            for s_idx in sup_idxs:
                si, fg, bg = _build_sup(s_idx)
                sup_imgs.append(si); fg_masks.append(fg); bg_masks.append(bg)
            sup_imgs_t = torch.cat(sup_imgs, dim=0)
            fg_masks_t = torch.cat(fg_masks, dim=0)
            bg_masks_t = torch.cat(bg_masks, dim=0)
            pred, _ = model.forward_kshot(sup_imgs_t, fg_masks_t, bg_masks_t, q_img_t)
        img_size = q_img_t.shape[-2:]
        pred = F.interpolate(pred, size=img_size, mode="bilinear", align_corners=False)
        return pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.float32)

    k      = getattr(cfg, "k_shot", 1)
    n_cols = 4
    col_w  = 2.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * col_w, n_rows * col_w),
        gridspec_kw={"wspace": 0.06, "hspace": 0.45},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title = f"R2D2 — {k}-shot test predictions  [{dataset_name}]" if dataset_name \
        else f"R2D2 — {k}-shot test predictions"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)

    col_titles = ["Query image", "Ground truth", "R2D2 prediction", "Error map"]
    for col, lbl in enumerate(col_titles):
        axes[0, col].set_title(lbl, fontsize=8, fontweight="bold", pad=4)

    idx_list = list(idx_teste)
    model.eval()
    with torch.no_grad():
        for row in range(n_rows):
            q_idx    = int(np.random.choice(idx_list))
            other    = [i for i in idx_list if i != q_idx]
            sup_idxs = [int(x) for x in np.random.choice(other, size=k, replace=len(other) < k)]

            img_np  = todas_imgs_np[q_idx]
            real    = msks_reais_np[q_idx]
            q_img_t = torch.tensor(_zscore(img_np), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            pred_bin  = _r2d2_pred(sup_idxs, q_img_t)
            d_ours    = dice_score(pred_bin, real)
            miou_ours = miou_score(pred_bin, real)

            axes[row, 0].set_ylabel(f"q#{q_idx}  sup{sup_idxs}", fontsize=6, labelpad=4)
            axes[row, 0].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 0].set_title(f"GT fg: {int(real.sum())} px", fontsize=7)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 1].imshow(_gt_rgba(real), interpolation="nearest")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 2].imshow(_pred_rgba(pred_bin), interpolation="nearest")
            axes[row, 2].set_title(f"Dice {d_ours:.3f}  mIoU {miou_ours:.3f}", fontsize=7)
            axes[row, 2].axis("off")

            axes[row, 3].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
            axes[row, 3].imshow(_error_rgba(pred_bin, real), interpolation="nearest")
            axes[row, 3].axis("off")

    import matplotlib.patches as mpatches
    fig.legend(handles=_legend_patches_mod("R2D2 prediction"), loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.04))

    safe_name = dataset_name.replace("/", "_").replace(" ", "_") if dataset_name else "test"
    out = out_dir / f"predictions_r2d2_{safe_name}.svg"
    save_fig(fig, out)
    print(f"  R2D2 prediction grid saved: {out}")


def plot_best_predictions_r2d2(
    model: Any,
    todas_imgs_np: np.ndarray,
    msks_reais_np: np.ndarray,
    episode_records: list,
    dices_r2d2: list,
    mious_r2d2: list,
    cfg: Any,
    device: Any,
    dataset_name: str = "",
    top_n: int = 6,
    results_dir: Path | str | None = None,
) -> None:
    """Save a grid of the best R2D2 episodes ranked by Dice and mIoU."""
    import torch.nn.functional as F
    from sslfss.data.dataset import _zscore
    from sslfss.data.propose_msk import propor_mascara
    from sslfss.data.sparsify import aplicar_esparsidade
    from sslfss.metrics import dice_score, miou_score

    out_dir = Path(results_dir) if results_dir is not None else Path(cfg.run_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    _gt_rgba    = _gt_rgba_mod
    _pred_rgba  = _pred_rgba_mod
    _error_rgba = _error_rgba_mod

    def _build_sup(s_idx):
        s_img = todas_imgs_np[int(s_idx)]
        if cfg.eval_mode == "few_shot":
            sp = aplicar_esparsidade(msks_reais_np[int(s_idx)])
        else:
            pseudo = propor_mascara(s_img)
            sp = aplicar_esparsidade(pseudo) if pseudo.max() > 0 else pseudo
        si = torch.tensor(_zscore(s_img), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        fg = torch.tensor((sp == 1).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        bg = torch.tensor((sp == 0).astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return si, fg, bg

    def _run_episode_r2d2(record):
        support_idxs, q_idx = record
        img_np  = todas_imgs_np[int(q_idx)]
        real    = msks_reais_np[int(q_idx)]
        q_img_t = torch.tensor(_zscore(img_np), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        img_size = q_img_t.shape[-2:]
        if len(support_idxs) == 1:
            si, fg, bg = _build_sup(support_idxs[0])
            pred, _ = model(si, fg, bg, q_img_t)
        else:
            sup_imgs = []; fg_masks = []; bg_masks = []
            for s_idx in support_idxs:
                si, fg, bg = _build_sup(s_idx)
                sup_imgs.append(si); fg_masks.append(fg); bg_masks.append(bg)
            sup_imgs_t = torch.cat(sup_imgs, dim=0)
            fg_masks_t = torch.cat(fg_masks, dim=0)
            bg_masks_t = torch.cat(bg_masks, dim=0)
            pred, _ = model.forward_kshot(sup_imgs_t, fg_masks_t, bg_masks_t, q_img_t)
        pred = F.interpolate(pred, size=img_size, mode="bilinear", align_corners=False)
        pred_bin = pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.float32)
        return pred_bin, dice_score(pred_bin, real), miou_score(pred_bin, real)

    def _top_indices(scores, n):
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

    top_dice_idx = _top_indices(dices_r2d2, top_n)
    top_miou_idx = _top_indices(mious_r2d2, top_n)

    n_rows = 2 * top_n
    n_cols = 4
    col_w  = 2.8
    k      = getattr(cfg, "k_shot", 1)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * col_w, n_rows * col_w),
        gridspec_kw={"wspace": 0.06, "hspace": 0.50},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    title = f"R2D2 — Best {k}-shot predictions  [{dataset_name}]" if dataset_name \
        else f"R2D2 — Best {k}-shot predictions"
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.005)

    col_titles     = ["Query image", "Ground truth", "R2D2 prediction", "Error map"]
    section_labels = {0: f"▲ Top-{top_n} by Dice", top_n: f"▲ Top-{top_n} by mIoU"}

    model.eval()
    with torch.no_grad():
        for section, ranked_idx in [(0, top_dice_idx), (top_n, top_miou_idx)]:
            hdr_color = "#1a5a8a" if section == 0 else "#6b2d8b"
            for local_row, ep_i in enumerate(ranked_idx):
                row = section + local_row

                if local_row == 0:
                    for col, lbl in enumerate(col_titles):
                        prefix = section_labels[section] + "\n" if col == 0 else ""
                        axes[row, col].set_title(
                            prefix + lbl, fontsize=8, fontweight="bold", pad=4, color=hdr_color,
                        )
                else:
                    for col, lbl in enumerate(col_titles):
                        axes[row, col].set_title(lbl, fontsize=7, pad=3)

                support_idxs, q_idx = episode_records[ep_i]
                pred_bin, d_val, m_val = _run_episode_r2d2((support_idxs, q_idx))
                img_np = todas_imgs_np[int(q_idx)]
                real   = msks_reais_np[int(q_idx)]

                axes[row, 0].set_ylabel(f"q#{q_idx}  sup{support_idxs}", fontsize=6, labelpad=3)

                axes[row, 0].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
                axes[row, 0].set_title(f"GT fg: {int(real.sum())} px", fontsize=7)
                axes[row, 0].axis("off")

                axes[row, 1].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
                axes[row, 1].imshow(_gt_rgba(real), interpolation="nearest")
                axes[row, 1].axis("off")

                axes[row, 2].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
                axes[row, 2].imshow(_pred_rgba(pred_bin), interpolation="nearest")
                axes[row, 2].set_title(f"Dice {d_val:.3f}  mIoU {m_val:.3f}", fontsize=7)
                axes[row, 2].axis("off")

                axes[row, 3].imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
                axes[row, 3].imshow(_error_rgba(pred_bin, real), interpolation="nearest")
                axes[row, 3].axis("off")

    if top_n > 0 and n_rows > top_n:
        y_sep = 1.0 - top_n / n_rows
        fig.add_artist(plt.Line2D([0.01, 0.99], [y_sep, y_sep],
                                  transform=fig.transFigure,
                                  color="#999999", linewidth=1.5, linestyle="--"))

    import matplotlib.patches as mpatches
    fig.legend(handles=_legend_patches_mod("R2D2 prediction"), loc="lower center", ncol=5,
               fontsize=8, framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.02))

    safe_name = dataset_name.replace("/", "_").replace(" ", "_") if dataset_name else "test"
    out = out_dir / f"best_predictions_r2d2_{safe_name}.svg"
    save_fig(fig, out)
    print(f"  R2D2 best-predictions grid saved: {out}")


def visualizar_comparacao(
    idx: int,
    imagens: list,
    pseudo_msks: list,
    mascaras_reais: list,
    results_dir: Path | str,
) -> None:
    """Save a 4-panel figure comparing the original image, SLIC superpixels, pseudo-label overlay, and ground-truth overlay for a single slice."""
    from skimage.segmentation import mark_boundaries, slic as _slic

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    from sslfss.plot_style import PALETTE
    img = imagens[idx]
    msk = mascaras_reais[idx]
    pm  = pseudo_msks[idx]
    segs = _slic(img, n_segments=50, compactness=0.10, channel_axis=None, start_label=0)

    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH * 0.30))
    fig.suptitle(f"Slice {idx} — pseudo-label validation")

    axes[0].imshow(img, cmap="gray");                     axes[0].set_title("Original image")
    axes[1].imshow(mark_boundaries(img, segs), cmap="gray"); axes[1].set_title("SLIC superpixels")

    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(pm, cmap="Oranges", alpha=0.5);        axes[2].set_title("Pseudo-label (SLIC)")

    axes[3].imshow(img, cmap="gray")
    axes[3].imshow(msk, cmap="Greens", alpha=0.5);        axes[3].set_title("Ground truth")

    for ax in axes:
        ax.axis("off")

    out = results_dir / f"pseudo_label_comparison_{idx}.svg"
    save_fig(fig, out)
    print(f"Figure saved: {out}")
