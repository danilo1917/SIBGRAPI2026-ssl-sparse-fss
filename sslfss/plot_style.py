from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# 180 mm double column in inches (1 inch = 25.4 mm)
FIG_WIDTH  = 180 / 25.4          # ≈ 7.09 in
FIG_HEIGHT = FIG_WIDTH * 0.62    # ≈ 4.40 in  (golden-ratio-ish)

# Order: black, orange, sky-blue, bluish-green, yellow, blue, vermilion, pink
_WONG = [
    "#000000",  # 0 – Black
    "#E69F00",  # 1 – Orange
    "#56B4E9",  # 2 – Sky blue
    "#009E73",  # 3 – Bluish green
    "#F0E442",  # 4 – Yellow  (avoid on white; used for fills only)
    "#0072B2",  # 5 – Blue
    "#D55E00",  # 6 – Vermilion
    "#CC79A7",  # 7 – Reddish purple
]

PALETTE: dict[str, str] = {
    "maml":        "#0072B2",   # Blue
    "anil":        "#D55E00",   # Vermilion
    "panet":       "#009E73",   # Bluish green
    "r2d2":        "#56B4E9",   # Sky blue
    "unet_sparse": "#E69F00",   # Orange
    "unet_dense":  "#CC79A7",   # Reddish purple
    "alpnet":      "#000000",   # Black
    "ss":          "#777777",   # Medium grey (SS baseline dashed line)
}

# Single source of truth for human-readable labels, shared by every plotting
# script so legends, axis labels, and titles are typographically consistent.
DISPLAY_NAMES: dict[str, str] = {
    "maml":        "MAML",
    "anil":        "ANIL",
    "panet":       "PANet",
    "r2d2":        "R2D2",
    "alpnet":      "ALPNet",
    "unet_sparse": "U-Net (sparse)",
    "unet_dense":  "U-Net (dense)",
    "ss":          "SS baseline",
}

DATASET_NAMES: dict[str, str] = {
    "jsrt":      "JSRT",
    "panoramic": "Panoramic",
}


def method_label(key: str) -> str:
    """Return the publication label for a method key (falls back to UPPER-CASE)."""
    return DISPLAY_NAMES.get(key, key.upper())


def dataset_label(key: str) -> str:
    """Return the publication label for a dataset key (falls back to .title())."""
    return DATASET_NAMES.get(key, key.title())


# Using both shape and colour makes lines distinguishable without colour vision.
MARKERS: dict[str, str] = {
    "maml":        "o",
    "anil":        "s",
    "panet":       "^",
    "r2d2":        "D",
    "unet_sparse": "v",
    "unet_dense":  "P",
    "alpnet":      "*",
}

# Exclude yellow (#4) as it is illegible on white backgrounds.
WONG_CYCLE: list[str] = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#009E73",  # Bluish green
    "#56B4E9",  # Sky blue
    "#E69F00",  # Orange
    "#CC79A7",  # Reddish purple
    "#000000",  # Black
]


def wong_colors(n: int) -> list[str]:
    """Return *n* colors cycling through the Wong palette (no yellow)."""
    return [WONG_CYCLE[i % len(WONG_CYCLE)] for i in range(n)]


# TP = Bluish green, FP = Sky blue, FN = Orange
# These three are distinguishable even under the most common deficiencies.
_TP_RGBA  = [0.000, 0.619, 0.451, 0.80]   # #009E73 + alpha
_FP_RGBA  = [0.337, 0.706, 0.914, 0.80]   # #56B4E9 + alpha
_FN_RGBA  = [0.902, 0.624, 0.000, 0.80]   # #E69F00 + alpha

# GT overlay: bluish green, semi-transparent
_GT_RGBA  = [0.000, 0.619, 0.451, 0.55]   # #009E73

# Prediction overlay: blue, semi-transparent
_PRED_RGBA = [0.000, 0.447, 0.698, 0.60]  # #0072B2

# Support sparse-label overlay: vermilion, semi-transparent
_SUP_RGBA  = [0.839, 0.369, 0.000, 0.75]  # #D55E00


def gt_rgba(msk: np.ndarray) -> np.ndarray:
    """Green-channel RGBA overlay for the ground-truth mask."""
    rgba = np.zeros((*msk.shape, 4), dtype=np.float32)
    rgba[msk > 0.5] = _GT_RGBA
    return rgba


def pred_rgba(pred: np.ndarray) -> np.ndarray:
    """Blue-channel RGBA overlay for a binary prediction mask."""
    rgba = np.zeros((*pred.shape, 4), dtype=np.float32)
    rgba[pred > 0.5] = _PRED_RGBA
    return rgba


def error_rgba(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Three-class error map: TP (green) / FP (sky-blue) / FN (orange)."""
    rgba = np.zeros((*pred.shape, 4), dtype=np.float32)
    p, g = pred > 0.5, gt > 0.5
    rgba[p &  g] = _TP_RGBA
    rgba[p & ~g] = _FP_RGBA
    rgba[~p & g] = _FN_RGBA
    return rgba


def sup_sparse_rgba(msk: np.ndarray) -> np.ndarray:
    """Vermilion overlay for a sparse support annotation."""
    rgba = np.zeros((*msk.shape, 4), dtype=np.float32)
    rgba[msk > 0.5] = _SUP_RGBA
    return rgba


def legend_patches(method_label: str = "Prediction") -> list[mpatches.Patch]:
    """Return five standard legend patches for prediction / error-map figures."""
    return [
        mpatches.Patch(color="#009E73", alpha=0.55, label="Ground truth"),
        mpatches.Patch(color="#0072B2", alpha=0.60, label=method_label),
        mpatches.Patch(color="#009E73", alpha=0.80, label="TP"),
        mpatches.Patch(color="#56B4E9", alpha=0.80, label="FP"),
        mpatches.Patch(color="#E69F00", alpha=0.80, label="FN"),
    ]


def error_legend_patches() -> list[mpatches.Patch]:
    """Compact legend for error maps only (no GT / pred overlays shown)."""
    return [
        mpatches.Patch(color="#009E73", alpha=0.80, label="TP"),
        mpatches.Patch(color="#56B4E9", alpha=0.80, label="FP"),
        mpatches.Patch(color="#E69F00", alpha=0.80, label="FN"),
    ]


def sparsification_legend_patches(method_label: str = "Prediction") -> list[mpatches.Patch]:
    """Legend for sparsification grid (adds support-sparse-label entry)."""
    return legend_patches(method_label) + [
        mpatches.Patch(color="#D55E00", alpha=0.75, label="Sparse support label"),
    ]


def apply_style() -> None:
    """Apply publication-quality rcParams to the current Matplotlib session."""
    plt.rcParams.update({
        # Font
        "font.family":          "sans-serif",
        "font.sans-serif":      ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.titleweight":     "bold",
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      8,
        "figure.titlesize":     10,
        "figure.titleweight":   "bold",
        # Lines
        "lines.linewidth":      1.8,
        "lines.markersize":     5,
        "patch.linewidth":      0.8,
        # Axes
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.8,
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.linestyle":       ":",
        "grid.linewidth":       0.6,
        "grid.alpha":           0.4,
        # Figure
        "figure.dpi":           150,
        "savefig.dpi":          150,
        "figure.constrained_layout.use": True,
        # SVG
        "svg.fonttype":         "none",   # embed text as text, not paths
    })


def save_fig(fig: plt.Figure, path: Path | str, *, dpi: int = 300) -> None:
    """Save *fig* as both SVG and PDF, creating parent directories as needed."""
    path = Path(path).with_suffix(".svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path} (+ .pdf)")


def style_boxplot(bp: dict, colors: Sequence[str], alpha: float = 0.75) -> None:
    """Apply consistent boxplot styling: white face with coloured median line."""
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
        patch.set_alpha(1.0)
    for median, color in zip(bp["medians"], colors):
        median.set_color(color)
        median.set_linewidth(2.0)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(0.8)
        whisker.set_color("#555555")
    for cap in bp["caps"]:
        cap.set_linewidth(0.8)
        cap.set_color("#555555")
    for flier in bp.get("fliers", []):
        flier.set_markersize(3)
        flier.set_alpha(0.5)
