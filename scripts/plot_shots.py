import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sslfss.stats import query_mean_ci

METHODS = ["panet", "maml", "r2d2", "alpnet"]
COLOR = {"panet": "#009E73", "maml": "#0072B2", "r2d2": "#E69F00", "alpnet": "#CC79A7"}
MARKER = {"panet": "^", "maml": "o", "r2d2": "D", "alpnet": "*"}
LABEL = {"panet": "PANet", "maml": "MAML", "r2d2": "R2D2", "alpnet": "ALPNet (dense supports)"}
DATASET_LABEL = {"jsrt": "JSRT", "panoramic": "Panoramic"}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linestyle": ":",
    "grid.alpha": 0.45,
    "lines.linewidth": 1.9,
    "svg.fonttype": "none",
}


def curve(df: pd.DataFrame, k_values: list[int], confidence: float) -> np.ndarray:
    """Mean and confidence bounds of the per-query Dice for each k."""
    stats = []
    for k in k_values:
        sub = df[df["k"] == k]
        stats.append(query_mean_ci(sub["dice"], sub["query_idx"], confidence))
    return np.array(stats).T


def plot_dataset(dataset: str, results_dir: Path, out_dir: Path, confidence: float) -> None:
    frames = {}
    for method in METHODS:
        path = results_dir / method / "shots" / dataset / "results.csv"
        if path.exists():
            frames[method] = pd.read_csv(path)
    if not frames:
        return

    k_values = sorted({int(k) for df in frames.values() for k in df["k"].unique()})
    x = np.arange(len(k_values))

    width = 180 / 25.4
    fig, ax = plt.subplots(figsize=(width, width * 0.48))
    for method, df in frames.items():
        mean, lo, hi = curve(df, k_values, confidence)
        ax.plot(x, mean, f"{MARKER[method]}-", color=COLOR[method], label=LABEL[method],
                markersize=8 if method == "alpnet" else 6, zorder=3)
        ax.fill_between(x, np.clip(lo, 0, 1), np.clip(hi, 0, 1), color=COLOR[method], alpha=0.14, zorder=2)

    ax.set_xticks(x, [str(k) for k in k_values])
    ax.set_xlim(-0.3, len(k_values) - 0.7)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_xlabel("Number of support images (k)")
    ax.set_ylabel("Dice score")
    ax.set_title(DATASET_LABEL.get(dataset, dataset))
    ax.legend(loc="lower center", ncol=len(frames), frameon=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = out_dir / f"shots_{dataset}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Dice versus number of support images for every method.")
    p.add_argument("--results_dir", default="results/pseudo")
    p.add_argument("--out_dir", default="results/pseudo/figures")
    p.add_argument("--confidence", type=float, default=0.99)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    datasets = sorted({path.parent.name for path in results_dir.glob("*/shots/*/results.csv")})
    if not datasets:
        sys.exit(f"No shots results found under {results_dir}")

    plt.rcParams.update(STYLE)
    for dataset in datasets:
        plot_dataset(dataset, results_dir, Path(args.out_dir), args.confidence)


if __name__ == "__main__":
    main()
