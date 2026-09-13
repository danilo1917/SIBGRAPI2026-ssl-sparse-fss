import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sslfss.stats import query_mean_ci

METHODS = ["maml", "panet", "r2d2", "alpnet"]
SPARSE_SUPPORT = {"maml", "panet", "r2d2"}
LABEL = {"maml": "MAML/MetaSGD", "panet": "PANet", "r2d2": "R2D2", "alpnet": "ALPNet"}
DATASET_LABEL = {"jsrt": "JSRT", "panoramic": "Panoramic"}


def compute(dataset: str, results_dir: Path, k_values: list[int] | None, confidence: float):
    """Return the k values and {method: [(mean, lo, hi) per k]} for one dataset."""
    frames = {}
    for method in METHODS:
        path = results_dir / method / "shots" / dataset / "results.csv"
        if path.exists():
            frames[method] = pd.read_csv(path)
    if k_values is None:
        k_values = sorted({int(k) for df in frames.values() for k in df["k"].unique()})
    rows = {}
    for method, df in frames.items():
        rows[method] = []
        for k in k_values:
            sub = df[df["k"] == k]
            rows[method].append(query_mean_ci(sub["dice"], sub["query_idx"], confidence))
    return k_values, rows


def best_sparse(rows: dict, n_k: int) -> list[str | None]:
    best = []
    for i in range(n_k):
        candidates = {m: rows[m][i][0] for m in rows if m in SPARSE_SUPPORT and not np.isnan(rows[m][i][0])}
        best.append(max(candidates, key=candidates.get) if candidates else None)
    return best


def latex(dataset: str, k_values: list[int], rows: dict, confidence: float) -> str:
    best = best_sparse(rows, len(k_values))
    name = DATASET_LABEL.get(dataset, dataset)
    lines = [
        r"\begin{table}[htb]",
        r"  \centering",
        f"  \\caption{{Dice per number of support images $k$ on {name}: mean $\\pm$ half-width of the "
        f"{round(confidence * 100)}\\,\\% confidence interval over query images. Bold: best method with "
        f"sparse supports. $^\\dagger$Dense supports.}}",
        f"  \\label{{tab:shots_{dataset}}}",
        f"  \\begin{{tabular}}{{l{'c' * len(k_values)}}}",
        r"    \toprule",
        "    Method & " + " & ".join(f"$k={k}$" for k in k_values) + r" \\",
        r"    \midrule",
    ]
    for method in METHODS:
        if method not in rows:
            continue
        cells = []
        for i, (mean, lo, hi) in enumerate(rows[method]):
            cell = "--" if np.isnan(mean) else f"{mean:.3f} $\\pm$ {(hi - lo) / 2:.3f}"
            cells.append(f"\\textbf{{{cell}}}" if best[i] == method else cell)
        label = LABEL[method] + (r"$^\dagger$" if method not in SPARSE_SUPPORT else "")
        lines.append(f"    {label} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def text(dataset: str, k_values: list[int], rows: dict) -> str:
    lines = [DATASET_LABEL.get(dataset, dataset), f"{'':14s}" + "".join(f"{f'k={k}':>9s}" for k in k_values)]
    for method in METHODS:
        if method in rows:
            lines.append(f"{LABEL[method]:14s}" + "".join(f"{mean:9.3f}" for mean, _, _ in rows[method]))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the shots-study table (mean and confidence interval per k).")
    p.add_argument("--results_dir", default="results/pseudo")
    p.add_argument("--out", default="results/pseudo/tables/shots.tex")
    p.add_argument("--confidence", type=float, default=0.99)
    p.add_argument("--k", nargs="+", type=int, default=None, help="k values to include (default: all)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    datasets = sorted({path.parent.name for path in results_dir.glob("*/shots/*/results.csv")})
    if not datasets:
        sys.exit(f"No shots results found under {results_dir}")

    tables = []
    for dataset in datasets:
        k_values, rows = compute(dataset, results_dir, args.k, args.confidence)
        if not rows:
            continue
        print(text(dataset, k_values, rows), end="\n\n")
        tables.append(latex(dataset, k_values, rows, args.confidence))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(tables) + "\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
