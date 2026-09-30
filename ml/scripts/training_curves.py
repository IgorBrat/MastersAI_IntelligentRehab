"""Plotting leave-one-subject-out training curves and judging the epoch budget"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  

OUT = pathlib.Path("reports/training_curves.png")


def folds_of(report: dict) -> list[dict]:
    """Collecting fold records from either a pooled or a per-exercise report"""
    if report.get("folds"):
        return report["folds"]
    out: list[dict] = []
    for sub in (report.get("per_exercise") or {}).values():
        out.extend(sub.get("folds") or [])
    return out


def curve_matrix(folds: list[dict], key: str) -> np.ndarray | None:
    """Padding each fold's curve to the longest with NaN, so folds can be averaged"""
    curves = [f.get(key) or [] for f in folds]
    curves = [c for c in curves if c]
    if not curves:
        return None
    width = max(len(c) for c in curves)
    matrix = np.full((len(curves), width), np.nan)
    for i, c in enumerate(curves):
        matrix[i, : len(c)] = c
    return matrix


def verdict(folds: list[dict], cap: int) -> str:
    """Stating whether the epoch budget bound the result or early stopping did"""
    runs = [f["epochs_run"] for f in folds if "epochs_run" in f]
    if not runs:
        return "no epoch data"
    hit = sum(r >= cap for r in runs)
    if hit == 0:
        return (
            f"early stopping fired on every fold (median {int(np.median(runs))} epochs). "
            f"Raising the budget cannot change this run."
        )
    return (
        f"{hit} of {len(runs)} folds hit the {cap}-epoch cap "
        f"(median {int(np.median(runs))}). Those folds were TRUNCATED -- rerun higher."
    )


def main(paths: list[str]) -> None:
    reports = []
    for path in paths:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        folds = folds_of(data)
        if any(f.get("train_loss") for f in folds):
            reports.append((pathlib.Path(path).stem, data, folds))
        else:
            print(f"  {pathlib.Path(path).name}: no curves stored -- rerun training")

    if not reports:
        print("\nNo report carries curves yet. Re-run any train command, then this script.")
        return

    fig, axes = plt.subplots(
        1, len(reports), figsize=(5.2 * len(reports), 4.2), squeeze=False
    )
    for ax, (name, data, folds) in zip(axes[0], reports):
        cap = (data.get("config") or {}).get("epochs", 0)
        for key, colour, label in (
            ("train_loss", "#4C72B0", "train"),
            ("val_loss", "#C44E52", "validation"),
        ):
            matrix = curve_matrix(folds, key)
            if matrix is None:
                continue
            mean = np.nanmean(matrix, axis=0)
            lo = np.nanpercentile(matrix, 25, axis=0)
            hi = np.nanpercentile(matrix, 75, axis=0)
            epochs = np.arange(1, mean.size + 1)
            ax.plot(epochs, mean, color=colour, label=label, linewidth=1.8)
            ax.fill_between(epochs, lo, hi, color=colour, alpha=0.18, linewidth=0)

        stops = [f["epochs_run"] for f in folds if "epochs_run" in f]
        if stops:
            ax.scatter(
                stops,
                np.full(len(stops), ax.get_ylim()[0]),
                marker="|", s=90, color="#555555",
                label=f"early stop (median {int(np.median(stops))})",
            )
        if cap:
            ax.axvline(cap, color="#888888", linestyle=":", linewidth=1.2)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("multi-task loss")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        print(f"\n{name}: {verdict(folds, cap or 30)}")

    fig.suptitle(
        "LOSO training curves - mean across folds, IQR shaded, dotted line = epoch cap",
        fontsize=10,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(
        set(glob.glob("reports/*hybrid*.json"))
        | set(glob.glob("reports/flexion_*.json"))
    )
    main(args)
