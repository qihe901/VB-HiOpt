"""Session-wise confusion matrices (paper Figure 2)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import collect_confusion_from_run, load_npz
from .style import SEED_CM_LABELS, SEEDIV_CM_LABELS, apply_style, save_figure


def row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=np.float64)
    if cm.max() <= 1.5:
        return cm
    denom = np.where(cm.sum(axis=1, keepdims=True) > 0, cm.sum(axis=1, keepdims=True), 1.0)
    return cm / denom


def _panel(ax, cm: np.ndarray, labels: tuple[str, ...], title: str, ylabel: bool) -> None:
    frac = row_normalize(cm)
    im = ax.imshow(frac, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=14)
    if ylabel:
        ax.set_ylabel("True", fontsize=14)
    ax.set_title(title, fontsize=16, pad=8)
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = frac[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > thresh else "black", fontsize=11)
    return im


def plot_confusion_matrices(
    matrices: list[np.ndarray],
    out: Path,
    *,
    dataset: str = "seed",
    titles: tuple[str, ...] | None = None,
    dpi: int = 300,
) -> Path:
    apply_style()
    ds = str(dataset).lower().replace("-", "")
    labels = SEED_CM_LABELS if ds == "seed" else SEEDIV_CM_LABELS
    prefix = "SEED" if ds == "seed" else "SEED-IV"
    n_col = len(matrices)
    titles = titles or tuple(f"Session {i + 1}" for i in range(n_col))
    fig, axes = plt.subplots(1, n_col, figsize=(5.6 * n_col, 5.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    last = None
    for j, ax in enumerate(axes):
        last = _panel(ax, matrices[j], labels, f"{prefix}  {titles[j]}", ylabel=j == 0)
    fig.colorbar(last, ax=axes.tolist(), location="right", shrink=0.86, pad=0.01)
    return save_figure(fig, out, dpi=dpi)


def demo_matrices() -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(2)
    seed, seediv = [], []
    for rec in ((0.82, 0.87, 0.94), (0.76, 0.82, 0.94), (0.80, 0.85, 0.93)):
        cm = np.eye(3) * rec
        for i, r in enumerate(rec):
            off = rng.dirichlet(np.ones(2)) * (1 - r)
            cm[i, [j for j in range(3) if j != i]] = off
        seed.append(cm)
    for rec in ((0.79, 0.72, 0.63, 0.83), (0.72, 0.85, 0.73, 0.79), (0.82, 0.69, 0.83, 0.79)):
        cm = np.eye(4) * rec
        for i, r in enumerate(rec):
            off = rng.dirichlet(np.ones(3)) * (1 - r)
            cm[i, [j for j in range(4) if j != i]] = off
        seediv.append(cm)
    return seed, seediv


def load_session_npy(folder: Path, prefix: str) -> list[np.ndarray]:
    mats = []
    for session in (1, 2, 3):
        path = folder / f"{prefix}_s{session}_confusion_matrix_norm.npy"
        if not path.is_file():
            path = folder / f"{prefix}_s{session}_confusion_matrix.npy"
        mats.append(np.load(path))
    return mats


def _dataset(args) -> str:
    ds = str(getattr(args, "dataset", "") or "").lower().replace("-", "")
    if ds in {"seed", "seediv"}:
        return ds
    raise ValueError("Specify --dataset seed or --dataset seediv")


def main_cli(args) -> Path:
    ds = _dataset(args)
    if args.demo and not getattr(args, "cm_dir", None) and not getattr(args, "npz", None):
        seed, seediv = demo_matrices()
        mats = seed if ds == "seed" else seediv
        return plot_confusion_matrices(mats, Path(args.out), dataset=ds)
    cm_dir = getattr(args, "cm_dir", None) or (
        getattr(args, "seed_cm_dir", None) if ds == "seed" else getattr(args, "seediv_cm_dir", None)
    )
    if cm_dir:
        prefix = "seed" if ds == "seed" else "seediv"
        return plot_confusion_matrices(load_session_npy(Path(cm_dir), prefix), Path(args.out), dataset=ds)
    npz_list = getattr(args, "npz", None) or (
        getattr(args, "seed_npz", None) if ds == "seed" else getattr(args, "seediv_npz", None)
    )
    if npz_list:
        mats = [load_npz(Path(p))["confusion"] for p in npz_list]
        return plot_confusion_matrices(mats, Path(args.out), dataset=ds)
    run = getattr(args, "run", None) or (
        getattr(args, "seed_run", None) if ds == "seed" else getattr(args, "seediv_run", None)
    )
    if run:
        mats = list(collect_confusion_from_run(Path(run)).values())[:3]
        return plot_confusion_matrices(mats, Path(args.out), dataset=ds)
    raise ValueError("Provide --demo, --cm_dir, --npz, or --run with --dataset seed|seediv")
