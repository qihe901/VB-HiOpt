"""Category-wise relative spectral attention (paper Figure 4)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import load_npz
from .style import BAND_TICKS, BAR_COLORS, apply_style, save_figure


def softmax_weights(scale: np.ndarray) -> np.ndarray:
    z = np.asarray(scale, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def class_mean_weights(scale: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = softmax_weights(scale)
    n_cls = int(y.max()) + 1
    out = np.zeros((n_cls, w.shape[1]), dtype=np.float64)
    for cls in range(n_cls):
        mask = y == cls
        if np.any(mask):
            out[cls] = w[mask].mean(axis=0)
    return out


def _grouped_bars(ax, matrix: np.ndarray, labels: list[str], title: str | None) -> None:
    relative = np.asarray(matrix, dtype=np.float64) - 1.0 / matrix.shape[1]
    n_cls, n_band = relative.shape
    x = np.arange(n_band)
    width = min(0.18, 0.72 / max(n_cls, 1))
    offsets = (np.arange(n_cls) - (n_cls - 1) / 2.0) * width
    for i, name in enumerate(labels):
        ax.bar(
            x + offsets[i],
            relative[i],
            width=width * 0.92,
            label=str(name).capitalize() if str(name).islower() else str(name),
            color=BAR_COLORS.get(str(name), BAR_COLORS.get(str(name).capitalize(), f"C{i}")),
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(list(BAND_TICKS[:n_band]), fontsize=12)
    ax.set_xlabel("Frequency Band", fontsize=11)
    ax.set_ylabel("Relative attention", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=min(4, n_cls), loc="upper center", bbox_to_anchor=(0.5, 1.02))


def plot_spectral_attention(
    matrix: np.ndarray,
    labels: list[str],
    out: Path,
    *,
    dataset: str = "seed",
    dpi: int = 300,
) -> Path:
    apply_style()
    ds = str(dataset).lower().replace("-", "")
    title = "SEED" if ds == "seed" else "SEED-IV"
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _grouped_bars(ax, matrix, labels, title)
    fig.tight_layout()
    return save_figure(fig, out, dpi=dpi)


def matrix_from_npz(path: Path) -> tuple[np.ndarray, list[str]]:
    data = load_npz(path)
    if "emotion_frequency_matrix" in data:
        labels = [str(x) for x in data["emotion_labels"].tolist()]
        return np.asarray(data["emotion_frequency_matrix"], dtype=np.float64), labels
    y = data["y"] if "y" in data else data["labels"]
    scale = data["scale"] if "scale" in data else data["band_scales"]
    labels = [f"class {i}" for i in range(int(y.max()) + 1)]
    return class_mean_weights(scale, y), labels


def reorder(matrix: np.ndarray, labels: list[str], order: list[str]) -> tuple[np.ndarray, list[str]]:
    lower = [x.lower() for x in labels]
    idx = []
    used = []
    for name in order:
        if name.lower() in lower:
            i = lower.index(name.lower())
            idx.append(i)
            used.append(labels[i])
    if not idx:
        return matrix, labels
    return matrix[idx], used


def demo_scale(num_classes: int, n: int = 400, f: int = 5):
    rng = np.random.default_rng(4 + num_classes)
    y = np.repeat(np.arange(num_classes), n // num_classes)
    bias = rng.normal(scale=0.15, size=(num_classes, f))
    scale = 1.0 + bias[y] + rng.normal(scale=0.05, size=(y.size, f))
    return class_mean_weights(scale, y), y


def _dataset(args) -> str:
    ds = str(getattr(args, "dataset", "") or "").lower().replace("-", "")
    if ds in {"seed", "seediv"}:
        return ds
    raise ValueError("Specify --dataset seed or --dataset seediv")


def main_cli(args) -> Path:
    ds = _dataset(args)
    order = ["Negative", "Neutral", "Positive"] if ds == "seed" else ["Happy", "Sad", "Fear", "Neutral"]
    if args.demo and not getattr(args, "npz", None) and not getattr(args, "seed_npz", None) and not getattr(args, "seediv_npz", None):
        matrix, _ = demo_scale(3 if ds == "seed" else 4)
        return plot_spectral_attention(matrix, order, Path(args.out), dataset=ds)
    path = getattr(args, "npz", None) or (
        getattr(args, "seed_npz", None) if ds == "seed" else getattr(args, "seediv_npz", None)
    )
    if not path:
        raise ValueError("Provide --npz (or --seed_npz / --seediv_npz) with --dataset")
    matrix, labels = matrix_from_npz(Path(path))
    matrix, labels = reorder(matrix, labels, order)
    return plot_spectral_attention(matrix, labels, Path(args.out), dataset=ds)
