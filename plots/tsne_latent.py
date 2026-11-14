"""t-SNE of Raw DE / VAE / w/o class prior / VB-HiOpt (paper Figure 3)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .io import load_npz
from .style import TSNE_COLORS, TSNE_LABELS, apply_style, save_figure

PANELS = (
    ("raw_de", "(a) Raw DE"),
    ("vae", "(b) VAE"),
    ("no_prior", "(c) w/o class prior"),
    ("vbhiopt", "(d) VB-HiOpt"),
)


def embed(x: np.ndarray, seed: int = 42, perplexity: float = 30.0, max_samples: int = 2500):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    idx = np.arange(x.shape[0])
    if max_samples and x.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
    z = StandardScaler().fit_transform(x)
    n = z.shape[0]
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca", learning_rate="auto")
    return tsne.fit_transform(z), idx


def plot_tsne_latent(
    panels: dict[str, tuple[np.ndarray, np.ndarray]],
    out: Path,
    seed_tsne: int = 42,
    dpi: int = 300,
    max_samples: int = 2500,
) -> Path:
    apply_style()
    keys = [k for k, _ in PANELS if k in panels]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.3))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        title = dict(PANELS)[key]
        x, y = panels[key]
        xy, idx = embed(x, seed=seed_tsne, max_samples=max_samples)
        y = np.asarray(y)[idx]
        for cls in sorted(np.unique(y)):
            cls = int(cls)
            mask = y == cls
            ax.scatter(xy[mask, 0], xy[mask, 1], c=TSNE_COLORS.get(cls, "#888"), s=10, alpha=0.55, edgecolors="none")
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=TSNE_COLORS[c], markeredgecolor="none", label=TSNE_LABELS[c])
        for c in (1, 2, 3, 0)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        edgecolor="0.35",
        facecolor="white",
        framealpha=0.95,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.14, wspace=0.18)
    return save_figure(fig, out, dpi=dpi)


def demo_panels(num_classes: int = 4, n: int = 400, dim: int = 16) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(4)
    y = np.repeat(np.arange(num_classes), n // num_classes)
    centres = rng.normal(scale=0.4, size=(num_classes, dim))
    spread = {"raw_de": 2.4, "vae": 1.6, "no_prior": 1.1, "vbhiopt": 0.55}
    return {k: ((centres[y] + rng.normal(scale=s, size=(y.size, dim))).astype(np.float32), y.copy()) for k, s in spread.items()}


def _feat(data: dict, preferred: tuple[str, ...]) -> np.ndarray:
    for key in preferred:
        if key in data:
            return data[key]
    raise KeyError(preferred)


def main_cli(args) -> Path:
    if args.demo and not getattr(args, "seediv_vbhiopt", None) and not getattr(args, "vbhiopt_npz", None):
        return plot_tsne_latent(demo_panels(), Path(args.out), seed_tsne=args.seed)
    panels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mapping = (
        ("raw_de", getattr(args, "seediv_raw", None) or getattr(args, "de_npz", None), ("de_flat", "x")),
        ("vae", getattr(args, "seediv_vae", None) or getattr(args, "vae_npz", None), ("mu_q",)),
        ("no_prior", getattr(args, "seediv_noprior", None) or getattr(args, "noprior_npz", None), ("mu_q",)),
        ("vbhiopt", getattr(args, "seediv_vbhiopt", None) or getattr(args, "vbhiopt_npz", None), ("mu_q",)),
    )
    for key, path, feat_keys in mapping:
        if not path:
            continue
        data = load_npz(Path(path))
        y = data["labels"] if "labels" in data else data["y"]
        panels[key] = (_feat(data, feat_keys), y)
    if "raw_de" not in panels and "vbhiopt" in panels:
        data = load_npz(Path(mapping[3][1]))
        if "de_flat" in data:
            panels["raw_de"] = (data["de_flat"], data["labels"])
    return plot_tsne_latent(panels, Path(args.out), seed_tsne=args.seed)
