"""Emotion-wise EEG channel chords (paper Figure 5)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, PathPatch, Patch, Wedge
from matplotlib.path import Path as MplPath

from .channels import SEED62_CHANNELS
from .io import load_npz
from .style import apply_style, save_figure

ROI_CHANNELS: dict[str, tuple[str, ...]] = {
    "Prefrontal": ("FP1", "FPZ", "FP2", "AF3", "AF4"),
    "Frontal": ("F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"),
    "Fronto-central": ("FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6"),
    "Temporal": ("FT7", "FT8", "T7", "T8", "TP7", "TP8"),
    "Central": ("C5", "C3", "C1", "CZ", "C2", "C4", "C6"),
    "Centro-parietal": ("CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6"),
    "Parietal": ("P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"),
    "Parieto-occipital": ("PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8"),
    "Occipital": ("O1", "OZ", "O2", "CB1", "CB2"),
}
OUTER_GROUPS = (
    ("Frontal", ROI_CHANNELS["Prefrontal"] + ROI_CHANNELS["Frontal"] + ROI_CHANNELS["Fronto-central"]),
    ("Temporal", ROI_CHANNELS["Temporal"]),
    ("Central", ROI_CHANNELS["Central"]),
    ("Parietal", ROI_CHANNELS["Centro-parietal"] + ROI_CHANNELS["Parietal"]),
    ("Occipital", ROI_CHANNELS["Parieto-occipital"] + ("O1", "OZ", "O2")),
)
REGION_COLOR = {
    "Frontal": "#4C78A8",
    "Temporal": "#F58518",
    "Central": "#54A24B",
    "Parietal": "#E45756",
    "Occipital": "#72B7B2",
}
CHANNEL_REGION = {ch: region for region, chs in OUTER_GROUPS for ch in chs}
ANATOMICAL = [ch for _, chs in OUTER_GROUPS for ch in chs]
EXCLUDE = {"CB1", "CB2"}
KEY_LABELS = {
    "FP1", "FP2", "FZ", "F3", "F4",
    "T7", "T8", "CZ", "C3", "C4",
    "PZ", "P3", "P4", "OZ", "O1", "O2",
}
SEED_TO_MNE = {
    "FP1": "Fp1", "FPZ": "Fpz", "FP2": "Fp2",
    "FZ": "Fz", "CZ": "Cz", "CPZ": "CPz", "PZ": "Pz",
    "POZ": "POz", "OZ": "Oz", "FCZ": "FCz",
}


def _mne_name(ch: str) -> str:
    if ch in SEED_TO_MNE:
        return SEED_TO_MNE[ch]
    if len(ch) == 2:
        return ch.capitalize()
    return ch


def _soft_hex(hex_color: str, mix: float = 0.45) -> str:
    r, g, b, _ = to_rgba(hex_color)
    r = 1.0 - mix * (1.0 - r)
    g = 1.0 - mix * (1.0 - g)
    b = 1.0 - mix * (1.0 - b)
    return f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}"


def _connection_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "chord_soft",
        [
            "#F4F6F8",
            _soft_hex(REGION_COLOR["Frontal"]),
            _soft_hex(REGION_COLOR["Occipital"]),
            _soft_hex(REGION_COLOR["Temporal"]),
            _soft_hex(REGION_COLOR["Parietal"]),
        ],
    )


def _topk_undirected(adj: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = 0.5 * (np.asarray(adj, dtype=np.float64) + np.asarray(adj, dtype=np.float64).T)
    np.fill_diagonal(a, 0.0)
    iu = np.triu_indices_from(a, k=1)
    order = np.argsort(a[iu])[::-1][:k]
    return iu[0][order], iu[1][order], a[iu][order]


def _angles(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = 2.0 * np.pi / n
    gap = step * 0.06
    starts = np.pi / 2 - np.arange(n) * step - (step - gap)
    ends = np.pi / 2 - np.arange(n) * step
    return 0.5 * (starts + ends), starts, ends


def _radial_rotation(angle: float) -> float:
    deg = float(np.degrees(angle))
    if np.cos(angle) < 0:
        deg += 180.0
    return deg


def _bezier(p0: np.ndarray, p1: np.ndarray, pull: float = 0.30) -> PathPatch:
    control = pull * (p0 + p1)
    path = MplPath([p0, control, p1], [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    return PathPatch(path, facecolor="none", capstyle="round", zorder=2)


def _draw_channel_ring(ax, names: list[str]) -> np.ndarray:
    n = len(names)
    centers, starts, ends = _angles(n)
    r_ring, ring_width = 1.32, 0.26
    for name, a0, a1 in zip(names, starts, ends):
        color = REGION_COLOR.get(CHANNEL_REGION.get(name, "Central"), "#888888")
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                r_ring,
                np.degrees(a0),
                np.degrees(a1),
                width=ring_width,
                facecolor=to_rgba(color, 0.45),
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        )
    i = 0
    while i < n:
        region = CHANNEL_REGION.get(names[i], "Other")
        j = i
        while j + 1 < n and CHANNEL_REGION.get(names[j + 1], "Other") == region:
            j += 1
        for ang in (starts[j], ends[i]):
            ax.plot(
                [(r_ring - ring_width) * np.cos(ang), r_ring * np.cos(ang)],
                [(r_ring - ring_width) * np.sin(ang), r_ring * np.sin(ang)],
                color="#555555",
                linewidth=1.0,
                zorder=5,
            )
        i = j + 1
    ch_r = r_ring - 0.5 * ring_width
    for ang, name in zip(centers, names):
        is_key = name in KEY_LABELS
        ax.text(
            ch_r * np.cos(ang),
            ch_r * np.sin(ang),
            _mne_name(name),
            ha="center",
            va="center",
            fontsize=7.4 if is_key else 6.4,
            color="#1a1a1a",
            fontweight="bold" if is_key else "normal",
            rotation=_radial_rotation(ang),
            rotation_mode="anchor",
            zorder=6,
        )
    ax.add_patch(Circle((0.0, 0.0), r_ring, fill=False, linewidth=0.8, edgecolor="#333333", zorder=5))
    ax.add_patch(Circle((0.0, 0.0), r_ring - ring_width, fill=False, linewidth=0.7, edgecolor="#333333", zorder=2))
    edge_r = r_ring - ring_width
    return np.column_stack([edge_r * np.cos(centers), edge_r * np.sin(centers)])


def _draw_panel(ax, adj: np.ndarray, title: str, channel_names: list[str], top_k: int, cmap) -> None:
    order = [channel_names.index(ch) for ch in ANATOMICAL if ch in channel_names and ch not in EXCLUDE]
    names = [channel_names[i] for i in order]
    coords = _draw_channel_ring(ax, names)
    sub = adj[np.ix_(order, order)]
    ii, jj, w = _topk_undirected(sub, top_k)
    if w.size:
        nw = (w - w.min()) / max(float(w.max() - w.min()), 1e-12)
        for a, b, t in zip(ii, jj, nw):
            patch = _bezier(coords[a], coords[b])
            patch.set_edgecolor(cmap(float(t)))
            patch.set_linewidth(0.8 + 3.4 * float(t))
            patch.set_alpha(0.82)
            ax.add_patch(patch)
    ax.set_title(title, fontsize=11, pad=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.36, 1.36)
    ax.set_ylim(-1.36, 1.36)


def plot_channel_chords(
    adj: np.ndarray,
    labels: list[str],
    out: Path,
    top_k: int = 30,
    dpi: int = 300,
    channel_names: list[str] | None = None,
) -> Path:
    apply_style()
    names = list(channel_names or SEED62_CHANNELS)
    cmap = _connection_cmap()
    n = int(adj.shape[0])
    fig = plt.figure(figsize=(5.2 * n, 5.4))
    gs = GridSpec(1, n, figure=fig, wspace=0.02, left=0.01, right=0.93, top=0.90, bottom=0.16)
    axes = [fig.add_subplot(gs[0, j]) for j in range(n)]
    for ax, panel, title in zip(axes, adj, labels):
        _draw_panel(ax, panel, title, names, top_k, cmap)
    handles = [
        Patch(facecolor=_soft_hex(REGION_COLOR[r]), edgecolor="#888888", linewidth=0.5, label=r)
        for r, _ in OUTER_GROUPS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=11,
        handlelength=1.6,
        handleheight=1.2,
        columnspacing=1.2,
        handletextpad=0.5,
        bbox_to_anchor=(0.48, 0.02),
    )
    cax = fig.add_axes([0.945, 0.28, 0.012, 0.50])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=8, length=2, width=0.6)
    return save_figure(fig, out, dpi=dpi)


def demo_adj(num_classes: int, c: int = 62) -> np.ndarray:
    rng = np.random.default_rng(5 + num_classes)
    out = np.zeros((num_classes, c, c))
    for k in range(num_classes):
        a = rng.random((c, c)) * 0.15
        hubs = rng.choice(c, size=8, replace=False)
        a[np.ix_(hubs, hubs)] += 0.8
        a = 0.5 * (a + a.T)
        np.fill_diagonal(a, 0.0)
        out[k] = a
    return out


def load_adj(path: Path) -> tuple[np.ndarray, list[str]]:
    path = Path(path)
    if path.suffix == ".npy":
        adj = np.load(path)
        n = adj.shape[0]
        labels = ["Negative", "Neutral", "Positive"] if n == 3 else ["Happy", "Sad", "Fear", "Neutral"]
        return adj, labels
    data = load_npz(path)
    adj = data["adj"] if "adj" in data else data["emotion_adj_matrix"]
    labels = (
        [str(x) for x in data["emotion_labels"].tolist()]
        if "emotion_labels" in data
        else [f"c{i}" for i in range(adj.shape[0])]
    )
    return adj, labels


def _dataset(args) -> str:
    ds = str(getattr(args, "dataset", "") or "").lower().replace("-", "")
    if ds in {"seed", "seediv"}:
        return ds
    raise ValueError("Specify --dataset seed or --dataset seediv")


def main_cli(args) -> Path:
    ds = _dataset(args)
    labels = ["Negative", "Neutral", "Positive"] if ds == "seed" else ["Happy", "Sad", "Fear", "Neutral"]
    if args.demo and not getattr(args, "npz", None) and not getattr(args, "seed_npz", None) and not getattr(args, "seediv_npz", None):
        return plot_channel_chords(demo_adj(len(labels)), labels, Path(args.out), top_k=args.top_k)
    path = getattr(args, "npz", None) or (
        getattr(args, "seed_npz", None) if ds == "seed" else getattr(args, "seediv_npz", None)
    )
    if not path:
        raise ValueError("Provide --npz (or --seed_npz / --seediv_npz) with --dataset")
    adj, file_labels = load_adj(Path(path))
    if len(file_labels) == len(labels):
        file_labels = labels
    return plot_channel_chords(adj, file_labels, Path(args.out), top_k=args.top_k)
