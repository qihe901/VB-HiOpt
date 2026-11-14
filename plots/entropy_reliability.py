"""Predictive entropy and reliability diagram (paper Figure 6)."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import load_npz

_TIMES = ["Times New Roman", "Nimbus Roman No9 L", "Nimbus Roman", "DejaVu Serif"]


def predictive_entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def reliability_bins(prob: np.ndarray, y: np.ndarray, num_bins: int = 15, min_count: int = 1):
    conf = np.asarray(prob).max(axis=1)
    pred = np.asarray(prob).argmax(axis=1)
    y = np.asarray(y)
    correct = pred == y
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    xs, ys = [], []
    ece = 0.0
    n = max(len(y), 1)
    for i in range(num_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        count = int(mask.sum())
        if count < min_count:
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(correct[mask].mean())
        xs.append(avg_conf)
        ys.append(avg_acc)
        ece += (count / n) * abs(avg_acc - avg_conf)
    return np.asarray(xs), np.asarray(ys), float(ece)


def _render_entropy(entropy: np.ndarray, correct: np.ndarray, dpi: int) -> np.ndarray:
    """Figure 6a: light-blue violins + white boxplots with orange medians."""
    plt.rcParams.update(plt.rcParamsDefault)
    correct = np.asarray(correct).astype(bool)
    c_ent = np.asarray(entropy, dtype=np.float64)[correct]
    w_ent = np.asarray(entropy, dtype=np.float64)[~correct]
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    parts = ax.violinplot(
        [c_ent, w_ent],
        positions=[1, 2],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.52,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#7EB6D9")
        body.set_edgecolor("#7EB6D9")
        body.set_alpha(0.72)
        body.set_linewidth(1.0)
        body.set_zorder(1)
    ax.boxplot(
        [c_ent, w_ent],
        positions=[1, 2],
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#FF7F0E", "linewidth": 1.6},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.0},
        zorder=3,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Correct", "Wrong"])
    ax.set_ylabel("Prediction entropy")
    ax.set_ylim(0.0, 1.45)
    ax.set_xlim(0.40, 2.60)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", length=4, width=1.0, colors="black")
    fig.tight_layout()
    return _fig_rgb(fig, dpi)


def _render_reliability(
    softmax_prob: np.ndarray,
    softmax_y: np.ndarray,
    vb_prob: np.ndarray,
    vb_y: np.ndarray,
    dpi: int,
) -> np.ndarray:
    """Standalone 6b, matching 819/绘图/plot_reliability_diagram.py paper style."""
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _TIMES,
            "mathtext.fontset": "stix",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    x_s, y_s, ece_s = reliability_bins(softmax_prob, softmax_y)
    x_v, y_v, ece_v = reliability_bins(vb_prob, vb_y)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#1F77B4", linewidth=1.5, label="Perfect calibration")
    ax.plot(
        x_s,
        y_s,
        "o-",
        color="#FF7F0E",
        linewidth=1.6,
        markersize=6,
        markerfacecolor="#FF7F0E",
        markeredgecolor="#FF7F0E",
        label=f"Softmax (ECE={ece_s:.3f})",
    )
    ax.plot(
        x_v,
        y_v,
        "o-",
        color="#2CA02C",
        linewidth=1.6,
        markersize=6,
        markerfacecolor="#2CA02C",
        markeredgecolor="#2CA02C",
        label=f"VB-HiOpt (ECE={ece_v:.3f})",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.35, color="#B0B0B0")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.tick_params(direction="out", length=4, width=1.0)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return _fig_rgb(fig, dpi)


def _fig_rgb(fig: plt.Figure, dpi: int) -> np.ndarray:
    buf = BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight", facecolor="white", format="png")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _hstack(img_a: np.ndarray, img_b: np.ndarray, gap: int = 28) -> np.ndarray:
    def _rgba(img: np.ndarray) -> np.ndarray:
        img = np.asarray(img)
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img, img, img, np.ones_like(img)], axis=-1)
        if img.shape[-1] == 3:
            img = np.concatenate([img, np.ones((*img.shape[:2], 1), dtype=img.dtype)], axis=-1)
        return img

    a, b = _rgba(img_a), _rgba(img_b)
    h = max(a.shape[0], b.shape[0])

    def _pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        top = (h - img.shape[0]) // 2
        pad = np.ones((h, img.shape[1], img.shape[2]), dtype=img.dtype)
        pad[..., :3] = 1.0
        pad[top : top + img.shape[0]] = img
        return pad

    a, b = _pad(a), _pad(b)
    spacer = np.ones((h, gap, 4), dtype=a.dtype)
    return np.concatenate([a, spacer, b], axis=1)


def plot_entropy_reliability(
    vb_prob: np.ndarray,
    vb_y: np.ndarray,
    softmax_prob: np.ndarray,
    softmax_y: np.ndarray,
    out: Path,
    dpi: int = 300,
    entropy: np.ndarray | None = None,
    correct: np.ndarray | None = None,
) -> Path:
    if entropy is None:
        entropy = predictive_entropy(vb_prob)
    if correct is None:
        correct = np.asarray(vb_prob).argmax(axis=1) == np.asarray(vb_y)
    img_a = _render_entropy(np.asarray(entropy), np.asarray(correct), dpi)
    img_b = _render_reliability(softmax_prob, softmax_y, vb_prob, vb_y, dpi)
    canvas = _hstack(img_a, img_b)
    fig, ax = plt.subplots(figsize=(canvas.shape[1] / dpi, canvas.shape[0] / dpi), dpi=dpi)
    ax.imshow(canvas)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(fig)
    return path


def _split_pred(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if "probabilities" in data:
        y = data["labels"]
        return data["probabilities"], y, data.get("entropy"), data.get("correct")
    return data["prob"], data["y"], data.get("entropy"), data.get("correct")


def demo_probs(n: int = 800, k: int = 4, calibrated: bool = True):
    rng = np.random.default_rng(6 if calibrated else 7)
    y = rng.integers(0, k, size=n)
    logits = rng.normal(scale=0.8 if calibrated else 1.6, size=(n, k))
    logits[np.arange(n), y] += 2.2 if calibrated else 0.4
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32), y


def main_cli(args) -> Path:
    if args.demo and not getattr(args, "vbhiopt_npz", None):
        vb_p, vb_y = demo_probs(calibrated=True)
        sm_p, sm_y = demo_probs(calibrated=False)
        return plot_entropy_reliability(vb_p, vb_y, sm_p, sm_y, Path(args.out))
    vb = load_npz(Path(args.vbhiopt_npz))
    sm = load_npz(Path(args.softmax_npz))
    vb_p, vb_y, ent, corr = _split_pred(vb)
    sm_p, sm_y, _, _ = _split_pred(sm)
    return plot_entropy_reliability(
        vb_p,
        vb_y,
        sm_p,
        sm_y,
        Path(args.out),
        entropy=None if ent is None else np.asarray(ent),
        correct=None if corr is None else np.asarray(corr),
    )
