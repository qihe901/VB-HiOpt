"""Typography and class colours matching the manuscript figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

DPI = 300

# Confusion-matrix display (matches 819 combined session plots).
SEED_CM_LABELS = ("negative", "neutral", "positive")
SEEDIV_CM_LABELS = ("neutral", "sad", "fear", "happy")
SEEDIV_CM_ORDER = (0, 2, 3, 1)  # Neutral, Sad, Fear, Happy from storage Neutral=0,Happy=1,Sad=2,Fear=3

# t-SNE / spectral class colours (manuscript Figure 3 / 4).
TSNE_COLORS = {0: "#4daf4a", 1: "#e41a1c", 2: "#984ea3", 3: "#ff7f00"}
TSNE_LABELS = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Fear"}
SEED_TSNE_COLORS = {0: "#C44536", 1: "#5B8C5A", 2: "#E07A3D"}
SEED_TSNE_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

BAR_COLORS = {
    "Negative": "#C44536",
    "Neutral": "#5B8C5A",
    "Positive": "#E07A3D",
    "Happy": "#E07A3D",
    "Sad": "#3A7CA5",
    "Fear": "#C44536",
    "negative": "#C44536",
    "neutral": "#5B8C5A",
    "positive": "#E07A3D",
    "happy": "#E07A3D",
    "sad": "#3A7CA5",
    "fear": "#C44536",
}

BAND_TICKS = (r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma$")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int = DPI) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    return path
