"""Alternating vs joint optimization curves (paper Figure 7)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import collect_logs_from_run, load_training_log
from .style import apply_style, save_figure

PANELS = (
    ("total_loss", "Total loss"),
    ("inner_grad_norm", "Encoder gradient norm"),
    ("classifier_grad_norm", "Classifier gradient norm"),
    ("stationarity_proxy", "Stationarity measure"),
)
ALT_COLOR = "#2E86AB"
JOINT_COLOR = "#E67E22"


def aggregate(logs: list[dict[str, np.ndarray]], key: str):
    length = min(len(log["epoch"]) for log in logs)
    series = []
    for log in logs:
        if key not in log:
            if key == "total_loss":
                series.append(log["inner_loss"][:length] + log["classifier_loss"][:length])
                continue
            raise KeyError(key)
        series.append(log[key][:length])
    arr = np.stack(series, axis=0)
    return logs[0]["epoch"][:length], arr.mean(axis=0), arr.std(axis=0, ddof=0)


def load_strategy_csv(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by.setdefault(row.get("strategy", "alternating"), []).append(row)

    def _col(items, name):
        return np.asarray([float(r[name]) for r in items], dtype=np.float64)

    out = {}
    for strat, items in by.items():
        block = {"epoch": _col(items, "epoch")}
        for key, _ in PANELS:
            if f"{key}_mean" in items[0]:
                block[key] = _col(items, f"{key}_mean")
                block[f"{key}_std"] = _col(items, f"{key}_std")
            elif key in items[0]:
                block[key] = _col(items, key)
        out[strat] = block
    return out


def _band(ax, epoch, mean, std, color, label) -> None:
    ax.plot(epoch, mean, color=color, lw=1.8, label=label)
    if std is not None and np.any(std > 0):
        ax.fill_between(epoch, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def plot_optimization_curves(
    alternating: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
    joint: list[dict[str, np.ndarray]] | dict[str, np.ndarray] | None,
    out: Path,
    dpi: int = 300,
) -> Path:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True)

    def _get(src, key):
        if isinstance(src, dict) and "epoch" in src:
            ep = src["epoch"]
            mean = src[key]
            std = src.get(f"{key}_std")
            return ep, mean, std
        return aggregate(src, key)

    for ax, (key, title) in zip(axes.flat, PANELS):
        ep, mean, std = _get(alternating, key)
        _band(ax, ep, mean, std, ALT_COLOR, "Alternating")
        if joint is not None:
            ep_j, mean_j, std_j = _get(joint, key)
            _band(ax, ep_j, mean_j, std_j, JOINT_COLOR, "Joint")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Figure 7 Optimization dynamics (mean_over_folds)", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, out, dpi=dpi)


def demo_logs(n_folds: int = 8, epochs: int = 80, joint: bool = False):
    rng = np.random.default_rng(8 if joint else 9)
    logs = []
    t = np.arange(epochs, dtype=np.float64)
    for _ in range(n_folds):
        g_enc = 1.4 * np.exp(-t / (12 if joint else 22)) + 0.05 + rng.normal(scale=0.03, size=epochs)
        g_cls = (0.6 if joint else 1.3) * np.exp(-t / (10 if joint else 20)) + 0.04 + rng.normal(scale=0.03, size=epochs)
        loss = 2.2 * np.exp(-t / (18 if joint else 28)) + 0.35 + rng.normal(scale=0.04 if joint else 0.08, size=epochs)
        logs.append(
            {
                "epoch": t,
                "total_loss": np.clip(loss, 0.05, None),
                "inner_grad_norm": np.clip(g_enc, 0.0, None),
                "classifier_grad_norm": np.clip(g_cls, 0.0, None),
                "stationarity_proxy": np.clip(g_enc, 0, None) ** 2 + np.clip(g_cls, 0, None) ** 2,
            }
        )
    return logs


def main_cli(args) -> Path:
    if getattr(args, "curves_csv", None):
        blocks = load_strategy_csv(Path(args.curves_csv))
        alt = blocks.get("alternating") or next(iter(blocks.values()))
        joint = blocks.get("joint")
        return plot_optimization_curves(alt, joint, Path(args.out))
    if args.demo and not getattr(args, "alternating_run", None):
        return plot_optimization_curves(demo_logs(joint=False), demo_logs(joint=True), Path(args.out))
    alt = collect_logs_from_run(Path(args.alternating_run)) if args.alternating_run else [load_training_log(Path(p)) for p in args.alternating_log]
    joint = None
    if args.joint_run:
        joint = collect_logs_from_run(Path(args.joint_run))
    elif args.joint_log:
        joint = [load_training_log(Path(p)) for p in args.joint_log]
    return plot_optimization_curves(alt, joint, Path(args.out))
