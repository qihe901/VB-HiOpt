"""Load exported arrays used by Figures 2–7."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_npz(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_confusion_from_run(run_dir: Path) -> dict[str, np.ndarray]:
    """Stack last-epoch confusion matrices under ``subject_*/metrics.json`` or fold dirs."""
    run_dir = Path(run_dir)
    matrices: dict[str, np.ndarray] = {}
    for metrics_path in sorted(run_dir.glob("**/metrics.json")):
        payload = load_json(metrics_path)
        cm = payload.get("confusion_matrix")
        if cm is None:
            continue
        key = str(metrics_path.parent.relative_to(run_dir)).replace("\\", "/")
        matrices[key] = np.asarray(cm, dtype=np.float64)
    if not matrices:
        raise FileNotFoundError(f"No confusion matrices in {run_dir}")
    return matrices


def load_training_log(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty training log: {path}")
    keys = [k for k in rows[0] if k]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        try:
            out[key] = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        except (TypeError, ValueError):
            continue
    return out


def collect_logs_from_run(run_dir: Path) -> list[dict[str, np.ndarray]]:
    logs = []
    for csv_path in sorted(Path(run_dir).glob("**/training_log.csv")):
        logs.append(load_training_log(csv_path))
    if not logs:
        raise FileNotFoundError(f"No training_log.csv under {run_dir}")
    return logs
