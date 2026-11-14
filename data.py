"""Load subject-wise DE arrays ``[N, C, F]`` for within-subject CV and LOSO.

Expected layouts (features are never shipped with this repository):

- Flat: ``<root>/DE/<subject>.npy`` plus ``--label_path`` or ``<root>/label/<dim>/all.npy``
- Sessions (SEED / SEED-IV): ``<root>/<session>/<subject>.npy``
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def remap_labels(ys: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], Dict[int, int]]:
    raw = np.sort(np.unique(np.concatenate([np.asarray(y).reshape(-1) for y in ys]))).astype(np.int64)
    if raw.size < 2:
        raise ValueError(f"Need at least two classes, got {raw.tolist()}.")
    mapping = {int(v): i for i, v in enumerate(raw.tolist())}
    mapped = [np.asarray([mapping[int(v)] for v in np.asarray(y).reshape(-1)], dtype=np.int64) for y in ys]
    return mapped, mapping


def class_weights_from_labels(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency weights for the class-weighted CE in Sec. 3.4.1."""
    counts = np.bincount(np.asarray(y).reshape(-1), minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (np.clip(counts, 1.0, None) * num_classes)
    return weights.astype(np.float32)


def _load_label_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    return np.load(path).astype(np.int64).reshape(-1)


def _align_flat_labels(labels_all: np.ndarray, subject_idx: int, n_trials: int, n_subjects: int) -> np.ndarray:
    n = int(labels_all.size)
    if n == n_trials:
        return labels_all
    if n == n_subjects * n_trials:
        start = subject_idx * n_trials
        return labels_all[start : start + n_trials]
    raise ValueError(
        f"Label length {n} does not match n_windows={n_trials} or S*N={n_subjects * n_trials}."
    )


def _align_session_labels(
    labels_all: np.ndarray,
    subject_idx: int,
    session_idx: int,
    n_sessions: int,
    n_trials: int,
    n_subjects: int,
) -> np.ndarray:
    n = int(labels_all.size)
    if n == n_trials:
        return labels_all
    if n == n_subjects * n_trials:
        return _align_flat_labels(labels_all, subject_idx, n_trials, n_subjects)
    if n == n_subjects * n_sessions * n_trials:
        start = subject_idx * n_sessions * n_trials + session_idx * n_trials
        return labels_all[start : start + n_trials]
    if n == n_sessions * n_trials:
        return labels_all[session_idx * n_trials : (session_idx + 1) * n_trials]
    raise ValueError("Label length does not match the subject/session layout.")


@dataclass(frozen=True)
class SubjectSplit:
    names: Tuple[str, ...]
    xs: Tuple[np.ndarray, ...]
    ys: Tuple[np.ndarray, ...]
    channels: int
    de_dim: int
    num_classes: int
    label_map: Dict[int, int]
    meta: Dict[str, Any]


def load_subjects(
    data_root: Union[str, Path],
    label_path: Optional[Union[str, Path]] = None,
    feature_subdir: str = "DE",
    sessions: Optional[Sequence[Union[str, int]]] = None,
) -> SubjectSplit:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    labels_all = _load_label_array(Path(label_path)) if label_path else None

    session_ids: List[str] = []
    if sessions:
        session_ids = [str(v) for v in sessions]
    else:
        numeric = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=int)
        if numeric and not (root / feature_subdir).is_dir():
            session_ids = numeric

    if session_ids:
        first = root / session_ids[0]
        names = sorted(
            [p.stem for p in first.glob("*.npy") if p.name != "labels.npy" and not p.stem.endswith("_labels")],
            key=lambda v: int(v) if v.isdigit() else v,
        )
        if not names:
            raise FileNotFoundError(f"No subject .npy files in {first}")
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        n_subjects = len(names)
        n_sessions = len(session_ids)
        session_labels = _load_label_array(first / "labels.npy") if labels_all is None and (first / "labels.npy").is_file() else None
        for subject_idx, name in enumerate(names):
            x_parts: List[np.ndarray] = []
            y_parts: List[np.ndarray] = []
            for session_idx, session in enumerate(session_ids):
                path = root / session / f"{name}.npy"
                if not path.is_file():
                    raise FileNotFoundError(path)
                x = np.load(path).astype(np.float32)
                if x.ndim != 3:
                    raise ValueError(f"Expected [N,C,F] in {path}, got {x.shape}")
                local = root / session / "labels.npy"
                source = labels_all
                if source is None and local.is_file():
                    source = _load_label_array(local)
                elif source is None:
                    source = session_labels
                if source is None:
                    raise FileNotFoundError("Provide --label_path or per-session labels.npy")
                y = _align_session_labels(source, subject_idx, session_idx, n_sessions, x.shape[0], n_subjects)
                x_parts.append(x)
                y_parts.append(y)
            xs.append(np.concatenate(x_parts, axis=0))
            ys.append(np.concatenate(y_parts, axis=0))
        layout = "sessions"
    else:
        feature_root = root / feature_subdir if feature_subdir not in {"", "."} else root
        if not feature_root.is_dir():
            raise FileNotFoundError(f"Feature directory not found: {feature_root}")
        names = sorted(
            [p.stem for p in feature_root.glob("*.npy") if p.name != "labels.npy" and not p.stem.endswith("_labels")],
            key=lambda v: int(v) if v.isdigit() else v,
        )
        if not names:
            raise FileNotFoundError(f"No subject .npy files in {feature_root}")
        if labels_all is None:
            for cand in (feature_root / "labels.npy", root / "labels.npy"):
                if cand.is_file():
                    labels_all = _load_label_array(cand)
                    break
        xs = []
        ys = []
        n_subjects = len(names)
        for subject_idx, name in enumerate(names):
            x = np.load(feature_root / f"{name}.npy").astype(np.float32)
            if x.ndim != 3:
                raise ValueError(f"Expected [N,C,F] for subject {name}, got {x.shape}")
            local = feature_root / f"{name}_labels.npy"
            if local.is_file():
                y = _load_label_array(local)
            elif labels_all is not None:
                y = _align_flat_labels(labels_all, subject_idx, x.shape[0], n_subjects)
            else:
                raise FileNotFoundError("Provide --label_path or <subject>_labels.npy")
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"X/y length mismatch for subject {name}: {x.shape[0]} vs {y.shape[0]}")
            xs.append(x)
            ys.append(y)
        layout = "flat"

    ys, label_map = remap_labels(ys)
    channels = int(xs[0].shape[1])
    de_dim = int(xs[0].shape[2])
    num_classes = int(max(int(y.max()) for y in ys) + 1)
    return SubjectSplit(
        names=tuple(names),
        xs=tuple(xs),
        ys=tuple(ys),
        channels=channels,
        de_dim=de_dim,
        num_classes=num_classes,
        label_map=label_map,
        meta={"layout": layout, "root": str(root), "sessions": session_ids},
    )
