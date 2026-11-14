"""SEED / SEED-IV 62-channel names and circular anatomical order for Figure 5."""
from __future__ import annotations

SEED62_CHANNELS: tuple[str, ...] = (
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
)

# Clockwise order around the scalp for chord diagrams (excludes CB1/CB2).
CIRCULAR_ORDER: tuple[str, ...] = (
    "FP1", "AF3", "F7", "F5", "F3", "F1", "FC5", "FC3", "FC1",
    "FT7", "T7", "TP7", "C5", "C3", "C1",
    "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1",
    "PO7", "PO5", "PO3", "O1", "OZ", "O2", "PO4", "PO6", "PO8",
    "P2", "P4", "P6", "P8", "CP2", "CP4", "CP6",
    "C2", "C4", "C6", "T8", "FT8", "TP8",
    "FC2", "FC4", "FC6", "F2", "F4", "F6", "F8", "AF4", "FP2",
    "FPZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ",
)

REGION_OF: dict[str, str] = {}
for _name, _chs in (
    ("Frontal", ("FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6")),
    ("Temporal", ("FT7", "FT8", "T7", "T8", "TP7", "TP8")),
    ("Central", ("C5", "C3", "C1", "CZ", "C2", "C4", "C6")),
    ("Parietal", ("CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8")),
    ("Occipital", ("PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2")),
):
    for _ch in _chs:
        REGION_OF[_ch] = _name

REGION_COLORS = {
    "Frontal": "#4C78A8",
    "Temporal": "#F58518",
    "Central": "#54A24B",
    "Parietal": "#E45756",
    "Occipital": "#72B7B2",
}


def channel_index(name: str) -> int:
    return SEED62_CHANNELS.index(name)
