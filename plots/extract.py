"""Export arrays from a trained ``last_epoch.pt`` for Figures 3–6.

    python plot_figures.py extract --ckpt last_epoch.pt --data_root ... --label_path ... \\
        --out_dir exports/subject_1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import BayesianClassifier, ClassConditionalPriorEncoder

from .entropy_reliability import predictive_entropy


@torch.no_grad()
def _load_models(ckpt_path: Path, device: torch.device):
    payload = torch.load(ckpt_path, map_location=device)
    args = payload.get("args", {})
    encoder_state = payload["encoder"]
    # Infer dims from state when CLI args are incomplete.
    channels = int(args.get("channels", encoder_state["channel_pos"].shape[1]))
    de_dim = int(args.get("de_dim", encoder_state["channel_projection.weight_mu"].shape[1]))
    latent_dim = int(args.get("latent_dim", encoder_state["posterior_mu_layer.weight_mu"].shape[0]))
    num_classes = int(args.get("num_classes", encoder_state["class_prior_mu"].shape[0]))
    encoder = ClassConditionalPriorEncoder(
        channels=channels,
        de_dim=de_dim,
        latent_dim=latent_dim,
        num_classes=num_classes,
        dropout=float(args.get("dropout", 0.35)),
        spatial_layers=int(args.get("spatial_layers", 2)),
        graph_dropout=float(args.get("dropout", 0.35)),
        graph_rank=int(args.get("graph_rank", 12)),
        use_band_attention=True,
        use_spatial_graph=True,
        fusion_dims=tuple(args.get("fusion_dims", [128])),
        channel_hidden=int(args.get("channel_hidden", 64)),
    ).to(device)
    classifier = BayesianClassifier(
        latent_dim=latent_dim,
        num_classes=num_classes,
        hidden_dim=int(args.get("classifier_hidden", 128)),
        dropout=float(args.get("dropout", 0.35)),
    ).to(device)
    encoder.load_state_dict(encoder_state, strict=False)
    classifier.load_state_dict(payload["classifier"], strict=False)
    encoder.eval()
    classifier.eval()
    return encoder, classifier, args


def extract_arrays(
    ckpt: Path,
    x: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    mc_samples: int = 12,
    batch_size: int = 64,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, classifier, _ = _load_models(Path(ckpt), device)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    mu, scales, adjs, probs, targets = [], [], [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        mu_q, _, _ = encoder.encode_posterior(xb, sample_weights=False)
        mu.append(mu_q.cpu().numpy())
        scales.append(encoder.band_scales(xb).squeeze(1).cpu().numpy())
        if encoder.spatial_blocks:
            adjs.append(encoder.relation_matrices(xb, layer="last").cpu().numpy())
        samples = []
        for _ in range(max(1, mc_samples)):
            mu_s, logvar_s, _ = encoder.encode_posterior(xb, sample_weights=True)
            z = encoder.reparameterize(mu_s, logvar_s, sample_z=True)
            logits = classifier(z, sample_weights=True)
            samples.append(torch.softmax(logits, dim=1))
        mean_p = torch.stack(samples, dim=0).mean(dim=0)
        probs.append(mean_p.cpu().numpy())
        targets.append(yb.numpy())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_all = np.concatenate(targets)
    scale = np.concatenate(scales)
    prob = np.concatenate(probs)
    np.savez_compressed(out_dir / "latent.npz", mu_q=np.concatenate(mu), x=np.asarray(x), y=y_all)
    np.savez_compressed(out_dir / "spectral.npz", scale=scale, y=y_all)
    if adjs:
        adj = np.concatenate(adjs)
        class_adj = []
        for cls in range(int(y_all.max()) + 1):
            mask = y_all == cls
            class_adj.append(adj[mask].mean(axis=0) if np.any(mask) else adj.mean(axis=0))
        np.savez_compressed(out_dir / "adjacency.npz", adj=np.stack(class_adj), y=y_all)
    np.savez_compressed(
        out_dir / "predictions.npz",
        prob=prob,
        y=y_all,
        entropy=predictive_entropy(prob),
        correct=(prob.argmax(axis=1) == y_all),
    )
    (out_dir / "extract_meta.json").write_text(
        json.dumps({"n": int(len(y_all)), "ckpt": str(ckpt), "mc_samples": mc_samples}, indent=2),
        encoding="utf-8",
    )
    return out_dir


def main_cli(args: argparse.Namespace) -> Path:
    from data import load_subjects

    data = load_subjects(args.data_root, label_path=args.label_path, feature_subdir=args.feature_subdir, sessions=args.sessions)
    names = list(data.names)
    if args.subject not in names:
        raise ValueError(f"Unknown subject {args.subject!r}. Available: {names}")
    idx = names.index(args.subject)
    return extract_arrays(Path(args.ckpt), data.xs[idx], data.ys[idx], Path(args.out_dir), mc_samples=args.eval_samples)
