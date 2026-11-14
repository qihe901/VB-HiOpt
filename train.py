"""Training entry for VB-HiOpt, aligned with Sec. 3.4–3.5 and Sec. 4.2.

Implements Gauss–Seidel alternating updates (Eqs. 47–50), stage-wise KL
weights (Eq. 46), AdamW with separate encoder/classifier learning rates,
gradient clipping, mild band-scale + Gaussian perturbation, last-epoch
reporting, stratified within-subject 5-fold CV, and LOSO.

Not included (absent from the PDF): train-set z-score (``train_global``),
channel/band dropout, and oracle best-test-epoch selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from data import SubjectSplit, class_weights_from_labels, load_subjects
from loss import VBHiOptLoss
from models import BayesianClassifier, ClassConditionalPriorEncoder


def set_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def set_parameter_list_requires_grad(parameters: Sequence[nn.Parameter], value: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad = value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_de(x: torch.Tensor, noise_std: float, scale_std: float) -> torch.Tensor:
    """Mild frequency-band scaling and Gaussian noise (Sec. 3.5)."""
    out = x
    if scale_std > 0:
        scale = 1.0 + scale_std * torch.randn(x.size(0), 1, x.size(2), device=x.device)
        out = out * scale.clamp(0.80, 1.20)
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return out


def _clip(parameters: Sequence[nn.Parameter], max_norm: float) -> float:
    if max_norm is None or max_norm <= 0:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(list(parameters), max_norm))


def _metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, Any]:
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    f1s: List[float] = []
    recalls: List[float] = []
    for truth, guess in zip(target.tolist(), pred.tolist()):
        confusion[int(truth), int(guess)] += 1
    for cls in range(num_classes):
        tp = int(((pred == cls) & (target == cls)).sum())
        fp = int(((pred == cls) & (target != cls)).sum())
        fn = int(((pred != cls) & (target == cls)).sum())
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        recalls.append(float(recall))
        f1s.append(float(f1))
    return {
        "accuracy": float((pred == target).float().mean()),
        "f1_macro": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion_matrix": confusion.tolist(),
    }


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    num_samples: int,
    sample_z: bool,
    sample_weights: bool,
) -> Dict[str, Any]:
    """Deterministic decode (S=1, no sampling) or MC mean / entropy / variance (Eqs. 39–42)."""
    encoder.eval()
    classifier.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    variances: List[torch.Tensor] = []

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        samples: List[torch.Tensor] = []
        for _ in range(max(1, num_samples)):
            mu_q, logvar_q, _ = encoder.encode_posterior(x, sample_weights=sample_weights)
            z = encoder.reparameterize(mu_q, logvar_q, sample_z=sample_z)
            logits = classifier(z, sample_weights=sample_weights)
            samples.append(torch.softmax(logits, dim=1))
        stacked = torch.stack(samples, dim=0)
        mean_p = stacked.mean(dim=0)
        preds.append(mean_p.argmax(dim=1).cpu())
        targets.append(y.cpu())
        entropies.append((-(mean_p * mean_p.clamp_min(1e-12).log()).sum(dim=1)).cpu())
        variances.append(stacked.var(dim=0, unbiased=False).mean(dim=1).cpu())

    pred = torch.cat(preds)
    target = torch.cat(targets)
    result = _metrics(pred, target, num_classes)
    result["entropy"] = float(torch.cat(entropies).mean())
    result["variance"] = float(torch.cat(variances).mean())
    result["num_samples"] = int(max(1, num_samples))
    result["sample_z"] = bool(sample_z)
    result["sample_weights"] = bool(sample_weights)
    return result


def build_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def build_models(args: argparse.Namespace, channels: int, de_dim: int, num_classes: int, device: torch.device):
    encoder = ClassConditionalPriorEncoder(
        channels=channels,
        de_dim=de_dim,
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        channel_hidden=args.channel_hidden,
        fusion_dims=args.fusion_dims,
        dropout=args.dropout,
        spatial_layers=args.spatial_layers,
        graph_dropout=args.dropout,
        graph_rank=args.graph_rank,
        use_band_attention=True,
        use_spatial_graph=True,
        band_attention_hidden=args.band_attention_hidden,
        band_residual_scale=args.band_residual_scale,
    ).to(device)
    classifier = BayesianClassifier(
        latent_dim=args.latent_dim,
        num_classes=num_classes,
        hidden_dim=args.classifier_hidden,
        dropout=args.dropout,
    ).to(device)
    return encoder, classifier


def train_alternating_epoch(
    encoder: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    opt_encoder: optim.Optimizer,
    opt_classifier: optim.Optimizer,
    loss_fn: VBHiOptLoss,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, float]:
    inner_params = encoder.inner_parameters()
    sample_z = epoch >= args.stochastic_z_start
    sample_weights = epoch >= args.weight_sampling_start
    sums = {
        "inner_loss": 0.0,
        "classifier_loss": 0.0,
        "ce": 0.0,
        "latent_kl": 0.0,
        "inner_grad_norm": 0.0,
        "classifier_grad_norm": 0.0,
        "stationarity_proxy": 0.0,
    }
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        n = int(y.numel())
        x_in = augment_de(x, args.noise_std, args.scale_std)

        encoder.train()
        classifier.eval()
        set_parameter_list_requires_grad(inner_params, True)
        set_requires_grad(classifier, False)
        opt_encoder.zero_grad(set_to_none=True)
        inner_loss, inner_terms = loss_fn.posterior_objective(
            encoder,
            classifier,
            x_in,
            y,
            num_samples=args.num_samples,
            sample_encoder_weights=sample_weights,
            sample_z=sample_z,
        )
        inner_loss.backward()
        inner_grad = _clip(inner_params, args.max_grad_norm)
        opt_encoder.step()

        encoder.eval()
        classifier.train()
        set_parameter_list_requires_grad(inner_params, False)
        set_requires_grad(classifier, True)
        opt_classifier.zero_grad(set_to_none=True)
        classifier_loss, _ = loss_fn.classifier_objective(
            encoder,
            classifier,
            x_in,
            y,
            num_samples=args.num_samples,
            sample_classifier_weights=sample_weights,
            sample_z=sample_z,
        )
        classifier_loss.backward()
        classifier_grad = _clip(list(classifier.parameters()), args.max_grad_norm)
        opt_classifier.step()

        sums["inner_loss"] += float(inner_loss.detach()) * n
        sums["classifier_loss"] += float(classifier_loss.detach()) * n
        sums["ce"] += float(inner_terms["ce"].detach()) * n
        sums["latent_kl"] += float(inner_terms["latent_kl"].detach()) * n
        sums["inner_grad_norm"] += inner_grad * n
        sums["classifier_grad_norm"] += classifier_grad * n
        sums["stationarity_proxy"] += (inner_grad * inner_grad + classifier_grad * classifier_grad) * n
        total += n

    set_parameter_list_requires_grad(inner_params, True)
    set_requires_grad(classifier, True)
    denom = max(1, total)
    return {key: value / denom for key, value in sums.items()}


def train_last_epoch(
    args: argparse.Namespace,
    device: torch.device,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    fold_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    fold_dir.mkdir(parents=True, exist_ok=True)
    channels, de_dim = int(x_train.shape[1]), int(x_train.shape[2])
    encoder, classifier = build_models(args, channels, de_dim, num_classes, device)
    weights = torch.as_tensor(class_weights_from_labels(y_train, num_classes), dtype=torch.float32, device=device)
    loss_fn = VBHiOptLoss(
        dataset_size=len(y_train),
        class_weights=weights,
        latent_kl_weight=args.latent_kl_weight,
        param_kl_weight=args.param_kl_weight,
        prior_aux_weight=args.prior_aux_weight,
        latent_kl_delay=args.latent_kl_delay,
        latent_kl_warmup=args.latent_kl_warmup,
        param_kl_delay=args.param_kl_delay,
        param_kl_warmup=args.param_kl_warmup,
        prior_aux_warmup=args.prior_aux_warmup,
        label_smoothing=args.label_smoothing,
    ).to(device)

    opt_encoder = optim.AdamW(encoder.inner_parameters(), lr=args.lr_encoder, weight_decay=args.weight_decay)
    opt_classifier = optim.AdamW(classifier.parameters(), lr=args.lr_classifier, weight_decay=args.weight_decay)
    train_loader = build_loader(x_train, y_train, args.batch_size, shuffle=True, seed=seed)
    test_loader = build_loader(x_test, y_test, args.batch_size, shuffle=False, seed=seed)

    log_path = fold_dir / "training_log.csv"
    fields = [
        "epoch",
        "inner_loss",
        "classifier_loss",
        "ce",
        "latent_kl",
        "inner_grad_norm",
        "classifier_grad_norm",
        "stationarity_proxy",
        "total_loss",
        "beta_z",
        "beta_w",
        "beta_p",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(args.epochs):
            loss_fn.set_epoch(epoch)
            terms = train_alternating_epoch(
                encoder, classifier, train_loader, opt_encoder, opt_classifier, loss_fn, device, args, epoch
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    **terms,
                    "total_loss": terms["inner_loss"] + terms["classifier_loss"],
                    "beta_z": loss_fn.latent_kl_weight,
                    "beta_w": loss_fn.param_kl_weight,
                    "beta_p": loss_fn.prior_aux_weight,
                }
            )
            if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % args.log_interval == 0:
                print(
                    f"  epoch {epoch:03d} inner={terms['inner_loss']:.4f} "
                    f"dec={terms['classifier_loss']:.4f} ce={terms['ce']:.4f}",
                    flush=True,
                )

    det = evaluate(encoder, classifier, test_loader, device, num_classes, 1, False, False)
    mc = evaluate(
        encoder,
        classifier,
        test_loader,
        device,
        num_classes,
        args.eval_samples,
        True,
        True,
    )
    torch.save(
        {"encoder": encoder.state_dict(), "classifier": classifier.state_dict(), "args": vars(args)},
        fold_dir / "last_epoch.pt",
    )
    result = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "selection": "last_training_epoch",
        "deterministic": {k: det[k] for k in ("accuracy", "f1_macro", "balanced_accuracy", "entropy", "variance")},
        "monte_carlo": {k: mc[k] for k in ("accuracy", "f1_macro", "balanced_accuracy", "entropy", "variance")},
        "confusion_matrix": det["confusion_matrix"],
    }
    (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(np.zeros(len(y)), y)]


def run_ws5fold(args: argparse.Namespace, data: SubjectSplit, save_dir: Path, device: torch.device) -> Dict[str, Any]:
    subject_rows: List[Dict[str, Any]] = []
    for subject_idx, name in enumerate(data.names):
        if args.only_subjects and name not in args.only_subjects:
            continue
        x = data.xs[subject_idx]
        y = data.ys[subject_idx]
        subject_dir = save_dir / f"subject_{name}"
        fold_seed = args.seed + subject_idx * 1009
        splits = _stratified_folds(y, args.n_folds, fold_seed)
        fold_metrics: List[Dict[str, Any]] = []
        print(f"\n=== subject {name} window-level stratified {args.n_folds}-fold ===", flush=True)
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if args.only_fold_indices is not None and fold_idx not in args.only_fold_indices:
                continue
            print(f" subject={name} fold={fold_idx}", flush=True)
            fold_metrics.append(
                train_last_epoch(
                    args,
                    device,
                    x[train_idx],
                    y[train_idx],
                    x[test_idx],
                    y[test_idx],
                    data.num_classes,
                    subject_dir / f"fold_{fold_idx}",
                    fold_seed + fold_idx * 17,
                )
            )
        acc = np.array([m["deterministic"]["accuracy"] for m in fold_metrics], dtype=np.float64)
        row = {
            "subject": name,
            "mean_acc": float(acc.mean()),
            "std_acc": float(acc.std(ddof=0)),
            "folds": fold_metrics,
        }
        (subject_dir / "subject_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        subject_rows.append(row)
        print(f"[subject {name}] last-epoch det acc={row['mean_acc']:.4f}", flush=True)

    acc = np.array([r["mean_acc"] for r in subject_rows], dtype=np.float64)
    summary = {
        "protocol": "within_subject_stratified_kfold",
        "n_folds": args.n_folds,
        "selection": "last_training_epoch",
        "mean_acc": float(acc.mean()) if acc.size else 0.0,
        "std_acc": float(acc.std(ddof=0)) if acc.size else 0.0,
        "subjects": subject_rows,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_loso(args: argparse.Namespace, data: SubjectSplit, save_dir: Path, device: torch.device) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for test_idx, name in enumerate(data.names):
        if args.only_subjects and name not in args.only_subjects:
            continue
        train_idx = [i for i in range(len(data.names)) if i != test_idx]
        x_train = np.concatenate([data.xs[i] for i in train_idx], axis=0)
        y_train = np.concatenate([data.ys[i] for i in train_idx], axis=0)
        print(f"\n=== LOSO hold-out subject {name} ===", flush=True)
        metrics = train_last_epoch(
            args,
            device,
            x_train,
            y_train,
            data.xs[test_idx],
            data.ys[test_idx],
            data.num_classes,
            save_dir / f"subject_{name}",
            args.seed + test_idx * 1009,
        )
        row = {"subject": name, **metrics}
        rows.append(row)
        print(f"[LOSO {name}] last-epoch det acc={metrics['deterministic']['accuracy']:.4f}", flush=True)

    acc = np.array([r["deterministic"]["accuracy"] for r in rows], dtype=np.float64)
    summary = {
        "protocol": "leave_one_subject_out",
        "selection": "last_training_epoch",
        "mean_acc": float(acc.mean()) if acc.size else 0.0,
        "std_acc": float(acc.std(ddof=0)) if acc.size else 0.0,
        "subjects": rows,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def dummy_split(args: argparse.Namespace) -> SubjectSplit:
    rng = np.random.default_rng(args.seed)
    xs = []
    ys = []
    for _ in range(2):
        y = np.repeat(np.arange(args.num_classes), 20)
        rng.shuffle(y)
        x = rng.normal(size=(y.shape[0], args.channels, args.de_dim)).astype(np.float32)
        xs.append(x)
        ys.append(y.astype(np.int64))
    return SubjectSplit(
        names=("1", "2"),
        xs=tuple(xs),
        ys=tuple(ys),
        channels=args.channels,
        de_dim=args.de_dim,
        num_classes=args.num_classes,
        label_map={i: i for i in range(args.num_classes)},
        meta={"layout": "dummy"},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VB-HiOpt training (PDF Sec. 3.4–3.5, 4.2)")
    parser.add_argument("--protocol", choices=("ws5fold", "loso", "smoke"), default="ws5fold")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument("--feature_subdir", type=str, default="DE")
    parser.add_argument("--dimension", type=str, default=None, help="Use <data_root>/label/<dimension>/all.npy")
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--save_dir", type=str, default="runs/vbhiopt")
    parser.add_argument("--only_subjects", nargs="*", default=None)
    parser.add_argument("--only_fold_indices", type=int, nargs="*", default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--channels", type=int, default=62)
    parser.add_argument("--de_dim", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=48)
    parser.add_argument("--channel_hidden", type=int, default=64)
    parser.add_argument("--fusion_dims", type=int, nargs="*", default=[128])
    parser.add_argument("--classifier_hidden", type=int, default=128)
    parser.add_argument("--spatial_layers", type=int, default=2)
    parser.add_argument("--graph_rank", type=int, default=12)
    parser.add_argument("--band_attention_hidden", type=int, default=24)
    parser.add_argument("--band_residual_scale", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr_encoder", type=float, default=5e-4)
    parser.add_argument("--lr_classifier", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--max_grad_norm", type=float, default=3.0, help="PDF applies clipping; threshold unstated.")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--eval_samples", type=int, default=12, help="MC samples S in Eqs. 39–42; S unstated in Sec. 4.2.")
    parser.add_argument("--stochastic_z_start", type=int, default=0)
    parser.add_argument("--weight_sampling_start", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.025)
    parser.add_argument("--scale_std", type=float, default=0.04)
    parser.add_argument("--log_interval", type=int, default=20)

    parser.add_argument("--latent_kl_weight", type=float, default=1.5e-3)
    parser.add_argument("--param_kl_weight", type=float, default=1e-6)
    parser.add_argument("--prior_aux_weight", type=float, default=0.05)
    parser.add_argument("--latent_kl_delay", type=int, default=15)
    parser.add_argument("--latent_kl_warmup", type=int, default=40)
    parser.add_argument("--param_kl_delay", type=int, default=0)
    parser.add_argument("--param_kl_warmup", type=int, default=20)
    parser.add_argument("--prior_aux_warmup", type=int, default=15)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.protocol == "smoke":
        args.epochs = min(args.epochs, 2)
        args.n_folds = 2
        args.eval_samples = 2
        data = dummy_split(args)
        summary = run_ws5fold(args, data, save_dir, device)
        print(json.dumps({"mean_acc": summary["mean_acc"], "protocol": summary["protocol"]}, indent=2))
        return

    if not args.data_root:
        raise ValueError("--data_root is required unless --protocol smoke")
    label_path = args.label_path
    if label_path is None and args.dimension:
        label_path = str(Path(args.data_root) / "label" / args.dimension / "all.npy")
    data = load_subjects(args.data_root, label_path=label_path, feature_subdir=args.feature_subdir, sessions=args.sessions)
    args.channels, args.de_dim, args.num_classes = data.channels, data.de_dim, data.num_classes
    (save_dir / "data_meta.json").write_text(json.dumps(data.meta | {"label_map": data.label_map}, indent=2), encoding="utf-8")

    if args.protocol == "ws5fold":
        summary = run_ws5fold(args, data, save_dir, device)
    else:
        summary = run_loso(args, data, save_dir, device)
    print(
        f"{summary['protocol']} last-epoch det acc={summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
