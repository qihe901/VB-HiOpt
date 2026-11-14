"""CLI: ``python plot_figures.py confusion --dataset seed --demo``."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import (
    channel_chords,
    confusion_matrices,
    entropy_reliability,
    extract as extract_mod,
    optimization_curves,
    spectral_attention,
    tsne_latent,
)


def _add_out(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--out", default=default)
    parser.add_argument("--demo", action="store_true", help="Draw the figure from synthetic arrays.")
    parser.add_argument("--dpi", type=int, default=300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot_figures.py",
        description="Reproduce paper Figures 2–7 from exported arrays or --demo.",
    )
    sub = parser.add_subparsers(dest="figure", required=True)

    p2 = sub.add_parser("confusion", aliases=["fig2"], help="Session-wise confusion matrices (one dataset)")
    _add_out(p2, "figures/fig2.png")
    p2.add_argument("--dataset", required=True, choices=("seed", "seediv"), help="Which dataset to draw")
    p2.add_argument("--cm_dir", help="Folder of session confusion .npy for the chosen dataset")
    p2.add_argument("--npz", nargs="*", help="Per-session npz with key confusion")
    p2.add_argument("--run", help="Training run folder containing metrics.json")
    p2.add_argument("--seed_npz", nargs="*")
    p2.add_argument("--seediv_npz", nargs="*")
    p2.add_argument("--seed_run")
    p2.add_argument("--seediv_run")
    p2.add_argument("--seed_cm_dir")
    p2.add_argument("--seediv_cm_dir")

    p3 = sub.add_parser("tsne", aliases=["fig3"], help="t-SNE of Raw DE / VAE / w/o prior / VB-HiOpt")
    _add_out(p3, "figures/fig3.png")
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--de_npz")
    p3.add_argument("--vae_npz")
    p3.add_argument("--noprior_npz")
    p3.add_argument("--vbhiopt_npz")
    for prefix in ("seed", "seediv"):
        p3.add_argument(f"--{prefix}_raw")
        p3.add_argument(f"--{prefix}_vae")
        p3.add_argument(f"--{prefix}_noprior")
        p3.add_argument(f"--{prefix}_vbhiopt")

    p4 = sub.add_parser("spectral", aliases=["fig4"], help="Category-wise spectral Δw (one dataset)")
    _add_out(p4, "figures/fig4.png")
    p4.add_argument("--dataset", required=True, choices=("seed", "seediv"), help="Which dataset to draw")
    p4.add_argument("--npz", help="Frequency-attention npz for the chosen dataset")
    p4.add_argument("--seed_npz")
    p4.add_argument("--seediv_npz")

    p5 = sub.add_parser("chord", aliases=["fig5"], help="Dynamic channel relation chords (one dataset)")
    _add_out(p5, "figures/fig5.png")
    p5.add_argument("--dataset", required=True, choices=("seed", "seediv"), help="Which dataset to draw")
    p5.add_argument("--npz", help="Adjacency .npy/.npz for the chosen dataset")
    p5.add_argument("--seed_npz")
    p5.add_argument("--seediv_npz")
    p5.add_argument("--top_k", type=int, default=30)

    p6 = sub.add_parser("uncertainty", aliases=["fig6"], help="Predictive entropy and reliability")
    _add_out(p6, "figures/fig6.png")
    p6.add_argument("--vbhiopt_npz")
    p6.add_argument("--softmax_npz")

    p7 = sub.add_parser("curves", aliases=["fig7"], help="Alternating vs joint training trajectories")
    _add_out(p7, "figures/fig7.png")
    p7.add_argument("--alternating_run")
    p7.add_argument("--joint_run")
    p7.add_argument("--alternating_log", nargs="*")
    p7.add_argument("--joint_log", nargs="*")
    p7.add_argument("--curves_csv")

    p_all = sub.add_parser("all", help="Render fig2–fig7 with --demo")
    p_all.add_argument("--out_dir", default="figures")

    p_od = sub.add_parser("from_outdir", help="Draw fig2–fig7 from manuscript outdir arrays")
    p_od.add_argument("--repo_root", default=None, help="Folder that contains outdir/ (default: parent of VB-HiOpt)")
    p_od.add_argument("--out_dir", default="figures")
    p_od.add_argument(
        "--dataset",
        choices=("seed", "seediv", "both"),
        default="both",
        help="For fig2/fig4/fig5: seed, seediv, or both as separate files (default both)",
    )

    p_ex = sub.add_parser("extract", help="Export latent / spectral / adj / MC arrays from a checkpoint")
    p_ex.add_argument("--ckpt", required=True)
    p_ex.add_argument("--data_root", required=True)
    p_ex.add_argument("--label_path")
    p_ex.add_argument("--feature_subdir", default="DE")
    p_ex.add_argument("--sessions", nargs="*")
    p_ex.add_argument("--subject", required=True)
    p_ex.add_argument("--out_dir", default="exports")
    p_ex.add_argument("--eval_samples", type=int, default=12)
    return parser


def _repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[2]


def run_from_outdir(repo_root: Path, out_dir: Path, dataset: str = "both") -> None:
    outdir = repo_root / "outdir"
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = ("seed", "seediv") if dataset == "both" else (dataset,)
    jobs = []
    fig4_npz = {
        "seed": outdir / "seed" / "819" / "s3_p0_full" / "band_attention" / "frequency_attention_all.npz",
        "seediv": outdir / "seediv" / "819" / "s3_p0" / "band_attention" / "frequency_attention_all.npz",
    }
    fig5_npz = {
        "seed": outdir / "seed" / "chord_prior" / "emotion_adj_matrix_seed_prior_best.npy",
        "seediv": outdir / "seediv" / "819" / "s3_p0" / "dynamic_adjacency" / "emotion_adj_matrix_prior_best.npy",
    }
    for ds in datasets:
        suffix = f"_{ds}" if dataset == "both" else ""
        jobs.append(
            (
                f"fig2{suffix}",
                confusion_matrices.main_cli,
                argparse.Namespace(
                    demo=False,
                    dataset=ds,
                    out=str(out_dir / f"fig2{suffix}.png"),
                    cm_dir=str(outdir / ds / "confusion_matrices_best_per_subject"),
                    seed_cm_dir=None,
                    seediv_cm_dir=None,
                    npz=None,
                    seed_npz=None,
                    seediv_npz=None,
                    run=None,
                    seed_run=None,
                    seediv_run=None,
                ),
            )
        )
        jobs.append(
            (
                f"fig4{suffix}",
                spectral_attention.main_cli,
                argparse.Namespace(
                    demo=False,
                    dataset=ds,
                    out=str(out_dir / f"fig4{suffix}.png"),
                    npz=str(fig4_npz[ds]),
                    seed_npz=None,
                    seediv_npz=None,
                ),
            )
        )
        jobs.append(
            (
                f"fig5{suffix}",
                channel_chords.main_cli,
                argparse.Namespace(
                    demo=False,
                    dataset=ds,
                    out=str(out_dir / f"fig5{suffix}.png"),
                    npz=str(fig5_npz[ds]),
                    seed_npz=None,
                    seediv_npz=None,
                    top_k=30,
                ),
            )
        )
    jobs.extend(
        [
            (
                "fig3",
                tsne_latent.main_cli,
                argparse.Namespace(
                    demo=False,
                    out=str(out_dir / "fig3.png"),
                    seed=42,
                    de_npz=str(outdir / "seediv" / "819" / "s3_p0" / "latent_viz" / "latent_all_vbhiopt.npz"),
                    vae_npz=str(outdir / "seediv" / "819" / "s3_p0" / "latent_viz" / "vae" / "latent_all_vae_session.npz"),
                    noprior_npz=str(outdir / "seediv" / "819" / "s3_p0" / "latent_viz" / "no_prior" / "latent_all_no_prior_session.npz"),
                    vbhiopt_npz=str(outdir / "seediv" / "819" / "s3_p0" / "latent_viz" / "latent_all_vbhiopt.npz"),
                    seed_raw=None,
                    seed_vae=None,
                    seed_noprior=None,
                    seed_vbhiopt=None,
                    seediv_raw=None,
                    seediv_vae=None,
                    seediv_noprior=None,
                    seediv_vbhiopt=None,
                ),
            ),
            (
                "fig6",
                entropy_reliability.main_cli,
                argparse.Namespace(
                    demo=False,
                    out=str(out_dir / "fig6.png"),
                    vbhiopt_npz=str(outdir / "seediv" / "819" / "s3_p0" / "uncertainty_viz" / "predictions_all_vbhiopt.npz"),
                    softmax_npz=str(outdir / "seediv" / "819" / "s3_p0" / "uncertainty_viz" / "predictions_all_softmax.npz"),
                ),
            ),
            (
                "fig7",
                optimization_curves.main_cli,
                argparse.Namespace(
                    demo=False,
                    out=str(out_dir / "fig7.png"),
                    curves_csv=str(outdir / "seed" / "819" / "table5_comparison_s3" / "optimization_curves.csv"),
                    alternating_run=None,
                    joint_run=None,
                    alternating_log=None,
                    joint_log=None,
                ),
            ),
        ]
    )
    for name, fn, ns in jobs:
        path = fn(ns)
        print(name, path)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.figure == "all":
        out_dir = Path(args.out_dir)
        mapping = (
            ("fig2_seed", confusion_matrices, argparse.Namespace(demo=True, dataset="seed", out=str(out_dir / "fig2_seed.png"), cm_dir=None, npz=None, run=None, seed_npz=None, seediv_npz=None, seed_run=None, seediv_run=None, seed_cm_dir=None, seediv_cm_dir=None)),
            ("fig2_seediv", confusion_matrices, argparse.Namespace(demo=True, dataset="seediv", out=str(out_dir / "fig2_seediv.png"), cm_dir=None, npz=None, run=None, seed_npz=None, seediv_npz=None, seed_run=None, seediv_run=None, seed_cm_dir=None, seediv_cm_dir=None)),
            ("fig3", tsne_latent, argparse.Namespace(demo=True, out=str(out_dir / "fig3.png"), seed=42, seed_raw=None, seed_vae=None, seed_noprior=None, seed_vbhiopt=None, seediv_raw=None, seediv_vae=None, seediv_noprior=None, seediv_vbhiopt=None)),
            ("fig4_seed", spectral_attention, argparse.Namespace(demo=True, dataset="seed", out=str(out_dir / "fig4_seed.png"), npz=None, seed_npz=None, seediv_npz=None)),
            ("fig4_seediv", spectral_attention, argparse.Namespace(demo=True, dataset="seediv", out=str(out_dir / "fig4_seediv.png"), npz=None, seed_npz=None, seediv_npz=None)),
            ("fig5_seed", channel_chords, argparse.Namespace(demo=True, dataset="seed", out=str(out_dir / "fig5_seed.png"), npz=None, seed_npz=None, seediv_npz=None, top_k=40)),
            ("fig5_seediv", channel_chords, argparse.Namespace(demo=True, dataset="seediv", out=str(out_dir / "fig5_seediv.png"), npz=None, seed_npz=None, seediv_npz=None, top_k=40)),
            ("fig6", entropy_reliability, argparse.Namespace(demo=True, out=str(out_dir / "fig6.png"), vbhiopt_npz=None, softmax_npz=None)),
            ("fig7", optimization_curves, argparse.Namespace(demo=True, out=str(out_dir / "fig7.png"), alternating_run=None, joint_run=None, alternating_log=None, joint_log=None)),
        )
        for name, mod, ns in mapping:
            path = mod.main_cli(ns)
            print(name, path)
        return
    if args.figure == "from_outdir":
        run_from_outdir(_repo_root(args.repo_root), Path(args.out_dir), dataset=args.dataset)
        return
    dispatch = {
        "confusion": confusion_matrices.main_cli,
        "fig2": confusion_matrices.main_cli,
        "tsne": tsne_latent.main_cli,
        "fig3": tsne_latent.main_cli,
        "spectral": spectral_attention.main_cli,
        "fig4": spectral_attention.main_cli,
        "chord": channel_chords.main_cli,
        "fig5": channel_chords.main_cli,
        "uncertainty": entropy_reliability.main_cli,
        "fig6": entropy_reliability.main_cli,
        "curves": optimization_curves.main_cli,
        "fig7": optimization_curves.main_cli,
        "extract": extract_mod.main_cli,
    }
    result = dispatch[args.figure](args)
    print(result)


if __name__ == "__main__":
    main()
