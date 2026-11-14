#!/usr/bin/env python3
"""Entry point for paper Figures 2–7.

Examples
--------
    python plot_figures.py all --out_dir figures
    python plot_figures.py from_outdir --dataset seed --out_dir figures
    python plot_figures.py confusion --dataset seed --demo --out figures/fig2_seed.png
    python plot_figures.py spectral --dataset seediv --npz exports/seediv/spectral.npz
    python plot_figures.py chord --dataset seed --npz exports/seed/adj.npy --top_k 30
    python plot_figures.py extract --ckpt last_epoch.pt --data_root <DE> --subject 1 --out_dir exports/s1
"""
from __future__ import annotations

from plots.__main__ import main

if __name__ == "__main__":
    main()
