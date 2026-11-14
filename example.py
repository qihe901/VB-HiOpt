"""Minimal forward pass on a dummy DE batch. No dataset required.

For the PDF training loop (alternating updates, 5-fold / LOSO, MC eval):

    python train.py --protocol smoke
    python train.py --protocol ws5fold --data_root <DE_root> --label_path <labels.npy>
    python train.py --protocol loso --data_root <DE_root> --dimension valence

Paper Figures 2–7 (demo layout, or from exported arrays):

    python plot_figures.py all --out_dir figures
"""
from __future__ import annotations

import torch

from loss import VBHiOptLoss
from models import BayesianClassifier, ClassConditionalPriorEncoder


def main() -> None:
    torch.manual_seed(0)
    batch, channels, de_dim, num_classes, latent_dim = 8, 62, 5, 3, 48

    encoder = ClassConditionalPriorEncoder(
        channels=channels,
        de_dim=de_dim,
        latent_dim=latent_dim,
        num_classes=num_classes,
        use_band_attention=True,
        use_spatial_graph=True,
    )
    classifier = BayesianClassifier(latent_dim=latent_dim, num_classes=num_classes)
    criterion = VBHiOptLoss(
        dataset_size=batch,
        class_weights=torch.ones(num_classes),
        prior_aux_weight=0.05,
    )
    criterion.set_epoch(20)

    x = torch.randn(batch, channels, de_dim)
    y = torch.randint(0, num_classes, (batch,))

    inner, inner_terms = criterion.posterior_objective(
        encoder, classifier, x, y, num_samples=1, sample_encoder_weights=True, sample_z=True
    )
    outer, outer_terms = criterion.classifier_objective(
        encoder, classifier, x, y, num_samples=1, sample_classifier_weights=True, sample_z=True
    )

    with torch.no_grad():
        mu_q, logvar_q, _ = encoder.encode_posterior(x, sample_weights=False)
        logits = classifier(mu_q, sample_weights=False)
        pred = logits.argmax(dim=-1)

    print("x", tuple(x.shape), "logits", tuple(logits.shape))
    print("inner", float(inner.detach()), {k: float(v.detach()) for k, v in inner_terms.items()})
    print("outer", float(outer.detach()), {k: float(v.detach()) for k, v in outer_terms.items()})
    print("pred", pred.tolist())


if __name__ == "__main__":
    main()
