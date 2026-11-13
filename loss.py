"""Two-block variational objective used by VB-HiOpt."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VBHiOptLoss(nn.Module):
    """Inner (encoder / q(z|x)) and outer (classifier) variational updates."""

    def __init__(
        self,
        dataset_size: int,
        class_weights: torch.Tensor,
        latent_kl_weight: float = 0.0015,
        param_kl_weight: float = 1e-6,
        prior_aux_weight: float = 0.0,
        latent_kl_delay: int = 15,
        latent_kl_warmup: int = 40,
        param_kl_delay: int = 0,
        param_kl_warmup: int = 20,
        prior_aux_warmup: int = 15,
        kl_free_bits: float = 0.0,
        var_floor: float = 1e-4,
        label_smoothing: float = 0.05,
        standard_gaussian_prior: bool = False,
    ) -> None:
        super().__init__()
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if min(latent_kl_weight, param_kl_weight, prior_aux_weight) < 0:
            raise ValueError("loss weights must be non-negative.")

        self.dataset_size = int(dataset_size)
        self.register_buffer("class_weights", class_weights.float())
        self.standard_gaussian_prior = bool(standard_gaussian_prior)

        self.target_latent_kl_weight = float(latent_kl_weight)
        self.target_param_kl_weight = float(param_kl_weight)
        self.target_prior_aux_weight = float(prior_aux_weight)
        self.latent_kl_weight = 0.0
        self.param_kl_weight = 0.0
        self.prior_aux_weight = 0.0

        self.latent_kl_delay = int(latent_kl_delay)
        self.latent_kl_warmup = int(latent_kl_warmup)
        self.param_kl_delay = int(param_kl_delay)
        self.param_kl_warmup = int(param_kl_warmup)
        self.prior_aux_warmup = int(prior_aux_warmup)
        self.kl_free_bits = float(kl_free_bits)
        self.var_floor = float(var_floor)
        self.label_smoothing = float(label_smoothing)

    @staticmethod
    def _schedule(epoch: int, target: float, delay: int, warmup: int) -> float:
        if epoch < delay:
            return 0.0
        if warmup <= 0:
            return target
        ratio = min(1.0, (epoch - delay + 1) / float(warmup))
        return target * ratio

    def set_epoch(self, epoch: int) -> None:
        self.latent_kl_weight = self._schedule(
            epoch, self.target_latent_kl_weight, self.latent_kl_delay, self.latent_kl_warmup
        )
        self.param_kl_weight = self._schedule(
            epoch, self.target_param_kl_weight, self.param_kl_delay, self.param_kl_warmup
        )
        self.prior_aux_weight = self._schedule(
            epoch, self.target_prior_aux_weight, 0, self.prior_aux_warmup
        )

    def classification_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            y,
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.label_smoothing,
        )

    def conditional_gaussian_kl(
        self,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
        logvar_q = logvar_q.clamp(-8.0, 4.0)
        logvar_p = logvar_p.clamp(-8.0, 4.0)
        var_q = logvar_q.exp().clamp_min(self.var_floor)
        var_p = logvar_p.exp().clamp_min(self.var_floor)
        per_dim = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0)
        per_dim = torch.nan_to_num(per_dim, nan=0.0, posinf=1e4, neginf=0.0)
        if self.kl_free_bits > 0:
            per_dim = per_dim.clamp_min(self.kl_free_bits)
        return per_dim.mean(dim=1).mean()

    def posterior_objective(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        num_samples: int,
        sample_encoder_weights: bool,
        sample_z: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ce = x.new_zeros(())
        latent_kl = x.new_zeros(())
        prior_aux = x.new_zeros(())

        if not self.standard_gaussian_prior:
            mu_p, logvar_p = encoder.class_prior(y)
        for _ in range(max(1, num_samples)):
            mu_q, logvar_q, _ = encoder.encode_posterior(x, sample_weights=sample_encoder_weights)
            if self.standard_gaussian_prior:
                mu_p = torch.zeros_like(mu_q)
                logvar_p = torch.zeros_like(logvar_q)
            z = encoder.reparameterize(mu_q, logvar_q, sample_z=sample_z)
            logits = classifier(z, sample_weights=False)
            ce = ce + self.classification_loss(logits, y)
            latent_kl = latent_kl + self.conditional_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
            if self.prior_aux_weight > 0 and not self.standard_gaussian_prior:
                prior_aux = prior_aux + self.classification_loss(encoder.class_prior_scores(z), y)

        divisor = float(max(1, num_samples))
        ce = ce / divisor
        latent_kl = latent_kl / divisor
        prior_aux = prior_aux / divisor
        encoder_kl = encoder.kl_divergence() / self.dataset_size
        loss = (
            ce
            + self.latent_kl_weight * latent_kl
            + self.prior_aux_weight * prior_aux
            + self.param_kl_weight * encoder_kl
        )
        return loss, {
            "ce": ce,
            "latent_kl": latent_kl,
            "prior_aux": prior_aux,
            "encoder_kl": encoder_kl,
        }

    def classifier_objective(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        num_samples: int,
        sample_classifier_weights: bool,
        sample_z: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            mu_q, logvar_q, _ = encoder.encode_posterior(x, sample_weights=False)

        ce = x.new_zeros(())
        for _ in range(max(1, num_samples)):
            z = encoder.reparameterize(mu_q, logvar_q, sample_z=sample_z)
            logits = classifier(z, sample_weights=sample_classifier_weights)
            ce = ce + self.classification_loss(logits, y)
        ce = ce / float(max(1, num_samples))
        classifier_kl = classifier.kl_divergence() / self.dataset_size
        loss = ce + self.param_kl_weight * classifier_kl
        return loss, {"ce": ce, "classifier_kl": classifier_kl}

    @torch.no_grad()
    def full_terms(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mu_q, logvar_q, _ = encoder.encode_posterior(x, sample_weights=False)
        mu_p, logvar_p = encoder.class_prior(y)
        logits = classifier(mu_q, sample_weights=False)
        ce = self.classification_loss(logits, y)
        latent_kl = self.conditional_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
        prior_aux = self.classification_loss(encoder.class_prior_scores(mu_q), y)
        encoder_kl = encoder.kl_divergence() / self.dataset_size
        classifier_kl = classifier.kl_divergence() / self.dataset_size
        loss = (
            ce
            + self.latent_kl_weight * latent_kl
            + self.prior_aux_weight * prior_aux
            + self.param_kl_weight * (encoder_kl + classifier_kl)
        )
        return {
            "loss": loss,
            "ce": ce,
            "latent_kl": latent_kl,
            "prior_aux": prior_aux,
            "encoder_kl": encoder_kl,
            "classifier_kl": classifier_kl,
        }
