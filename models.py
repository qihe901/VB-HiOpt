"""VB-HiOpt network modules.

Input tensor layout is differential entropy ``x`` of shape ``[B, C, F]``
(batch, EEG channels, frequency bands).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """Diagonal-Gaussian Bayesian linear layer with local reparameterisation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        bias: bool = True,
        init_rho: float = -5.5,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        if prior_sigma <= 0:
            raise ValueError("prior_sigma must be positive.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.prior_mu = float(prior_mu)
        self.prior_sigma = float(prior_sigma)
        self.use_bias = bool(bias)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5.0))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), float(init_rho)))

        if bias:
            bound = 1.0 / math.sqrt(in_features)
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            nn.init.uniform_(self.bias_mu, -bound, bound)
            self.bias_rho = nn.Parameter(torch.full((out_features,), float(init_rho)))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

    @staticmethod
    def sigma_from_rho(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-8

    def forward(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        bias_mu = self.bias_mu if self.use_bias else None
        output_mean = F.linear(x, self.weight_mu, bias_mu)
        if not sample:
            return output_mean

        weight_var = self.sigma_from_rho(self.weight_rho).pow(2)
        bias_var = self.sigma_from_rho(self.bias_rho).pow(2) if self.use_bias else None
        output_var = F.linear(x.pow(2), weight_var, bias_var).clamp_min(1e-8)
        return output_mean + torch.sqrt(output_var) * torch.randn_like(output_mean)

    def kl_divergence(self) -> torch.Tensor:
        kl = self._gaussian_kl(self.weight_mu, self.sigma_from_rho(self.weight_rho))
        if self.use_bias:
            kl = kl + self._gaussian_kl(self.bias_mu, self.sigma_from_rho(self.bias_rho))
        return kl

    def _gaussian_kl(self, mu_q: torch.Tensor, sigma_q: torch.Tensor) -> torch.Tensor:
        mu_p = torch.as_tensor(self.prior_mu, dtype=mu_q.dtype, device=mu_q.device)
        sigma_p = torch.as_tensor(self.prior_sigma, dtype=mu_q.dtype, device=mu_q.device)
        value = torch.log(sigma_p / sigma_q)
        value = value + (sigma_q.pow(2) + (mu_q - mu_p).pow(2)) / (2.0 * sigma_p.pow(2)) - 0.5
        return value.sum()


def _forward_layer(layer: nn.Module, x: torch.Tensor, sample_weights: bool) -> torch.Tensor:
    if isinstance(layer, BayesianLinear):
        return layer(x, sample=sample_weights)
    return layer(x)


def _bounded_logvar(raw: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    """Smoothly bound log-variance while preserving gradients."""
    middle = 0.5 * (minimum + maximum)
    radius = 0.5 * (maximum - minimum)
    return middle + radius * torch.tanh(raw)


class BandAttention(nn.Module):
    """Identity-preserving adaptive spectral recalibration for DE features.

    The module is intentionally initialised as an exact identity mapping:

        x_out = x * (1 + r * tanh(a(x)))

    where the final projection that produces ``a(x)`` is zero-initialised.
    Therefore ``a(x)=0`` and ``x_out=x`` at initialisation. During training the
    network learns bounded, sample-adaptive residual reweighting in the range
    ``[1-r, 1+r]``. This retains the original DE representation at the start of
    optimisation while preserving adaptive spectral selectivity.
    """

    def __init__(self, de_dim: int, hidden: int = 24, residual_scale: float = 0.5) -> None:
        super().__init__()
        if de_dim <= 0:
            raise ValueError("de_dim must be positive.")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1].")

        hidden = max(int(hidden), int(de_dim) * 3)
        self.de_dim = int(de_dim)
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.Linear(de_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, de_dim),
        )

        # Exact identity initialisation: raw residual == 0 for every input.
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Return sample-wise multiplicative band scales with shape [B,1,DE]."""
        if x.dim() != 3 or x.size(-1) != self.de_dim:
            raise ValueError(f"Expected x [B,C,{self.de_dim}], got {tuple(x.shape)}.")
        raw = self.net(x.mean(dim=1)).unsqueeze(1)
        return 1.0 + self.residual_scale * torch.tanh(raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale(x)


class AdaptiveChannelGraphBlock(nn.Module):
    """Sample-adaptive residual channel-relation block."""

    def __init__(
        self,
        channels: int,
        dim: int,
        dropout: float = 0.15,
        graph_rank: int = 8,
    ) -> None:
        super().__init__()
        if channels <= 0 or dim <= 0:
            raise ValueError("channels and dim must be positive.")

        self.channels = int(channels)
        self.dim = int(dim)
        relation_dim = max(4, min(int(graph_rank), dim))

        self.pre_norm = nn.LayerNorm(dim)
        self.rel_q = nn.Linear(dim, relation_dim, bias=False)
        self.rel_k = nn.Linear(dim, relation_dim, bias=False)
        self.graph_value = nn.Linear(dim, dim, bias=False)
        self.graph_dropout = nn.Dropout(dropout)
        self.graph_gate_raw = nn.Parameter(torch.tensor(-2.5))
        self.temperature_raw = nn.Parameter(torch.tensor(0.0))

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ffn_gate_raw = nn.Parameter(torch.tensor(-2.0))

    def relation_matrix(self, h_norm: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.rel_q(h_norm), dim=-1, eps=1e-6)
        k = F.normalize(self.rel_k(h_norm), dim=-1, eps=1e-6)
        score = torch.matmul(q, k.transpose(-1, -2))
        score = 0.5 * (score + score.transpose(-1, -2))
        temperature = 0.35 + F.softplus(self.temperature_raw)
        return torch.softmax(score / temperature, dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3 or h.size(1) != self.channels or h.size(2) != self.dim:
            raise ValueError(f"Expected [B,{self.channels},{self.dim}], got {tuple(h.shape)}.")

        h_norm = self.pre_norm(h)
        graph_message = torch.matmul(self.relation_matrix(h_norm), h_norm)
        graph_message = self.graph_value(graph_message)
        h = h + torch.sigmoid(self.graph_gate_raw) * self.graph_dropout(graph_message)
        return h + torch.sigmoid(self.ffn_gate_raw) * self.ffn(self.ffn_norm(h))


class AttentiveStatsPool(nn.Module):
    """Attention, mean, standard-deviation and max pooling across EEG channels."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(16, dim // 3)
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        alpha = torch.softmax(self.score(h).squeeze(-1), dim=1)
        h_att = torch.sum(h * alpha.unsqueeze(-1), dim=1)
        h_mean = h.mean(dim=1)
        h_std = h.var(dim=1, unbiased=False).clamp_min(1e-8).sqrt()
        h_max = h.max(dim=1).values
        return torch.cat([h_att, h_mean, h_std, h_max], dim=1)


class ClassConditionalPriorEncoder(nn.Module):
    """EEG encoder for q_phi(z|x) with a learnable class-conditional Gaussian prior p(z|y)."""

    def __init__(
        self,
        channels: int,
        de_dim: int,
        latent_dim: int,
        num_classes: int,
        channel_hidden: int = 64,
        fusion_dims: Iterable[int] = (128,),
        dropout: float = 0.15,
        spatial_layers: int = 1,
        graph_dropout: float = 0.15,
        graph_rank: int = 8,
        use_band_attention: bool = False,
        use_spatial_graph: bool = True,
        band_attention_hidden: int = 24,
        band_residual_scale: float = 0.5,
        prior_logvar_min: float = -4.0,
        prior_logvar_max: float = 0.5,
        posterior_logvar_min: float = -4.0,
        posterior_logvar_max: float = 0.5,
        class_prior_scale: float = 2.0,
        kl_scale: float = 1.0,
        unbounded_logvar: bool = False,
    ) -> None:
        super().__init__()
        if min(channels, de_dim, latent_dim, num_classes, channel_hidden) <= 0:
            raise ValueError("All dimensions must be positive.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        if not unbounded_logvar and (
            prior_logvar_min >= prior_logvar_max or posterior_logvar_min >= posterior_logvar_max
        ):
            raise ValueError("log-variance minima must be smaller than maxima.")

        self.channels = int(channels)
        self.de_dim = int(de_dim)
        self.latent_dim = int(latent_dim)
        self.num_classes = int(num_classes)
        self.prior_logvar_min = float(prior_logvar_min)
        self.prior_logvar_max = float(prior_logvar_max)
        self.posterior_logvar_min = float(posterior_logvar_min)
        self.posterior_logvar_max = float(posterior_logvar_max)
        self.class_prior_scale = float(class_prior_scale)
        self.kl_scale = float(kl_scale)
        self.unbounded_logvar = bool(unbounded_logvar)
        self.use_band_attention = bool(use_band_attention)
        self.use_spatial_graph = bool(use_spatial_graph)

        self.band_attention: nn.Module = (
            BandAttention(
                de_dim,
                hidden=band_attention_hidden,
                residual_scale=band_residual_scale,
            )
            if use_band_attention
            else nn.Identity()
        )
        self.channel_projection = BayesianLinear(self.de_dim, channel_hidden)
        self.channel_norm = nn.LayerNorm(channel_hidden)
        self.channel_dropout = nn.Dropout(dropout)
        self.channel_pos = nn.Parameter(torch.zeros(1, self.channels, channel_hidden))
        nn.init.trunc_normal_(self.channel_pos, std=0.02)

        if self.use_spatial_graph:
            self.spatial_blocks = nn.ModuleList(
                [
                    AdaptiveChannelGraphBlock(
                        channels=self.channels,
                        dim=channel_hidden,
                        dropout=graph_dropout,
                        graph_rank=graph_rank,
                    )
                    for _ in range(max(1, int(spatial_layers)))
                ]
            )
        else:
            self.spatial_blocks = nn.ModuleList()
        self.spatial_norm = nn.LayerNorm(channel_hidden)
        self.pool = AttentiveStatsPool(channel_hidden)

        prev = channel_hidden * 4
        fusion: List[nn.Module] = []
        for width in fusion_dims:
            width = int(width)
            if width <= 0:
                raise ValueError("fusion dimensions must be positive.")
            fusion.extend(
                [
                    BayesianLinear(prev, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        self.fusion = nn.ModuleList(fusion)
        self.feature_dim = prev

        self.posterior_mu_layer = BayesianLinear(prev, latent_dim)
        self.posterior_logvar_layer = BayesianLinear(prev, latent_dim)

        self.class_prior_mu = nn.Parameter(torch.empty(num_classes, latent_dim))
        self.class_prior_logvar = nn.Parameter(torch.full((num_classes, latent_dim), 0.25))
        self._initialise_latent_heads()
        self._initialise_class_priors()

    def _initialise_latent_heads(self) -> None:
        nn.init.normal_(self.posterior_logvar_layer.weight_mu, mean=0.0, std=0.01)
        if self.posterior_logvar_layer.bias_mu is not None:
            nn.init.constant_(self.posterior_logvar_layer.bias_mu, -0.35)

    def _initialise_class_priors(self) -> None:
        with torch.no_grad():
            raw = torch.randn(self.num_classes, self.latent_dim)
            raw = raw - raw.mean(dim=0, keepdim=True)
            raw = F.normalize(raw, dim=1)
            self.class_prior_mu.copy_(self.class_prior_scale * raw)

    def _check_x(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,C,DE], got {tuple(x.shape)}.")
        if x.size(1) != self.channels or x.size(2) != self.de_dim:
            raise ValueError(f"Expected [B,{self.channels},{self.de_dim}], got {tuple(x.shape)}.")

    @torch.no_grad()
    def band_scales(self, x: torch.Tensor) -> torch.Tensor:
        """Expose spectral scales for paper visualisation without changing inference."""
        self._check_x(x)
        if isinstance(self.band_attention, BandAttention):
            return self.band_attention.scale(x)
        return torch.ones(
            x.size(0), 1, self.de_dim, dtype=x.dtype, device=x.device
        )

    def _spatial_hidden(self, x: torch.Tensor, sample_weights: bool = False) -> torch.Tensor:
        """Channel-wise hidden states after band + spatial encoding."""
        self._check_x(x)
        x = self.band_attention(x)
        batch = x.size(0)
        h = x.reshape(batch * self.channels, self.de_dim)
        h = self.channel_projection(h, sample=sample_weights)
        h = self.channel_dropout(self.channel_norm(F.gelu(h)))
        h = h.reshape(batch, self.channels, -1) + self.channel_pos
        for block in self.spatial_blocks:
            h = block(h)
        return self.spatial_norm(h)

    @torch.no_grad()
    def relation_matrices(self, x: torch.Tensor, layer: str = "last") -> torch.Tensor:
        """Expose sample-wise channel adjacency matrices with shape [B, C, C].

        ``layer`` can be ``last`` (final spatial block) or ``mean`` (average over blocks).
        """
        self._check_x(x)
        x = self.band_attention(x)
        batch = x.size(0)
        h = x.reshape(batch * self.channels, self.de_dim)
        h = self.channel_projection(h, sample=False)
        h = self.channel_dropout(self.channel_norm(F.gelu(h)))
        h = h.reshape(batch, self.channels, -1) + self.channel_pos

        relations: List[torch.Tensor] = []
        for block in self.spatial_blocks:
            h_norm = block.pre_norm(h)
            relations.append(block.relation_matrix(h_norm))
            h = block(h)

        if layer == "mean":
            return torch.stack(relations, dim=0).mean(dim=0)
        if layer != "last":
            raise ValueError(f"Unsupported layer mode: {layer!r}. Use 'last' or 'mean'.")
        return relations[-1]

    @torch.no_grad()
    def channel_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Expose attentive pooling channel weights with shape [B, C]."""
        h = self._spatial_hidden(x, sample_weights=False)
        return torch.softmax(self.pool.score(h).squeeze(-1), dim=1)

    def extract_features(self, x: torch.Tensor, sample_weights: bool = False) -> torch.Tensor:
        self._check_x(x)
        h = self._spatial_hidden(x, sample_weights=sample_weights)

        out = self.pool(h)
        for layer in self.fusion:
            out = _forward_layer(layer, out, sample_weights)
        return out

    def _map_logvar(self, raw: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
        if self.unbounded_logvar:
            return raw
        return _bounded_logvar(raw, minimum, maximum)

    def posterior_from_features(
        self,
        h: torch.Tensor,
        sample_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_q = self.posterior_mu_layer(h, sample=sample_weights)
        raw_logvar = self.posterior_logvar_layer(h, sample=sample_weights)
        logvar_q = self._map_logvar(
            raw_logvar,
            self.posterior_logvar_min,
            self.posterior_logvar_max,
        )
        return mu_q, logvar_q

    def encode_posterior(
        self,
        x: torch.Tensor,
        sample_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.extract_features(x, sample_weights=sample_weights)
        mu_q, logvar_q = self.posterior_from_features(h, sample_weights=sample_weights)
        return mu_q, logvar_q, h

    def class_prior(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if y.dim() != 1:
            raise ValueError("y must have shape [B].")
        if y.numel() == 0:
            raise ValueError("y cannot be empty.")
        if y.min().item() < 0 or y.max().item() >= self.num_classes:
            raise ValueError("y contains an invalid class index.")
        logvar_all = self._map_logvar(
            self.class_prior_logvar,
            self.prior_logvar_min,
            self.prior_logvar_max,
        )
        return self.class_prior_mu[y], logvar_all[y]

    def class_prior_scores(self, z: torch.Tensor) -> torch.Tensor:
        """Gaussian class evidence used only by the optional prior auxiliary objective."""
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"Expected z [B,{self.latent_dim}], got {tuple(z.shape)}.")
        mu_all = self.class_prior_mu.unsqueeze(0)
        logvar_all = self._map_logvar(
            self.class_prior_logvar,
            self.prior_logvar_min,
            self.prior_logvar_max,
        ).unsqueeze(0)
        centred = z.unsqueeze(1) - mu_all
        inv_var = torch.exp(-logvar_all)
        score = -0.5 * (logvar_all + centred.pow(2) * inv_var).mean(dim=-1)
        return torch.nan_to_num(score, nan=-1e4, neginf=-1e4, posinf=1e4)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample_z: bool = True) -> torch.Tensor:
        if not sample_z:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def inner_parameters(self) -> List[nn.Parameter]:
        return list(self.parameters())

    def kl_divergence(self) -> torch.Tensor:
        modules: Sequence[nn.Module] = (
            self.channel_projection,
            self.fusion,
            self.posterior_mu_layer,
            self.posterior_logvar_layer,
        )
        total = torch.zeros((), device=next(self.parameters()).device)
        for module in modules:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, BayesianLinear):
                        total = total + layer.kl_divergence()
            elif isinstance(module, BayesianLinear):
                total = total + module.kl_divergence()
        return self.kl_scale * total


class VanillaVAEEncoder(nn.Module):
    """Flat MLP encoder for a standard VAE baseline (Figure 4b).

    Flattens ``[B, C, DE]`` and maps through MLP layers to ``q(z|x)``.
    Shares the same public API as :class:`ClassConditionalPriorEncoder` so
    training / extraction scripts can swap encoders without code changes.
    """

    def __init__(
        self,
        channels: int,
        de_dim: int,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        fusion_dims: Iterable[int] = (128,),
        dropout: float = 0.15,
        posterior_logvar_min: float = -4.0,
        posterior_logvar_max: float = 0.5,
        kl_scale: float = 1.0,
        unbounded_logvar: bool = False,
    ) -> None:
        super().__init__()
        if min(channels, de_dim, latent_dim, num_classes, hidden_dim) <= 0:
            raise ValueError("All dimensions must be positive.")
        if not unbounded_logvar and posterior_logvar_min >= posterior_logvar_max:
            raise ValueError("posterior log-variance min must be smaller than max.")

        self.channels = int(channels)
        self.de_dim = int(de_dim)
        self.latent_dim = int(latent_dim)
        self.num_classes = int(num_classes)
        self.posterior_logvar_min = float(posterior_logvar_min)
        self.posterior_logvar_max = float(posterior_logvar_max)
        self.kl_scale = float(kl_scale)
        self.unbounded_logvar = bool(unbounded_logvar)

        input_dim = self.channels * self.de_dim
        prev = input_dim
        trunk: List[nn.Module] = []
        for width in (hidden_dim, *fusion_dims):
            width = int(width)
            if width <= 0:
                raise ValueError("fusion dimensions must be positive.")
            trunk.extend(
                [
                    BayesianLinear(prev, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = width
        self.trunk = nn.ModuleList(trunk)
        self.feature_dim = prev

        self.posterior_mu_layer = BayesianLinear(prev, latent_dim)
        self.posterior_logvar_layer = BayesianLinear(prev, latent_dim)

        self.class_prior_mu = nn.Parameter(torch.zeros(num_classes, latent_dim))
        self.class_prior_logvar = nn.Parameter(torch.zeros(num_classes, latent_dim))
        self._initialise_latent_heads()

    def _initialise_latent_heads(self) -> None:
        nn.init.normal_(self.posterior_logvar_layer.weight_mu, mean=0.0, std=0.01)
        if self.posterior_logvar_layer.bias_mu is not None:
            nn.init.constant_(self.posterior_logvar_layer.bias_mu, -0.35)

    def _check_x(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,C,DE], got {tuple(x.shape)}.")
        if x.size(1) != self.channels or x.size(2) != self.de_dim:
            raise ValueError(f"Expected [B,{self.channels},{self.de_dim}], got {tuple(x.shape)}.")

    def extract_features(self, x: torch.Tensor, sample_weights: bool = False) -> torch.Tensor:
        self._check_x(x)
        out = x.reshape(x.size(0), -1)
        for layer in self.trunk:
            out = _forward_layer(layer, out, sample_weights)
        return out

    def posterior_from_features(
        self,
        h: torch.Tensor,
        sample_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_q = self.posterior_mu_layer(h, sample=sample_weights)
        raw_logvar = self.posterior_logvar_layer(h, sample=sample_weights)
        if self.unbounded_logvar:
            logvar_q = raw_logvar
        else:
            logvar_q = _bounded_logvar(
                raw_logvar,
                self.posterior_logvar_min,
                self.posterior_logvar_max,
            )
        return mu_q, logvar_q

    def encode_posterior(
        self,
        x: torch.Tensor,
        sample_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.extract_features(x, sample_weights=sample_weights)
        mu_q, logvar_q = self.posterior_from_features(h, sample_weights=sample_weights)
        return mu_q, logvar_q, h

    def class_prior(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if y.dim() != 1:
            raise ValueError("y must have shape [B].")
        if y.min().item() < 0 or y.max().item() >= self.num_classes:
            raise ValueError("y contains an invalid class index.")
        return self.class_prior_mu[y], self.class_prior_logvar[y]

    def class_prior_scores(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"Expected z [B,{self.latent_dim}], got {tuple(z.shape)}.")
        mu_all = self.class_prior_mu.unsqueeze(0)
        logvar_all = self.class_prior_logvar.unsqueeze(0)
        centred = z.unsqueeze(1) - mu_all
        inv_var = torch.exp(-logvar_all)
        score = -0.5 * (logvar_all + centred.pow(2) * inv_var).mean(dim=-1)
        return torch.nan_to_num(score, nan=-1e4, neginf=-1e4, posinf=1e4)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample_z: bool = True) -> torch.Tensor:
        if not sample_z:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def inner_parameters(self) -> List[nn.Parameter]:
        return list(self.parameters())

    def kl_divergence(self) -> torch.Tensor:
        total = torch.zeros((), device=next(self.parameters()).device)
        for layer in self.trunk:
            if isinstance(layer, BayesianLinear):
                total = total + layer.kl_divergence()
        total = total + self.posterior_mu_layer.kl_divergence()
        total = total + self.posterior_logvar_layer.kl_divergence()
        return self.kl_scale * total


class BayesianClassifier(nn.Module):
    """Bayesian residual classifier operating on the latent representation z."""

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.30,
        kl_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0 or num_classes <= 1:
            raise ValueError("Invalid classifier dimensions.")
        self.latent_dim = int(latent_dim)
        self.num_classes = int(num_classes)
        self.kl_scale = float(kl_scale)

        self.hidden1 = BayesianLinear(latent_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.hidden2 = BayesianLinear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = BayesianLinear(hidden_dim, num_classes)
        if self.output.bias_mu is not None:
            nn.init.zeros_(self.output.bias_mu)

    def forward(self, z: torch.Tensor, sample_weights: bool = False) -> torch.Tensor:
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"Expected z [B,{self.latent_dim}], got {tuple(z.shape)}.")
        h = self.dropout(F.gelu(self.norm1(self.hidden1(z, sample=sample_weights))))
        residual = h
        h = self.dropout(F.gelu(self.norm2(self.hidden2(h, sample=sample_weights))))
        return self.output(h + residual, sample=sample_weights)

    def kl_divergence(self) -> torch.Tensor:
        total = torch.zeros((), device=next(self.parameters()).device)
        for layer in self.modules():
            if isinstance(layer, BayesianLinear):
                total = total + layer.kl_divergence()
        return self.kl_scale * total
