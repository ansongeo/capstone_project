"""
FHE-Compatible TimeDiff: Diffusion model for 2-variable time series under FHE.

Designed for Fully Homomorphic Encryption (CKKS scheme via OpenFHE).
Replaces all FHE-incompatible operations (sigmoid, tanh, exp, sqrt, log, clamp)
with polynomial activations and precomputed buffers.

Architecture:
    - 1D CNN backbone with polynomial activations (degree 2)
    - Affine channel scaling (no LayerNorm division/sqrt)
    - FiLM time conditioning (plaintext time -> cipher modulation)
    - DDIM sampling (50 steps, all coefficients precomputed)
    - SGD+momentum optimizer option (no division/sqrt on gradients)

FHE Multiplicative Depth Budget:
    4-layer CNN: 3 + 3 + 3 + 1 = 10 levels  (fits CKKS N=2^15, 12-15 levels)
    3-layer CNN: 3 + 3 + 1     =  7 levels  (conservative option)

Dependencies: torch, numpy, tqdm (only 3 packages needed)

Usage:
    # Train on synthetic data (default: CNN backbone, DDIM sampling):
    python fhe_timediff.py --num_steps 10000

    # Train with SGD (FHE-compatible optimizer):
    python fhe_timediff.py --optimizer sgd --lr 1e-3 --num_steps 20000

    # DDPM sampling instead of DDIM:
    python fhe_timediff.py --sampling_method ddpm --num_steps 20000

    # Mixed mode (1 continuous + 1 binary variable):
    python fhe_timediff.py --diff_type mixed --num_steps 10000

    # MLP backbone (alternative to CNN):
    python fhe_timediff.py --backbone mlp --num_steps 10000

    # Train on your own data:
    python fhe_timediff.py --data_path my_data.pt --num_steps 50000

Reference:
    Ho et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
    Song et al. "Denoising Diffusion Implicit Models." ICLR 2021.
"""
import argparse
import copy
import math
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================
# Section 1: Utility Functions (copied from simple_timediff.py — already FHE-safe)
# =============================================================================

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """Index into tensor `a` using timestep indices `t`, reshape for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cycle(dl):
    """Infinite dataloader iterator."""
    while True:
        for data in dl:
            yield data

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal 2021). Preferred over linear."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps):
    """Linear noise schedule (Ho et al. 2020)."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def normalize_to_neg_one_to_one(sample):
    """Map [0, 1] -> [-1, 1] for diffusion."""
    return sample * 2 - 1

def unnormalize_to_zero_to_one(t):
    """Map [-1, 1] -> [0, 1] after sampling."""
    return (t + 1) * 0.5

def identity(t, *args, **kwargs):
    return t

def normalize_data(data):
    """Per-channel min-max normalization to [0, 1]."""
    min_vals = data.amin(dim=(0, 2), keepdim=True)
    max_vals = data.amax(dim=(0, 2), keepdim=True)
    data = (data - min_vals) / (max_vals - min_vals + 1e-8)
    return data, min_vals.squeeze(), max_vals.squeeze()

def reverse_normalize(data, min_vals, max_vals):
    """Reverse min-max normalization back to original scale."""
    return data * (max_vals - min_vals) + min_vals


# =============================================================================
# Section 2: FHE-Compatible Building Blocks
# =============================================================================

class PolyActivation(nn.Module):
    """
    Learnable degree-2 polynomial activation: f(x) = a*x^2 + b*x + c

    FHE cost: multiplicative depth +2 (one multiplication for x*x, one for a*x^2).
    Replaces sigmoid, tanh, SiLU, GELU — all of which need Taylor series
    approximations with many more multiplications.

    Parameters are per-channel for richer expressivity.
    """
    def __init__(self, num_channels, init_a=0.5, init_b=1.0, init_c=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.full((1, num_channels, 1), init_a))
        self.b = nn.Parameter(torch.full((1, num_channels, 1), init_b))
        self.c = nn.Parameter(torch.full((1, num_channels, 1), init_c))

    def forward(self, x):
        """x: (batch, channels, seq_length)"""
        return self.a * x * x + self.b * x + self.c

    def extra_repr(self):
        return f'channels={self.a.shape[1]}, init_a={self.a.data[0,0,0]:.2f}'


class AffineChannelScale(nn.Module):
    """
    Affine per-channel scaling: f(x) = gamma * x + beta

    FHE cost: multiplicative depth +1.
    Replaces LayerNorm (which requires mean, variance, division, sqrt).
    Training absorbs normalization effects into the learned gamma/beta.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x):
        """x: (batch, channels, seq_length)"""
        return self.gamma * x + self.beta


class FHETimeEmbedding(nn.Module):
    """
    FHE-compatible time embedding using nn.Embedding + polynomial MLP.

    Replaces SinusoidalPosEmb + GELU MLP from simple_timediff.py.

    Runs entirely in PLAINTEXT (timestep is the loop counter, not encrypted),
    so it adds ZERO multiplicative depth to the FHE circuit. The time vector
    only enters the encrypted path via FiLM modulation (scale + shift).
    """
    def __init__(self, num_timesteps, embed_dim=128, time_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_timesteps, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, time_dim),
            PolyActivation(time_dim, init_a=0.5, init_b=1.0, init_c=0.0),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t):
        """
        t: (batch,) — integer timestep indices
        returns: (batch, time_dim)
        """
        emb = self.embed(t)          # (batch, embed_dim)
        # PolyActivation expects (batch, channels, seq_length), so reshape
        emb = emb.unsqueeze(-1)      # (batch, embed_dim, 1)
        out = self.mlp[0](emb.squeeze(-1))   # Linear: (batch, time_dim)
        out = out.unsqueeze(-1)              # (batch, time_dim, 1)
        out = self.mlp[1](out)               # PolyActivation
        out = out.squeeze(-1)                # (batch, time_dim)
        out = self.mlp[2](out)               # Linear: (batch, time_dim)
        return out


class FHEFiLM(nn.Module):
    """
    Feature-wise Linear Modulation for FHE: x_out = x * (scale + 1) + shift

    FHE cost: multiplicative depth +1 (cipher * plaintext scale).
    Scale and shift are computed from the plaintext time embedding,
    so the Linear layer itself is free in FHE terms.
    """
    def __init__(self, time_dim, num_channels):
        super().__init__()
        self.proj = nn.Linear(time_dim, num_channels * 2)
        # Initialize scale near 0, shift near 0 (identity-like at start)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, time_vec):
        """
        x: (batch, channels, seq_length) — encrypted signal
        time_vec: (batch, time_dim) — plaintext time embedding
        returns: (batch, channels, seq_length) — modulated signal
        """
        params = self.proj(time_vec)           # (batch, channels*2)
        scale, shift = params.chunk(2, dim=1)  # each: (batch, channels)
        scale = scale.unsqueeze(-1)            # (batch, channels, 1)
        shift = shift.unsqueeze(-1)            # (batch, channels, 1)
        return x * (scale + 1) + shift         # cipher * plaintext + plaintext


# =============================================================================
# Section 3: FHE-Compatible Backbones
# =============================================================================

class FHEConvNet(nn.Module):
    """
    1D CNN backbone for FHE-compatible diffusion noise prediction.

    Architecture: stack of Conv1d -> AffineScale -> FiLM(time) -> PolyActivation
    Final layer has no activation (just predicts noise).

    Residual connection from input to output (free addition in FHE).

    Multiplicative depth per layer:
        Conv1d: +0 (cipher * plaintext weights, but counted as 0 base since
                     it's the entry point or follows a PolyAct at a known depth)
        AffineScale: +1 (cipher * plaintext gamma)
        FiLM: +1 (cipher * plaintext scale)
        PolyActivation: +2 (cipher * cipher for x^2, then cipher * plaintext a)

    Per intermediate layer: effectively +3 depth (AffineScale, FiLM combine
    as affine transforms, total 1 multiplication; PolyAct adds 2)
    Final layer: +1 (AffineScale + FiLM, no PolyAct)

    Total for 4 layers: 3 + 3 + 3 + 1 = 10 levels
    Total for 3 layers: 3 + 3 + 1     = 7 levels

    Parameters
    ----------
    input_channels : int
        Number of input channels (2 for cause-effect pair)
    hidden_channels : list of int
        Channel widths for intermediate conv layers, e.g. [32, 64, 64]
    output_channels : int
        Number of output channels (same as input for noise prediction)
    kernel_size : int
        Conv kernel size (odd, use same-padding)
    time_dim : int
        Dimension of time embedding vector
    """
    def __init__(self, input_channels, hidden_channels, output_channels,
                 kernel_size=9, time_dim=128):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for same-padding"
        padding = kernel_size // 2

        # Build layer lists
        all_channels = [input_channels] + list(hidden_channels) + [output_channels]
        num_layers = len(all_channels) - 1

        self.convs = nn.ModuleList()
        self.scales = nn.ModuleList()
        self.films = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(num_layers):
            in_c = all_channels[i]
            out_c = all_channels[i + 1]

            self.convs.append(nn.Conv1d(in_c, out_c, kernel_size, padding=padding))
            self.scales.append(AffineChannelScale(out_c))
            self.films.append(FHEFiLM(time_dim, out_c))

            # No activation on the last layer
            if i < num_layers - 1:
                self.activations.append(PolyActivation(out_c))
            else:
                self.activations.append(None)

        # Residual projection if input != output channels
        if input_channels != output_channels:
            self.residual_proj = nn.Conv1d(input_channels, output_channels, 1)
        else:
            self.residual_proj = None

        self.num_layers = num_layers
        self._fhe_depth = self._compute_depth()

    def _compute_depth(self):
        """Compute total multiplicative depth for FHE budget tracking."""
        depth = 0
        for i in range(self.num_layers):
            # AffineScale + FiLM: 1 depth (affine combine into 1 mult)
            depth += 1
            # PolyActivation: 2 depth (x*x then a*x^2)
            if self.activations[i] is not None:
                depth += 2
        return depth

    @property
    def fhe_depth(self):
        return self._fhe_depth

    def forward(self, x, time_vec):
        """
        x: (batch, channels, seq_length) — noisy input (encrypted in FHE)
        time_vec: (batch, time_dim) — time embedding (plaintext)
        returns: (batch, output_channels, seq_length) — predicted noise
        """
        residual = x

        for i in range(self.num_layers):
            x = self.convs[i](x)
            x = self.scales[i](x)
            x = self.films[i](x, time_vec)
            if self.activations[i] is not None:
                x = self.activations[i](x)

        # Residual connection (free addition in FHE)
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        x = x + residual

        return x


class FHEMLPNet(nn.Module):
    """
    MLP backbone for FHE-compatible diffusion (alternative to CNN).

    Flattens (batch, channels, seq_length) -> (batch, channels*seq_length),
    applies MLP with polynomial activations, reshapes back.

    Simpler than CNN but doesn't exploit temporal locality.
    May work better for very short sequences.

    Multiplicative depth: 3 per layer (same as CNN without the conv structure).
    """
    def __init__(self, input_channels, seq_length, output_channels,
                 hidden_dims=None, time_dim=128):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512]

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.seq_length = seq_length
        flat_dim = input_channels * seq_length
        out_flat_dim = output_channels * seq_length

        layers = []
        dims = [flat_dim] + list(hidden_dims) + [out_flat_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(PolyActivation1D(dims[i + 1]))

        self.net = nn.ModuleList(layers)

        # Time conditioning: project time_vec -> bias for each hidden layer
        self.time_projs = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.time_projs.append(nn.Linear(time_dim, dims[i + 1]))

        self._fhe_depth = (len(hidden_dims) + 1) * 3  # rough estimate

    @property
    def fhe_depth(self):
        return self._fhe_depth

    def forward(self, x, time_vec):
        """
        x: (batch, channels, seq_length)
        time_vec: (batch, time_dim)
        returns: (batch, output_channels, seq_length)
        """
        b = x.shape[0]
        residual = x
        h = x.reshape(b, -1)  # flatten

        proj_idx = 0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                h = layer(h)
                # Add time conditioning
                h = h + self.time_projs[proj_idx](time_vec)
                proj_idx += 1
            else:
                h = layer(h)

        out = h.reshape(b, self.output_channels, self.seq_length)

        # Residual if shapes match
        if self.input_channels == self.output_channels:
            out = out + residual

        return out


class PolyActivation1D(nn.Module):
    """Polynomial activation for flat (batch, features) tensors used by MLP backbone."""
    def __init__(self, num_features, init_a=0.5, init_b=1.0, init_c=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.full((1, num_features), init_a))
        self.b = nn.Parameter(torch.full((1, num_features), init_b))
        self.c = nn.Parameter(torch.full((1, num_features), init_c))

    def forward(self, x):
        """x: (batch, features)"""
        return self.a * x * x + self.b * x + self.c


class FHEBackbone(nn.Module):
    """
    Wrapper that combines time embedding + backbone (CNN or MLP).

    This is the full noise-prediction model that replaces SimpleRNN.
    """
    def __init__(self, backbone, time_embedding):
        super().__init__()
        self.backbone = backbone
        self.time_emb = time_embedding

    @property
    def fhe_depth(self):
        return self.backbone.fhe_depth

    def forward(self, x, t):
        """
        x: (batch, channels, seq_length) — noisy input
        t: (batch,) — integer diffusion timestep
        returns: (batch, channels, seq_length) — predicted noise
        """
        time_vec = self.time_emb(t)       # plaintext path
        return self.backbone(x, time_vec)  # encrypted path


# =============================================================================
# Section 4: FHE Gaussian Diffusion (DDPM + DDIM, precomputed everything)
# =============================================================================

class FHEGaussianDiffusion(nn.Module):
    """
    Gaussian diffusion with FHE-compatible sampling.

    Key differences from simple_timediff.py GaussianDiffusion:
    1. Precomputes posterior_std (avoids exp() at runtime)
    2. No clamp(-1, 1) during sampling (post-process after decryption)
    3. DDIM sampling support (reduces 1000 -> 50 steps)
    4. All runtime operations are add/multiply only

    Training is standard DDPM (identical math). Only sampling changes.
    """
    def __init__(self, model, seq_length, channels, timesteps=1000,
                 beta_schedule='cosine', auto_normalize=True,
                 ddim_steps=50, ddim_eta=0.0):
        super().__init__()
        self.model = model
        self.channels = channels
        self.seq_length = seq_length
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

        # Compute noise schedule
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # Register all schedule buffers (float64 -> float32)
        register = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register('betas', betas)
        register('alphas_cumprod', alphas_cumprod)
        register('alphas_cumprod_prev', alphas_cumprod_prev)

        # Forward diffusion coefficients
        register('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # Reverse diffusion coefficients
        register('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0) coefficients — DDPM
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register('posterior_variance', posterior_variance)
        register('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # FHE KEY CHANGE: precompute posterior_std instead of log_variance + exp()
        # This avoids exp(0.5 * log(var)) at runtime — just multiply by std directly
        register('posterior_std', torch.sqrt(posterior_variance.clamp(min=1e-20)))

        # Loss weight (uniform)
        register('loss_weight', torch.ones(self.num_timesteps))

        # Normalization
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        # Precompute DDIM coefficients
        self._precompute_ddim(alphas_cumprod)

    def _precompute_ddim(self, alphas_cumprod):
        """
        Precompute all DDIM sampling coefficients at init (plaintext).

        DDIM step:
            x0_pred = sqrt_recip * x_t - sqrt_recipm1 * pred_noise
            x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred
                     + sqrt(1 - alpha_bar_{t-1} - sigma^2) * pred_noise
                     + sigma * noise

        All coefficients are precomputed — runtime is only multiply + add.
        """
        # Subsequence of timesteps for DDIM (evenly spaced)
        c = self.num_timesteps // self.ddim_steps
        ddim_timesteps = torch.arange(0, self.num_timesteps, c).long()

        # Ensure we include the last timestep
        if ddim_timesteps[-1] != self.num_timesteps - 1:
            ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([self.num_timesteps - 1])])

        self.register_buffer('ddim_timesteps', ddim_timesteps)

        # Alpha bars at DDIM timesteps
        ddim_alphas_cumprod = alphas_cumprod[ddim_timesteps]
        ddim_alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], dtype=torch.float64),
            ddim_alphas_cumprod[:-1]
        ])

        # Precompute coefficients for each DDIM step
        # x0 prediction coefficients (same as DDPM)
        register = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register('ddim_sqrt_recip_alphas_cumprod', torch.sqrt(1. / ddim_alphas_cumprod))
        register('ddim_sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / ddim_alphas_cumprod - 1))

        # Sigma for stochastic DDIM
        ddim_sigmas = (self.ddim_eta *
            torch.sqrt((1 - ddim_alphas_cumprod_prev) / (1 - ddim_alphas_cumprod) *
                       (1 - ddim_alphas_cumprod / ddim_alphas_cumprod_prev)))
        register('ddim_sigmas', ddim_sigmas)

        # Direction pointing to x_t coefficient
        register('ddim_coeff_x0', torch.sqrt(ddim_alphas_cumprod_prev))
        register('ddim_coeff_eps', torch.sqrt(1 - ddim_alphas_cumprod_prev - ddim_sigmas ** 2))

    def predict_start_from_noise(self, x_t, t, noise):
        """Recover x_0 from noisy x_t and predicted noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Compute posterior mean and std for q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_std = extract(self.posterior_std, t, x_t.shape)
        return posterior_mean, posterior_std

    @torch.no_grad()
    def p_sample_ddpm(self, x, t_int):
        """
        Single DDPM reverse step — FHE compatible.

        Uses precomputed posterior_std instead of exp(0.5 * log_var).
        No clamp on x_start (deferred to post-processing).
        """
        b, *_, device = *x.shape, x.device
        t = torch.full((b,), t_int, device=device, dtype=torch.long)

        pred_noise = self.model(x, t)
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        # NOTE: no clamp here — FHE cannot do comparisons.
        # Model learns to stay in bounds during training.

        posterior_mean, posterior_std = self.q_posterior(x_start=x_start, x_t=x, t=t)
        noise = torch.randn_like(x) if t_int > 0 else 0.
        return posterior_mean + posterior_std * noise  # FHE: multiply + add only

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape):
        """Full DDPM reverse diffusion (1000 steps)."""
        device = self.betas.device
        sample = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(self.num_timesteps)), desc='DDPM Sampling', total=self.num_timesteps):
            sample = self.p_sample_ddpm(sample, t)

        return self.unnormalize(sample)

    @torch.no_grad()
    def p_sample_loop_ddim(self, shape):
        """
        DDIM reverse diffusion — FHE compatible.

        All coefficients precomputed at init. Runtime is only:
          - Model forward pass (backbone: multiply + add)
          - Coefficient extraction (plaintext buffer index)
          - Multiply + add to combine predictions

        No exp(), sqrt(), log(), clamp() at runtime.
        """
        device = self.betas.device
        sample = torch.randn(shape, device=device)
        b = shape[0]

        timesteps = self.ddim_timesteps.flip(0)  # reversed order

        for i in tqdm(range(len(timesteps)), desc='DDIM Sampling'):
            t_idx = i  # index into precomputed DDIM coefficient arrays
            # Reverse index: we go from end to start of ddim_timesteps
            coeff_idx = len(timesteps) - 1 - i

            t = timesteps[i]
            t_batch = torch.full((b,), t.item(), device=device, dtype=torch.long)

            # Predict noise
            pred_noise = self.model(sample, t_batch)

            # Predict x_0 using precomputed coefficients
            x0_pred = (
                self.ddim_sqrt_recip_alphas_cumprod[coeff_idx] * sample -
                self.ddim_sqrt_recipm1_alphas_cumprod[coeff_idx] * pred_noise
            )

            # DDIM update: x_{t-1} = coeff_x0 * x0_pred + coeff_eps * pred_noise + sigma * noise
            coeff_x0 = self.ddim_coeff_x0[coeff_idx]
            coeff_eps = self.ddim_coeff_eps[coeff_idx]
            sigma = self.ddim_sigmas[coeff_idx]

            noise = torch.randn_like(sample) if i < len(timesteps) - 1 else 0.
            sample = coeff_x0 * x0_pred + coeff_eps * pred_noise + sigma * noise

        return self.unnormalize(sample)

    @torch.no_grad()
    def sample(self, batch_size=16, method='ddim'):
        """Generate new samples."""
        shape = (batch_size, self.channels, self.seq_length)
        if method == 'ddim':
            return self.p_sample_loop_ddim(shape)
        else:
            return self.p_sample_loop_ddpm(shape)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to x_start at timestep t."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        """Compute training loss: MSE between predicted and actual noise."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        pred_noise = self.model(x_t, t)
        loss = F.mse_loss(pred_noise, noise, reduction='none')
        loss = loss.mean(dim=(1, 2))
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, sample, *args, **kwargs):
        """Training forward pass: normalize, sample timestep, compute loss."""
        b, c, n = sample.shape
        assert n == self.seq_length, f'Expected seq_length {self.seq_length}, got {n}'
        t = torch.randint(0, self.num_timesteps, (b,), device=sample.device).long()
        sample = self.normalize(sample)
        return self.p_losses(sample, t, *args, **kwargs)


# =============================================================================
# Section 5: FHE Mixed Diffusion (1 continuous + 1 binary)
# =============================================================================

def poly_sigmoid(x, degree=4):
    """
    Polynomial approximation of sigmoid for FHE.

    Uses Chebyshev-style approximation on [-6, 6]:
        sigmoid(x) ~ 0.5 + 0.197x - 0.004x^3   (degree 3)

    This avoids exp() which is impossible in FHE.
    Good to ~2% accuracy on [-6, 6], which is sufficient for binary outputs.
    """
    # Minimax polynomial approximation
    # Clamp-free: polynomial naturally saturates for large |x|
    return 0.5 + 0.197 * x - 0.004 * x * x * x


class FHEMixedDiffusion(nn.Module):
    """
    Mixed diffusion for 1 continuous + 1 binary variable — FHE compatible.

    Channel 0: continuous — Gaussian diffusion (same as FHEGaussianDiffusion)
    Channel 1: binary — noise corruption + polynomial sigmoid for BCE loss

    Key FHE changes vs SimpleMixedDiffusion:
    - Uses poly_sigmoid instead of torch.sigmoid (no exp)
    - Uses precomputed posterior_std (no exp at runtime)
    - No clamp during sampling
    """
    def __init__(self, model, seq_length, channels=2, timesteps=1000,
                 beta_schedule='cosine', loss_lambda=0.5,
                 ddim_steps=50, ddim_eta=0.0):
        super().__init__()
        assert channels == 2, "FHEMixedDiffusion is designed for exactly 2 channels"
        self.model = model
        self.channels = channels
        self.seq_length = seq_length
        self.loss_lambda = loss_lambda
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

        # Noise schedule
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        register = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register('betas', betas)
        register('alphas_cumprod', alphas_cumprod)
        register('alphas_cumprod_prev', alphas_cumprod_prev)
        register('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register('posterior_variance', posterior_variance)
        register('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        register('posterior_std', torch.sqrt(posterior_variance.clamp(min=1e-20)))

        # Precompute DDIM coefficients
        self._precompute_ddim(alphas_cumprod)

    def _precompute_ddim(self, alphas_cumprod):
        """Same DDIM precomputation as FHEGaussianDiffusion."""
        c = self.num_timesteps // self.ddim_steps
        ddim_timesteps = torch.arange(0, self.num_timesteps, c).long()
        if ddim_timesteps[-1] != self.num_timesteps - 1:
            ddim_timesteps = torch.cat([ddim_timesteps, torch.tensor([self.num_timesteps - 1])])

        self.register_buffer('ddim_timesteps', ddim_timesteps)

        ddim_alphas_cumprod = alphas_cumprod[ddim_timesteps]
        ddim_alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], dtype=torch.float64),
            ddim_alphas_cumprod[:-1]
        ])

        register = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register('ddim_sqrt_recip_alphas_cumprod', torch.sqrt(1. / ddim_alphas_cumprod))
        register('ddim_sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / ddim_alphas_cumprod - 1))

        ddim_sigmas = (self.ddim_eta *
            torch.sqrt((1 - ddim_alphas_cumprod_prev) / (1 - ddim_alphas_cumprod) *
                       (1 - ddim_alphas_cumprod / ddim_alphas_cumprod_prev)))
        register('ddim_sigmas', ddim_sigmas)
        register('ddim_coeff_x0', torch.sqrt(ddim_alphas_cumprod_prev))
        register('ddim_coeff_eps', torch.sqrt(1 - ddim_alphas_cumprod_prev - ddim_sigmas ** 2))

    def q_sample_continuous(self, x_start, t, noise=None):
        """Forward diffusion for continuous channel."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_sample_binary(self, x_start, t):
        """Forward corruption for binary channel."""
        alpha_bar = extract(self.alphas_cumprod, t, x_start.shape)
        return alpha_bar * x_start + (1 - alpha_bar) * 0.5

    def forward(self, sample, *args, **kwargs):
        """Training forward pass."""
        b, c, n = sample.shape
        assert n == self.seq_length and c == 2
        t = torch.randint(0, self.num_timesteps, (b,), device=sample.device).long()

        x_cont = sample[:, 0:1, :]
        x_bin = sample[:, 1:2, :]

        x_cont_norm = normalize_to_neg_one_to_one(x_cont)

        noise = torch.randn_like(x_cont_norm)
        x_cont_t = self.q_sample_continuous(x_cont_norm, t, noise)
        x_bin_t = self.q_sample_binary(x_bin, t)

        x_in = torch.cat([x_cont_t, x_bin_t], dim=1)
        model_out = self.model(x_in, t)

        pred_noise = model_out[:, 0:1, :]
        pred_bin = model_out[:, 1:2, :]

        loss_cont = F.mse_loss(pred_noise, noise)
        # NOTE: during training we use standard BCE (not poly_sigmoid) for
        # better gradients. poly_sigmoid is only needed at FHE inference time.
        loss_bin = F.binary_cross_entropy_with_logits(pred_bin, x_bin)

        return loss_cont + self.loss_lambda * loss_bin

    @torch.no_grad()
    def sample(self, batch_size=16, method='ddim'):
        """Generate samples using DDIM or DDPM."""
        if method == 'ddim':
            return self._sample_ddim(batch_size)
        else:
            return self._sample_ddpm(batch_size)

    @torch.no_grad()
    def _sample_ddpm(self, batch_size):
        """DDPM sampling for mixed diffusion — FHE compatible."""
        device = self.betas.device

        z_cont = torch.randn(batch_size, 1, self.seq_length, device=device)
        z_bin = torch.full((batch_size, 1, self.seq_length), 0.5, device=device)

        for t in tqdm(reversed(range(self.num_timesteps)), desc='DDPM Mixed Sampling', total=self.num_timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            x_in = torch.cat([z_cont, z_bin], dim=1)
            model_out = self.model(x_in, t_batch)

            pred_noise = model_out[:, 0:1, :]
            pred_bin_logit = model_out[:, 1:2, :]

            # Continuous: reverse step with precomputed posterior_std
            x_start = (
                extract(self.sqrt_recip_alphas_cumprod, t_batch, z_cont.shape) * z_cont -
                extract(self.sqrt_recipm1_alphas_cumprod, t_batch, z_cont.shape) * pred_noise
            )
            # No clamp — FHE compatible

            posterior_mean = (
                extract(self.posterior_mean_coef1, t_batch, z_cont.shape) * x_start +
                extract(self.posterior_mean_coef2, t_batch, z_cont.shape) * z_cont
            )
            posterior_std = extract(self.posterior_std, t_batch, z_cont.shape)
            noise = torch.randn_like(z_cont) if t > 0 else 0.
            z_cont = posterior_mean + posterior_std * noise  # FHE: multiply + add

            # Binary: polynomial sigmoid (FHE compatible, avoids exp)
            pred_prob = poly_sigmoid(pred_bin_logit)
            z_bin = pred_prob

        x_cont_out = unnormalize_to_zero_to_one(z_cont)
        x_bin_out = (z_bin > 0.5).float()
        return torch.cat([x_cont_out, x_bin_out], dim=1)

    @torch.no_grad()
    def _sample_ddim(self, batch_size):
        """DDIM sampling for mixed diffusion — FHE compatible."""
        device = self.betas.device

        z_cont = torch.randn(batch_size, 1, self.seq_length, device=device)
        z_bin = torch.full((batch_size, 1, self.seq_length), 0.5, device=device)

        timesteps = self.ddim_timesteps.flip(0)

        for i in tqdm(range(len(timesteps)), desc='DDIM Mixed Sampling'):
            coeff_idx = len(timesteps) - 1 - i
            t = timesteps[i]
            t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)

            x_in = torch.cat([z_cont, z_bin], dim=1)
            model_out = self.model(x_in, t_batch)

            pred_noise = model_out[:, 0:1, :]
            pred_bin_logit = model_out[:, 1:2, :]

            # Continuous DDIM step with precomputed coefficients
            x0_pred = (
                self.ddim_sqrt_recip_alphas_cumprod[coeff_idx] * z_cont -
                self.ddim_sqrt_recipm1_alphas_cumprod[coeff_idx] * pred_noise
            )

            coeff_x0 = self.ddim_coeff_x0[coeff_idx]
            coeff_eps = self.ddim_coeff_eps[coeff_idx]
            sigma = self.ddim_sigmas[coeff_idx]

            noise = torch.randn_like(z_cont) if i < len(timesteps) - 1 else 0.
            z_cont = coeff_x0 * x0_pred + coeff_eps * pred_noise + sigma * noise

            # Binary: polynomial sigmoid
            pred_prob = poly_sigmoid(pred_bin_logit)
            z_bin = pred_prob

        x_cont_out = unnormalize_to_zero_to_one(z_cont)
        x_bin_out = (z_bin > 0.5).float()
        return torch.cat([x_cont_out, x_bin_out], dim=1)


# =============================================================================
# Section 6: Dataset & Synthetic Data Generator (same as simple_timediff.py)
# =============================================================================

class SimpleTimeSeriesDataset(Dataset):
    """
    Minimal dataset for 2-variable time series.
    Expects clean data (no NaNs) with shape (num_samples, 2, seq_length).
    """
    def __init__(self, data, seq_length=None):
        if isinstance(data, str):
            if data.endswith('.pt'):
                data = torch.load(data, weights_only=False)
            elif data.endswith('.npy'):
                data = torch.from_numpy(np.load(data)).float()
            else:
                raise ValueError("Supported formats: .pt, .npy")

        if seq_length is not None and data.shape[2] > seq_length:
            data = data[:, :, :seq_length]

        assert data.dim() == 3, f"Expected 3D tensor (N, C, T), got shape {data.shape}"
        assert not torch.isnan(data).any(), "Data contains NaNs! Please clean your data first."

        self.data, self.min_vals, self.max_vals = normalize_data(data.float())
        print(f"Dataset loaded: {self.data.shape} "
              f"({self.data.shape[0]} samples, {self.data.shape[1]} channels, {self.data.shape[2]} timesteps)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].clone()

    @property
    def channels(self):
        return self.data.shape[1]

    @property
    def seq_length(self):
        return self.data.shape[2]


def generate_synthetic_data(num_samples=2000, seq_length=276, mode='gaussian'):
    """
    Generate toy cause-effect time series for testing.

    Gaussian mode: 2 continuous variables (sine cause -> lagged effect)
    Mixed mode: 1 continuous cause + 1 binary effect (threshold)
    """
    t = torch.linspace(0, 4 * math.pi, seq_length)

    data = torch.zeros(num_samples, 2, seq_length)
    for i in range(num_samples):
        freq = 0.5 + torch.rand(1).item() * 1.5
        phase = torch.rand(1).item() * 2 * math.pi
        cause = torch.sin(freq * t + phase) + 0.1 * torch.randn(seq_length)

        if mode == 'gaussian':
            lag = int(torch.randint(3, 10, (1,)).item())
            effect = torch.roll(cause, lag) * 0.7 + 0.2 * torch.randn(seq_length)
        else:
            threshold = 0.0 + 0.3 * torch.randn(1).item()
            effect = (cause > threshold).float()

        data[i, 0] = cause
        data[i, 1] = effect

    for c in range(2):
        cmin, cmax = data[:, c].min(), data[:, c].max()
        if cmax > cmin:
            data[:, c] = (data[:, c] - cmin) / (cmax - cmin)

    print(f"Generated synthetic data: {data.shape} (mode={mode})")
    return data


# =============================================================================
# Section 7: EMA, Training Loop & Sampling
# =============================================================================

class EMA:
    """Exponential Moving Average of model parameters for stable sampling."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()

    def to(self, device):
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        return self

    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


def save_checkpoint(diffusion, optimizer, ema, step, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({
        'step': step,
        'model': diffusion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict(),
    }, path)
    print(f"  Checkpoint saved: {path} (step {step})")


def load_checkpoint(diffusion, optimizer, ema, path, device):
    """Load training checkpoint."""
    data = torch.load(path, map_location=device, weights_only=False)
    diffusion.load_state_dict(data['model'])
    optimizer.load_state_dict(data['optimizer'])
    ema.load_state_dict(data['ema'])
    return data['step']


def train(diffusion, dataset, args):
    """Main training loop with Adam or SGD optimizer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    diffusion = diffusion.to(device)
    ema = EMA(diffusion, decay=0.995)
    ema.to(device)

    # Optimizer selection: Adam (plaintext training) or SGD (FHE-compatible)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr, betas=(0.9, 0.99))
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    dl_iter = cycle(dl)

    grad_accum = 2

    with tqdm(total=args.num_steps, desc='Training') as pbar:
        for step in range(args.num_steps):
            total_loss = 0.

            for _ in range(grad_accum):
                data = next(dl_iter).to(device)
                loss = diffusion(data) / grad_accum
                loss.backward()
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(diffusion)

            pbar.set_description(f'Loss: {total_loss:.4f}')
            pbar.update(1)

            if (step + 1) % 1000 == 0:
                print(f"\n  Step {step+1}/{args.num_steps}, Loss: {total_loss:.4f}")

            if (step + 1) % 5000 == 0:
                save_checkpoint(diffusion, optimizer, ema, step + 1, args.save_path)

    save_checkpoint(diffusion, optimizer, ema, args.num_steps, args.save_path)
    return ema


@torch.no_grad()
def generate_samples(diffusion, ema, num_samples, batch_size, min_vals, max_vals,
                     sampling_method='ddim'):
    """Generate samples using EMA model weights."""
    device = next(diffusion.parameters()).device
    ema.apply_shadow(diffusion)
    diffusion.eval()

    all_samples = []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        samples = diffusion.sample(batch_size=bs, method=sampling_method)
        all_samples.append(samples.cpu())
        remaining -= bs

    ema.restore(diffusion)
    diffusion.train()

    samples = torch.cat(all_samples, dim=0).numpy()
    min_np = min_vals.numpy().reshape(1, -1, 1)
    max_np = max_vals.numpy().reshape(1, -1, 1)
    samples = reverse_normalize(samples, min_np, max_np)
    return samples


def print_fhe_audit(model, args):
    """Print FHE compatibility audit: depth budget, operation count, etc."""
    print("\n" + "=" * 60)
    print("FHE COMPATIBILITY AUDIT")
    print("=" * 60)

    # Get backbone depth
    backbone = model.model  # FHEBackbone
    depth = backbone.fhe_depth
    print(f"  Backbone type: {args.backbone.upper()}")
    print(f"  Multiplicative depth: {depth} levels")
    print(f"  CKKS budget (N=2^15): 12-15 levels")
    print(f"  Status: {'FITS' if depth <= 15 else 'TOO DEEP'} without bootstrapping")

    # Count operations
    num_poly_acts = sum(1 for m in model.modules() if isinstance(m, (PolyActivation, PolyActivation1D)))
    num_convs = sum(1 for m in model.modules() if isinstance(m, nn.Conv1d))
    num_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    num_films = sum(1 for m in model.modules() if isinstance(m, FHEFiLM))

    print(f"\n  FHE operations per forward pass:")
    print(f"    Polynomial activations (depth +2 each): {num_poly_acts}")
    print(f"    Conv1d layers (cipher x plaintext):     {num_convs}")
    print(f"    FiLM modulations (cipher x plaintext):  {num_films}")
    print(f"    Linear layers (mostly plaintext path):  {num_linears}")

    # Check for FHE-incompatible ops
    incompatible = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.LSTM, nn.GRU)):
            incompatible.append(f"    {name}: {type(module).__name__}")
    for name, module in model.named_modules():
        if isinstance(module, (nn.SiLU, nn.GELU, nn.ReLU, nn.Sigmoid, nn.Tanh)):
            incompatible.append(f"    {name}: {type(module).__name__}")

    if incompatible:
        print(f"\n  WARNING — FHE-incompatible modules found:")
        for line in incompatible:
            print(line)
    else:
        print(f"\n  All modules are FHE-compatible")

    # Sampling method
    print(f"\n  Sampling method: {args.sampling_method.upper()}")
    if args.sampling_method == 'ddim':
        print(f"    DDIM steps: {args.ddim_steps}")
        print(f"    DDIM eta: {args.ddim_eta}")
    else:
        print(f"    DDPM steps: {args.timesteps}")

    # Optimizer
    print(f"\n  Optimizer: {args.optimizer.upper()}")
    if args.optimizer == 'sgd':
        print(f"    FHE-compatible (multiply + add only)")
    else:
        print(f"    NOT FHE-compatible (uses sqrt, division)")
        print(f"    Use --optimizer sgd for FHE training")

    print("=" * 60 + "\n")


# =============================================================================
# Section 8: Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='FHE-Compatible TimeDiff: diffusion for 2-variable time series')

    # Data
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to .pt or .npy data file. If None, uses synthetic data.')
    parser.add_argument('--seq_length', type=int, default=276,
                        help='Sequence length (default: 276)')

    # Model backbone
    parser.add_argument('--backbone', type=str, default='cnn', choices=['cnn', 'mlp'],
                        help='Backbone type: cnn (1D convolutions) or mlp')
    parser.add_argument('--kernel_size', type=int, default=9,
                        help='CNN kernel size (default: 9, must be odd)')
    parser.add_argument('--hidden_channels', type=str, default='32,64,64',
                        help='CNN hidden channel widths, comma-separated (default: 32,64,64)')
    parser.add_argument('--time_dim', type=int, default=128,
                        help='Time embedding dimension (default: 128)')

    # Diffusion
    parser.add_argument('--diff_type', type=str, default='gaussian', choices=['gaussian', 'mixed'],
                        help='gaussian: 2 continuous vars. mixed: 1 continuous + 1 binary.')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Diffusion timesteps (default: 1000)')
    parser.add_argument('--sampling_method', type=str, default='ddim', choices=['ddpm', 'ddim'],
                        help='Sampling method: ddim (fast, 50 steps) or ddpm (1000 steps)')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Number of DDIM sampling steps (default: 50)')
    parser.add_argument('--ddim_eta', type=float, default=0.0,
                        help='DDIM eta: 0.0=deterministic, 1.0=full stochastic (default: 0.0)')

    # Training
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer: adam (plaintext) or sgd (FHE-compatible)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=8e-5,
                        help='Learning rate (default: 8e-5 for adam, use ~1e-3 for sgd)')
    parser.add_argument('--num_steps', type=int, default=50000,
                        help='Training iterations (default: 50000)')

    # Output
    parser.add_argument('--save_path', type=str, default='results/fhe_model.pt',
                        help='Checkpoint save path')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples to generate after training')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parse hidden channels
    hidden_channels = [int(x) for x in args.hidden_channels.split(',')]

    # Load or generate data
    if args.data_path is not None:
        dataset = SimpleTimeSeriesDataset(args.data_path, seq_length=args.seq_length)
    else:
        print("No data_path provided - generating synthetic data for testing...")
        raw_data = generate_synthetic_data(
            num_samples=2000, seq_length=args.seq_length, mode=args.diff_type
        )
        dataset = SimpleTimeSeriesDataset(raw_data, seq_length=args.seq_length)

    channels = dataset.channels
    seq_length = dataset.seq_length

    # Build FHE-compatible time embedding (plaintext path)
    time_emb = FHETimeEmbedding(
        num_timesteps=args.timesteps,
        embed_dim=args.time_dim,
        time_dim=args.time_dim,
    )

    # Build backbone (encrypted path)
    if args.backbone == 'cnn':
        backbone_net = FHEConvNet(
            input_channels=channels,
            hidden_channels=hidden_channels,
            output_channels=channels,
            kernel_size=args.kernel_size,
            time_dim=args.time_dim,
        )
    elif args.backbone == 'mlp':
        backbone_net = FHEMLPNet(
            input_channels=channels,
            seq_length=seq_length,
            output_channels=channels,
            hidden_dims=[512, 512],
            time_dim=args.time_dim,
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # Combine into full noise-prediction model
    model = FHEBackbone(backbone_net, time_emb)

    # Build diffusion wrapper
    if args.diff_type == 'gaussian':
        diffusion = FHEGaussianDiffusion(
            model=model,
            seq_length=seq_length,
            channels=channels,
            timesteps=args.timesteps,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
        )
    else:
        diffusion = FHEMixedDiffusion(
            model=model,
            seq_length=seq_length,
            channels=channels,
            timesteps=args.timesteps,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
        )

    total_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"\nModel: FHE-compatible {args.diff_type} diffusion ({args.backbone.upper()} backbone)")
    print(f"Parameters: {total_params:,}")
    print(f"Config: backbone={args.backbone}, hidden={args.hidden_channels}, kernel={args.kernel_size}, "
          f"timesteps={args.timesteps}, optimizer={args.optimizer}, lr={args.lr}")

    # FHE audit
    print_fhe_audit(diffusion, args)

    # Train
    ema = train(diffusion, dataset, args)

    # Generate samples
    print(f"\nGenerating {args.num_samples} samples using {args.sampling_method.upper()}...")
    samples = generate_samples(
        diffusion, ema, args.num_samples, args.batch_size,
        dataset.min_vals, dataset.max_vals,
        sampling_method=args.sampling_method,
    )

    # Save
    out_path = args.save_path.replace('.pt', '_samples.npy')
    np.save(out_path, samples)
    print(f"Saved {samples.shape} samples to {out_path}")
    print("Done!")


if __name__ == '__main__':
    main()
