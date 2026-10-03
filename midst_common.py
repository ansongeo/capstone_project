"""
MIDST MIA -- code shared by Blocks 1-4
======================================
Dataset configs, preprocessing, model construction, training, sampling and
loss extraction were copied into every block; they live here now so that a
change is made once.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ema_pytorch import EMA
from models.ETDiff.mixed_diffusion import MixedDiffusion
from models.ETDiff.blocks import RNN


# =============================================================================
# DATASETS
# =============================================================================

DATASETS = {
    "eicu": {
        "train_path":     "data/eicu-extract/TRAIN-eicu_multiple_60_1440_276.pt",
        "test_path":      "data/eicu-extract/TEST-eicu_multiple_60_1440_276.pt",
        "synthetic_path": "samples/eicu.npy",
        "diffusion_kwargs": {
            "seq_length":                   272,
            "channels":                     9,
            "numerical_features_indices":   [0, 2, 4, 6],
            "categorical_features_indices": [1, 3, 5, 7, 8],
            "categorical_num_classes":      [2, 2, 2, 2, 2],
            "timesteps":                    1000,
            "beta_schedule":                "cosine",
            "auto_normalize":               True,
            "loss_lambda":                  0.8,
            "parametrization":              "x0",
        },
    },

    "mimic": {
        "train_path":     "data/mimic4-extract/TRAIN-mimic4_vitals_72h.pt",
        "test_path":      "data/mimic4-extract/TEST-mimic4_vitals_72h.pt",
        "synthetic_path": "samples/mimiciv.npy",
        "diffusion_kwargs": {
            "seq_length":                   72,
            "channels":                     11,
            "numerical_features_indices":   [0, 2, 4, 6, 8],
            "categorical_features_indices": [1, 3, 5, 7, 9, 10],
            "categorical_num_classes":      [2, 2, 2, 2, 2, 2],
            "timesteps":                    1000,
            "beta_schedule":                "cosine",
            "auto_normalize":               True,
            "loss_lambda":                  0.8,
            "parametrization":              "x0",
        },
    },
}


# =============================================================================
# PREPROCESSING  -- matches ETDiff TimeSeriesDataset
# =============================================================================

def replace_nan_with_mean(data: torch.Tensor) -> torch.Tensor:
    """Replace NaNs with per-channel mean -- matches ETDiff training."""
    mean = data.nanmean(dim=(0, 2))[None, :, None]
    data = torch.where(torch.isnan(data), mean, data)
    return data


def normalize_data(data: torch.Tensor, categorical_cols: list,
                   eps: float = 0) -> torch.Tensor:
    """
    Min-max normalize to [0,1] along (samples, time) per channel.
    Categorical columns are left untouched -- matches ETDiff training.
    """
    data_np = data.numpy()
    if categorical_cols is not None:
        indicators = data_np[:, categorical_cols, :].copy()
    mn = np.nanmin(data_np, axis=(0, 2))[None, :, None]
    mx = np.nanmax(data_np, axis=(0, 2))[None, :, None]
    data_np = (data_np - mn + eps) / (mx - mn + 2 * eps)
    if categorical_cols is not None:
        data_np[:, categorical_cols, :] = indicators
    return torch.tensor(data_np).float()


def preprocess(data: torch.Tensor, categorical_cols: list) -> torch.Tensor:
    """Full preprocessing pipeline matching ETDiff TimeSeriesDataset."""
    data = replace_nan_with_mean(data)
    data = normalize_data(data, categorical_cols)
    return data


def nan_to_zero(x):
    """Fallback for residual NaNs during batch processing."""
    return torch.nan_to_num(x, nan=0.0)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_real_pool(cfg):
    """
    Join train + test into a single unlabelled Dreal pool and preprocess.
    The attacker does not know the original split -- that is what we attack.
    Returns (X_real, n_train).
    """
    ds       = cfg[cfg["dataset"]]
    seq_len  = ds["diffusion_kwargs"]["seq_length"]
    cat_cols = ds["diffusion_kwargs"]["categorical_features_indices"]

    X_train = torch.load(ds["train_path"], map_location="cpu").float()[:, :, :seq_len]
    X_test  = torch.load(ds["test_path"],  map_location="cpu").float()[:, :, :seq_len]
    X_real  = torch.cat([X_train, X_test], dim=0)

    print(f"  X_train {tuple(X_train.shape)}  X_test {tuple(X_test.shape)}")
    print(f"  Dreal (joined pool) {tuple(X_real.shape)}")
    print(f"  Preprocessing: replace NaN with mean + min-max normalize ...")
    X_real = preprocess(X_real, cat_cols)
    return X_real, len(X_train)


def load_npy_or_pt(path, seq_len):
    """Load .npy or .pt synthetic data file."""
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu").float()[:, :, :seq_len]
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype.kind in ("f", "i", "u"):
        return torch.from_numpy(raw.astype(np.float32))[:, :, :seq_len]
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        inner = raw.item()
        if isinstance(inner, torch.Tensor):
            return inner.float()[:, :, :seq_len]
        return torch.from_numpy(np.array(inner, dtype=np.float32))[:, :, :seq_len]
    return torch.load(path, map_location="cpu").float()[:, :, :seq_len]


# =============================================================================
# MODEL
# =============================================================================

def build_inner_model(cfg):
    ds_cfg  = cfg[cfg["dataset"]]
    diff_kw = ds_cfg["diffusion_kwargs"]
    n_num   = len(diff_kw["numerical_features_indices"])
    n_cat   = sum(diff_kw["categorical_num_classes"])
    ch      = n_num + n_cat
    return RNN(
        input_channels  = ch,
        hidden_channels = 256,
        output_channels = ch,
        layers          = 3,
        model           = "lstm",
        dropout         = 0,
        bidirectional   = False,
        self_condition  = False,
        embed_dim       = 64,
        time_dim        = 256,
    )


def build_diffusion(cfg, inner_model):
    ds_cfg = cfg[cfg["dataset"]]
    return MixedDiffusion(model=inner_model, **ds_cfg["diffusion_kwargs"])


def train_diffusion_steps(diffusion, X, num_steps, batch_size, grad_accum,
                          lr, wd, cfg, desc):
    """
    Train like ETDiff.train() (models/ETDiff/et_diff.py) and return the EMA model:
      - one step = one optimiser update over grad_accum micro-batches
        (ETDiff's 700k steps are optimiser steps, i.e. 1.4M micro-batches),
      - Adam betas (0.9, 0.99), grad-norm clip 1.0,
      - EMA(0.995) updated every 10 steps, used for everything downstream.
    """
    device     = cfg["device"]
    diffusion  = diffusion.to(device).train()
    ema        = EMA(diffusion,
                     beta         = cfg.get("ema_decay", 0.995),
                     update_every = cfg.get("ema_update_every", 10)).to(device)

    optimizer  = torch.optim.Adam(diffusion.parameters(), lr=lr, weight_decay=wd,
                                  betas=(0.9, 0.99))
    loader     = DataLoader(TensorDataset(X), batch_size=batch_size,
                            shuffle=True, drop_last=True)
    data_iter  = iter(loader)
    pbar       = tqdm(range(num_steps), desc=desc, leave=False)

    def next_batch():
        nonlocal data_iter
        try:
            (xb,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (xb,) = next(data_iter)
        return nan_to_zero(xb).to(device)

    for step in pbar:
        accum_loss = 0.0
        for _ in range(grad_accum):
            loss = diffusion(next_batch()) / grad_accum
            loss.backward()
            accum_loss += loss.item()
        nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        ema.update()
        pbar.set_postfix(loss=f"{accum_loss:.4f}")

    return ema.ema_model.cpu()


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

@torch.no_grad()
def generate_synthetic(diffusion, n_samples, batch_size, device):
    from models.ETDiff.mixed_diffusion import ohe_to_categories

    diffusion = diffusion.to(device).eval()
    num_idx  = diffusion.numerical_features_indices
    cat_idx  = diffusion.categorical_features_indices
    channels = diffusion.channels
    seq_len  = diffusion.seq_length
    samples, remaining = [], n_samples

    while remaining > 0:
        b = min(batch_size, remaining)
        z_norm = torch.randn(
            (b, diffusion.num_numerical_features, seq_len), device=device
        )
        has_cat = (len(diffusion.categorical_num_classes) > 0 and
                   diffusion.categorical_num_classes[0] != 0)
        log_z = torch.zeros((b, 0, seq_len), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros(
                (b, len(diffusion.categorical_num_classes_expanded), seq_len),
                device=device
            )
            log_z = diffusion.cat_log_sample_categorical(uniform_logits)

        for i in reversed(range(diffusion.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = torch.cat([z_norm, log_z], dim=1).float()
            model_out     = diffusion.model(x, t, None)
            model_out_num = diffusion.extract_modeloutput(model_out, "numerical")
            model_out_cat = diffusion.extract_modeloutput(model_out, "categorical")
            z_norm, _     = diffusion.gauss_p_sample(
                model_out_num, z_norm, t, i, clip_denoised=True
            )
            if has_cat:
                log_z = diffusion.cat_p_sample(model_out_cat, log_z, t, None)

        z_norm = diffusion.unnormalize(z_norm)
        if has_cat:
            z_cat = ohe_to_categories(
                torch.exp(log_z).round(),
                diffusion.categorical_num_classes
            ).cpu().float()
        else:
            z_cat = torch.zeros((b, 0, seq_len))

        out = torch.zeros((b, channels, seq_len))
        out[:, num_idx, :] = z_norm.cpu()
        if len(cat_idx) > 0:
            out[:, cat_idx, :] = z_cat

        samples.append(out)
        remaining -= b

    diffusion.cpu()
    return torch.cat(samples, dim=0)


# =============================================================================
# LOSS EXTRACTION
# =============================================================================

@torch.no_grad()
def get_loss_vector(diffusion, X, cfg, n_samples=None, batch_size=128):
    """Returns averaged scalar loss per patient. Shape (N,)."""
    if n_samples is None:
        n_samples  = cfg["n_loss_samples"]
    device     = cfg["device"]
    diffusion  = diffusion.to(device).eval()
    total      = np.zeros(len(X), dtype=np.float64)

    try:
        for _ in range(n_samples):
            chunk = []
            for (xb,) in DataLoader(TensorDataset(X),
                                    batch_size=batch_size, shuffle=False):
                xb = nan_to_zero(xb).to(device)
                t  = torch.randint(0, diffusion.num_timesteps,
                                   (xb.size(0),), device=device).long()
                loss = _per_sample_loss(diffusion, xb, t)
                chunk.append(loss.cpu().numpy())
            total += np.concatenate(chunk)
    finally:
        diffusion.cpu()

    return (total / n_samples).astype(np.float32)


def _per_sample_loss(diffusion, x_start, t):
    from models.ETDiff.mixed_diffusion import index_to_log_onehot
    import torch.nn.functional as F
    from einops import reduce
    import models.ETDiff.utils as utils

    pt     = torch.ones_like(t).float() / diffusion.num_timesteps

    # --- numerical ---
    # diffusion.forward() calls normalize_numericals() before mixed_loss(),
    # applying normalize_to_neg_one_to_one (x*2-1) to numerical features.
    # We must apply the same normalization here so loss extraction uses
    # the same [-1,1] inputs as training -- without this the model sees
    # out-of-distribution inputs and assigns noisy, uninformative losses.
    x_num = diffusion.extract_features(x_start, "numerical")
    x_num = diffusion.normalize(x_num)          # [0,1] -> [-1,1]
    noise   = torch.randn_like(x_num)
    x_num_t = diffusion.gauss_q_sample(x_num, t, noise=noise)

    # --- categorical ---
    x_cat       = diffusion.extract_features(x_start, "categorical")
    log_x_cat   = index_to_log_onehot(x_cat.long(),
                                       diffusion.categorical_num_classes)
    log_x_cat_t = diffusion.cat_q_sample(log_x_start=log_x_cat, t=t)

    # --- model forward ---
    x_in      = torch.cat([x_num_t, log_x_cat_t], dim=1)
    model_out = diffusion.model(x_in, t, None)
    out_num   = diffusion.extract_modeloutput(model_out, "numerical")
    out_cat   = diffusion.extract_modeloutput(model_out, "categorical")

    # --- gaussian loss (per sample) ---
    # gauss_loss() in mixed_diffusion returns a scalar (.mean() at the end)
    # replicate without final mean to get shape (B,)
    g = F.mse_loss(out_num, noise, reduction="none")   # (B, num_ch, T)
    g = reduce(g, "b ... -> b (...)", "mean")           # (B, num_ch*T)
    g = g * utils.extract(diffusion.loss_weight, t, g.shape)
    loss_gauss = g.mean(dim=-1)                         # (B,)

    # --- categorical loss (per sample) ---
    # cat_loss() returns (B,) via sum_except_batch -- use directly
    loss_multi = diffusion.cat_loss(
        out_cat, log_x_cat, log_x_cat_t, t, pt, None
    ) / len(diffusion.categorical_num_classes)          # (B,)

    return diffusion.loss_lambda * loss_multi + loss_gauss  # (B,)


# =============================================================================
# META-CLASSIFIER
# =============================================================================

class MetaClassifierMLP(nn.Module):
    """
    Input : (batch, d)  -- loss features per (patient, shadow) pair
    Output: (batch,)    -- raw logit for P(member)
    """
    def __init__(self, input_dim, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev   = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
