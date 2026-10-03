"""
MIDST MIA -- code shared by Blocks 0-4
======================================
Dataset configs, preprocessing, model construction, training, sampling and
loss features were copied into every block; they live here now so that a
change is made once.
"""

import argparse
import os

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
        "target_hidden":  256,   # etdiff_train.py --hidden default for eicu
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
        "target_hidden":  68,    # etdiff_train.py: (11 channels + 6 categorical) * 4
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

    "physionet": {   # public stand-in, MIMIC-IV layout: preprocess/physionet2012.py
        "train_path":     "data/physionet2012/TRAIN-physionet2012_48h.pt",
        "test_path":      "data/physionet2012/TEST-physionet2012_48h.pt",
        "target_hidden":  68,    # same layout as MIMIC-IV
        "diffusion_kwargs": {
            "seq_length":                   48,
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


# TimeDiff recipe (etdiff_train.py defaults): used for the target in Block 0.
TARGET = {
    "num_steps":  700000,   # optimiser steps
    "batch_size": 32,
    "grad_accum": 2,
    "lr":         8e-5,
    "n_release":  20000,    # etdiff_train.py collect_samples(num_samples=20000)
    "seed":       1,
}
SPLIT_SEED = 0              # members / non-members drawn from the pooled data

# Synth-shadows (Block 2) and the proxy (Block 4) are probes: they should
# memorise the synthetic data they see as closely as possible, so they get
# 4x the target's capacity and 2x its steps (same lr, batch and EMA).
PROBE = {
    "hidden":    256,
    "num_steps": 1400000,
}


def parse_cli(cfg):
    """Command-line overrides shared by all blocks."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=cfg["dataset"], choices=list(DATASETS))
    p.add_argument("--output_dir", default=None,
                   help="default: block1_<dataset>")
    p.add_argument("--n_members", type=int, default=None,
                   help="target training-set size; the attacked pool is 2 x this "
                        "(default: half of all records)")
    a, _ = p.parse_known_args()
    cfg = dict(cfg, dataset=a.dataset, n_members=a.n_members)
    cfg["output_dir"] = a.output_dir or f"block1_{a.dataset}"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    return cfg


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

def load_real_data(cfg):
    """
    Train + test joined into one set of real records, NaN -> per-channel mean
    (ETDiff replace_nan_with_mean), raw clinical units. (n, C, T) float32.
    The designated train/test split is not used: Block 0 draws its own.
    """
    ds      = cfg[cfg["dataset"]]
    seq_len = ds["diffusion_kwargs"]["seq_length"]
    X_train = torch.load(ds["train_path"], map_location="cpu").float()[:, :, :seq_len]
    X_test  = torch.load(ds["test_path"],  map_location="cpu").float()[:, :, :seq_len]
    X = torch.cat([X_train, X_test], dim=0).numpy()
    mean = np.nanmean(X, axis=(0, 2), keepdims=True)
    return np.where(np.isnan(X), mean, X).astype(np.float32)


def member_split(n, n_members, seed=SPLIT_SEED):
    """
    Shuffle all n records; the first 2 * n_members form the attacked pool and
    the first n_members of those are the target's members (50/50).
    Returns (pool_idx, y_member) with y_member aligned to pool_idx.
    """
    perm = np.random.RandomState(seed).permutation(n)
    pool = np.sort(perm[:2 * n_members])
    mem  = np.sort(perm[:n_members])
    return pool, np.isin(pool, mem).astype(int)


def load_attack_pool(cfg):
    """The pool written by Block 0: (X_pool raw units, y_member)."""
    X = load_real_data(cfg)
    pool = np.load(os.path.join(cfg["output_dir"], "pool_idx.npy"))
    y    = np.load(os.path.join(cfg["output_dir"], "y_member.npy"))
    print(f"  attacked pool {tuple(X[pool].shape)}: {int(y.sum())} members / "
          f"{int((1 - y).sum())} non-members")
    return X[pool], y


def cat_idx(cfg):
    return cfg[cfg["dataset"]]["diffusion_kwargs"]["categorical_features_indices"]


def minmax(X):
    return X.min(axis=(0, 2), keepdims=True), X.max(axis=(0, 2), keepdims=True)


def normalize(X, mn, mx, cat):
    """Min-max to [0,1] per channel, categorical channels untouched (TimeSeriesDataset)."""
    Xn = (X - mn) / (mx - mn)
    Xn[:, cat] = X[:, cat]
    return Xn.astype(np.float32)


def denormalize(Xn, mn, mx, cat):
    X = Xn * (mx - mn) + mn
    X[:, cat] = Xn[:, cat]
    return X.astype(np.float32)


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

def build_inner_model(cfg, hidden=256):
    ds_cfg  = cfg[cfg["dataset"]]
    diff_kw = ds_cfg["diffusion_kwargs"]
    n_num   = len(diff_kw["numerical_features_indices"])
    n_cat   = sum(diff_kw["categorical_num_classes"])
    ch      = n_num + n_cat
    return RNN(
        input_channels  = ch,
        hidden_channels = hidden,
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
                          lr, wd, cfg, desc, seed=0):
    """
    Train like ETDiff.train() (models/ETDiff/et_diff.py) and return the EMA model:
      - one step = one optimiser update over grad_accum micro-batches
        (ETDiff's 700k steps are optimiser steps, i.e. 1.4M micro-batches),
      - Adam betas (0.9, 0.99), grad-norm clip 1.0,
      - EMA(0.995) updated every 10 steps, used for everything downstream.
    On a GPU this runs as one CUDA graph per step (midst_fast.py, ~5x faster,
    same loss); train_diffusion_steps_reference is the plain PyTorch loop.
    """
    if not str(cfg["device"]).startswith("cuda"):
        return train_diffusion_steps_reference(diffusion, X, num_steps, batch_size,
                                               grad_accum, lr, wd, cfg, desc)
    from midst_fast import train_graphed
    assert wd == 0, "the CUDA-graph trainer implements Adam without weight decay"
    kw = cfg[cfg["dataset"]]["diffusion_kwargs"]
    X = nan_to_zero(X).to(cfg["device"])
    return train_graphed(diffusion, X, num_steps,
                         kw["numerical_features_indices"],
                         kw["categorical_features_indices"],
                         batch=batch_size * grad_accum, lr=lr,
                         ema_decay=cfg.get("ema_decay", 0.995),
                         ema_every=cfg.get("ema_update_every", 10),
                         seed=seed, log=desc.strip())


def train_diffusion_steps_reference(diffusion, X, num_steps, batch_size, grad_accum,
                          lr, wd, cfg, desc):
    """Plain PyTorch version of train_diffusion_steps (same semantics, slower)."""
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


def train_model(cfg, Xn, num_steps, hidden, seed, desc,
                batch_size=32, grad_accum=2, lr=8e-5):
    """Seed, build a fresh MixedDiffusion and train it on Xn ([0,1] numerics)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    diffusion = build_diffusion(cfg, build_inner_model(cfg, hidden)).to(cfg["device"])
    return train_diffusion_steps(diffusion, torch.tensor(Xn), num_steps, batch_size,
                                 grad_accum, lr, 0.0, cfg, desc, seed=seed)


def fit_release(cfg, X_raw, num_steps, hidden, seed, desc, ckpt_path):
    """
    What etdiff_train.py does for a training set: normalise with that set's
    own min/max (TimeSeriesDataset), train, save, then release n_release
    samples mapped back to raw units with the same min/max.
    Returns (model, (mn, mx), released raw-unit samples).
    """
    cat = cat_idx(cfg)
    mn, mx = minmax(X_raw)
    model = train_model(cfg, normalize(X_raw, mn, mx, cat), num_steps, hidden, seed, desc)
    torch.save(model.state_dict(), ckpt_path)
    np.savez(ckpt_path + ".minmax.npz", mn=mn, mx=mx)
    S = generate_synthetic(model, TARGET["n_release"], 2500, cfg["device"], seed=seed)
    return model, (mn, mx), denormalize(S.numpy(), mn, mx, cat)


def train_probe(cfg, S_raw, pool_mn, pool_mx, seed, desc):
    """
    Synth-shadow / proxy: fit PROBE to raw-unit synthetic data S_raw.
    The data is scaled with the attacked pool's min/max (the scale the real
    records are scored in) and clipped to [0, 1], so the probe learns the
    synthetic data on the same scale as the records it later scores.
    """
    S = np.clip(normalize(S_raw, pool_mn, pool_mx, cat_idx(cfg)), 0, 1)
    return train_model(cfg, S, PROBE["num_steps"], PROBE["hidden"], seed, desc)


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

@torch.no_grad()
def generate_synthetic(diffusion, n_samples, batch_size, device, seed=None):
    from models.ETDiff.mixed_diffusion import ohe_to_categories

    if seed is not None:
        torch.manual_seed(seed)
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
# LOSS FEATURES
# =============================================================================
# Per record and per diffusion step t in T_GRID, the model's Gaussian (vital
# signs, eps-MSE) and categorical (missing-value flags + mortality) losses are
# computed separately under N_DRAW noise draws. The draws are frozen (same
# seed for every model), so the only difference between two models' features
# is the models, not sampling noise. Each record keeps summaries of its N_DRAW
# losses: mean, std, min, max, 10th/50th/90th percentile, over the first
# 100 / 300 / 600 draws.

T_GRID     = (0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750)
N_DRAW     = 600
BUDGETS    = (100, 300, 600)
NOISE_SEED = 1234
STATS      = ("mean", "std", "min", "max", "q10", "q50", "q90")


def _stats(L):
    """L: (n, N_DRAW) on GPU -> (n, len(BUDGETS), 7)."""
    out = []
    for b in BUDGETS:
        x = L[:, :b]
        q = torch.quantile(x, torch.tensor([.1, .5, .9], device=x.device), dim=1).T
        out.append(torch.stack([x.mean(1), x.std(1), x.min(1).values, x.max(1).values,
                                q[:, 0], q[:, 1], q[:, 2]], 1))
    return torch.stack(out, 1)


@torch.no_grad()
def loss_features(cfg, diffusion, Xn, rows=4096):
    """
    Xn: (n, C, T) records scaled like the model's training data.
    Returns (n, len(T_GRID), 2, len(BUDGETS), 7) float32:
    [Gaussian, categorical] loss summaries per t.
    """
    from midst_fast import FastMixedLoss
    kw  = cfg[cfg["dataset"]]["diffusion_kwargs"]
    num = kw["numerical_features_indices"]
    cat = kw["categorical_features_indices"]
    dev = cfg["device"]
    diffusion = diffusion.to(dev).eval()
    fl = FastMixedLoss(diffusion, num, cat)
    X = torch.tensor(Xn, device=dev)
    n, L = len(X), X.shape[2]
    g = torch.Generator(device="cpu").manual_seed(NOISE_SEED)
    eps = torch.randn(N_DRAW, len(num), L, generator=g).to(dev)
    gum = torch.rand(N_DRAW, len(cat), 2, L, generator=g).to(dev)
    out = torch.empty((n, len(T_GRID), 2, len(BUDGETS), 7))
    for ti, tv in enumerate(T_GRID):
        G = torch.empty((n, N_DRAW), device=dev)
        C = torch.empty((n, N_DRAW), device=dev)
        for s in range(0, n, rows):
            x = X[s:s + rows]
            b = len(x)
            t = torch.full((b,), tv, device=dev, dtype=torch.long)
            for r in range(N_DRAW):
                gs, cs, _ = fl(x, t=t, noise=eps[r].expand(b, -1, -1),
                               gumbel_u=gum[r].expand(b, -1, -1, -1), return_parts=True)
                G[s:s + b, r] = gs
                C[s:s + b, r] = cs
        out[:, ti, 0] = _stats(G).cpu()
        out[:, ti, 1] = _stats(C).cpu()
    return out.numpy().astype(np.float32)


def flat_features(F):
    """
    Classifier input from loss features (n, nT, 2, nB, 7) -> (n, d):
      - log |summary| for all 7 summaries, per t and loss type, at the
        largest draw budget (losses span orders of magnitude across t),
      - the change of the mean loss from one t to the next (how fast the
        loss grows with noise level, per loss type).
    """
    f = F[:, :, :, -1]                                  # (n, nT, 2, 7)
    slope = np.diff(f[..., 0], axis=1)                  # (n, nT-1, 2)
    return np.concatenate([np.log(np.abs(f) + 1e-8).reshape(len(F), -1),
                           slope.reshape(len(F), -1)], 1).astype(np.float32)


def features_dir(cfg):
    d = os.path.join(cfg["output_dir"], "features")
    os.makedirs(d, exist_ok=True)
    return d
