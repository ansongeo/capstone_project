"""
MIDST MIA -- Block 2: Train Synth-Shadow Models & Extract Losses
================================================================
Reads outputs from Block 1 (Dk_syn + split indices) and for each k:
  1. Trains a synth-shadow model on Dk_syn
  2. Computes relative loss on ALL of Dreal:
       relative_loss(x) = mean_loss(x) - mean_loss(Dk_syn)
  3. Accumulates loss_matrix (N, K) and label_matrix (N, K)

Outputs saved to output_dir/
  loss_matrix.npy   (N_real, K)
  label_matrix.npy  (N_real, K)

Run next: MIDST_block3_train_classifier.py
"""

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
# CONFIG  -- must match Block 1
# =============================================================================

CONFIG = {
    "dataset": "eicu",    # "eicu" | "mimic"

    "eicu": {
        "train_path":     "data/eicu-extract/TRAIN-eicu_multiple_60_1440_276.pt",
        "test_path":      "data/eicu-extract/TEST-eicu_multiple_60_1440_276.pt",
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

    "K":                 16,
    "split_train_frac":  0.5,   # must match Block 1

    # Synth-shadow training
    "shadow_num_steps":  600000,  # Dk_syn is 2000 samples so converges faster
    "shadow_batch_size": 32,
    "shadow_grad_accum": 2,
    "shadow_lr":         8e-5,
    "shadow_wd":         0.0,

    # Loss extraction
    "n_loss_samples":    20,

    # must match Block 1
    "split_seed":        42,

    # resume -- set to k index (0-based) to resume mid-run
    "resume_from_k":     0,

    "output_dir": "block1_eicu",
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
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

def load_real_data(cfg):
    """
    Join train + test into a single unlabelled Dreal pool and preprocess.
    The attacker does not know the original split -- that is what we attack.
    """
    ds      = cfg[cfg["dataset"]]
    seq_len = ds["diffusion_kwargs"]["seq_length"]
    cat_cols = ds["diffusion_kwargs"]["categorical_features_indices"]

    X_train = torch.load(ds["train_path"], map_location="cpu").float()[:, :, :seq_len]
    X_test  = torch.load(ds["test_path"],  map_location="cpu").float()[:, :, :seq_len]
    X_real  = torch.cat([X_train, X_test], dim=0)

    print(f"  X_train {tuple(X_train.shape)}  X_test {tuple(X_test.shape)}")
    print(f"  Dreal (joined pool) {tuple(X_real.shape)}")
    print(f"  Preprocessing: replace NaN with mean + min-max normalize ...")
    X_real = preprocess(X_real, cat_cols)
    return X_real


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
    """Train synth-shadow with EMA -- returns ema_model."""
    device     = cfg["device"]
    diffusion  = diffusion.to(device).train()
    ema        = EMA(diffusion,
                     beta         = cfg.get("ema_decay", 0.995),
                     update_every = cfg.get("ema_update_every", 10))
    ema        = ema.to(device)

    optimizer  = torch.optim.Adam(diffusion.parameters(), lr=lr, weight_decay=wd)
    loader     = DataLoader(TensorDataset(X), batch_size=batch_size,
                            shuffle=True, drop_last=True)
    data_iter  = iter(loader)
    pbar       = tqdm(range(num_steps), desc=desc, leave=False)
    accum_loss = 0.0
    optimizer.zero_grad()

    for step in pbar:
        try:
            (xb,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (xb,) = next(data_iter)

        xb   = nan_to_zero(xb).to(device)
        loss = diffusion(xb) / grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update()
            pbar.set_postfix(loss=f"{accum_loss:.4f}")
            accum_loss = 0.0

    return ema.ema_model.cpu()


# =============================================================================
# LOSS EXTRACTION
# =============================================================================

@torch.no_grad()
def get_loss_vector(diffusion, X, cfg, n_samples=None):
    """Returns averaged scalar loss per patient. Shape (N,)."""
    if n_samples is None:
        n_samples  = cfg["n_loss_samples"]
    device     = cfg["device"]
    batch_size = cfg.get("shadow_batch_size", 128)
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

    device = x_start.device
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
# MAIN BLOCK 2
# =============================================================================

def main():
    cfg       = CONFIG
    split_dir = os.path.join(cfg["output_dir"], "block1", "splits")
    synth_dir = os.path.join(cfg["output_dir"], "block1", "synth")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f"[config]  dataset={cfg['dataset']}  K={cfg['K']}  "
          f"device={cfg['device']}")

    print("\n[load] Loading real data ...")
    X_real = load_real_data(cfg)
    N      = len(X_real)
    K      = cfg["K"]

    # load or initialise loss/label matrices
    loss_path  = os.path.join(cfg["output_dir"], "loss_matrix.npy")
    label_path = os.path.join(cfg["output_dir"], "label_matrix.npy")

    if os.path.exists(loss_path) and os.path.exists(label_path):
        print(f"  Resuming from existing loss_matrix / label_matrix ...")
        loss_matrix  = np.load(loss_path)
        label_matrix = np.load(label_path)
    else:
        loss_matrix  = np.zeros((N, K), dtype=np.float32)
        label_matrix = np.zeros((N, K), dtype=np.float32)

    resume_from = cfg["resume_from_k"]
    print(f"\n[Block 2] Training synth-shadow models and extracting losses ...")

    for k in range(K):
        if k < resume_from:
            print(f"  [k={k+1}/{K}] Skipping (resume_from_k={resume_from})")
            continue

        synth_path     = os.path.join(synth_dir, f"k{k:03d}_Dk_syn.pt")
        train_idx_path = os.path.join(split_dir,  f"k{k:03d}_train_idx.npy")
        test_idx_path  = os.path.join(split_dir,  f"k{k:03d}_test_idx.npy")

        if not os.path.exists(synth_path):
            print(f"  [k={k+1}/{K}] Missing {synth_path} -- run Block 1 first.")
            continue

        # load split indices
        train_idx = np.load(train_idx_path)
        test_idx  = np.load(test_idx_path)
        label_matrix[train_idx, k] = 1.0

        # load Dk_syn -- already in model output space, no preprocessing needed
        Dk_syn = torch.load(synth_path, map_location="cpu").float()
        print(f"\n  [k={k+1}/{K}] Training synth-shadow on "
              f"{len(Dk_syn)} synthetic samples  "
              f"({cfg['shadow_num_steps']} steps) ...")

        synth_shadow = build_diffusion(cfg, build_inner_model(cfg))
        synth_shadow = train_diffusion_steps(
            synth_shadow, Dk_syn,
            num_steps  = cfg["shadow_num_steps"],
            batch_size = cfg["shadow_batch_size"],
            grad_accum = cfg["shadow_grad_accum"],
            lr         = cfg["shadow_lr"],
            wd         = cfg["shadow_wd"],
            cfg        = cfg,
            desc       = f"    synth-shadow k={k+1}",
        )

        # relative loss: subtract mean loss on Dk_syn as per-split baseline
        # so all K splits are on a comparable scale for the MLP
        print(f"  [k={k+1}/{K}] Extracting loss features ...")
        baseline          = get_loss_vector(synth_shadow, Dk_syn,  cfg).mean()
        loss_matrix[:, k] = get_loss_vector(synth_shadow, X_real, cfg) - baseline
        del synth_shadow, Dk_syn

        # diagnostics
        m_loss  = loss_matrix[train_idx, k].mean()
        nm_loss = loss_matrix[test_idx,  k].mean()
        print(f"  [k={k+1}/{K}] member rel_loss={m_loss:.4f}  "
              f"non-member rel_loss={nm_loss:.4f}  gap={nm_loss-m_loss:.4f}")

        # checkpoint after every split so progress isn't lost on VM timeout
        np.save(loss_path,  loss_matrix)
        np.save(label_path, label_matrix)
        print(f"  [k={k+1}/{K}] Checkpointed -> {loss_path}")

    print(f"\n[Block 2 done]")
    print(f"  loss_matrix  : {loss_matrix.shape}")
    print(f"  label_matrix : {label_matrix.shape}")
    print(f"  Saved to {cfg['output_dir']}/")
    print("  Run next: MIDST_block3_train_classifier.py")


if __name__ == "__main__":
    main()