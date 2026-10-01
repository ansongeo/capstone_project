"""
MIDST MIA -- Block 1: Generate Internal Synthetic Datasets
===========================================================
Supports eICU (N, 9, 272) and MIMIC-IV (N, 11, 72)

What this block does
--------------------
  For each k = 1..K:
    1. Sample a random 80/20 split of Dreal -> (Dk_train, Dk_test)
    2. Train a BASE SHADOW GENERATOR on Dk_train (real patients)
       using full training (num_steps iterations with gradient accumulation
       and EMA, matching the original ETDiff training setup) so that the
       generated synthetic data is high quality.
    3. Generate Dk_syn from the base shadow.
    4. Save Dk_syn + split indices to disk.

Outputs saved to output_dir/block1/
  splits/k{k:03d}_train_idx.npy   -- indices of Dk_train in X_real
  splits/k{k:03d}_test_idx.npy    -- indices of Dk_test  in X_real
  synth/k{k:03d}_Dk_syn.pt        -- generated synthetic tensor (N_syn, C, T)

Run next: MIDST_block2_extract_losses.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.ETDiff.mixed_diffusion import MixedDiffusion
from models.ETDiff.blocks import RNN


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "dataset": "eicu",    # "eicu" | "mimic"

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

    "K":                    16,      # number of 80/20 splits
    "split_train_frac":     0.8,     # 80% train / 20% test

    # Base shadow training -- full ETDiff-style training for quality synthetic data
    "base_num_steps":       600000,  # gradient steps -- 700k is full ETDiff;
                                     # 300k balances quality vs runtime on half the data
    "base_batch_size":      32,
    "base_grad_accum":      2,       # gradient accumulation steps
    "base_lr":              8e-5,    # matches ETDiff default
    "base_wd":              0.0,

    "synth_samples_per_shadow": 2000,  # reduced to speed up generation

    # split reproducibility -- each split k uses seed (split_seed + k)
    "split_seed":           42,

    # resume -- set to k index (0-based) to resume from a specific split
    "resume_from_k":        12,

    "output_dir": "block1_eicu",  # where to save splits and synthetic datasets
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_real_data(cfg):
    """
    The attacker sees Dreal as a single unlabelled pool.
    X_train and X_test are joined into X_real -- the attacker does NOT
    know the original split. y_member is ground truth kept only for
    final evaluation in Block 4, never used during the attack itself.
    """
    ds      = cfg[cfg["dataset"]]
    seq_len = ds["diffusion_kwargs"]["seq_length"]
    X_train = torch.load(ds["train_path"], map_location="cpu").float()
    X_test  = torch.load(ds["test_path"],  map_location="cpu").float()
    X_train = X_train[:, :, :seq_len]
    X_test  = X_test[:,  :, :seq_len]

    # join into single unlabelled pool -- this is all the attacker sees
    X_real = torch.cat([X_train, X_test], dim=0)

    # preprocess to match ETDiff training pipeline:
    #   1. replace NaNs with per-channel mean
    #   2. min-max normalize to [0,1], categorical columns left untouched
    ds_cfg         = cfg[cfg["dataset"]]
    categorical_cols = ds_cfg["diffusion_kwargs"]["categorical_features_indices"]
    print(f"  Preprocessing: replace NaN with mean + min-max normalize ...")
    X_real = preprocess(X_real, categorical_cols)

    # y_member is ground truth: 1 = was in original Dtrain, 0 = was in Dtest
    # saved here for use in Block 4 evaluation only
    y_member = torch.cat([torch.ones(len(X_train)),
                          torch.zeros(len(X_test))]).numpy()
    np.save(os.path.join(cfg["output_dir"], "y_member.npy"), y_member)

    print(f"  X_train {tuple(X_train.shape)}  X_test {tuple(X_test.shape)}")
    print(f"  Dreal (joined pool) {tuple(X_real.shape)}")
    print(f"  ground truth: {int(y_member.sum())} members / "
          f"{int((1-y_member).sum())} non-members")
    return X_real, y_member


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

    mn  = np.nanmin(data_np, axis=(0, 2))
    mx  = np.nanmax(data_np, axis=(0, 2))
    mn  = mn[None, :, None]
    mx  = mx[None, :, None]
    data_np = (data_np - mn + eps) / (mx - mn + 2 * eps)

    if categorical_cols is not None:
        data_np[:, categorical_cols, :] = indicators   # restore categoricals

    return torch.tensor(data_np).float()


def preprocess(data: torch.Tensor, categorical_cols: list) -> torch.Tensor:
    """Full preprocessing pipeline matching ETDiff TimeSeriesDataset."""
    data = replace_nan_with_mean(data)
    data = normalize_data(data, categorical_cols)
    return data


def nan_to_zero(x):
    """Fallback for in-loop NaN cleanup after preprocessing."""
    return torch.nan_to_num(x, nan=0.0)


# =============================================================================
# MODEL CONSTRUCTION
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
    Train for a fixed number of gradient steps rather than epochs.
    Matches the ETDiff training setup (num_steps + gradient accumulation).
    """
    device    = cfg["device"]
    diffusion = diffusion.to(device).train()
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr, weight_decay=wd)
    loader    = DataLoader(TensorDataset(X), batch_size=batch_size,
                           shuffle=True, drop_last=True)
    data_iter = iter(loader)
    pbar      = tqdm(range(num_steps), desc=desc, leave=False)

    step        = 0
    accum_loss  = 0.0
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
            pbar.set_postfix(loss=f"{accum_loss:.4f}")
            accum_loss = 0.0

    return diffusion.cpu()


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
# MAIN BLOCK 1
# =============================================================================

def main():
    cfg = CONFIG
    split_dir = os.path.join(cfg["output_dir"], "block1", "splits")
    synth_dir = os.path.join(cfg["output_dir"], "block1", "synth")
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f"[config]  dataset={cfg['dataset']}  K={cfg['K']}  "
          f"device={cfg['device']}")
    print(f"  base_num_steps={cfg['base_num_steps']}  "
          f"batch={cfg['base_batch_size']}  "
          f"grad_accum={cfg['base_grad_accum']}")

    print("\n[load] Loading real data ...")
    X_real, y_member = load_real_data(cfg)
    N = len(X_real)

    K            = cfg["K"]
    resume_from  = cfg["resume_from_k"]
    device       = cfg["device"]

    print(f"\n[Block 1] Generating K={K} internal synthetic datasets ...")
    print(f"  Outputs -> {cfg['output_dir']}/block1/\n")

    for k in range(K):
        split_train_path = os.path.join(split_dir, f"k{k:03d}_train_idx.npy")
        split_test_path  = os.path.join(split_dir, f"k{k:03d}_test_idx.npy")
        synth_path       = os.path.join(synth_dir,  f"k{k:03d}_Dk_syn.pt")

        # resume: skip already-completed splits
        if k < resume_from:
            print(f"  [k={k+1}/{K}] Skipping (resume_from_k={resume_from})")
            continue

        if os.path.exists(synth_path):
            print(f"  [k={k+1}/{K}] Already exists, skipping.")
            continue

        # Step 1a: split -- use deterministic per-k seed so resuming
        # across VM sessions always produces the same split for split k
        rng       = np.random.RandomState(seed=cfg.get("split_seed", 42) + k)
        perm      = rng.permutation(N)
        train_idx = perm[:int(N * cfg["split_train_frac"])]
        test_idx  = perm[int(N * cfg["split_train_frac"]):]
        np.save(split_train_path, train_idx)
        np.save(split_test_path,  test_idx)

        # Step 1b: train base shadow
        print(f"\n  [k={k+1}/{K}] Training base shadow on "
              f"{len(train_idx)} real patients "
              f"({cfg['base_num_steps']} steps) ...")
        base_gen = build_diffusion(cfg, build_inner_model(cfg))
        base_gen = train_diffusion_steps(
            base_gen, X_real[train_idx],
            num_steps  = cfg["base_num_steps"],
            batch_size = cfg["base_batch_size"],
            grad_accum = cfg["base_grad_accum"],
            lr         = cfg["base_lr"],
            wd         = cfg["base_wd"],
            cfg        = cfg,
            desc       = f"    base shadow k={k+1}",
        )

        # Step 1c: generate Dk_syn
        print(f"  [k={k+1}/{K}] Generating "
              f"{cfg['synth_samples_per_shadow']} synthetic samples ...")
        Dk_syn = generate_synthetic(
            base_gen, cfg["synth_samples_per_shadow"],
            cfg["base_batch_size"], device
        )
        del base_gen

        torch.save(Dk_syn, synth_path)
        print(f"  [k={k+1}/{K}] Saved -> {synth_path}  shape={tuple(Dk_syn.shape)}")
        del Dk_syn

    print(f"\n[Block 1 done]  Splits and synthetic datasets saved to "
          f"{cfg['output_dir']}/block1/")
    print("  Run next: MIDST_block2_extract_losses.py")


if __name__ == "__main__":
    main()
