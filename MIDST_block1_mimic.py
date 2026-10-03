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

from midst_common import (DATASETS, load_real_pool, build_inner_model,
                          build_diffusion, train_diffusion_steps,
                          generate_synthetic)


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "dataset": "mimic",    # "eicu" | "mimic"

    **DATASETS,

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
    "resume_from_k":        0,

    "output_dir": "block1_mimic",  # where to save splits and synthetic datasets
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
    X_real, n_train = load_real_pool(cfg)

    # y_member is ground truth: 1 = was in original Dtrain, 0 = was in Dtest
    # saved here for use in Block 4 evaluation only
    y_member = np.zeros(len(X_real))
    y_member[:n_train] = 1
    np.save(os.path.join(cfg["output_dir"], "y_member.npy"), y_member)
    print(f"  ground truth: {int(y_member.sum())} members / "
          f"{int((1-y_member).sum())} non-members")
    return X_real, y_member


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
            use_ema    = False,
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