"""
MIDST MIA -- Block 1: Generate Internal Synthetic Datasets
===========================================================
Supports eICU (N, 9, 272) and MIMIC-IV (N, 11, 72)

What this block does
--------------------
  For each k = 1..K:
    1. Sample a random 50/50 split of the attacked pool -> (Dk_train, Dk_test)
    2. Train a BASE SHADOW GENERATOR on Dk_train (real patients) with exactly
       the target's settings (Block 0: TimeDiff recipe, same hidden size,
       steps and own-min/max normalisation), so its synthetic data looks
       like the target's release.
    3. Release Dk_syn from the base shadow: 20,000 records in raw units,
       the same size as the target's release.
    4. Save Dk_syn + split indices to disk.
    5. White-box view: the base shadow's own loss features on the pool
       (it plays the target's role for the white-box attack).

Outputs saved to output_dir/block1/
  splits/k{k:03d}_train_idx.npy   -- indices of Dk_train in the attacked pool
  splits/k{k:03d}_test_idx.npy    -- indices of Dk_test  in the attacked pool
  synth/k{k:03d}_Dk_syn.npy       -- released synthetic data (20000, C, T)
  models/k{k:03d}_shadow.pt       -- base shadow weights (+ .minmax.npz)
Outputs saved to output_dir/features/
  sh{k:02d}.npy, lab{k:02d}.npy   -- shadow k's loss features and labels

Run after: MIDST_block0_mimic.py
Run next:  MIDST_block2_mimic.py
"""

import os
import numpy as np
import torch

from midst_common import (DATASETS, TARGET, parse_cli, load_attack_pool,
                          fit_release, cat_idx, normalize, loss_features,
                          features_dir)


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    "dataset": "mimic",    # "eicu" | "mimic"

    **DATASETS,

    "K":                    32,      # number of shadow models (50/50 splits)

    # Base shadows use the target's recipe (midst_common.TARGET) and hidden
    # size (target_hidden), so Dk_syn is distributed like the real release.

    # split reproducibility -- each split k uses seed (split_seed + k)
    "split_seed":           1000,

    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# MAIN BLOCK 1
# =============================================================================

def main():
    cfg = parse_cli(CONFIG)
    split_dir = os.path.join(cfg["output_dir"], "block1", "splits")
    synth_dir = os.path.join(cfg["output_dir"], "block1", "synth")
    model_dir = os.path.join(cfg["output_dir"], "block1", "models")
    for d in (split_dir, synth_dir, model_dir):
        os.makedirs(d, exist_ok=True)

    K      = cfg["K"]
    hidden = cfg[cfg["dataset"]]["target_hidden"]
    print(f"[config]  dataset={cfg['dataset']}  K={K}  device={cfg['device']}")
    print(f"  base shadows = target recipe: hidden={hidden}  "
          f"steps={TARGET['num_steps']}  release={TARGET['n_release']}")

    print("\n[load] Loading the attacked pool (Block 0) ...")
    X_pool, _ = load_attack_pool(cfg)
    N = len(X_pool)

    print(f"\n[Block 1] Generating K={K} internal synthetic datasets ...")
    print(f"  Outputs -> {cfg['output_dir']}/block1/\n")

    for k in range(1, K + 1):
        split_train_path = os.path.join(split_dir, f"k{k:03d}_train_idx.npy")
        split_test_path  = os.path.join(split_dir, f"k{k:03d}_test_idx.npy")
        synth_path       = os.path.join(synth_dir, f"k{k:03d}_Dk_syn.npy")
        feat_path        = os.path.join(features_dir(cfg), f"sh{k:02d}.npy")

        if os.path.exists(synth_path) and os.path.exists(feat_path):
            print(f"  [k={k}/{K}] Already exists, skipping.")
            continue

        # Step 1a: 50/50 split, like the target's members / non-members.
        # Deterministic per-k seed so resuming always gives the same split.
        rng       = np.random.RandomState(seed=cfg["split_seed"] + k)
        member    = np.zeros(N, bool)
        member[rng.permutation(N)[:N // 2]] = True
        train_idx = np.where(member)[0]
        test_idx  = np.where(~member)[0]
        np.save(split_train_path, train_idx)
        np.save(split_test_path,  test_idx)

        # Step 1b + 1c: train the base shadow like the target, release Dk_syn
        print(f"\n  [k={k}/{K}] Training base shadow on "
              f"{len(train_idx)} real patients ({TARGET['num_steps']} steps) ...")
        shadow, (mn, mx), Dk_syn = fit_release(
            cfg, X_pool[train_idx], TARGET["num_steps"], hidden, 100 + k,
            f"shadow{k}", os.path.join(model_dir, f"k{k:03d}_shadow.pt"))
        np.save(synth_path, Dk_syn)
        print(f"  [k={k}/{K}] Saved -> {synth_path}  shape={Dk_syn.shape}")

        # Step 1d: white-box training rows -- the shadow scores the pool in
        # its own normalisation, as the target does in Block 0
        label = np.zeros(N, dtype=int)
        label[train_idx] = 1
        np.save(os.path.join(features_dir(cfg), f"lab{k:02d}.npy"), label)
        np.save(feat_path, loss_features(cfg, shadow,
                                         normalize(X_pool, mn, mx, cat_idx(cfg))))
        del shadow

    print(f"\n[Block 1 done]  Splits and synthetic datasets saved to "
          f"{cfg['output_dir']}/block1/")
    print("  Run next: MIDST_block2_mimic.py")


if __name__ == "__main__":
    main()