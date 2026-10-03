"""
MIDST MIA -- Block 2: Train Synth-Shadow Models & Extract Loss Features
=======================================================================
Reads outputs from Block 1 (Dk_syn + split indices) and for each k:
  1. Trains a synth-shadow model on Dk_syn (an overfitting probe:
     midst_common.PROBE, data on the pool's min/max scale)
  2. Computes loss features on ALL of the attacked pool: per diffusion step
     t, the vital-sign and categorical losses under 600 frozen noise draws,
     summarised per record (midst_common.loss_features)
  3. Saves them with the record labels for shadow k

Outputs saved to output_dir/features/
  ss{k:02d}.npy    (N, n_t, 2, n_budgets, 7)  synth-shadow k's loss features
  lab{k:02d}.npy   (N,)                        1 = record was in Dk_train

Run next: MIDST_block3_mimic.py
"""

import os
import numpy as np
import torch

from midst_common import (DATASETS, PROBE, parse_cli, load_attack_pool,
                          cat_idx, minmax, normalize, train_probe,
                          loss_features, features_dir)


# =============================================================================
# CONFIG  -- must match Block 1
# =============================================================================

CONFIG = {
    "dataset": "mimic",    # "eicu" | "mimic"

    **DATASETS,

    "K":                 32,    # must match Block 1

    # Synth-shadows use midst_common.PROBE (hidden 256, 1.4M steps)


    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# MAIN BLOCK 2
# =============================================================================

def main():
    cfg       = parse_cli(CONFIG)
    split_dir = os.path.join(cfg["output_dir"], "block1", "splits")
    synth_dir = os.path.join(cfg["output_dir"], "block1", "synth")
    feat_dir  = features_dir(cfg)

    print(f"[config]  dataset={cfg['dataset']}  K={cfg['K']}  "
          f"device={cfg['device']}")

    print("\n[load] Loading real data ...")
    X_pool, _ = load_attack_pool(cfg)
    pool_mn, pool_mx = minmax(X_pool)
    X_real = normalize(X_pool, pool_mn, pool_mx, cat_idx(cfg))
    N      = len(X_real)
    K      = cfg["K"]

    print(f"\n[Block 2] Training synth-shadow models and extracting losses ...")

    for k in range(1, K + 1):
        synth_path = os.path.join(synth_dir, f"k{k:03d}_Dk_syn.npy")
        feat_path  = os.path.join(feat_dir, f"ss{k:02d}.npy")

        if os.path.exists(feat_path):
            print(f"  [k={k}/{K}] Already exists, skipping.")
            continue
        if not os.path.exists(synth_path):
            print(f"  [k={k}/{K}] Missing {synth_path} -- run Block 1 first.")
            continue

        # labels for shadow k: 1 = record was in Dk_train
        label = np.zeros(N, dtype=int)
        label[np.load(os.path.join(split_dir, f"k{k:03d}_train_idx.npy"))] = 1
        np.save(os.path.join(feat_dir, f"lab{k:02d}.npy"), label)

        # load Dk_syn -- raw units, released like the target's Dsynth
        Dk_syn_raw = np.load(synth_path)
        print(f"\n  [k={k}/{K}] Training synth-shadow on "
              f"{len(Dk_syn_raw)} synthetic samples  "
              f"(hidden {PROBE['hidden']}, {PROBE['num_steps']} steps) ...")
        synth_shadow = train_probe(cfg, Dk_syn_raw, pool_mn, pool_mx, 500 + k,
                                   f"synthshadow{k}")
        torch.save(synth_shadow.state_dict(), os.path.join(
            cfg["output_dir"], "block1", "models", f"k{k:03d}_synthshadow.pt"))

        print(f"  [k={k}/{K}] Extracting loss features ...")
        F = loss_features(cfg, synth_shadow, X_real)
        np.save(feat_path, F)
        del synth_shadow

        # diagnostic: mean loss at t=0, members vs non-members
        for j, kind in enumerate(("vitals", "categorical")):
            m, nm = F[label == 1, 0, j, -1, 0].mean(), F[label == 0, 0, j, -1, 0].mean()
            print(f"  [k={k}/{K}] {kind:11s} loss t=0: member {m:.4f}  "
                  f"non-member {nm:.4f}  gap {nm - m:.4f}")

    print(f"\n[Block 2 done]  features -> {feat_dir}/")
    print("  Run next: MIDST_block3_mimic.py")


if __name__ == "__main__":
    main()