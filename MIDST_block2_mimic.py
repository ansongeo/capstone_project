"""
MIDST MIA -- Block 2: Train Synth-Shadow Models & Extract Losses
================================================================
Reads outputs from Block 1 (Dk_syn + split indices) and for each k:
  1. Trains a synth-shadow model on Dk_syn (an overfitting probe:
     midst_common.PROBE, data on the pool's min/max scale)
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

from midst_common import (DATASETS, PROBE, parse_cli, load_attack_pool,
                          cat_idx, minmax, normalize, train_probe,
                          get_loss_vector)


# =============================================================================
# CONFIG  -- must match Block 1
# =============================================================================

CONFIG = {
    "dataset": "mimic",    # "eicu" | "mimic"

    **DATASETS,

    "K":                 32,    # must match Block 1

    # Synth-shadows use midst_common.PROBE (hidden 256, 1.4M steps)

    # Loss extraction
    "n_loss_samples":    20,
    "loss_batch_size":   32,

    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# MAIN BLOCK 2
# =============================================================================

def main():
    cfg       = parse_cli(CONFIG)
    split_dir = os.path.join(cfg["output_dir"], "block1", "splits")
    synth_dir = os.path.join(cfg["output_dir"], "block1", "synth")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f"[config]  dataset={cfg['dataset']}  K={cfg['K']}  "
          f"device={cfg['device']}")

    print("\n[load] Loading real data ...")
    X_pool, _ = load_attack_pool(cfg)
    pool_mn, pool_mx = minmax(X_pool)
    X_real = torch.tensor(normalize(X_pool, pool_mn, pool_mx, cat_idx(cfg)))
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

    print(f"\n[Block 2] Training synth-shadow models and extracting losses ...")

    for k in range(1, K + 1):
        synth_path     = os.path.join(synth_dir, f"k{k:03d}_Dk_syn.npy")
        train_idx_path = os.path.join(split_dir,  f"k{k:03d}_train_idx.npy")
        test_idx_path  = os.path.join(split_dir,  f"k{k:03d}_test_idx.npy")

        if not os.path.exists(synth_path):
            print(f"  [k={k}/{K}] Missing {synth_path} -- run Block 1 first.")
            continue

        # load split indices
        train_idx = np.load(train_idx_path)
        test_idx  = np.load(test_idx_path)
        label_matrix[train_idx, k - 1] = 1.0

        # load Dk_syn -- raw units, released like the target's Dsynth
        Dk_syn_raw = np.load(synth_path)
        print(f"\n  [k={k}/{K}] Training synth-shadow on "
              f"{len(Dk_syn_raw)} synthetic samples  "
              f"(hidden {PROBE['hidden']}, {PROBE['num_steps']} steps) ...")
        synth_shadow = train_probe(cfg, Dk_syn_raw, pool_mn, pool_mx, 500 + k,
                                   f"synthshadow{k}")
        torch.save(synth_shadow.state_dict(), os.path.join(
            cfg["output_dir"], "block1", "models", f"k{k:03d}_synthshadow.pt"))
        Dk_syn = torch.tensor(np.clip(normalize(Dk_syn_raw, pool_mn, pool_mx,
                                                cat_idx(cfg)), 0, 1))

        # relative loss: subtract mean loss on Dk_syn as per-split baseline
        # so all K splits are on a comparable scale for the MLP
        print(f"  [k={k}/{K}] Extracting loss features ...")
        baseline          = get_loss_vector(synth_shadow, Dk_syn,  cfg,
                                            batch_size=cfg["loss_batch_size"]).mean()
        loss_matrix[:, k - 1] = get_loss_vector(synth_shadow, X_real, cfg,
                                            batch_size=cfg["loss_batch_size"]) - baseline
        del synth_shadow, Dk_syn

        # diagnostics
        m_loss  = loss_matrix[train_idx, k - 1].mean()
        nm_loss = loss_matrix[test_idx,  k - 1].mean()
        print(f"  [k={k}/{K}] member rel_loss={m_loss:.4f}  "
              f"non-member rel_loss={nm_loss:.4f}  gap={nm_loss-m_loss:.4f}")

        # checkpoint after every split so progress isn't lost on VM timeout
        np.save(loss_path,  loss_matrix)
        np.save(label_path, label_matrix)
        print(f"  [k={k}/{K}] Checkpointed -> {loss_path}")

    print(f"\n[Block 2 done]")
    print(f"  loss_matrix  : {loss_matrix.shape}")
    print(f"  label_matrix : {label_matrix.shape}")
    print(f"  Saved to {cfg['output_dir']}/")
    print("  Run next: MIDST_block3_train_classifier.py")


if __name__ == "__main__":
    main()