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

from midst_common import (DATASETS, load_real_pool, build_inner_model,
                          build_diffusion, train_diffusion_steps,
                          get_loss_vector)


# =============================================================================
# CONFIG  -- must match Block 1
# =============================================================================

CONFIG = {
    "dataset": "mimic",    # "eicu" | "mimic"

    **DATASETS,

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

    "output_dir": "block1_mimic",
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}


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
    X_real, _ = load_real_pool(cfg)
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
        baseline          = get_loss_vector(synth_shadow, Dk_syn,  cfg,
                                            batch_size=cfg["shadow_batch_size"]).mean()
        loss_matrix[:, k] = get_loss_vector(synth_shadow, X_real, cfg,
                                            batch_size=cfg["shadow_batch_size"]) - baseline
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