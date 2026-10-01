"""
tsne_first5_splits.py
---------------------
Reconstructs Dk_train and Dk_test from saved split indices, runs
eval_samples.py t-SNE for k=0..4, then stitches the 5 individual
PNGs into one combined figure: tsne_first5_splits.png

Usage:
    python tsne_first5_splits.py
"""

import os
import subprocess
import tempfile
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =============================================================================
# CONFIG -- edit these to match your setup
# =============================================================================

REAL_TRAIN_PATH = "data/eicu-extract/TRAIN-eicu_multiple_60_1440_276.pt"
REAL_TEST_PATH  = "data/eicu-extract/TEST-eicu_multiple_60_1440_276.pt"
BLOCK1_DIR      = "block1_eicu/block1"        # output_dir/block1
DATA_NAME       = "eicu"
SEQ_LEN         = 272
T_SNE_NUM       = 2000                          # samples per t-SNE plot
OUTPUT_PNG      = "tsne_new5_eicu.png"
N_SPLITS        = 5                             # how many splits to visualise

# =============================================================================
# LOAD FULL DREAL
# =============================================================================

print("[load] Loading Dreal ...")
X_train = torch.load(REAL_TRAIN_PATH, map_location="cpu").float()[:, :, :SEQ_LEN]
X_test  = torch.load(REAL_TEST_PATH,  map_location="cpu").float()[:, :, :SEQ_LEN]
X_real  = torch.cat([X_train, X_test], dim=0)
print(f"  Dreal {tuple(X_real.shape)}")

# =============================================================================
# RUN T-SNE FOR EACH SPLIT
# =============================================================================

split_dir = os.path.join(BLOCK1_DIR, "splits")
synth_dir = os.path.join(BLOCK1_DIR, "synth")
tmp_dir   = tempfile.mkdtemp(prefix="midst_tsne_")
img_paths = []

for k in range(N_SPLITS):
    train_idx_path = os.path.join(split_dir, f"k{k:03d}_train_idx.npy")
    test_idx_path  = os.path.join(split_dir, f"k{k:03d}_test_idx.npy")
    synth_path     = os.path.join(synth_dir,  f"k{k:03d}_Dk_syn.pt")

    if not os.path.exists(synth_path):
        print(f"  [k={k+1}] Missing {synth_path}, skipping.")
        continue
    if not os.path.exists(train_idx_path):
        print(f"  [k={k+1}] Missing split indices, skipping.")
        continue

    # reconstruct Dk_train and Dk_test from indices
    train_idx = np.load(train_idx_path)
    test_idx  = np.load(test_idx_path)
    Dk_train  = X_real[train_idx]
    Dk_test   = X_real[test_idx]

    # save as temp .pt files so eval_samples.py can load them
    dk_train_path = os.path.join(tmp_dir, f"k{k:03d}_Dk_train.pt")
    dk_test_path  = os.path.join(tmp_dir, f"k{k:03d}_Dk_test.pt")
    torch.save(Dk_train, dk_train_path)
    torch.save(Dk_test,  dk_test_path)

    img_path = os.path.join(tmp_dir, f"tsne_k{k:03d}.png")
    img_paths.append((k, img_path))

    print(f"\n[k={k+1}/{N_SPLITS}] Running t-SNE ...")
    print(f"  Dk_train={tuple(Dk_train.shape)}  "
          f"Dk_test={tuple(Dk_test.shape)}  "
          f"Dk_syn={synth_path}")

    cmd = [
        "python3", "eval_samples.py",
        "--data_name",   DATA_NAME,
        "--metric",      "t-sne",
        "--sync_path",   synth_path,
        "--train_path",  dk_train_path,
        "--test_path",   dk_test_path,
        "--img_name",    img_path,
        "--t_sne_num",   str(T_SNE_NUM),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [k={k+1}] eval_samples.py failed, skipping.")
        img_paths.pop()

# =============================================================================
# STITCH INTO ONE PNG
# =============================================================================

print(f"\n[stitch] Combining {len(img_paths)} plots into {OUTPUT_PNG} ...")

n = len(img_paths)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
if n == 1:
    axes = [axes]

for ax, (k, img_path) in zip(axes, img_paths):
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(f"k={k+1}", fontsize=14)
    else:
        ax.text(0.5, 0.5, f"k={k+1}\n(missing)",
                ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")

fig.suptitle(f"t-SNE: Dk_train vs Dk_syn for k=1..{n}  "
             f"(dataset={DATA_NAME})", fontsize=16)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"[done] Saved -> {OUTPUT_PNG}")

# cleanup temp files
import shutil
shutil.rmtree(tmp_dir)