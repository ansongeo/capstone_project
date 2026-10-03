"""
MIDST MIA -- Block 3: Train Meta-Classifier
============================================
Reads the synth-shadow loss features and labels from Block 2 and trains a
gradient-boosted tree classifier (LightGBM) on every (patient, shadow) pair:

    M: R^d -> P(member)
    input  = the patient's loss summaries under synth-shadow k
    output = membership probability

One classifier is trained per shadow count in K_SWEEP (first K shadows), so
Block 4 can show whether more shadows still help.

Outputs saved to output_dir/
  meta_classifier_K{K}.joblib

Run next: MIDST_block4_mimic.py
"""

import os
import joblib
import numpy as np
import lightgbm as lgb

from midst_common import DATASETS, parse_cli, flat_features, features_dir


# =============================================================================
# CONFIG -- output_dir must match Block 2
# =============================================================================

CONFIG = {
    "K_sweep": (4, 8, 16, 32, 64),
    "lgbm": dict(n_estimators=600, learning_rate=0.03, num_leaves=31,
                 min_child_samples=100, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.5, verbose=-1, n_jobs=8),

    "dataset":    "mimic",
    "device":     "cpu",
}


# =============================================================================
# DATASET BUILDING
# =============================================================================

def shadow_ids(cfg):
    d = features_dir(cfg)
    return sorted(int(f[2:4]) for f in os.listdir(d) if f.startswith("ss"))


def build_meta_dataset(cfg, ks):
    """
    Pool the (features, label) pairs of shadows ks from Block 2.

    Returns
    -------
    features : (N*K, d)  float32
    labels   : (N*K,)    int
    """
    d = features_dir(cfg)
    features = np.concatenate([flat_features(np.load(os.path.join(d, f"ss{k:02d}.npy")))
                               for k in ks])
    labels   = np.concatenate([np.load(os.path.join(d, f"lab{k:02d}.npy")) for k in ks])
    print(f"  pooled dataset : {features.shape}  "
          f"positive rate={100*labels.mean():.1f}%  (K={len(ks)} shadows)")
    return features, labels


# =============================================================================
# MAIN BLOCK 3
# =============================================================================

def main():
    cfg = parse_cli(dict(CONFIG, **DATASETS))
    ks  = shadow_ids(cfg)
    print(f"[config]  output_dir={cfg['output_dir']}  shadows available={len(ks)}")

    for K in cfg["K_sweep"]:
        if K > len(ks):
            break
        print(f"\n[Block 3] K={K}: building training data and fitting LightGBM ...")
        meta_X, meta_y = build_meta_dataset(cfg, ks[:K])
        clf = lgb.LGBMClassifier(**cfg["lgbm"]).fit(meta_X, meta_y)
        save_path = os.path.join(cfg["output_dir"], f"meta_classifier_K{K}.joblib")
        joblib.dump(clf, save_path)
        print(f"  saved {save_path}")

    print("\n[Block 3 done]")
    print("  Run next: MIDST_block4_mimic.py")


if __name__ == "__main__":
    main()
