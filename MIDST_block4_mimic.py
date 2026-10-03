"""
MIDST MIA -- Block 4: Inference & Evaluation
=============================================
Reads meta_classifier.pt from Block 3 and Dsynth from Block 0.
Trains a proxy generator on released Dsynth, computes the same loss
features as Block 2 on all of the attacked pool, applies the
meta-classifier, and evaluates AUC.

The proxy is trained on Dsynth (released by the defender) and substitutes
for the unobserved target model weights.

Outputs saved to output_dir/
  features/proxy.npy         -- proxy loss features (N, n_t, 2, n_budgets, 7)
  inference_scores.npy       -- MLP logit scores per patient (N,)
  midst_roc.png
  midst_scores.png

Run after: MIDST_block3_mimic.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from midst_common import (DATASETS, PROBE, parse_cli, load_attack_pool,
                          cat_idx, minmax, normalize, train_probe,
                          loss_features, flat_features, features_dir,
                          MetaClassifierMLP)


# =============================================================================
# CONFIG -- output_dir and dataset must match Blocks 1-3
# =============================================================================

CONFIG = {
    "dataset": "mimic",    # "eicu" | "mimic"

    **DATASETS,

    # MLP architecture -- must match Block 3
    "mlp_hidden":     [256, 128, 64],
    "mlp_dropout":    0.3,
    "mlp_batch_size": 512,

    # Proxy generator trained on released Dsynth: midst_common.PROBE, the
    # same overfitting settings as the synth-shadows (Block 2)

    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(cfg):
    """
    Load the attacked pool and its ground truth (Block 0) and the target's
    released synthetic data Dsynth (Block 0, raw units).
    """
    X_pool, y_member = load_attack_pool(cfg)
    pool_mn, pool_mx = minmax(X_pool)
    X_real = normalize(X_pool, pool_mn, pool_mx, cat_idx(cfg))

    X_syn = np.load(os.path.join(cfg["output_dir"], "released.npy"))
    print(f"  Dsynth (released) {tuple(X_syn.shape)}")
    return X_real, y_member, X_syn, (pool_mn, pool_mx)


# =============================================================================
# EVALUATION & PLOTS
# =============================================================================

def evaluate(scores, y_member):
    auc = roc_auc_score(y_member, scores)
    fpr, tpr, _ = roc_curve(y_member, scores)
    print(f"\n{'='*57}")
    print(f"  MIDST MIA  --  train (members) vs test (non-members)")
    print(f"{'='*57}")
    print(f"  AUC                     : {auc:.4f}")
    results = {"auc": auc, "fpr": fpr, "tpr": tpr}
    for target_fpr in [0.001, 0.005, 0.01, 0.05]:
        idx     = max(0, np.searchsorted(fpr, target_fpr, side="right") - 1)
        tpr_val = float(tpr[idx])
        results[f"tpr@{target_fpr}"] = tpr_val
        print(f"  TPR @ FPR={target_fpr:<6}      : {tpr_val:.4f}")
    print(f"  (random baseline  AUC=0.5000, TPR=FPR)")
    print(f"{'='*57}")
    return results


def plot_roc(results, output_dir):
    plt.figure(figsize=(7, 6))
    plt.plot(results["fpr"], results["tpr"], lw=2,
             label=f"MIDST MIA  AUC={results['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MIDST MIA -- Member vs Non-member ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "midst_roc.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] {path}")


def plot_score_distributions(scores, y_member, output_dir):
    m  = scores[y_member == 1]
    nm = scores[y_member == 0]
    plt.figure(figsize=(7, 4))
    kw = dict(bins=80, alpha=0.6, density=True)
    plt.hist(m,  **kw, color="steelblue", label=f"Members     n={len(m)}")
    plt.hist(nm, **kw, color="tomato",    label=f"Non-members n={len(nm)}")
    plt.xlabel("Meta-classifier score (logit)")
    plt.ylabel("Density")
    plt.title("MIDST MIA -- Score distributions")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "midst_scores.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[plot] {path}")


# =============================================================================
# MAIN BLOCK 4
# =============================================================================

def main():
    cfg    = parse_cli(CONFIG)
    device = cfg["device"]
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"[config]  dataset={cfg['dataset']}  device={device}  "
          f"output_dir={cfg['output_dir']}")

    # load data
    print("\n[load] Loading data ...")
    X_real, y_member, X_syn_raw, (pool_mn, pool_mx) = load_all_data(cfg)

    clf_path = os.path.join(cfg["output_dir"], "meta_classifier.pt")
    assert os.path.exists(clf_path), \
        f"Missing {clf_path} -- run Block 3 first."

    # train proxy generator on released Dsynth
    # proxy substitutes for the unobserved target model weights
    print(f"\n[Block 4] Training proxy on Dsynth "
          f"(hidden {PROBE['hidden']}, {PROBE['num_steps']} steps) ...")
    proxy = train_probe(cfg, X_syn_raw, pool_mn, pool_mx, 900, "proxy")
    torch.save(proxy.state_dict(), os.path.join(cfg["output_dir"], "proxy.pt"))

    print("\n[Block 4] Extracting proxy loss features ...")
    feat_path = os.path.join(features_dir(cfg), "proxy.npy")
    F = loss_features(cfg, proxy, X_real)
    np.save(feat_path, F)
    del proxy
    X_attack = flat_features(F)

    # load meta-classifier saved by Block 3
    meta_clf = MetaClassifierMLP(
        input_dim    = X_attack.shape[1],
        hidden_sizes = cfg["mlp_hidden"],
        dropout      = cfg["mlp_dropout"],
    )
    meta_clf.load_state_dict(torch.load(clf_path, map_location="cpu"))
    meta_clf.eval()
    print(f"  Loaded meta-classifier from {clf_path}")

    # score with meta-classifier
    meta_clf = meta_clf.to(device)
    scores   = []
    loader   = DataLoader(
        TensorDataset(torch.from_numpy(X_attack)),
        batch_size=cfg["mlp_batch_size"], shuffle=False
    )
    with torch.no_grad():
        for (xb,) in loader:
            scores.append(meta_clf(xb.to(device)).cpu().numpy())
    scores = np.concatenate(scores)

    # save outputs
    np.save(os.path.join(cfg["output_dir"], "inference_scores.npy"), scores)
    print(f"\n  member scores     mean={scores[y_member==1].mean():.3f}  "
          f"std={scores[y_member==1].std():.3f}")
    print(f"  non-member scores mean={scores[y_member==0].mean():.3f}  "
          f"std={scores[y_member==0].std():.3f}")

    # evaluate
    print("\n[eval] Evaluating ...")
    results = evaluate(scores, y_member)

    # plots
    print("\n[plots] Saving ...")
    plot_roc(results, cfg["output_dir"])
    plot_score_distributions(scores, y_member, cfg["output_dir"])

    print(f"\n[Block 4 done]  All outputs in ./{cfg['output_dir']}/")


if __name__ == "__main__":
    main()