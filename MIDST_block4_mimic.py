"""
MIDST MIA -- Block 4: Inference & Evaluation
=============================================
Reads meta_classifier.pt from Block 3 and Dsynth from Block 0.
Trains a proxy generator on released Dsynth, computes relative losses
on all of Dreal, applies the meta-classifier, and evaluates AUC.

  relative_loss(x) = mean_loss_proxy(x) - mean_loss_proxy(Dsynth)

The proxy is trained on Dsynth (released by the defender) and substitutes
for the unobserved target model weights.

Outputs saved to output_dir/
  inference_scores.npy       -- MLP logit scores per patient (N,)
  proxy_losses.npy           -- relative proxy loss per patient (N,)
  midst_roc.png
  midst_scores.png

Run after: MIDST_block3_train_classifier.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from midst_common import (DATASETS, parse_cli, preprocess, load_attack_pool,
                          cat_idx, minmax, normalize, build_inner_model,
                          build_diffusion,
                          train_diffusion_steps, get_loss_vector,
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

    # Proxy generator trained on released Dsynth
    # Use same steps as base shadow (Block 1) for consistent quality
    "proxy_num_steps":  600000,
    "proxy_batch_size": 32,
    "proxy_grad_accum": 2,
    "proxy_lr":         8e-5,
    "proxy_wd":         0.0,
    "ema_decay":        0.995,
    "ema_update_every": 10,

    "n_loss_samples": 20,

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
    ds       = cfg[cfg["dataset"]]
    cat_cols = ds["diffusion_kwargs"]["categorical_features_indices"]

    X_pool, y_member = load_attack_pool(cfg)
    X_real = torch.tensor(normalize(X_pool, *minmax(X_pool), cat_idx(cfg)))

    X_syn = torch.from_numpy(np.load(os.path.join(cfg["output_dir"], "released.npy")))
    print(f"  Dsynth (released) {tuple(X_syn.shape)}")
    print(f"  Preprocessing Dsynth ...")
    X_syn = preprocess(X_syn, cat_cols)

    return X_real, y_member, X_syn


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
    X_real, y_member, X_syn_released = load_all_data(cfg)

    # load meta-classifier saved by Block 3
    clf_path = os.path.join(cfg["output_dir"], "meta_classifier.pt")
    assert os.path.exists(clf_path), \
        f"Missing {clf_path} -- run Block 3 first."
    meta_clf = MetaClassifierMLP(
        input_dim    = 1,
        hidden_sizes = cfg["mlp_hidden"],
        dropout      = cfg["mlp_dropout"],
    )
    meta_clf.load_state_dict(torch.load(clf_path, map_location="cpu"))
    meta_clf.eval()
    print(f"  Loaded meta-classifier from {clf_path}")

    # train proxy generator on released Dsynth
    # proxy substitutes for the unobserved target model weights
    print(f"\n[Block 4] Training proxy on Dsynth "
          f"({cfg['proxy_num_steps']} steps) ...")
    proxy = build_diffusion(cfg, build_inner_model(cfg))
    proxy = train_diffusion_steps(
        proxy, X_syn_released,
        num_steps  = cfg["proxy_num_steps"],
        batch_size = cfg["proxy_batch_size"],
        grad_accum = cfg["proxy_grad_accum"],
        lr         = cfg["proxy_lr"],
        wd         = cfg["proxy_wd"],
        cfg        = cfg,
        desc       = "  proxy",
    )

    # compute relative losses on Dreal
    # baseline = mean proxy loss on Dsynth (matches shadow pipeline convention
    # where baseline = mean loss on Dk_syn)
    print("\n[Block 4] Extracting proxy loss features ...")
    proxy_baseline = get_loss_vector(
        proxy, X_syn_released, cfg, batch_size=cfg["proxy_batch_size"]
    ).mean()
    print(f"  proxy baseline (mean loss on Dsynth): {proxy_baseline:.4f}")

    proxy_losses = get_loss_vector(
        proxy, X_real, cfg, batch_size=cfg["proxy_batch_size"]
    ) - proxy_baseline                                     # (N_real,)
    del proxy

    print(f"  proxy_losses (raw)  mean={proxy_losses.mean():.4f}  "
          f"std={proxy_losses.std():.4f}  "
          f"min={proxy_losses.min():.4f}  max={proxy_losses.max():.4f}")

    # clip proxy losses at the same p99 threshold used in Block 3
    # so inference inputs match the training distribution seen by the MLP
    loss_matrix_path = os.path.join(cfg["output_dir"], "loss_matrix.npy")
    if os.path.exists(loss_matrix_path):
        loss_matrix = np.load(loss_matrix_path)
        p99_threshold = np.mean([
            np.percentile(loss_matrix[:, k], 99)
            for k in range(loss_matrix.shape[1])
        ])
        print(f"  Clipping at mean p99={p99_threshold:.3f} "
              f"(from Block 2 loss_matrix)")
        proxy_losses = np.clip(proxy_losses, -np.inf, p99_threshold)
    else:
        print(f"  WARNING: loss_matrix.npy not found -- skipping clip")

    print(f"  proxy_losses (clipped)  mean={proxy_losses.mean():.4f}  "
          f"std={proxy_losses.std():.4f}  "
          f"min={proxy_losses.min():.4f}  max={proxy_losses.max():.4f}")

    # score with meta-classifier
    meta_clf = meta_clf.to(device)
    scores   = []
    loader   = DataLoader(
        TensorDataset(torch.from_numpy(proxy_losses.reshape(-1, 1))),
        batch_size=cfg["mlp_batch_size"], shuffle=False
    )
    with torch.no_grad():
        for (xb,) in loader:
            scores.append(meta_clf(xb.to(device)).cpu().numpy())
    scores = np.concatenate(scores)

    # save outputs
    np.save(os.path.join(cfg["output_dir"], "inference_scores.npy"), scores)
    np.save(os.path.join(cfg["output_dir"], "proxy_losses.npy"),     proxy_losses)
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