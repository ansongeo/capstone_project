"""
MIDST MIA -- Block 3: Train Meta-Classifier
============================================
Reads the synth-shadow loss features and labels from Block 2 and trains an
MLP meta-classifier on every (patient, shadow) pair:

    M: R^d -> P(member)
    input  = the patient's loss summaries under synth-shadow k
    output = raw logit for membership probability

With 50/50 splits the positive rate is exactly 50%.

Outputs saved to output_dir/
  meta_classifier.pt   -- MLP weights (best val AUC checkpoint)

Run next: MIDST_block4_mimic.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from midst_common import (DATASETS, parse_cli, MetaClassifierMLP, flat_features,
                          features_dir)


# =============================================================================
# CONFIG -- output_dir must match Block 2
# =============================================================================

CONFIG = {
    "mlp_hidden":     [256, 128, 64],
    "mlp_epochs":     50,
    "mlp_batch_size": 512,
    "mlp_lr":         1e-3,
    "mlp_wd":         1e-4,
    "mlp_dropout":    0.3,
    "mlp_patience":   50,   # early stopping: epochs without improvement

    "dataset":    "mimic",
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# DATASET BUILDING
# =============================================================================

def build_meta_dataset(cfg):
    """
    Pool all N * K (features, label) pairs from Block 2.

    Returns
    -------
    features : (N*K, d)  float32
    labels   : (N*K,)    float32
    """
    d  = features_dir(cfg)
    ks = sorted(int(f[2:4]) for f in os.listdir(d) if f.startswith("ss"))
    features = np.concatenate([flat_features(np.load(os.path.join(d, f"ss{k:02d}.npy")))
                               for k in ks]).astype(np.float32)
    labels   = np.concatenate([np.load(os.path.join(d, f"lab{k:02d}.npy"))
                               for k in ks]).astype(np.float32)
    idx      = np.random.permutation(len(labels))
    print(f"  pooled dataset : {features.shape}  "
          f"positive rate={100*labels.mean():.1f}%  (K={len(ks)} shadows)")
    return features[idx], labels[idx]


# =============================================================================
# TRAINING
# =============================================================================

def train_meta_classifier(features, labels, cfg):
    """
    Train MLP with early stopping on validation AUC.
    Returns best model (cpu).
    """
    device = cfg["device"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        features, labels,
        test_size=0.15, random_state=42, stratify=labels
    )
    print(f"  train={len(X_tr)}  val={len(X_val)}")

    mlp = MetaClassifierMLP(
        input_dim    = features.shape[1],
        hidden_sizes = cfg["mlp_hidden"],
        dropout      = cfg["mlp_dropout"],
    ).to(device)

    # compute pos_weight to handle class imbalance regardless of split_train_frac
    # for 50/50: pos_weight=1.0 (no effect); for 80/20: pos_weight=0.25
    pos_rate = float(y_tr.mean())
    pos_w    = torch.tensor([(1.0 - pos_rate) / pos_rate],
                             dtype=torch.float32, device=device)
    print(f"  positive rate={100*pos_rate:.1f}%  pos_weight={pos_w.item():.3f}")
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.Adam(mlp.parameters(),
                             lr=cfg["mlp_lr"], weight_decay=cfg["mlp_wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["mlp_epochs"]
    )

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=cfg["mlp_batch_size"], shuffle=True
    )

    best_auc, best_state = 0.0, None
    patience_counter = 0

    for epoch in range(cfg["mlp_epochs"]):
        mlp.train()
        epoch_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(mlp(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()

        mlp.eval()
        with torch.no_grad():
            val_logits = mlp(
                torch.from_numpy(X_val).to(device)
            ).cpu().numpy()
        auc = roc_auc_score(y_val, val_logits)

        if auc > best_auc:
            best_auc         = auc
            best_state       = {k: v.cpu().clone()
                                for k, v in mlp.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{cfg['mlp_epochs']}  "
                  f"train_loss={epoch_loss/len(tr_loader):.4f}  "
                  f"val_AUC={auc:.4f}  best={best_auc:.4f}  "
                  f"patience={patience_counter}/{cfg['mlp_patience']}")

        if patience_counter >= cfg["mlp_patience"]:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {cfg['mlp_patience']} epochs)")
            break

    print(f"\n  Best val AUC (meta-classifier): {best_auc:.4f}")
    mlp.load_state_dict(best_state)
    return mlp.cpu()


# =============================================================================
# MAIN BLOCK 3
# =============================================================================

def main():
    cfg = parse_cli(dict(CONFIG, **DATASETS))
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"[config]  device={cfg['device']}  output_dir={cfg['output_dir']}")

    print("\n[Block 3] Building meta-classifier training data ...")
    meta_X, meta_y = build_meta_dataset(cfg)

    print("\n[Block 3] Training meta-classifier ...")
    meta_clf = train_meta_classifier(meta_X, meta_y, cfg)

    save_path = os.path.join(cfg["output_dir"], "meta_classifier.pt")
    torch.save(meta_clf.state_dict(), save_path)
    print(f"\n[Block 3 done]  Meta-classifier saved to {save_path}")
    print("  Run next: MIDST_block4_mimic.py")


if __name__ == "__main__":
    main()