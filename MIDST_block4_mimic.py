"""
MIDST MIA -- Block 4: Inference & Evaluation
=============================================
Reads meta_classifier.pt from Block 3.
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

from ema_pytorch import EMA
from models.ETDiff.mixed_diffusion import MixedDiffusion
from models.ETDiff.blocks import RNN


# =============================================================================
# CONFIG -- output_dir and dataset must match Blocks 1-3
# =============================================================================

CONFIG = {
    "dataset": "mimic",    # "eicu" | "mimic"

    "eicu": {
        "train_path":     "data/eicu-extract/TRAIN-eicu_multiple_60_1440_276.pt",
        "test_path":      "data/eicu-extract/TEST-eicu_multiple_60_1440_276.pt",
        "synthetic_path": "samples/eicu.npy",
        "diffusion_kwargs": {
            "seq_length":                   272,
            "channels":                     9,
            "numerical_features_indices":   [0, 2, 4, 6],
            "categorical_features_indices": [1, 3, 5, 7, 8],
            "categorical_num_classes":      [2, 2, 2, 2, 2],
            "timesteps":                    1000,
            "beta_schedule":                "cosine",
            "auto_normalize":               True,
            "loss_lambda":                  0.8,
            "parametrization":              "x0",
        },
    },

    "mimic": {
        "train_path":     "data/mimic4-extract/TRAIN-mimic4_vitals_72h.pt",
        "test_path":      "data/mimic4-extract/TEST-mimic4_vitals_72h.pt",
        "synthetic_path": "samples/mimiciv.npy",
        "diffusion_kwargs": {
            "seq_length":                   72,
            "channels":                     11,
            "numerical_features_indices":   [0, 2, 4, 6, 8],
            "categorical_features_indices": [1, 3, 5, 7, 9, 10],
            "categorical_num_classes":      [2, 2, 2, 2, 2, 2],
            "timesteps":                    1000,
            "beta_schedule":                "cosine",
            "auto_normalize":               True,
            "loss_lambda":                  0.8,
            "parametrization":              "x0",
        },
    },

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

    "output_dir": "block1_mimic",
    "device":     "cuda" if torch.cuda.is_available() else "cpu",
}


# =============================================================================
# PREPROCESSING -- matches ETDiff TimeSeriesDataset
# =============================================================================

def replace_nan_with_mean(data: torch.Tensor) -> torch.Tensor:
    """Replace NaNs with per-channel mean -- matches ETDiff training."""
    mean = data.nanmean(dim=(0, 2))[None, :, None]
    data = torch.where(torch.isnan(data), mean, data)
    return data


def normalize_data(data: torch.Tensor, categorical_cols: list,
                   eps: float = 0) -> torch.Tensor:
    """
    Min-max normalize to [0,1] along (samples, time) per channel.
    Categorical columns are left untouched -- matches ETDiff training.
    """
    data_np = data.numpy()
    if categorical_cols is not None:
        indicators = data_np[:, categorical_cols, :].copy()
    mn = np.nanmin(data_np, axis=(0, 2))[None, :, None]
    mx = np.nanmax(data_np, axis=(0, 2))[None, :, None]
    data_np = (data_np - mn + eps) / (mx - mn + 2 * eps)
    if categorical_cols is not None:
        data_np[:, categorical_cols, :] = indicators
    return torch.tensor(data_np).float()


def preprocess(data: torch.Tensor, categorical_cols: list) -> torch.Tensor:
    """Full preprocessing pipeline matching ETDiff TimeSeriesDataset."""
    data = replace_nan_with_mean(data)
    data = normalize_data(data, categorical_cols)
    return data


def nan_to_zero(x):
    """Fallback for residual NaNs during batch processing."""
    return torch.nan_to_num(x, nan=0.0)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data(cfg):
    """
    Load and preprocess Dreal and Dsynth.
    y_member is loaded from Block 1's saved y_member.npy to ensure
    consistency with the split indices used during shadow training.
    """
    ds       = cfg[cfg["dataset"]]
    seq_len  = ds["diffusion_kwargs"]["seq_length"]
    cat_cols = ds["diffusion_kwargs"]["categorical_features_indices"]

    # load Dreal as single unlabelled pool -- same as Blocks 1 and 2
    X_train = torch.load(ds["train_path"], map_location="cpu").float()[:, :, :seq_len]
    X_test  = torch.load(ds["test_path"],  map_location="cpu").float()[:, :, :seq_len]
    X_real  = torch.cat([X_train, X_test], dim=0)

    print(f"  X_train {tuple(X_train.shape)}  X_test {tuple(X_test.shape)}")
    print(f"  Dreal (joined pool) {tuple(X_real.shape)}")
    print(f"  Preprocessing Dreal ...")
    X_real = preprocess(X_real, cat_cols)

    # load ground truth labels saved by Block 1
    y_member_path = os.path.join(cfg["output_dir"], "y_member.npy")
    assert os.path.exists(y_member_path), \
        f"Missing {y_member_path} -- run Block 1 first."
    y_member = np.load(y_member_path)
    print(f"  ground truth: {int(y_member.sum())} members / "
          f"{int((1-y_member).sum())} non-members")

    # load and preprocess released Dsynth
    X_syn = _load_npy_or_pt(ds["synthetic_path"], seq_len)
    print(f"  Dsynth (released) {tuple(X_syn.shape)}")
    print(f"  Preprocessing Dsynth ...")
    X_syn = preprocess(X_syn, cat_cols)

    return X_real, y_member, X_syn


def _load_npy_or_pt(path, seq_len):
    """Load .npy or .pt synthetic data file."""
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu").float()[:, :, :seq_len]
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype.kind in ("f", "i", "u"):
        return torch.from_numpy(raw.astype(np.float32))[:, :, :seq_len]
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        inner = raw.item()
        if isinstance(inner, torch.Tensor):
            return inner.float()[:, :, :seq_len]
        return torch.from_numpy(np.array(inner, dtype=np.float32))[:, :, :seq_len]
    return torch.load(path, map_location="cpu").float()[:, :, :seq_len]


# =============================================================================
# MODEL
# =============================================================================

def build_inner_model(cfg):
    ds_cfg  = cfg[cfg["dataset"]]
    diff_kw = ds_cfg["diffusion_kwargs"]
    n_num   = len(diff_kw["numerical_features_indices"])
    n_cat   = sum(diff_kw["categorical_num_classes"])
    ch      = n_num + n_cat
    return RNN(
        input_channels  = ch,
        hidden_channels = 256,
        output_channels = ch,
        layers          = 3,
        model           = "lstm",
        dropout         = 0,
        bidirectional   = False,
        self_condition  = False,
        embed_dim       = 64,
        time_dim        = 256,
    )


def build_diffusion(cfg, inner_model):
    ds_cfg = cfg[cfg["dataset"]]
    return MixedDiffusion(model=inner_model, **ds_cfg["diffusion_kwargs"])


def train_diffusion_steps(diffusion, X, num_steps, batch_size, grad_accum,
                          lr, wd, cfg, desc):
    """Train proxy with EMA -- returns ema_model."""
    device     = cfg["device"]
    diffusion  = diffusion.to(device).train()
    ema        = EMA(diffusion,
                     beta         = cfg.get("ema_decay", 0.995),
                     update_every = cfg.get("ema_update_every", 10))
    ema        = ema.to(device)

    optimizer  = torch.optim.Adam(diffusion.parameters(), lr=lr, weight_decay=wd)
    loader     = DataLoader(TensorDataset(X), batch_size=batch_size,
                            shuffle=True, drop_last=True)
    data_iter  = iter(loader)
    pbar       = tqdm(range(num_steps), desc=desc, leave=False)
    accum_loss = 0.0
    optimizer.zero_grad()

    for step in pbar:
        try:
            (xb,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (xb,) = next(data_iter)

        xb   = nan_to_zero(xb).to(device)
        loss = diffusion(xb) / grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update()
            pbar.set_postfix(loss=f"{accum_loss:.4f}")
            accum_loss = 0.0

    return ema.ema_model.cpu()


# =============================================================================
# LOSS EXTRACTION
# =============================================================================

@torch.no_grad()
def get_loss_vector(diffusion, X, cfg, n_samples=None):
    """Returns averaged scalar loss per patient. Shape (N,)."""
    if n_samples is None:
        n_samples  = cfg["n_loss_samples"]
    device     = cfg["device"]
    batch_size = cfg.get("proxy_batch_size", 128)
    diffusion  = diffusion.to(device).eval()
    total      = np.zeros(len(X), dtype=np.float64)

    try:
        for _ in range(n_samples):
            chunk = []
            for (xb,) in DataLoader(TensorDataset(X),
                                    batch_size=batch_size, shuffle=False):
                xb = nan_to_zero(xb).to(device)
                t  = torch.randint(0, diffusion.num_timesteps,
                                   (xb.size(0),), device=device).long()
                loss = _per_sample_loss(diffusion, xb, t)
                chunk.append(loss.cpu().numpy())
            total += np.concatenate(chunk)
    finally:
        diffusion.cpu()

    return (total / n_samples).astype(np.float32)


def _per_sample_loss(diffusion, x_start, t):
    from models.ETDiff.mixed_diffusion import index_to_log_onehot
    import torch.nn.functional as F
    from einops import reduce
    import models.ETDiff.utils as utils

    device = x_start.device
    pt     = torch.ones_like(t).float() / diffusion.num_timesteps

    # apply same [-1,1] normalization used during training
    x_num   = diffusion.extract_features(x_start, "numerical")
    x_num   = diffusion.normalize(x_num)
    noise   = torch.randn_like(x_num)
    x_num_t = diffusion.gauss_q_sample(x_num, t, noise=noise)

    x_cat       = diffusion.extract_features(x_start, "categorical")
    log_x_cat   = index_to_log_onehot(x_cat.long(),
                                       diffusion.categorical_num_classes)
    log_x_cat_t = diffusion.cat_q_sample(log_x_start=log_x_cat, t=t)

    x_in      = torch.cat([x_num_t, log_x_cat_t], dim=1)
    model_out = diffusion.model(x_in, t, None)
    out_num   = diffusion.extract_modeloutput(model_out, "numerical")
    out_cat   = diffusion.extract_modeloutput(model_out, "categorical")

    # gaussian loss per sample (B,)
    g = F.mse_loss(out_num, noise, reduction="none")
    g = reduce(g, "b ... -> b (...)", "mean")
    g = g * utils.extract(diffusion.loss_weight, t, g.shape)
    loss_gauss = g.mean(dim=-1)

    # categorical loss per sample (B,) via cat_loss
    loss_multi = diffusion.cat_loss(
        out_cat, log_x_cat, log_x_cat_t, t, pt, None
    ) / len(diffusion.categorical_num_classes)

    return diffusion.loss_lambda * loss_multi + loss_gauss


# =============================================================================
# META-CLASSIFIER
# =============================================================================

class MetaClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev   = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
    cfg    = CONFIG
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
        proxy, X_syn_released, cfg
    ).mean()
    print(f"  proxy baseline (mean loss on Dsynth): {proxy_baseline:.4f}")

    proxy_losses = get_loss_vector(
        proxy, X_real, cfg
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