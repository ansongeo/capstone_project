# MIDST membership inference attack on TimeDiff

Shadow-model membership inference attack on TimeDiff/ETDiff
(MixedDiffusion). Scripts: `MIDST_block{0..4}_mimic.py`, shared code in
`midst_common.py`, fast training/loss in `midst_fast.py`.

## Run

```bash
./run_midst.sh                                    # MIMIC-IV, 64 shadows, GPUs 0 and 1
DATASET=eicu ./run_midst.sh
python preprocess/physionet2012.py                # public stand-in (no credentials)
DATASET=physionet K=32 GPUS="0" ./run_midst.sh
```

Each block also runs on its own (`--dataset`, `--output_dir`,
`--n_members`; Blocks 1 and 2 take `--shadows 3` or `--shadows 1-16`).
Every step skips outputs that already exist, so a rerun resumes.

## Pipeline

| Block | What it does | Settings |
| --- | --- | --- |
| 0 | Joins all records, shuffles, keeps a pool of 2 x `n_members`; half are members. Trains the **target** on the members; releases 20,000 synthetic records. | TimeDiff recipe: 700k optimiser steps of 32 x 2, lr 8e-5, Adam(0.9, 0.99), EMA 0.995; hidden 68 (MIMIC-IV) / 256 (eICU) |
| 1 | For each k: random 50/50 split of the pool; trains **base shadow** k on its half like the target; releases 20,000 records. | Same as the target |
| 2 | Trains **synth-shadow** k on shadow k's release; scores every pool record. | Overfitting probe: hidden 256, 1.4M steps |
| 3 | Fits the **meta-classifier** on the synth-shadows' scores (labels = each shadow's split), at K = 4, 8, 16, 32, 64. | LightGBM |
| 4 | Trains the **proxy** on the target's release like a synth-shadow; scores the pool; applies the classifiers; writes `result.json`. | Same as Block 2 |

Scores per record: at 16 diffusion steps t, the vital-sign and the
categorical (missing-value flags + mortality) losses under 600 frozen noise
draws, summarised (mean, std, min, max, 10th/50th/90th percentiles) and
calibrated per record against the record's values in the other shadows.

**Black-box** (the MIDST threat model): synth-shadow scores train the
classifier, proxy scores are attacked. **White-box** (reference only, needs
the target's weights): base-shadow scores train the classifier, the target's
own scores are attacked.

## Outputs (`block1_<dataset>/`)

- `result.json`: AUC and TPR at FPR 0.1%-10% per view and K, plus each
  single mean-loss feature's AUC.
- `midst_roc_{blackbox,whitebox}.png`, `midst_scores_*.png`,
  `inference_scores_*.npy`.
- `features/`: loss features of every model (`target`, `sh*`, `ss*`,
  `proxy`) and the shadow labels (`lab*`), about 30 MB per model for
  MIMIC-IV.

## Cost

With `midst_fast.py` a 700k-step target or shadow takes ~25 min and a
1.4M-step probe ~95 min on one GPU (RTX 6000 Ada, MIMIC-IV shape); several
jobs share a GPU well. 64 shadows: roughly 20 h on two GPUs.

## Results on the PhysioNet stand-in

5,908 members vs 5,908 non-members, 32 shadows: black-box AUC 0.521,
white-box AUC 0.628.
