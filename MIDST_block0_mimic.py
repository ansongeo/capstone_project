"""
MIDST MIA -- Block 0: Train the Target and Release its Synthetic Data
=====================================================================
Supports eICU (N, 9, 272) and MIMIC-IV (N, 11, 72)

What this block does
--------------------
  1. Joins the designated train and test files into one set of real records
     and shuffles it. The first 2 * n_members records form the attacked pool;
     a random half of the pool (n_members) are the target's members, the
     other half are non-members (50/50).
  2. Trains the TARGET on the members with the TimeDiff recipe
     (etdiff_train.py defaults: 700k optimiser steps of 32 x 2, lr 8e-5,
     Adam(0.9, 0.99), EMA 0.995; hidden 68 for MIMIC-IV, 256 for eICU),
     normalised with the members' own min/max like TimeSeriesDataset.
  3. Releases 20,000 synthetic records in raw units, like etdiff_train.py.

Drawing our own split, instead of using the designated train file, gives a
balanced 50/50 evaluation and lets the training-set size be varied with
--n_members.

Outputs saved to output_dir/
  pool_idx.npy           -- indices of the attacked pool in the joined data
  y_member.npy           -- ground truth over the pool (Block 4 evaluation only)
  target.pt (+ .minmax.npz)
  released.npy           -- the target's synthetic release (20000, C, T)

Run next: MIDST_block1_mimic.py
"""

import os
import numpy as np

from midst_common import (DATASETS, TARGET, parse_cli, load_real_data,
                          member_split, fit_release)


CONFIG = {
    "dataset": "mimic",    # "eicu" | "mimic"
    **DATASETS,
    "device":  "cuda",
}


def main():
    cfg = parse_cli(CONFIG)
    out = cfg["output_dir"]
    if os.path.exists(os.path.join(out, "released.npy")):
        print(f"[Block 0] {out}/released.npy exists, skipping.")
        return

    X = load_real_data(cfg)
    n_members = cfg["n_members"] or len(X) // 2
    pool, y_member = member_split(len(X), n_members)
    np.save(os.path.join(out, "pool_idx.npy"), pool)
    np.save(os.path.join(out, "y_member.npy"), y_member)
    print(f"[Block 0] {len(X)} records; pool {len(pool)} = "
          f"{int(y_member.sum())} members + {int((1 - y_member).sum())} non-members")

    hidden = cfg[cfg["dataset"]]["target_hidden"]
    print(f"[Block 0] Training target (hidden {hidden}, "
          f"{TARGET['num_steps']} steps) ...")
    _, _, released = fit_release(cfg, X[pool][y_member == 1], TARGET["num_steps"],
                                 hidden, TARGET["seed"], "target",
                                 os.path.join(out, "target.pt"))
    np.save(os.path.join(out, "released.npy"), released)
    print(f"[Block 0 done]  released {released.shape} -> {out}/released.npy")
    print("  Run next: MIDST_block1_mimic.py")


if __name__ == "__main__":
    main()
