"""
PhysioNet/CinC Challenge 2012 -> the MIMIC-IV vitals layout, as a public stand-in
for testing the MIDST attack without credentialed data.

    python preprocess/physionet2012.py            # downloads sets A-C (~40 MB) if missing

Writes data/physionet2012/{TRAIN,TEST}-physionet2012_48h.pt in the layout of
mimic4_vitals_72h (N, 11, 48):
  [HR, HR_null, SBP, SBP_null, DBP, DBP_null, RR, RR_null, MAP, MAP_null, mortality]
hourly (first value in each hour), NaN where missing (null flag = 1), invasive
blood pressure with the non-invasive one as fallback. Stays with no heart rate
at all are dropped (11,816 of 12,000 remain). All stays go into TRAIN and TEST
is empty: Block 0 draws its own member split from the joined records.
"""

import glob
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import torch

URL = "https://physionet.org/files/challenge-2012/1.0.0/"
RAW = "data/physionet2012/raw"
OUT = "data/physionet2012"
VITALS = [("HR",), ("SysABP", "NISysABP"), ("DiasABP", "NIDiasABP"),
          ("RespRate",), ("MAP", "NIMAP")]
HOURS = 48


def download():
    os.makedirs(RAW, exist_ok=True)
    for s in "abc":
        for name in (f"set-{s}.tar.gz", f"Outcomes-{s}.txt"):
            path = os.path.join(RAW, name)
            if not os.path.exists(path):
                print(f"  downloading {name} ...")
                urllib.request.urlretrieve(URL + name, path)
        if not os.path.isdir(os.path.join(RAW, f"set-{s}")):
            with tarfile.open(os.path.join(RAW, f"set-{s}.tar.gz")) as tf:
                tf.extractall(RAW)


def stay(path, died):
    df = pd.read_csv(path)
    hh, mm = df.Time.str.split(":", expand=True).astype(int).T.values
    df["hour"] = hh
    df["min"] = hh * 60 + mm
    df = df.sort_values("min")
    x = np.full((11, HOURS), np.nan, np.float32)
    for i, names in enumerate(VITALS):
        v = np.full(HOURS, np.nan, np.float32)
        for name in names:                   # invasive first, then non-invasive
            s = df[(df.Parameter == name) & (df.Value > 0)].groupby("hour").Value.first()
            for h, val in s.items():
                if 0 <= h < HOURS and np.isnan(v[h]):
                    v[h] = val
        x[2 * i] = v
        x[2 * i + 1] = np.isnan(v).astype(np.float32)
    x[10] = died
    return x


def main():
    download()
    died = pd.concat([pd.read_csv(os.path.join(RAW, f"Outcomes-{s}.txt")) for s in "abc"]
                     ).set_index("RecordID")["In-hospital_death"]
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, "set-*", "*.txt"))):
        rid = int(os.path.basename(path)[:-4])
        rows.append(stay(path, float(died.loc[rid])))
    X = np.stack(rows)
    X = X[~np.isnan(X[:, 0]).all(1)]
    torch.save(torch.from_numpy(X), os.path.join(OUT, "TRAIN-physionet2012_48h.pt"))
    torch.save(torch.zeros((0,) + X.shape[1:]), os.path.join(OUT, "TEST-physionet2012_48h.pt"))
    print(f"{X.shape[0]} stays, mortality {X[:, 10, 0].mean():.3f} -> {OUT}/")


if __name__ == "__main__":
    main()
