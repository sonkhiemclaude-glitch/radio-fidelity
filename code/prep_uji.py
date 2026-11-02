"""Second corpus: UJIIndoorLoc (UCI 310) -- indoor WiFi at 2.4 GHz.

Chosen because every axis differs from the Antwerp corpus: indoors rather than
outdoors, 2.4 GHz rather than 868 MHz, three multi-storey buildings rather than
a city, and -- the point of including it -- *no transmitter coordinates at all*.
An indoor simulator therefore has to estimate the access-point positions before
it can model anything, which is exactly the situation the admission theory is
about, one rung lower down than the outdoor case.

Access-point positions are estimated by the standard power-weighted centroid of
the reference points that heard them, and their floor by the modal floor of the
same observations.  That estimate uses pooled real coordinates, including subsequent evaluation
positions. This original indoor arm is transductive calibration, not a
zero-label independent deployment evaluation.

Outputs `data/uji.npz`.
"""
from __future__ import annotations
import os, sys, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(HERE, "..", "raw", "uji", "UJIndoorLoc")
OUT = os.environ.get("REV_DATA_DIR", os.path.join(HERE, "..", "data"))

NOT_HEARD = 100          # the corpus writes 100 for "access point not detected"
FLOOR_H = 3.9            # storey height, metres (the campus buildings)
MIN_OBS = 20             # access points seen fewer times are dropped


def main():
    tr = pd.read_csv(os.path.join(RAW, "trainingData.csv"))
    va = pd.read_csv(os.path.join(RAW, "validationData.csv"))
    df = pd.concat([tr, va], ignore_index=True)
    waps = [c for c in df.columns if c.startswith("WAP")]
    R = df[waps].to_numpy(np.float64)
    heard = R != NOT_HEARD
    x = df["LONGITUDE"].to_numpy(float)
    y = df["LATITUDE"].to_numpy(float)
    fl = df["FLOOR"].to_numpy(int)
    bld = df["BUILDINGID"].to_numpy(int)
    phone = df["PHONEID"].to_numpy(int)
    user = df["USERID"].to_numpy(int)
    src = np.r_[np.zeros(len(tr), np.int8), np.ones(len(va), np.int8)]

    keep = heard.sum(0) >= MIN_OBS
    R = R[:, keep]; heard = heard[:, keep]
    waps = [w for w, k in zip(waps, keep) if k]
    print(f"{len(df)} fingerprints, {len(waps)} access points with >= {MIN_OBS} "
          f"observations ({keep.size - keep.sum()} dropped)")

    # --- access-point position estimate: power-weighted centroid -----------
    lin = np.where(heard, 10 ** (R / 10.0), 0.0)
    wsum = lin.sum(0)
    ap_x = (lin * x[:, None]).sum(0) / wsum
    ap_y = (lin * y[:, None]).sum(0) / wsum
    ap_f = np.zeros(len(waps))
    ap_b = np.zeros(len(waps), int)
    for j in range(len(waps)):
        m = heard[:, j]
        w = lin[m, j]
        ap_f[j] = np.average(fl[m], weights=w)
        ap_b[j] = np.bincount(bld[m], weights=w).argmax()
    z = fl * FLOOR_H
    ap_z = ap_f * FLOOR_H
    print(f"  AP centroid spread: x sd {ap_x.std():.0f} m, y sd {ap_y.std():.0f} m")
    print(f"  buildings {np.bincount(bld)}, floors {np.bincount(fl)}")
    print(f"  devices {len(np.unique(phone))}, users {len(np.unique(user))}")

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "uji.npz"),
                        R=R.astype(np.float32), heard=heard,
                        x=x, y=y, z=z, floor=fl, bld=bld,
                        phone=phone, user=user, src=src,
                        ap_x=ap_x, ap_y=ap_y, ap_z=ap_z, ap_b=ap_b,
                        ap_floor=ap_f, waps=np.array(waps),
                        floor_h=FLOOR_H, not_heard=NOT_HEARD)
    print(f"-> {os.path.join(OUT,'uji.npz')}")


if __name__ == "__main__":
    main()
