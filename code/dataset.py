"""Shared loading, link-feature construction and caching.

Two link sets are built once and reused by every experiment:

`real`   the 3.5e5 (message, gateway) pairs that were actually received, with
         their measured RSSI -- the evaluation set, and the only place real
         labels ever appear;
`grid`   the (road reference point, gateway) pairs, with no measurement
         attached -- the set the simulator populates to build a training
         fingerprint database.

Geographic blocking.  RF measurements are strongly autocorrelated at the metre
scale, so a random split puts near-duplicate points on both sides and flatters
every method.  All splits here are by 500 m spatial blocks, assigned to folds
by a hash of the block index, so a test point is never within a block of a
training point.
"""
from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.environ.get("PHYSICS_DATA_DIR", os.path.join(HERE, "..", "data")))
RESULTS = os.path.abspath(os.environ.get("PHYSICS_RESULTS_DIR", os.path.join(HERE, "..", "results")))
FIGS = os.path.abspath(os.environ.get("PHYSICS_FIGURES_DIR", os.path.join(HERE, "..", "figures")))
BLOCK_M = 500.0


def load(max_gw_km: float = 15.0):
    """Real corpus plus geometry, with far-outlier gateways removed.

    One gateway in the operator's registry sits 45 km from the campaign area
    yet reports a median RSSI of -109 dBm, which is inconsistent with any link
    budget by more than 30 dB; its registered coordinates cannot be right, and
    it is dropped rather than modelled.
    """
    d = np.load(os.path.join(DATA, "lorawan.npz"), allow_pickle=True)
    gx, gy = d["gw_x"], d["gw_y"]
    keep = np.hypot(gx, gy) <= max_gw_km * 1000
    remap = -np.ones(len(gx), np.int64)
    remap[keep] = np.arange(keep.sum())
    om, og = d["obs_msg"], d["obs_gw"].astype(np.int64)
    ok = keep[og]
    out = {
        "x": d["x"], "y": d["y"], "t": d["t"], "dev": d["dev"], "sf": d["sf"],
        "gw_x": gx[keep], "gw_y": gy[keep],
        "gw_id": d["gw_id"][keep],
        "obs_msg": om[ok], "obs_gw": remap[og[ok]],
        "obs_rssi": d["obs_rssi"][ok].astype(np.float64),
        "obs_esp": d["obs_esp"][ok].astype(np.float64),
        "n_gw": int(keep.sum()),
        "n_msg": len(d["x"]),
        "dropped_gw": int((~keep).sum()),
    }
    return out


def link_geometry(px, py, gx, gy, gw_idx, geom, cache: str | None = None):
    """Distance plus the two line-of-sight integrals for a set of links."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return z["d"], z["n_cross"], z["depth"]
    x1 = np.asarray(px, float); y1 = np.asarray(py, float)
    x2 = gx[gw_idx]; y2 = gy[gw_idx]
    d = np.hypot(x2 - x1, y2 - y1)
    nc, dep = geom.los(x1, y1, x2, y2)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, d=d, n_cross=nc, depth=dep)
    return d, nc, dep


def block_id(x, y, block=BLOCK_M):
    return (np.floor(x / block).astype(np.int64) * 100_003
            + np.floor(y / block).astype(np.int64))


def block_folds(x, y, n_folds=5, seed=7, block=BLOCK_M):
    """Assign every point to one of `n_folds` geographic folds."""
    b = block_id(x, y, block)
    uniq = np.unique(b)
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, n_folds, size=len(uniq))
    m = dict(zip(uniq.tolist(), lab.tolist()))
    return np.array([m[v] for v in b.tolist()], np.int64)


N_MSG_CAND = 10_000        # messages sampled for the candidate-link corpus


def candidates(d, geom, n_msg=N_MSG_CAND, seed=3108, cache=None):
    """Every (message, gateway) pair for a random subsample of messages, with
    the measured RSSI where the gateway heard it and a censoring threshold
    where it did not.

    This is the corpus the propagation model is *fitted* on.  Using only the
    received links -- which is what the published record invites -- conditions
    on a tail event and biases every coefficient; see `tobit`.

    Subsampling is by message, so the candidate-link set is an unbiased sample
    of the population of links the network could have carried.  All 130k
    messages would give 4e6 links and a 1 GB design matrix for no statistical
    gain.
    """
    import simulator as S
    n_gw = d["n_gw"]
    rng = np.random.default_rng(seed)
    msgs = np.sort(rng.choice(d["n_msg"], size=min(n_msg, d["n_msg"]),
                              replace=False))
    cm = np.repeat(msgs, n_gw)
    cg = np.tile(np.arange(n_gw), len(msgs))
    dist, ncr, dep = link_geometry(d["x"][cm], d["y"][cm], d["gw_x"], d["gw_y"],
                                   cg, geom, cache=cache)
    key_obs = d["obs_msg"].astype(np.int64) * n_gw + d["obs_gw"]
    key_c = cm.astype(np.int64) * n_gw + cg
    order = np.argsort(key_obs)
    ko = key_obs[order]
    pos = np.clip(np.searchsorted(ko, key_c), 0, len(ko) - 1)
    got = ko[pos] == key_c
    y = np.full(len(cg), np.nan)
    y[got] = d["obs_rssi"][order][pos[got]]
    thr = np.array([S.SENS[int(s)] for s in d["sf"][cm]])
    return dict(msg=cm, gw=cg, dist=dist, n_cross=ncr, depth=dep,
                y=y, got=got, thr=thr, msgs=msgs,
                x=d["x"][cm], y_pos=d["y"][cm], sf=d["sf"][cm])
