"""Experiment 7 -- independent replication indoors.

Every axis of the Antwerp study is changed: 2.4 GHz instead of 868 MHz, indoors
instead of outdoors, three multi-storey buildings instead of a city, a
five-rung ladder built from different physics, and -- most importantly -- a
simulator that has to *estimate* its own transmitter positions, because
UJIIndoorLoc publishes none.  If the U-shape is an artefact of the outdoor
corpus it should not survive any of that.

Ladder
    G0  free space from the estimated AP position
    G1  + indoor clutter offset              C (dB at 10 m)
    G2  + excess path-loss exponent          n - 2
    G3  + per-AP offsets                     delta_j (centred)
    G4  + loss per floor crossed             beta (dB/floor)
    G5  + inter-building loss                omega (dB)

With 382 access points the design matrix has 385 columns; at 4e5 real links and
2e6 simulated links it is never materialised.  Predictions are formed directly
from the four feature vectors, and the oracle fit accumulates its normal
equations in chunks.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import learners as L

F_HZ = 2.437e9          # WiFi channel 6
P_TX = 20.0             # dBm EIRP, a typical enterprise access point
SEED = 3108
N_REP = 12
GRID_M = 5.0            # simulated reference-point pitch inside each floor
SENS = -95.0            # client detection threshold, dBm
FLOOR_DBM = -105.0      # value written for "access point not detected"

D0_IN = 10.0                # reference distance for the indoor exponent rung

PRI = dict(
    # ITU-R P.1238 office at 2.4 GHz exceeds free space by about 10 dB at 10 m
    offset=dict(mu=10.0, tau=5.0),
    exponent=dict(mu=1.0, tau=0.5),       # n ~ 3.0 indoors, plausible 2.5-4.0
    ap=dict(mu=0.0, tau=6.0),             # unknown AP power/antenna/mounting
    floor=dict(mu=15.0, tau=6.0),         # 10-20 dB per floor at 2.4 GHz
    building=dict(mu=20.0, tau=8.0),      # exterior wall plus separation
)
LEVELS = ["G0 free space", "G1 +clutter offset", "G2 +exponent",
          "G3 +AP offsets", "G4 +floor loss", "G5 +building loss"]
N_LEVELS = len(LEVELS)


def free_space(d):
    d = np.maximum(np.asarray(d, float), 1.0)
    return P_TX - (20 * np.log10(d) + 20 * np.log10(F_HZ) - 147.55)


def level_cols(level, n_ap):
    c = []
    if level >= 1: c.append(0)
    if level >= 2: c.append(1)
    if level >= 3: c += list(range(2, 2 + n_ap))
    if level >= 4: c.append(2 + n_ap)
    if level >= 5: c.append(3 + n_ap)
    return np.array(c, int)


def predict(level, phi, dist, ap, nfloor, diffb, n_ap):
    """Simulated RSSI without ever forming the design matrix."""
    out = free_space(dist)
    if level >= 1:
        out = out - phi[0]
    if level >= 2:
        out = out + phi[1] * (-10 * np.log10(np.maximum(dist, 1.0) / D0_IN))
    if level >= 3:
        out = out + phi[2 + np.asarray(ap, int)] - phi[2: 2 + n_ap].mean()
    if level >= 4:
        out = out + phi[2 + n_ap] * (-nfloor)
    if level >= 5:
        out = out + phi[3 + n_ap] * (-diffb)
    return out


def design_chunk(dist, ap, nfloor, diffb, n_ap, sl):
    m = sl.stop - sl.start
    X = np.zeros((m, 2 + n_ap + 2))
    X[:, 0] = -1.0
    X[:, 1] = -10 * np.log10(np.maximum(dist[sl], 1.0) / D0_IN)
    X[:, 2: 2 + n_ap] = -1.0 / n_ap
    X[np.arange(m), 2 + np.asarray(ap[sl], int)] += 1.0
    X[:, 2 + n_ap] = -nfloor[sl]
    X[:, 3 + n_ap] = -diffb[sl]
    return X


def oracle_fit(dist, ap, nfloor, diffb, r0, n_ap, level, chunk=100_000):
    """Least squares restricted to one level's columns, accumulating X^T X and
    X^T r in chunks so the 385-column design is never materialised."""
    cols = level_cols(level, n_ap)
    phi = np.zeros(2 + n_ap + 2)
    if len(cols) == 0:
        return phi
    p = len(cols)
    XtX = np.zeros((p, p)); Xtr = np.zeros(p)
    n = len(dist)
    for a in range(0, n, chunk):
        sl = slice(a, min(a + chunk, n))
        X = design_chunk(dist, ap, nfloor, diffb, n_ap, sl)[:, cols]
        XtX += X.T @ X; Xtr += X.T @ r0[sl]
    sol, *_ = np.linalg.lstsq(XtX, -Xtr, rcond=None)
    phi[cols] = sol
    return phi


def main():
    t0 = time.time()
    z = np.load(os.path.join(D.DATA, "uji.npz"), allow_pickle=True)
    R, heard = z["R"].astype(np.float64), z["heard"]
    x, y, zz, fl, bld = z["x"], z["y"], z["z"], z["floor"], z["bld"]
    ap_x, ap_y, ap_z, ap_b = z["ap_x"], z["ap_y"], z["ap_z"], z["ap_b"]
    n_pt, n_ap = R.shape
    fh = float(z["floor_h"])

    i, j = np.nonzero(heard)
    rssi = R[i, j]
    dist = np.sqrt((x[i] - ap_x[j]) ** 2 + (y[i] - ap_y[j]) ** 2
                   + (zz[i] - ap_z[j]) ** 2)
    nfloor = np.abs(zz[i] - ap_z[j]) / fh
    diffb = (bld[i] != ap_b[j]).astype(float)
    r0 = free_space(dist) - rssi
    print(f"{n_pt} fingerprints, {n_ap} APs, {len(rssi)} heard links "
          f"(median distance {np.median(dist):.0f} m, "
          f"median floors crossed {np.median(nfloor):.1f})")
    print(f"  G0 residual rmse {np.sqrt((r0**2).mean()):.1f} dB")

    oracle = [oracle_fit(dist, j, nfloor, diffb, r0, n_ap, lv)
              for lv in range(N_LEVELS)]
    po = oracle[N_LEVELS - 1]
    print(f"oracle: C = {po[0]:.1f} dB at {D0_IN:.0f} m, n = {2+po[1]:.2f}, "
          f"AP offset sd {po[2:2+n_ap].std():.1f} dB, floor loss "
          f"{po[2+n_ap]:.1f} dB, building loss {po[3+n_ap]:.1f} dB "
          f"({time.time()-t0:.0f}s)")

    def draw(mode, level, gen):
        p = np.zeros(2 + n_ap + 2)
        if mode == "oracle":
            return oracle[level].copy()
        rand = (mode == "randomised")
        pick = (lambda m, t: m if rand else gen.normal(m, t))
        p[0] = pick(PRI["offset"]["mu"], PRI["offset"]["tau"])
        p[1] = pick(PRI["exponent"]["mu"], PRI["exponent"]["tau"])
        p[2:2 + n_ap] = (PRI["ap"]["mu"] if rand else
                         gen.normal(PRI["ap"]["mu"], PRI["ap"]["tau"], n_ap))
        p[2 + n_ap] = pick(PRI["floor"]["mu"], PRI["floor"]["tau"])
        p[3 + n_ap] = pick(PRI["building"]["mu"], PRI["building"]["tau"])
        return p

    # ---- simulated reference grid: a lattice inside each (building, floor) --
    from scipy.spatial import cKDTree
    gx, gy, gz, gb, gf = [], [], [], [], []
    for b in np.unique(bld):
        for f in np.unique(fl[bld == b]):
            m = (bld == b) & (fl == f)
            if m.sum() < 30:
                continue
            xs = np.arange(x[m].min(), x[m].max() + GRID_M, GRID_M)
            ys = np.arange(y[m].min(), y[m].max() + GRID_M, GRID_M)
            X, Y = np.meshgrid(xs, ys)
            P = np.c_[X.ravel(), Y.ravel()]
            dd, _ = cKDTree(np.c_[x[m], y[m]]).query(P)
            P = P[dd <= 6.0]
            gx.append(P[:, 0]); gy.append(P[:, 1])
            gz.append(np.full(len(P), f * fh)); gb.append(np.full(len(P), b))
            gf.append(np.full(len(P), f))
    gx = np.concatenate(gx); gy = np.concatenate(gy)
    gz = np.concatenate(gz); gb = np.concatenate(gb); gf = np.concatenate(gf)
    n_ref = len(gx)
    print(f"simulated reference grid: {n_ref} points at {GRID_M:.0f} m pitch "
          f"({n_ref*n_ap/1e6:.1f} M simulated links)")

    gi = np.tile(np.arange(n_ap), n_ref)
    gd = np.sqrt((np.repeat(gx, n_ap) - ap_x[gi]) ** 2
                 + (np.repeat(gy, n_ap) - ap_y[gi]) ** 2
                 + (np.repeat(gz, n_ap) - ap_z[gi]) ** 2)
    gnf = np.abs(np.repeat(gz, n_ap) - ap_z[gi]) / fh
    gdb = (np.repeat(gb, n_ap) != ap_b[gi]).astype(np.float32)

    q_fp = np.where(heard, R, FLOOR_DBM).astype(np.float32)
    q_xy = np.c_[x, y]

    rows = []
    for mode in ("oracle", "guess", "randomised"):
        reps = N_REP if mode == "guess" else 1
        for level in range(N_LEVELS):
            mse, e50, e90, facc, bacc = [], [], [], [], []
            for r in range(reps):
                gen = np.random.default_rng(SEED + 811 * r + 7 * level)
                p = draw(mode, level, gen)
                pred = predict(level, p, dist, j, nfloor, diffb, n_ap)
                mse.append(float(np.mean((pred - rssi) ** 2)))
                gp = predict(level, p, gd, gi, gnf, gdb, n_ap)
                fp = gp.reshape(n_ref, n_ap).astype(np.float32)
                fp[fp < SENS] = FLOOR_DBM
                live = np.flatnonzero((fp > FLOOR_DBM).any(1))
                if len(live) < 50:
                    e50.append(np.nan); e90.append(np.nan)
                    facc.append(np.nan); bacc.append(np.nan); continue
                I, W = L.knn_indices(fp[live], q_fp, k=5)
                px = (W * gx[live][I]).sum(1); py = (W * gy[live][I]).sum(1)
                err = np.hypot(px - x, py - y)
                pf = np.round((W * gf[live][I]).sum(1))
                pb = np.round((W * gb[live][I]).sum(1))
                e50.append(float(np.median(err)))
                e90.append(float(np.percentile(err, 90)))
                facc.append(float(np.mean(pf == fl)))
                bacc.append(float(np.mean(pb == bld)))
            rows.append(dict(mode=mode, level=level, level_name=LEVELS[level],
                             reps=reps, rssi_mse=float(np.mean(mse)),
                             rssi_rmse=float(np.sqrt(np.mean(mse))),
                             err50=float(np.nanmean(e50)),
                             err50_sd=float(np.nanstd(e50)),
                             err90=float(np.nanmean(e90)),
                             floor_acc=float(np.nanmean(facc)),
                             bld_acc=float(np.nanmean(bacc)),
                             err50_reps=e50))
            print(f"  {mode:10s} G{level} rmse {np.sqrt(np.mean(mse)):6.2f} dB  "
                  f"median err {np.nanmean(e50):6.1f} m  floor "
                  f"{np.nanmean(facc)*100:5.1f}%  bldg {np.nanmean(bacc)*100:5.1f}%"
                  f"   ({time.time()-t0:.0f}s)", flush=True)

    # real-data ceiling, split by device so the shift is a real one
    ph = z["phone"]
    uniq = np.unique(ph)
    rng = np.random.default_rng(SEED)
    hold = set(rng.choice(uniq, size=max(1, len(uniq) // 3), replace=False).tolist())
    te = np.array([p in hold for p in ph])
    pxy = L.knn_localize(q_fp[~te], q_xy[~te], q_fp[te], k=5)
    cer = L.positioning_errors(pxy, q_xy[te])
    ceiling = dict(err50=float(np.median(cer)), err90=float(np.percentile(cer, 90)),
                   n_test=int(te.sum()), n_held_devices=len(hold))
    print(f"real-data ceiling (held-out devices): median {ceiling['err50']:.1f} m")

    out = dict(rows=rows, ceiling=ceiling, n_ref=int(n_ref), n_ap=int(n_ap),
               n_pt=int(n_pt), n_links=int(len(rssi)), levels=LEVELS,
               grid_m=GRID_M, n_rep=N_REP,
               g0_rmse=float(np.sqrt((r0 ** 2).mean())), d0=D0_IN,
               oracle=dict(offset=float(po[0]), exponent=float(2 + po[1]),
                           ap_sd=float(po[2:2 + n_ap].std()),
                           floor=float(po[2 + n_ap]),
                           building=float(po[3 + n_ap])),
               priors={k: dict(v) for k, v in PRI.items()})
    with open(os.path.join(D.RESULTS, "exp7_indoor.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> exp7_indoor.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
