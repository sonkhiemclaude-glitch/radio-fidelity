"""Experiment 9 -- how much of the fit is the subsample?

The ladder is fitted on a random subsample of messages, because the full corpus
would need a design matrix of a gigabyte and the subsample is by message and
therefore unbiased.  That is an argument; this is a measurement.  The
coefficients are re-fitted on geographically blocked bootstrap resamples of the
candidate corpus, warm-started from the full-sample solution, and the spread of
each is reported.

Blocks rather than links: RF measurements are autocorrelated at the metre scale,
so an i.i.d. bootstrap would understate the uncertainty of exactly the
coefficients that depend on geometry.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import simulator as S
import tobit as TB

B = 24
N_BLOCK = 120
SEED = 3108


def main():
    t0 = time.time()
    d = D.load(); n_gw = d["n_gw"]
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    C = D.candidates(d, geom, cache=os.path.join(D.DATA, "links_cand.npz"))
    Psi = S.features(C["dist"], C["gw"], C["n_cross"], C["depth"], n_gw)
    base = S.free_space(C["dist"])
    z = np.load(os.path.join(D.DATA, "tobit_fits.npz"))
    lv = S.N_LEVELS - 1
    cols = S.level_columns(lv, n_gw)
    phi_full = z["phi"][lv]; sig_full = float(z["sigma"][lv])

    blocks = D.block_folds(C["x"], C["y_pos"], n_folds=N_BLOCK, seed=SEED)
    members = [np.flatnonzero(blocks == b) for b in range(N_BLOCK)]
    rng = np.random.default_rng(SEED)
    rows = []
    for b in range(B):
        pick = rng.integers(0, N_BLOCK, N_BLOCK)
        idx = np.concatenate([members[p] for p in pick])
        A = np.ascontiguousarray(Psi[idx][:, cols])
        phi, sig, f = TB.fit_best(A, base[idx], C["y"][idx], C["got"][idx],
                                  C["thr"][idx], phi0=phi_full[cols],
                                  sigma0=sig_full)
        full = np.zeros(S.n_features(n_gw)); full[cols] = phi
        rows.append(dict(C=float(full[0]), n=float(2 + full[1]),
                         gw_sd=float(full[2:2 + n_gw].std()),
                         kappa=float(full[2 + n_gw]),
                         gamma=float(full[3 + n_gw]), sigma=float(sig)))
        if (b + 1) % 6 == 0:
            print(f"  {b+1}/{B} resamples ({time.time()-t0:.0f}s)", flush=True)

    keys = ("C", "n", "gw_sd", "kappa", "gamma", "sigma")
    summ = {}
    print(f"\n{'coefficient':>10} {'full sample':>12} {'bootstrap mean':>15} "
          f"{'sd':>9} {'95% CI':>22}")
    ref = dict(C=float(phi_full[0]), n=float(2 + phi_full[1]),
               gw_sd=float(phi_full[2:2 + n_gw].std()),
               kappa=float(phi_full[2 + n_gw]), gamma=float(phi_full[3 + n_gw]),
               sigma=sig_full)
    for k in keys:
        v = np.array([r[k] for r in rows])
        lo, hi = np.percentile(v, [2.5, 97.5])
        summ[k] = dict(full=ref[k], mean=float(v.mean()), sd=float(v.std()),
                       lo=float(lo), hi=float(hi))
        print(f"{k:>10} {ref[k]:12.4f} {v.mean():15.4f} {v.std():9.4f} "
              f"[{lo:9.4f}, {hi:9.4f}]")

    out = dict(B=B, n_block=N_BLOCK, level=lv, rows=rows, summary=summ,
               n_cand=int(len(C["dist"])), n_cand_msg=int(len(C["msgs"])),
               n_msg_total=int(d["n_msg"]))
    with open(os.path.join(D.RESULTS, "exp9_stability.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> exp9_stability.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
