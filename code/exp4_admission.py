"""Experiment 4 -- the admission test.

The criterion of Corollary 1 needs an oracle coefficient, which a modeller does
not have.  What a modeller can have is a handful of real measurements.  Theorem 5
turns those into a decision: walk the ladder from the bottom, admit a rung only
while a one-sided empirical-Bernstein bound on the paired per-link loss
difference is negative, and stop at the first failure.  Because the order is
fixed in advance the walk needs no multiplicity correction, and the family-wise
probability of admitting a harmful rung is at most alpha.

The test is cheap in the way that matters: it evaluates simulators against
measurements and never trains anything.  This experiment measures its error rate
and its power against the number of real measurements, and compares the level it
selects with the two policies the field actually uses -- "take the most detailed
simulator available", and "take the empirical argmin" with no error control.

A "measurement" here is a candidate link: a (message, gateway) pair, which is
either a received power or the information that nothing was received.  Both are
things a real calibration campaign collects.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import simulator as S
import theory as T
import tobit as TB
import priors as PR
from exp2_ladder import prior_draw, SEED

NS = [50, 100, 200, 500, 1000, 2000, 5000, 20000]
B = 400
ALPHA = 0.05
LOSS_RANGE = 15.0     # per-link losses are clipped to [0, LOSS_RANGE]

# Clipping is not cosmetic: a finite-sample bound needs a bounded loss, and at
# the top of the ladder a single link whose predicted power is wrong by a
# hundred decibels contributes an unbounded log-loss.  The clip is applied
# identically at every level, so it cannot favour one, and Experiment 8 varies
# it.


def main():
    t0 = time.time()
    d = D.load(); n_gw = d["n_gw"]
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    C = D.candidates(d, geom, cache=os.path.join(D.DATA, "links_cand.npz"))
    Psi = S.features(C["dist"], C["gw"], C["n_cross"], C["depth"], n_gw)
    base = S.free_space(C["dist"])
    z = np.load(os.path.join(D.DATA, "tobit_fits.npz"))

    gen = np.random.default_rng(SEED)
    phi, sig = prior_draw("uninformed", "guess", 0, gen, n_gw,
                          z["phi"][0], float(z["sigma"][0]))
    loss = []
    for lv in range(S.N_LEVELS):
        cols = S.level_columns(lv, n_gw)
        loss.append(np.clip(TB.link_loss(phi[cols], sig, Psi[:, cols], base,
                                         C["y"], C["got"], C["thr"]),
                            0.0, LOSS_RANGE))
    truth = np.array([l.mean() for l in loss])
    best = int(np.argmin(truth))
    # a rung is harmful if admitting it raises the risk above the level below;
    # a policy is harmful if the level it selects required admitting one
    harmful_rung = np.array([truth[j] > truth[j - 1] for j in range(1, len(truth))])
    harmful_level = np.array([bool(harmful_rung[:k].any())
                              for k in range(len(truth))])
    print("population risk by level (per-link censored log-loss, clipped):")
    for lv in range(S.N_LEVELS):
        print(f"  L{lv} {S.LEVELS[lv]:24s} {truth[lv]:8.5f}"
              + ("   <- argmin" if lv == best else "")
              + ("   [harmful]" if harmful_level[lv] else ""))

    # geographic blocks, so a resample of n links is a realistic campaign
    blocks = D.block_folds(C["x"], C["y_pos"], n_folds=200, seed=SEED)
    rng = np.random.default_rng(SEED)
    rows = []
    for n in NS:
        sel_test, sel_argmin, harmful, exact = [], [], 0, 0
        for b in range(B):
            order = rng.permutation(200)
            idx = []
            tot = 0
            for blk in order:
                p = np.flatnonzero(blocks == blk)
                idx.append(p); tot += len(p)
                if tot >= n:
                    break
            idx = np.concatenate(idx)[:n]
            diffs = [loss[k + 1][idx] - loss[k][idx]
                     for k in range(S.N_LEVELS - 1)]
            k_test = T.fixed_sequence_test(diffs, ALPHA, rng_range=2 * LOSS_RANGE)
            emp = np.array([l[idx].mean() for l in loss])
            k_arg = int(np.argmin(emp))
            sel_test.append(k_test); sel_argmin.append(k_arg)
            harmful += int(harmful_level[k_test])
            exact += int(k_test == best)
        rows.append(dict(
            n=n,
            test_mean_level=float(np.mean(sel_test)),
            test_exact=float(exact / B),
            test_harmful=float(harmful / B),
            test_risk=float(np.mean([truth[k] for k in sel_test])),
            argmin_mean_level=float(np.mean(sel_argmin)),
            argmin_exact=float(np.mean([k == best for k in sel_argmin])),
            argmin_harmful=float(np.mean([harmful_level[k] for k in sel_argmin])),
            argmin_risk=float(np.mean([truth[k] for k in sel_argmin])),
            top_risk=float(truth[-1]), base_risk=float(truth[0]),
            oracle_risk=float(truth[best])))
        r = rows[-1]
        print(f"  n={n:6d}  test: level {r['test_mean_level']:.2f} exact "
              f"{r['test_exact']*100:5.1f}% harmful {r['test_harmful']*100:4.1f}% "
              f"risk {r['test_risk']:.5f} | argmin: level "
              f"{r['argmin_mean_level']:.2f} exact {r['argmin_exact']*100:5.1f}% "
              f"harmful {r['argmin_harmful']*100:4.1f}% risk {r['argmin_risk']:.5f}",
              flush=True)

    out = dict(runtime_s=float(time.time() - t0),
               n_campaigns=int(B * len(NS)),
               levels=list(S.LEVELS), truth=truth.tolist(), best_level=best,
               harmful_level=harmful_level.tolist(),
               rows=rows, alpha=ALPHA, B=B, loss_range=LOSS_RANGE,
               sigma=float(sig))
    with open(os.path.join(D.RESULTS, "exp4_admission.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> exp4_admission.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
