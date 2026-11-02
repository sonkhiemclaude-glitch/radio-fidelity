"""Experiment 6 -- does a learned representation change the answer?

The theory is stated for a rich learner trained on the simulator's output, and
the localization experiment uses a nearest-neighbour readout.  Neither is what
the field is currently building.  The current recipe is: pretrain an encoder on
a large simulated corpus, freeze it, and read out a downstream task from a
small amount of real labelled data.

This experiment runs that recipe at every rung of the ladder.  The question is
not whether pretraining helps -- it does -- but whether *which simulator you
pretrained on* still matters once a learned representation and a little real
data are in the loop, and if so, up to how much real data.

There is a confound to control, and it is not incidental to this paper.  A more
pessimistic simulator predicts that fewer reference points are heard at all, so
the corpus it supplies for pretraining is *smaller*: the declared priors leave
11,282 usable reference points at the base of the ladder and 1,475 at the top.
Corpus size is a real consequence of the rung -- a modeller who chooses that
simulator really does get that much data -- but it is not the same thing as the
rung's physics, and a ranking that conflates them says nothing.  Both arms are
therefore run: `free`, in which each rung keeps whatever corpus it produces,
and `equal`, in which every rung is subsampled to the smallest corpus any rung
produces, so that only the physics differs.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import simulator as S
import priors as PR
import learners as L
from exp2_ladder import build_query, fingerprints_for, prior_draw, SF_MAIN, SEED

N_DRAW = 3                       # pretraining runs per level
N_PROBE = [0, 25, 100, 400, 1600, 6400]
EPOCHS = 60


def main():
    t0 = time.time()
    d = D.load(); n_gw = d["n_gw"]
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    rx, ry = geom.road_x, geom.road_y
    n_ref = len(rx)
    gi = np.tile(np.arange(n_gw), n_ref)
    px = np.repeat(rx, n_gw); py = np.repeat(ry, n_gw)
    gdist, gncr, gdep = D.link_geometry(px, py, d["gw_x"], d["gw_y"], gi, geom,
                                        cache=os.path.join(D.DATA, "links_grid.npz"))
    q = build_query(d, n=25_000)
    q_fp = fingerprints_for(q, d, n_gw)
    q_xy = np.c_[d["x"][q], d["y"][q]]
    fold = D.block_folds(d["x"][q], d["y"][q], n_folds=4, seed=SEED)
    thr = S.SENS[SF_MAIN]
    zf = np.load(os.path.join(D.DATA, "tobit_fits.npz"))
    field = S.ShadowField(rx.min() - 500, ry.min() - 500, rx.max() + 500,
                          ry.max() + 500, 25.0, 1.0, PR.SHADOW_ELL, n_gw,
                          seed=SEED)
    unit_shadow = field(px, py, gi)

    def sim_db(level, gen, cap=None):
        phi, sig = prior_draw("uninformed", "guess", level, gen, n_gw,
                              zf["phi"][level], float(zf["sigma"][level]))
        gp = S.rssi(level, phi, gdist, gi, gncr, gdep, n_gw, sig * unit_shadow)
        fp = gp.reshape(n_ref, n_gw).astype(np.float32)
        fp[fp < thr] = L.FLOOR
        live = (fp > L.FLOOR).any(1)
        X, Y = fp[live], np.c_[rx, ry][live]
        if cap is not None and len(X) > cap:
            take = np.sort(gen.choice(len(X), size=cap, replace=False))
            X, Y = X[take], Y[take]
        return X, Y

    # how large a pretraining corpus each rung supplies, before any probing
    n_sim_free = []
    for level in range(S.N_LEVELS):
        v = []
        for r in range(N_DRAW):
            gen = np.random.default_rng(SEED + 5081 * r)
            v.append(len(sim_db(level, gen)[0]))
        n_sim_free.append(int(min(v)))
    cap_equal = int(min(n_sim_free))
    print(f"  pretraining corpus per rung: {n_sim_free}; "
          f"equalised arm caps every rung at {cap_equal}")

    def run_arm(cap):
      rows = []
      for level in range(S.N_LEVELS):
        per_probe = {n: [] for n in N_PROBE}
        for r in range(N_DRAW):
            gen = np.random.default_rng(SEED + 5081 * r)
            Xs, Ys = sim_db(level, gen, cap)
            if len(Xs) < 200:
                continue
            model = L.pretrain_encoder(Xs, Ys, dim=32, hidden=128,
                                       epochs=EPOCHS, seed=SEED + r)
            Zs = L.embed(model, Xs)
            Zq = L.embed(model, q_fp)
            # zero-shot readout: probe fitted on the simulated data only
            for n_probe in N_PROBE:
                errs = []
                for f in range(4):
                    tr = np.flatnonzero(fold != f)
                    te = np.flatnonzero(fold == f)
                    if n_probe == 0:
                        pred = L.linear_probe(Zs, Ys, Zq[te])
                    else:
                        g2 = np.random.default_rng(SEED + 17 * f + r)
                        take = g2.choice(tr, size=min(n_probe, len(tr)),
                                         replace=False)
                        pred = L.linear_probe(Zq[take], q_xy[take], Zq[te])
                    errs.append(np.hypot(pred[:, 0] - q_xy[te, 0],
                                         pred[:, 1] - q_xy[te, 1]))
                per_probe[n_probe].append(float(np.median(np.concatenate(errs))))
        rows.append(dict(level=level, name=S.LEVELS[level],
                         n_sim=int(len(Xs)),
                         err={str(n): float(np.mean(v)) for n, v in per_probe.items()},
                         err_sd={str(n): float(np.std(v)) for n, v in per_probe.items()}))
        print(f"  L{level} " + "  ".join(
            f"n={n}: {np.mean(per_probe[n]):6.0f} m" for n in N_PROBE)
            + f"   ({time.time()-t0:.0f}s)", flush=True)
      return rows

    print("  --- arm: free (each rung keeps the corpus it produces) ---")
    rows = run_arm(None)
    print(f"  --- arm: equal (every rung capped at {cap_equal}) ---")
    rows_equal = run_arm(cap_equal)

    # real-only baseline at each probe budget, no pretraining at all
    real_only = {}
    for n_probe in N_PROBE:
        if n_probe == 0:
            continue
        errs = []
        for f in range(4):
            tr = np.flatnonzero(fold != f); te = np.flatnonzero(fold == f)
            g2 = np.random.default_rng(SEED + 91 * f)
            take = g2.choice(tr, size=min(n_probe, len(tr)), replace=False)
            pred = L.knn_localize(q_fp[take], q_xy[take], q_fp[te],
                                  k=min(5, len(take)))
            errs.append(L.positioning_errors(pred, q_xy[te]))
        real_only[str(n_probe)] = float(np.median(np.concatenate(errs)))
        print(f"  real-only n={n_probe}: {real_only[str(n_probe)]:.0f} m")

    def summarise(rr):
        best = {str(n): int(np.argmin([r["err"][str(n)] for r in rr]))
                for n in N_PROBE}
        spread = {str(n): float(max(r["err"][str(n)] for r in rr)
                                / min(r["err"][str(n)] for r in rr))
                  for n in N_PROBE}
        return best, spread

    best, spread = summarise(rows)
    best_eq, spread_eq = summarise(rows_equal)
    from scipy import stats
    order_agree = float(stats.spearmanr(
        [r["err"]["0"] for r in rows],
        [r["err"]["0"] for r in rows_equal]).statistic)
    print("\nargmin level by probe budget:      ", best)
    print("argmin level, corpus equalised:    ", best_eq)
    print("worst/best ratio across levels:    ", {k: round(v, 2) for k, v in spread.items()})
    print("worst/best ratio, corpus equalised:", {k: round(v, 2) for k, v in spread_eq.items()})
    print(f"zero-shot rung ordering, free vs equalised: Spearman {order_agree:.2f}")

    out = dict(rows=rows, rows_equal=rows_equal, real_only=real_only,
               best_level=best, level_spread=spread,
               best_level_equal=best_eq, level_spread_equal=spread_eq,
               n_sim_free=n_sim_free, cap_equal=cap_equal,
               order_agree=order_agree,
               n_probe=N_PROBE, n_draw=N_DRAW,
               epochs=EPOCHS, n_query=int(len(q)))
    with open(os.path.join(D.RESULTS, "exp6_foundation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> exp6_foundation.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
