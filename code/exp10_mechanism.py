"""Experiment 10 -- is the downstream advantage of the top rungs fidelity or
compensation?

Section 6.1 finds that the level minimising the link risk is not the level
minimising positioning error, and argues that the top rungs win downstream for
the wrong reason: their coefficients are grossly too large, the simulated
database becomes very sparse, and real fingerprints in this corpus are sparse
too, so a nearest-neighbour search over floor-filled vectors is matched on
*support* while being wrong about *levels*.

That is a mechanism claim and it deserves a mechanism test rather than an
anecdote.  Two things are measured here.

1.  A continuous sweep of the geometric priors, from the declared literature
    values down to nothing.  If the top rungs won on fidelity, shrinking their
    coefficients towards the (small) truth would help them; if they won on
    support matching, there is an intermediate scale that matches the real
    sparsity best and the positioning argmin moves away from the top of the
    ladder once the simulated database stops being sparse enough.

2.  Across every (scale, level) cell of that sweep, the relationship between
    positioning error and the *sparsity mismatch* -- how far the simulated
    database's mean number of live gateways per reference point is from the
    real corpus's mean number per message.  If support drives the metric, the
    mismatch should predict the error across cells that differ in every other
    respect.
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

N_REP = 4
SCALES = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0)


def spearman(a, b):
    from scipy import stats
    r = stats.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def main():
    t0 = time.time()
    d = D.load(); n_gw = d["n_gw"]
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    zf = np.load(os.path.join(D.DATA, "tobit_fits.npz"))
    rx, ry = geom.road_x, geom.road_y
    n_ref = len(rx)
    gi = np.tile(np.arange(n_gw), n_ref)
    px = np.repeat(rx, n_gw); py = np.repeat(ry, n_gw)
    gdist, gncr, gdep = D.link_geometry(px, py, d["gw_x"], d["gw_y"], gi, geom,
                                        cache=os.path.join(D.DATA, "links_grid.npz"))
    ref_xy = np.c_[rx, ry]
    field = S.ShadowField(rx.min() - 500, ry.min() - 500, rx.max() + 500,
                          ry.max() + 500, 25.0, 1.0, PR.SHADOW_ELL, n_gw, seed=SEED)
    unit_shadow = field(px, py, gi)
    thr = S.SENS[SF_MAIN]

    q = build_query(d, n=12_000)
    q_fp = fingerprints_for(q, d, n_gw)
    q_xy = np.c_[d["x"][q], d["y"][q]]
    # the real support: how many gateways a real message names
    real_density = float((q_fp > L.FLOOR).sum(1).mean())

    rows = []
    for sc in SCALES:
        cp = dict(mu=PR.CROSSINGS["mu"] * sc, tau=PR.CROSSINGS["tau"] * sc)
        dp = dict(mu=PR.DEPTH["mu"] * sc, tau=PR.DEPTH["tau"] * sc)
        errs, dens = [], []
        for level in range(S.N_LEVELS):
            ev, dv = [], []
            for r in range(N_REP):
                gen = np.random.default_rng(SEED + 977 * r)
                phi, sig = prior_draw("uninformed", "guess", level, gen, n_gw,
                                      zf["phi"][level], float(zf["sigma"][level]))
                phi[2 + n_gw] = gen.normal(cp["mu"], cp["tau"])
                phi[3 + n_gw] = gen.normal(dp["mu"], dp["tau"])
                gp = S.rssi(level, phi, gdist, gi, gncr, gdep, n_gw,
                            sig * unit_shadow)
                fp = gp.reshape(n_ref, n_gw).astype(np.float32)
                fp[fp < thr] = L.FLOOR
                dv.append(float((fp > L.FLOOR).sum(1).mean()))
                live = (fp > L.FLOOR).any(1)
                if live.sum() < 50:
                    ev.append(np.nan); continue
                pxy = L.knn_localize(fp[live], ref_xy[live], q_fp, k=5)
                ev.append(float(np.median(L.positioning_errors(pxy, q_xy))))
            errs.append(float(np.nanmean(ev)))
            dens.append(float(np.mean(dv)))
        rows.append(dict(scale=sc, err=errs, density=dens,
                         argmin=int(np.nanargmin(errs))))
        print(f"  scale {sc:5.2f}  argmin F{rows[-1]['argmin']}  err "
              + " ".join(f"{e:6.0f}" for e in errs)
              + "   density " + " ".join(f"{x:5.2f}" for x in dens)
              + f"   ({time.time()-t0:.0f}s)", flush=True)

    # --- mechanism: does sparsity mismatch predict positioning error? -------
    E, MM, LV, SCn = [], [], [], []
    for r in rows:
        for lv in range(S.N_LEVELS):
            if not np.isfinite(r["err"][lv]):
                continue
            E.append(r["err"][lv])
            MM.append(abs(np.log((r["density"][lv] + 1e-3) / real_density)))
            LV.append(lv); SCn.append(r["scale"])
    E, MM = np.array(E), np.array(MM)
    rho_all, p_all = spearman(MM, E)
    # the same, holding the level fixed -- so the correlation cannot be an
    # artefact of both quantities being monotone in the rung
    within = []
    for lv in range(S.N_LEVELS):
        m = np.array(LV) == lv
        if m.sum() >= 4 and np.ptp(MM[m]) > 0:
            within.append(spearman(MM[m], E[m])[0])

    out = dict(rows=rows, real_density=real_density, scales=list(SCALES),
               n_rep=N_REP, n_query=len(q),
               mismatch=dict(rho=rho_all, p=p_all, n=int(len(E)),
                             within_level=within,
                             within_level_mean=float(np.mean(within))
                             if within else float("nan")),
               argmin_by_scale={f"{s:g}": r["argmin"]
                                for s, r in zip(SCALES, rows)},
               runtime_s=time.time() - t0)
    os.makedirs(D.RESULTS, exist_ok=True)
    with open(os.path.join(D.RESULTS, "exp10_mechanism.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  sparsity mismatch vs positioning error: rho={rho_all:.2f} "
          f"(p={p_all:.1e}, n={len(E)}); within-level mean rho="
          f"{out['mismatch']['within_level_mean']:.2f}")
    print(f"  real support {real_density:.2f} gateways per message")
    print(f"  done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
