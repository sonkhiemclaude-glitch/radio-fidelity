"""Experiment 8 -- robustness.

The nuisance parameters of the evaluation protocol, and the two prior choices
that a reader could reasonably disagree with.  The claim under test is that the
level the ladder selects does not move.

  1  neighbourhood size k
  2  the value written for "not received"
  3  reference-point pitch
  4  spreading factor
  5  the alternative "penetration-loss transplant" prior for the building rung
  6  the modeller's assumed residual scale
  7  random versus geographically blocked evaluation (the leakage tax)
  8  a second real corpus of a different radio technology, with no geometry
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import simulator as S
import tobit as TB
import priors as PR
import learners as L
from exp2_ladder import (build_query, fingerprints_for, prior_draw,
                         SF_MAIN, SEED)

N_REP = 6


def main():
    t0 = time.time()
    d = D.load(); n_gw = d["n_gw"]
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    C = D.candidates(d, geom, cache=os.path.join(D.DATA, "links_cand.npz"))
    Psi = S.features(C["dist"], C["gw"], C["n_cross"], C["depth"], n_gw)
    base = S.free_space(C["dist"])
    zf = np.load(os.path.join(D.DATA, "tobit_fits.npz"))
    out = {}

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

    # ---------------- localization ladder under protocol variants ----------
    def ladder_loc(k=5, floor=L.FLOOR, pitch=1, sf=SF_MAIN, reps=N_REP,
                   crossings_prior=None, depth_prior=None, sigma_prior=None,
                   n_query=12_000):
        q = build_query(d, sf=sf, n=n_query)
        q_fp = fingerprints_for(q, d, n_gw)
        q_fp = np.where(q_fp <= L.FLOOR, floor, q_fp)
        q_xy = np.c_[d["x"][q], d["y"][q]]
        mask = np.zeros(n_ref, bool); mask[np.arange(0, n_ref, pitch)] = True
        lm = np.repeat(mask, n_gw)
        thr_sf = S.SENS[sf]
        res = []
        for level in range(S.N_LEVELS):
            vals = []
            for r in range(reps):
                gen = np.random.default_rng(SEED + 977 * r)
                phi, sig = prior_draw("uninformed", "guess", level, gen, n_gw,
                                      zf["phi"][level], float(zf["sigma"][level]))
                if crossings_prior is not None:
                    phi[2 + n_gw] = gen.normal(crossings_prior["mu"],
                                               crossings_prior["tau"])
                if depth_prior is not None:
                    phi[3 + n_gw] = gen.normal(depth_prior["mu"],
                                               depth_prior["tau"])
                if sigma_prior is not None:
                    sig = sigma_prior
                gp = S.rssi(level, phi, gdist[lm], gi[lm], gncr[lm], gdep[lm],
                            n_gw, sig * unit_shadow[lm])
                fp = gp.reshape(int(mask.sum()), n_gw).astype(np.float32)
                fp[fp < thr_sf] = floor
                live = (fp > floor).any(1)
                if live.sum() < 50:
                    vals.append(np.nan); continue
                pxy = L.knn_localize(fp[live], ref_xy[mask][live], q_fp,
                                     k=min(k, int(live.sum())))
                vals.append(float(np.median(L.positioning_errors(pxy, q_xy))))
            res.append(float(np.nanmean(vals)))
        return res

    variants = {
        "base": dict(),
        "k=1": dict(k=1), "k=3": dict(k=3), "k=9": dict(k=9), "k=15": dict(k=15),
        "floor=-130": dict(floor=-130.0), "floor=-150": dict(floor=-150.0),
        "pitch=100m": dict(pitch=2), "pitch=200m": dict(pitch=4),
        "SF9": dict(sf=9), "SF12": dict(sf=12),
        "crossings 12+-5": dict(crossings_prior=PR.CROSSINGS_TRANSPLANT),
        "geometry priors x0.1": dict(crossings_prior=PR.CROSSINGS_SMALL,
                                     depth_prior=PR.DEPTH_SMALL),
        "sigma=5": dict(sigma_prior=5.0), "sigma=12": dict(sigma_prior=12.0),
    }
    rows = {}
    for name, kw in variants.items():
        v = ladder_loc(**kw)
        rows[name] = dict(err=v, argmin=int(np.nanargmin(v)))
        print(f"  {name:20s} argmin L{rows[name]['argmin']}  "
              + " ".join(f"{e:6.0f}" for e in v) + f"   ({time.time()-t0:.0f}s)",
              flush=True)
    out["loc_variants"] = rows

    # ---------------- link-risk ladder under scale variants ----------------
    link = {}
    for sname, sval in (("sigma=5", 5.0), ("sigma=8", PR.SIGMA["mu"]),
                        ("sigma=12", 12.0), ("sigma=16", 16.0)):
        vals = []
        for level in range(S.N_LEVELS):
            v = []
            for r in range(N_REP):
                gen = np.random.default_rng(SEED + 977 * r)
                phi, _ = prior_draw("uninformed", "guess", level, gen, n_gw,
                                    zf["phi"][level], float(zf["sigma"][level]))
                cols = S.level_columns(level, n_gw)
                v.append(TB.risk(phi[cols], sval, Psi[:, cols], base, C["y"],
                                 C["got"], C["thr"]))
            vals.append(float(np.mean(v)))
        link[sname] = dict(risk=vals, argmin=int(np.argmin(vals)))
        print(f"  link risk {sname:10s} argmin L{link[sname]['argmin']}  "
              + " ".join(f"{e:.4f}" for e in vals))
    out["link_scale"] = link

    # ---------------- admission test under different loss clips -------------
    import theory as T
    from exp2_ladder import prior_draw as _pd
    gen0 = np.random.default_rng(SEED)
    phi0, sig0 = _pd("uninformed", "guess", 0, gen0, n_gw, zf["phi"][0],
                     float(zf["sigma"][0]))
    blocks = D.block_folds(C["x"], C["y_pos"], n_folds=200, seed=SEED)
    rngc = np.random.default_rng(SEED)
    clips = {}
    for cap in (8.0, 15.0, 30.0, 60.0):
        loss = [np.clip(TB.link_loss(phi0[S.level_columns(lv, n_gw)], sig0,
                                     Psi[:, S.level_columns(lv, n_gw)], base,
                                     C["y"], C["got"], C["thr"]), 0.0, cap)
                for lv in range(S.N_LEVELS)]
        truth = np.array([l.mean() for l in loss])
        harmful = np.array([bool(np.any(np.diff(truth)[:k] > 0))
                            for k in range(S.N_LEVELS)])
        sel, bad = [], 0
        for b in range(120):
            order = rngc.permutation(200)
            idx, tot = [], 0
            for blk in order:
                p_ = np.flatnonzero(blocks == blk); idx.append(p_); tot += len(p_)
                if tot >= 2000:
                    break
            idx = np.concatenate(idx)[:2000]
            k = T.fixed_sequence_test([loss[j + 1][idx] - loss[j][idx]
                                       for j in range(S.N_LEVELS - 1)],
                                      0.05, rng_range=2 * cap)
            sel.append(k); bad += int(harmful[k])
        clips[f"clip={cap:.0f}"] = dict(
            argmin=int(np.argmin(truth)), mean_level=float(np.mean(sel)),
            harmful=float(bad / 120), truth=truth.tolist())
        print(f"  clip {cap:5.0f}: population argmin L{np.argmin(truth)}, "
              f"test selects L{np.mean(sel):.2f}, harmful {100*bad/120:.1f}%")
    out["clip"] = clips

    # ---------------- gateways whose availability looks unstable ------------
    t_med = np.median(d["t"])
    early = d["t"][d["obs_msg"]] < t_med
    r_early = np.bincount(d["obs_gw"][early], minlength=n_gw)
    r_late = np.bincount(d["obs_gw"][~early], minlength=n_gw)
    ratio = (r_late + 1) / (r_early + 1)
    unstable = np.flatnonzero((ratio > 2.0) | (ratio < 0.5))
    keep = ~np.isin(C["gw"], unstable)
    print(f"  {len(unstable)} of {n_gw} gateways change reception rate by more "
          f"than 2x between campaign halves")
    stab = {}
    for name, m in (("all", np.ones(len(C["gw"]), bool)), ("stable", keep)):
        v = []
        for level in range(S.N_LEVELS):
            cols = S.level_columns(level, n_gw)
            vv = []
            for r in range(N_REP):
                gen = np.random.default_rng(SEED + 977 * r)
                phi, sg = prior_draw("uninformed", "guess", level, gen, n_gw,
                                     zf["phi"][level], float(zf["sigma"][level]))
                vv.append(TB.risk(phi[cols], sg, Psi[m][:, cols], base[m],
                                  C["y"][m], C["got"][m], C["thr"][m]))
            v.append(float(np.mean(vv)))
        stab[name] = dict(risk=v, argmin=int(np.argmin(v)))
        print(f"  gateways {name:7s}: argmin L{stab[name]['argmin']}  "
              + " ".join(f"{x:.4f}" for x in v))
    out["gateway_stability"] = dict(n_unstable=int(len(unstable)), **stab)

    # ---------------- the leakage tax --------------------------------------
    q = build_query(d, n=20_000)
    q_fp = fingerprints_for(q, d, n_gw)
    q_xy = np.c_[d["x"][q], d["y"][q]]
    rngl = np.random.default_rng(SEED)
    rand_fold = rngl.integers(0, 5, len(q))
    blk_fold = D.block_folds(d["x"][q], d["y"][q], n_folds=5, seed=SEED)
    tax = {}
    for name, fold in (("random", rand_fold), ("blocked", blk_fold)):
        errs = []
        for f in range(5):
            tr, te = fold != f, fold == f
            pxy = L.knn_localize(q_fp[tr], q_xy[tr], q_fp[te], k=5)
            errs.append(L.positioning_errors(pxy, q_xy[te]))
        e = np.concatenate(errs)
        tax[name] = dict(err50=float(np.median(e)), err90=float(np.percentile(e, 90)))
    tax["inflation"] = tax["blocked"]["err50"] / tax["random"]["err50"]
    print(f"  leakage tax: random split {tax['random']['err50']:.0f} m vs blocked "
          f"{tax['blocked']['err50']:.0f} m ({tax['inflation']:.2f}x optimism)")
    out["leakage"] = tax

    with open(os.path.join(D.RESULTS, "exp8_robustness.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> exp8_robustness.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
