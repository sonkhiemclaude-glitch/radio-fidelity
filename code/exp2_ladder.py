"""Experiment 2 -- the ladder.

Walks the six fidelity levels and measures, on real measurements, what each rung
actually buys.  Three coefficient regimes are compared at every level:

`oracle`      the coefficients refitted to the real data *at that level* by
              censored maximum likelihood.  Not available to a modeller; it
              isolates the structural bias of the simulator and is monotone by
              construction.
`guess`       a single value drawn from the literature prior, which is what a
              modeller actually does.
`randomised`  the fixed prior mean. The rich squared-loss identity is a link-level
              interpretation, not a claim of mixture training for k-NN.

Two tasks are scored.  The link task is the one the theory covers: the mean
censored negative log-likelihood of the simulator on the candidate-link
population, which under the rich-learner assumption is the risk of the model
trained on it.  The localization task is the downstream one the field cares
about: the fingerprint database is built by *sampling* the simulator at
reference points on the road network, and positioning error is measured on real
messages. Prior-based arms use no real calibration labels; the fitted
coefficient arm uses real observations and is a diagnostic comparator.
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
import statsx as SX

N_REP = 20
N_QUERY = 12_000
SF_MAIN = 7
SEED = 3108


def build_query(d, sf=SF_MAIN, n=N_QUERY, seed=SEED):
    """Real messages used as localization queries: one spreading factor, and a
    geographically blocked subsample so the query set is spread over the city
    rather than over the densest drive."""
    sel = np.flatnonzero(d["sf"] == sf)
    fold = D.block_folds(d["x"][sel], d["y"][sel], n_folds=8, seed=seed)
    rng = np.random.default_rng(seed)
    take = []
    for f in range(8):
        idx = sel[fold == f]
        k = min(len(idx), int(np.ceil(n / 8)))
        take.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(take))


def fingerprints_for(msgs, d, n_gw):
    """Dense real fingerprints for a set of message indices."""
    pos = -np.ones(d["n_msg"], np.int64)
    pos[msgs] = np.arange(len(msgs))
    keep = pos[d["obs_msg"]] >= 0
    return L.to_fingerprint(d["obs_rssi"][keep], d["obs_gw"][keep],
                            pos[d["obs_msg"][keep]], len(msgs), n_gw)


def prior_draw(regime, mode, level, gen, n_gw, phi_oracle, sigma_oracle):
    """A full coefficient vector plus a scale, in the layout of
    `simulator.features`."""
    p = np.zeros(S.n_features(n_gw))
    if mode == "oracle":
        return phi_oracle.copy(), sigma_oracle
    rand = (mode == "randomised")
    j = (lambda m, t: m if rand else gen.normal(m, t))
    p[0] = j(PR.OFFSET["mu"], PR.OFFSET["tau"])
    p[1] = j(PR.EXPONENT["mu"], PR.EXPONENT["tau"])
    p[2 + n_gw] = j(PR.CROSSINGS["mu"], PR.CROSSINGS["tau"])
    p[3 + n_gw] = j(PR.DEPTH["mu"], PR.DEPTH["tau"])
    if regime == "informed":
        base = phi_oracle[2: 2 + n_gw]
        p[2: 2 + n_gw] = base if rand else base + gen.normal(
            0, PR.GATEWAY_INFORMED["tau"], n_gw)
    else:
        p[2: 2 + n_gw] = (PR.GATEWAY_UNINFORMED["mu"] if rand else
                          gen.normal(PR.GATEWAY_UNINFORMED["mu"],
                                     PR.GATEWAY_UNINFORMED["tau"], n_gw))
    return p, max(1.0, j(PR.SIGMA["mu"], PR.SIGMA["tau"]))


def main():
    t0 = time.time()
    d = D.load()
    n_gw = d["n_gw"]
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    C = D.candidates(d, geom, cache=os.path.join(D.DATA, "links_cand.npz"))
    Psi = S.features(C["dist"], C["gw"], C["n_cross"], C["depth"], n_gw)
    base = S.free_space(C["dist"])
    z = np.load(os.path.join(D.DATA, "tobit_fits.npz"))
    oracle = [(z["phi"][lv], float(z["sigma"][lv])) for lv in range(S.N_LEVELS)]

    rx, ry = geom.road_x, geom.road_y
    n_ref = len(rx)
    gi = np.tile(np.arange(n_gw), n_ref)
    px = np.repeat(rx, n_gw); py = np.repeat(ry, n_gw)
    gdist, gncr, gdep = D.link_geometry(px, py, d["gw_x"], d["gw_y"], gi, geom,
                                        cache=os.path.join(D.DATA, "links_grid.npz"))
    print(f"grid: {n_ref} reference points x {n_gw} gateways = {len(gdist)} links "
          f"({time.time()-t0:.0f}s)")

    # one unit-scale shadowing field, reused across levels and scaled by the
    # simulator's own sigma, so that levels differ only in their mean function
    field = S.ShadowField(rx.min() - 500, ry.min() - 500, rx.max() + 500,
                          ry.max() + 500, 25.0, 1.0, PR.SHADOW_ELL, n_gw,
                          seed=SEED)
    unit_shadow = field(px, py, gi)

    q = build_query(d)
    q_fp = fingerprints_for(q, d, n_gw)
    q_xy = np.c_[d["x"][q], d["y"][q]]
    ref_xy = np.c_[rx, ry]
    thr = S.SENS[SF_MAIN]
    print(f"queries: {len(q)} SF{SF_MAIN} messages")

    ckpt = os.path.join(D.RESULTS, "exp2_partial.json")
    rows = []
    done = set()
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt))
        done = {(r["regime"], r["mode"], r["level"]) for r in rows}
        print(f"resuming: {len(rows)} rows already done")
    for regime in ("uninformed", "informed"):
        for mode in ("oracle", "guess", "randomised"):
            if mode == "oracle" and regime == "informed":
                continue
            reps = N_REP if mode == "guess" else 1
            for level in range(S.N_LEVELS):
                if (regime, mode, level) in done:
                    continue
                nll, err50, err_mean, err90, cover = [], [], [], [], []
                for r in range(reps):
                    # The draw depends on the replicate only, never on the
                    # level: a modeller draws one set of coefficients and then
                    # chooses which of them to switch on, so the comparison
                    # across levels is paired.  Seeding per level instead makes
                    # every level an independent sample and inflates the
                    # variance of the very difference the ladder is about.
                    gen = np.random.default_rng(SEED + 977 * r)
                    phi, sig = prior_draw(regime, mode, level, gen, n_gw,
                                          oracle[level][0], oracle[level][1])
                    cols = S.level_columns(level, n_gw)
                    nll.append(float(np.mean(TB.link_loss(
                        phi[cols] if len(cols) else np.zeros(0), sig,
                        Psi[:, cols], base, C["y"], C["got"], C["thr"]))))
                    gpred = S.rssi(level, phi, gdist, gi, gncr, gdep, n_gw,
                                   sig * unit_shadow)
                    fp = gpred.reshape(n_ref, n_gw).astype(np.float32)
                    fp[fp < thr] = L.FLOOR
                    live = (fp > L.FLOOR).any(1)
                    cover.append(float((fp > L.FLOOR).mean()))
                    if live.sum() < 50:
                        err50.append(np.nan); err_mean.append(np.nan)
                        err90.append(np.nan); continue
                    pxy = L.knn_localize(fp[live], ref_xy[live], q_fp, k=5)
                    e = L.positioning_errors(pxy, q_xy)
                    err50.append(float(np.median(e)))
                    err_mean.append(float(np.mean(e)))
                    err90.append(float(np.percentile(e, 90)))
                rows.append(dict(regime=regime, mode=mode, level=level,
                                 level_name=S.LEVELS[level], reps=reps,
                                 nll=float(np.mean(nll)),
                                 nll_sd=float(np.std(nll)),
                                 err50=float(np.nanmean(err50)),
                                 err50_sd=float(np.nanstd(err50)),
                                 err_mean=float(np.nanmean(err_mean)),
                                 err90=float(np.nanmean(err90)),
                                 sim_detect=float(np.mean(cover)),
                                 err50_reps=err50, nll_reps=nll))
                with open(ckpt, "w") as f:
                    json.dump(rows, f)
                print(f"  {regime:11s} {mode:10s} L{level} "
                      f"nll {np.mean(nll):7.4f}  detect "
                      f"{np.mean(cover)*100:5.2f}%  median err "
                      f"{np.nanmean(err50):7.0f} m   ({time.time()-t0:.0f}s)",
                      flush=True)

    # ---- paired comparisons between adjacent levels ------------------------
    # Every replicate uses one coefficient draw at every level, so the level
    # differences are paired and the question "is the minimum really at this
    # level" has a test rather than an eyeball.
    pairs = []
    for regime in ("uninformed", "informed"):
        rr = {r["level"]: r for r in rows
              if r["regime"] == regime and r["mode"] == "guess"}
        for k in range(S.N_LEVELS - 1):
            a = np.array(rr[k]["nll_reps"]); b = np.array(rr[k + 1]["nll_reps"])
            ea = np.array(rr[k]["err50_reps"]); eb = np.array(rr[k + 1]["err50_reps"])
            m, lo, hi = SX.bootstrap_ci(b - a)
            me, loe, hie = SX.bootstrap_ci(eb - ea)
            pairs.append(dict(
                regime=regime, rung=k + 1,
                nll_delta=m, nll_lo=lo, nll_hi=hi,
                nll_p=SX.paired_test(b, a)["p"],
                err_delta=me, err_lo=loe, err_hi=hie,
                err_p=SX.paired_test(eb, ea)["p"]))
    for regime in ("uninformed", "informed"):
        sub = [p for p in pairs if p["regime"] == regime]
        adj, rej = SX.holm_bonferroni([p["nll_p"] for p in sub])
        for p, a_, r_ in zip(sub, adj, rej):
            p["nll_p_holm"] = float(a_); p["nll_reject"] = bool(r_)
        adj, rej = SX.holm_bonferroni([p["err_p"] for p in sub])
        for p, a_, r_ in zip(sub, adj, rej):
            p["err_p_holm"] = float(a_); p["err_reject"] = bool(r_)
    print("\npaired rung increments (uninformed, literature prior):")
    for p in pairs:
        if p["regime"] != "uninformed":
            continue
        print(f"  F{p['rung']}: link {p['nll_delta']:+9.3f} "
              f"[{p['nll_lo']:+.3f}, {p['nll_hi']:+.3f}] "
              f"p_holm {p['nll_p_holm']:.4f} | "
              f"loc {p['err_delta']:+7.0f} m "
              f"[{p['err_lo']:+.0f}, {p['err_hi']:+.0f}] "
              f"p_holm {p['err_p_holm']:.4f}")

    fold = D.block_folds(d["x"][q], d["y"][q], n_folds=5, seed=SEED)
    errs = []
    for f in range(5):
        tr, te = fold != f, fold == f
        pxy = L.knn_localize(q_fp[tr], q_xy[tr], q_fp[te], k=5)
        errs.append(L.positioning_errors(pxy, q_xy[te]))
    errs = np.concatenate(errs)
    ceiling = dict(err50=float(np.median(errs)), err_mean=float(np.mean(errs)),
                   err90=float(np.percentile(errs, 90)))
    print(f"real-data ceiling (blocked 5-fold): median {ceiling['err50']:.0f} m")

    out = dict(runtime_s=float(time.time() - t0),
               rows=rows, pairs=pairs, ceiling=ceiling, n_ref=int(n_ref),
               n_query=int(len(q)), n_rep=N_REP, sf=SF_MAIN,
               levels=list(S.LEVELS),
               real_detect=float(C["got"].mean()))
    os.makedirs(D.RESULTS, exist_ok=True)
    with open(os.path.join(D.RESULTS, "exp2_ladder.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> exp2_ladder.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
