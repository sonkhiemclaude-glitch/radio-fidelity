"""Experiment 5 -- what unlabelled real data cannot tell you.

The dominant practice for judging a simulator without ground truth is to
compare distributions: match the simulated RSSI histogram to the measured one,
minimise an MMD between simulated and measured feature vectors, or train a
domain classifier and read its accuracy as a distance.  Proposition 5 says none
of this can order two simulators by real risk, because the real risk depends on
the *pairing* between covariate and response and an unlabelled sample carries
no information about that pairing.

Three things are measured here.  First, a constructive witness: two simulators
with byte-identical predicted-RSSI marginals and very different real risk, so
that every marginal-matching criterion is exactly indifferent between them.
Second, the practical version: four standard unlabelled proxies are computed
for every fidelity level and their ranking is compared with the truth.

Third -- and this is what makes the second a measurement rather than an
anecdote -- the same four proxies are computed over a *family* of simulators,
the six levels crossed with independent draws from the declared priors, so that
the correlations are estimated from tens of simulators rather than from six
points.  Two statistics are reported for the family: the correlation across the
whole family, and the mean correlation *within* a level, across draws.  The
second is the sharper one, because it asks whether a proxy can order two
simulators that model the same physics with different coefficients -- which is
exactly the comparison Proposition 5 says cannot be guaranteed.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import simulator as S
import tobit as TB
import priors as PR
import learners as L
from exp2_ladder import build_query, fingerprints_for, prior_draw, SF_MAIN, SEED

N_SUB = 4000          # subsample for the quadratic-cost proxies
N_DRAW_FAMILY = 8     # coefficient draws per level in the family arm
N_QUERY_FAMILY = 8000


def mmd2_rbf(X, Y, gamma=None, rng=None):
    """Unbiased squared MMD with an RBF kernel and the median heuristic."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    if gamma is None:
        Z = np.r_[X[: min(len(X), 500)], Y[: min(len(Y), 500)]]
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / max(med, 1e-9)

    def k(A, B):
        return np.exp(-gamma * (((A ** 2).sum(1)[:, None] - 2 * A @ B.T
                                 + (B ** 2).sum(1)[None, :])))
    n, m = len(X), len(Y)
    Kxx = k(X, X); Kyy = k(Y, Y); Kxy = k(X, Y)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                 - 2 * Kxy.mean())


def domain_auc(X, Y, seed=0):
    """AUC of a classifier separating simulated from real feature vectors --
    the empirical proxy for the A-distance."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    Z = np.r_[X, Y]; y = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    p = cross_val_predict(HistGradientBoostingClassifier(
        max_iter=120, random_state=seed), Z, y, cv=3, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    d = D.load(); n_gw = d["n_gw"]
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    om, og = d["obs_msg"], d["obs_gw"]
    dist, ncr, dep = D.link_geometry(d["x"][om], d["y"][om], d["gw_x"], d["gw_y"],
                                     og, geom, cache=os.path.join(D.DATA, "links_real.npz"))
    rssi = d["obs_rssi"]
    Psi = S.features(dist, og, ncr, dep, n_gw)
    r0 = S.free_space(dist) - rssi

    # ---------------- witness: identical marginal, different risk ----------
    # The impossibility is about the *pairing* between a link and a prediction,
    # and an unlabelled sample sees only the multiset of predictions.  So the
    # sharpest finite-sample witness is a permutation: two simulators that emit
    # exactly the same multiset of predicted powers and differ only in which
    # link gets which.  Their predicted-power distributions are identical by
    # construction -- every marginal-matching criterion is exactly indifferent
    # between them -- and their risks are as far apart as the pairing allows.
    vals = np.sort(S.free_space(dist) - r0.mean())      # the multiset on offer
    order_best = np.argsort(np.argsort(rssi))           # aligned with the truth
    order_worst = np.argsort(np.argsort(-rssi))         # anti-aligned
    A = vals[order_best]
    Bs = vals[order_worst]
    w1 = float(sps.wasserstein_distance(A, Bs))
    riskA = float(np.mean((A - rssi) ** 2)); riskB = float(np.mean((Bs - rssi) ** 2))
    print(f"witness: identical predicted-power multiset "
          f"(1-Wasserstein between the two = {w1:.2e} dB), "
          f"risks {riskA:.1f} vs {riskB:.1f} dB^2 "
          f"(ratio {max(riskA,riskB)/max(min(riskA,riskB),1e-9):.1f}x)")

    # ---------------- proxies across the ladder ----------------------------
    zf = np.load(os.path.join(D.DATA, "tobit_fits.npz"))
    gen = np.random.default_rng(SEED)
    phi, sig = prior_draw("uninformed", "guess", 0, gen, n_gw,
                          zf["phi"][0], float(zf["sigma"][0]))
    Cc = D.candidates(d, geom, cache=os.path.join(D.DATA, "links_cand.npz"))
    Psi_c = S.features(Cc["dist"], Cc["gw"], Cc["n_cross"], Cc["depth"], n_gw)
    base_c = S.free_space(Cc["dist"])

    rx, ry = geom.road_x, geom.road_y
    n_ref = len(rx)
    gi = np.tile(np.arange(n_gw), n_ref)
    px = np.repeat(rx, n_gw); py = np.repeat(ry, n_gw)
    gdist, gncr, gdep = D.link_geometry(px, py, d["gw_x"], d["gw_y"], gi, geom,
                                        cache=os.path.join(D.DATA, "links_grid.npz"))
    q = build_query(d, n=20_000)
    q_fp = fingerprints_for(q, d, n_gw)
    q_xy = np.c_[d["x"][q], d["y"][q]]
    ref_xy = np.c_[rx, ry]
    thr = S.SENS[SF_MAIN]
    field = S.ShadowField(rx.min() - 500, ry.min() - 500, rx.max() + 500,
                          ry.max() + 500, 25.0, 1.0, PR.SHADOW_ELL, n_gw,
                          seed=SEED)
    shadow = sig * field(px, py, gi)

    sub_real = rng.choice(len(q_fp), size=min(N_SUB, len(q_fp)), replace=False)

    # The real reception rate is a property of the *candidate* links -- every
    # (message, gateway) pair the query set offers -- not of the received ones,
    # and it has to be measured over the same population the simulated rate is.
    det_real = float((q_fp > L.FLOOR).sum() / (len(q) * n_gw))

    def proxies(fp, gp, live, q_use, qfp_use, qxy_use, sub_r, det_ref):
        sim_vals = gp[gp >= thr]
        w1_vals = float(sps.wasserstein_distance(
            rng.choice(sim_vals, size=min(50_000, len(sim_vals)), replace=False),
            rng.choice(rssi, size=min(50_000, len(rssi)), replace=False)))
        sub_sim = rng.choice(np.flatnonzero(live), size=min(N_SUB, live.sum()),
                             replace=False)
        Xs = fp[sub_sim]; Xr = qfp_use[sub_r]
        mmd = mmd2_rbf(Xs, Xr)
        auc = domain_auc(Xs, Xr, seed=SEED)
        det_sim = float((gp >= thr).mean())
        pxy = L.knn_localize(fp[live], ref_xy[live], qfp_use, k=5)
        err = float(np.median(L.positioning_errors(pxy, qxy_use)))
        return dict(loc_err=err, w1=w1_vals, mmd2=mmd, domain_auc=auc,
                    det_sim=det_sim, det_real=det_ref,
                    det_gap=abs(det_sim - det_ref))

    rows = []
    for lv in range(S.N_LEVELS):
        cols = S.level_columns(lv, n_gw)
        risk = float(np.mean(TB.link_loss(phi[cols], sig, Psi_c[:, cols],
                                          base_c, Cc["y"], Cc["got"], Cc["thr"])))
        gp = S.rssi(lv, phi, gdist, gi, gncr, gdep, n_gw, shadow)
        fp = gp.reshape(n_ref, n_gw).astype(np.float32)
        fp[fp < thr] = L.FLOOR
        live = (fp > L.FLOOR).any(1)
        pr = proxies(fp, gp, live, q, q_fp, q_xy, sub_real, det_real)
        rows.append(dict(level=lv, name=S.LEVELS[lv], rssi_risk=risk, **pr))
        print(f"  L{lv} risk {risk:8.2f}  loc {pr['loc_err']:6.0f} m | "
              f"W1 {pr['w1']:6.2f}  MMD2 {pr['mmd2']:8.5f}  "
              f"AUC {pr['domain_auc']:.3f}  detgap {pr['det_gap']:.3f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    def perm_p(a, b, n_perm=20_000, seed=0):
        """Exact-in-distribution two-sided permutation p for Spearman rho."""
        a = np.asarray(a, float); b = np.asarray(b, float)
        obs = abs(sps.spearmanr(a, b).statistic)
        g = np.random.default_rng(seed)
        import itertools, math
        if math.factorial(len(a)) <= n_perm:
            perms = list(itertools.permutations(range(len(a))))
            hits = sum(abs(sps.spearmanr(a, b[list(p)]).statistic) >= obs - 1e-12
                       for p in perms)
            return float(hits / len(perms))
        hits = sum(abs(sps.spearmanr(a, g.permutation(b)).statistic) >= obs - 1e-12
                   for _ in range(n_perm))
        return float((hits + 1) / (n_perm + 1))

    def rank_corr(vals, target, seed=0):
        rho, _ = sps.spearmanr(vals, target)
        return float(rho), perm_p(vals, target, seed=seed)

    KEYS = ("w1", "mmd2", "domain_auc", "det_gap")
    corr = {k: dict(vs_risk=rank_corr([r[k] for r in rows],
                                      [r["rssi_risk"] for r in rows]),
                    vs_loc=rank_corr([r[k] for r in rows],
                                     [r["loc_err"] for r in rows]))
            for k in KEYS}
    print("\nSpearman correlation of each unlabelled proxy with the truth "
          "(six ladder levels, exact permutation p):")
    for k, v in corr.items():
        print(f"  {k:11s} vs RSSI risk {v['vs_risk'][0]:+.3f} (p={v['vs_risk'][1]:.4f})"
              f"  vs localization {v['vs_loc'][0]:+.3f} (p={v['vs_loc'][1]:.4f})")

    # -------- the family arm: many simulators, not six ----------------------
    # Six points is not a measurement.  The same proxies are recomputed over the
    # six levels crossed with independent coefficient draws, which gives a
    # family large enough to put an interval on, and -- more to the point --
    # lets the proxies be scored on the comparison the impossibility is about:
    # two simulators at the *same* level with different coefficients.
    qf = build_query(d, n=N_QUERY_FAMILY, seed=SEED + 1)
    qf_fp = fingerprints_for(qf, d, n_gw)
    qf_xy = np.c_[d["x"][qf], d["y"][qf]]
    sub_rf = rng.choice(len(qf_fp), size=min(N_SUB, len(qf_fp)), replace=False)
    det_real_f = float((qf_fp > L.FLOOR).sum() / (len(qf) * n_gw))
    fam = []
    for dr in range(N_DRAW_FAMILY):
        gd = np.random.default_rng(SEED + 4441 * (dr + 1))
        phid, sigd = prior_draw("uninformed", "guess", S.N_LEVELS - 1, gd, n_gw,
                                zf["phi"][S.N_LEVELS - 1],
                                float(zf["sigma"][S.N_LEVELS - 1]))
        shd = sigd * field(px, py, gi)
        for lv in range(S.N_LEVELS):
            cols = S.level_columns(lv, n_gw)
            risk = float(np.mean(TB.link_loss(phid[cols], sigd, Psi_c[:, cols],
                                              base_c, Cc["y"], Cc["got"],
                                              Cc["thr"])))
            gp = S.rssi(lv, phid, gdist, gi, gncr, gdep, n_gw, shd)
            fp = gp.reshape(n_ref, n_gw).astype(np.float32)
            fp[fp < thr] = L.FLOOR
            live = (fp > L.FLOOR).any(1)
            if live.sum() < 50:
                continue
            pr = proxies(fp, gp, live, qf, qf_fp, qf_xy, sub_rf, det_real_f)
            fam.append(dict(level=lv, draw=dr, rssi_risk=risk, **pr))
        print(f"  family draw {dr + 1}/{N_DRAW_FAMILY} "
              f"({len(fam)} simulators, {time.time()-t0:.0f}s)", flush=True)

    def boot_rho(a, b, B=2000, seed=0):
        a = np.asarray(a, float); b = np.asarray(b, float)
        g = np.random.default_rng(seed)
        v = [sps.spearmanr(a[i], b[i]).statistic
             for i in (g.integers(0, len(a), len(a)) for _ in range(B))]
        v = np.array([x for x in v if np.isfinite(x)])
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    fam_corr = {}
    for k in KEYS:
        a = np.array([r[k] for r in fam])
        out_k = {}
        for tgt in ("rssi_risk", "loc_err"):
            b = np.array([r[tgt] for r in fam])
            rho = float(sps.spearmanr(a, b).statistic)
            lo, hi = boot_rho(a, b, seed=SEED)
            wl = []
            for lv in range(S.N_LEVELS):
                m = np.array([r["level"] == lv for r in fam])
                if m.sum() >= 4 and np.ptp(a[m]) > 0 and np.ptp(b[m]) > 0:
                    wl.append(float(sps.spearmanr(a[m], b[m]).statistic))
            out_k[tgt] = dict(rho=rho, lo=lo, hi=hi, within=wl,
                              within_mean=float(np.mean(wl)) if wl else float("nan"))
        fam_corr[k] = out_k
    print(f"\nfamily of {len(fam)} simulators "
          f"({N_DRAW_FAMILY} draws x {S.N_LEVELS} levels):")
    for k, v in fam_corr.items():
        print(f"  {k:11s} vs risk {v['rssi_risk']['rho']:+.2f} "
              f"[{v['rssi_risk']['lo']:+.2f},{v['rssi_risk']['hi']:+.2f}] "
              f"within-level {v['rssi_risk']['within_mean']:+.2f}   |  "
              f"vs loc {v['loc_err']['rho']:+.2f} "
              f"[{v['loc_err']['lo']:+.2f},{v['loc_err']['hi']:+.2f}] "
              f"within-level {v['loc_err']['within_mean']:+.2f}")

    out = dict(rows=rows, corr=corr, family=fam, family_corr=fam_corr,
               det_real=det_real, det_real_family=det_real_f,
               n_family=len(fam), n_draw_family=N_DRAW_FAMILY,
               witness=dict(w1_between=w1, riskA=riskA, riskB=riskB,
                            ratio=max(riskA, riskB) / min(riskA, riskB)),
               n_sub=N_SUB)
    with open(os.path.join(D.RESULTS, "exp5_unlabelled.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> exp5_unlabelled.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
