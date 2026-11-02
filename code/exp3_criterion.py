"""Experiment 3 -- is the criterion right?

Corollary 1 makes a quantitative, falsifiable prediction: a rung with
conditional oracle coefficient beta helps if and only if the modeller's prior
satisfies (m - beta)^2 + tau^2 < beta^2, a half-disc in the (m, tau) plane of
radius |beta| centred at (beta, 0).  For the censored likelihood the prediction
is only second-order accurate (Theorem 2), so measuring where the real sign flip
happens is a genuine test rather than an algebraic identity.

Two rungs are tested.  The clutter offset is the rung with the largest true
effect, so its admission region is large and the measurement is precise.  The
per-gateway offsets are the rung a modeller can actually buy information about,
by surveying the installations; there the criterion becomes an operational
threshold on how accurate that survey has to be.

The whole (m, tau) plane is swept from a single one-dimensional sweep of the
coefficient: the risk is evaluated on a fine grid of coefficient values once,
and any prior's expected risk is then a weighted average over that grid.
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
from exp1_premise import loss_derivatives
from exp2_ladder import build_query, fingerprints_for, SF_MAIN, SEED

N_GRID = 481          # coefficient values in the one-dimensional sweep
N_QUERY_LOC = 12_000
N_LOC_GRID = 25


def expected_over_prior(cs, vals, mu, tau):
    """E_pi[f(c)] for c ~ N(mu, tau^2), by quadrature on the precomputed sweep."""
    if tau <= 0:
        return float(np.interp(mu, cs, vals))
    w = np.exp(-0.5 * ((cs - mu) / tau) ** 2)
    s = w.sum()
    if s <= 0:
        return float(np.interp(mu, cs, vals))
    return float((w * vals).sum() / s)


def main():
    t0 = time.time()
    d = D.load(); n_gw = d["n_gw"]
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    C = D.candidates(d, geom, cache=os.path.join(D.DATA, "links_cand.npz"))
    Psi = S.features(C["dist"], C["gw"], C["n_cross"], C["depth"], n_gw)
    base = S.free_space(C["dist"])
    sig = PR.SIGMA["mu"]

    # ================= rung 1: the clutter offset (F0 -> F1) ===============
    A = np.ascontiguousarray(Psi[:, [0]])
    b_, H_ = TB.fit_block(A, base, C["y"], C["got"], C["thr"], sig)
    beta = float(b_[0]); H = float(H_[0, 0])
    base_risk = TB.risk(np.zeros(0), sig, Psi[:, []], base, C["y"], C["got"],
                        C["thr"])
    print(f"rung F1 (clutter offset): conditional oracle C* = {beta:.3f} dB, "
          f"H = {H:.5f}, base risk {base_risk:.5f}")

    cs = np.linspace(-1.2 * abs(beta), 3.2 * abs(beta), N_GRID)
    vals = np.array([TB.risk(np.array([c]), sig, A, base, C["y"], C["got"],
                             C["thr"]) for c in cs])
    print(f"  swept {N_GRID} coefficient values ({time.time()-t0:.0f}s); "
          f"empirical argmin at C = {cs[np.argmin(vals)]:.3f}")

    mus = np.linspace(-0.6 * beta, 2.6 * beta, 17)
    taus = np.linspace(0.0, 1.8 * beta, 13)
    meas = np.zeros((len(taus), len(mus)))
    pred = np.zeros_like(meas)
    for i, tau in enumerate(taus):
        for j, mu in enumerate(mus):
            meas[i, j] = expected_over_prior(cs, vals, mu, tau) - base_risk
            pred[i, j] = H * ((mu - beta) ** 2 + tau ** 2 - beta ** 2)
    err = np.abs(meas - pred)
    sign_ok = float(np.mean(np.sign(meas) == np.sign(pred)))
    scale = float(np.abs(pred).max())
    print(f"  second-order prediction vs measured: max abs err {err.max():.4f} "
          f"({err.max()/scale*100:.2f}% of range), sign agreement "
          f"{sign_ok*100:.1f}%")

    tau_edge, tau_pred = [], []
    for j, mu in enumerate(mus):
        adm = np.flatnonzero(meas[:, j] < 0)
        tau_edge.append(float(taus[adm.max()]) if len(adm) else -1.0)
        tau_pred.append(float(np.sqrt(max(0.0, beta ** 2 - (mu - beta) ** 2))))
    print(f"  admission boundary: measured max tau {max(tau_edge):.2f}, "
          f"predicted |C*| = {abs(beta):.2f}")

    # ---- does it carry to the downstream task? ----------------------------
    q = build_query(d, n=N_QUERY_LOC)
    q_fp = fingerprints_for(q, d, n_gw)
    q_xy = np.c_[d["x"][q], d["y"][q]]
    rx, ry = geom.road_x, geom.road_y
    n_ref = len(rx)
    gi = np.tile(np.arange(n_gw), n_ref)
    px = np.repeat(rx, n_gw); py = np.repeat(ry, n_gw)
    gdist, gncr, gdep = D.link_geometry(px, py, d["gw_x"], d["gw_y"], gi, geom,
                                        cache=os.path.join(D.DATA, "links_grid.npz"))
    ref_xy = np.c_[rx, ry]
    thr = S.SENS[SF_MAIN]
    field = S.ShadowField(rx.min() - 500, ry.min() - 500, rx.max() + 500,
                          ry.max() + 500, 25.0, 1.0, PR.SHADOW_ELL, n_gw, seed=SEED)
    shadow = sig * field(px, py, gi)

    def loc_err(c):
        phi = np.zeros(S.n_features(n_gw)); phi[0] = c
        gp = S.rssi(1, phi, gdist, gi, gncr, gdep, n_gw, shadow)
        fp = gp.reshape(n_ref, n_gw).astype(np.float32)
        fp[fp < thr] = L.FLOOR
        live = (fp > L.FLOOR).any(1)
        if live.sum() < 50:
            return np.nan
        pxy = L.knn_localize(fp[live], ref_xy[live], q_fp, k=5)
        return float(np.median(L.positioning_errors(pxy, q_xy)))

    cs_l = np.linspace(0.0, 2.6 * beta, N_LOC_GRID)
    loc_vals = np.array([loc_err(c) for c in cs_l])
    base_loc = loc_err(0.0)
    print(f"  localization sweep done ({time.time()-t0:.0f}s); best at "
          f"C = {cs_l[np.nanargmin(loc_vals)]:.1f} dB "
          f"({np.nanmin(loc_vals):.0f} m), at C = 0: {base_loc:.0f} m")
    mus_l = np.linspace(0.0, 2.4 * beta, 9)
    taus_l = [0.0, 0.5 * beta, 1.0 * beta]
    loc = np.array([[expected_over_prior(cs_l, loc_vals, mu, tau) - base_loc
                     for mu in mus_l] for tau in taus_l])
    pred_loc = np.array([[H * ((mu - beta) ** 2 + tau ** 2 - beta ** 2)
                          for mu in mus_l] for tau in taus_l])
    loc_sign = float(np.mean(np.sign(loc) == np.sign(pred_loc)))
    print(f"  downstream sign agreement with the criterion: {loc_sign*100:.1f}%")

    # ================= rung 2: how good must a site survey be? =============
    phi_m = np.zeros(S.n_features(n_gw))
    phi_m[0] = PR.OFFSET["mu"]; phi_m[1] = PR.EXPONENT["mu"]
    cols = list(range(2, 2 + n_gw))
    mu_lvl = base + Psi[:, [0, 1]] @ phi_m[[0, 1]]
    Ag = np.ascontiguousarray(Psi[:, cols])
    delta = TB.fit_offsets(C["gw"], mu_lvl, C["y"], C["got"], C["thr"], sig,
                           n_gw)
    mu2 = mu_lvl + Ag @ delta
    w = np.empty(len(mu2))
    w[C["got"]] = 1.0 / sig ** 2
    zz = (C["thr"][~C["got"]] - mu2[~C["got"]]) / sig
    lam = TB._mills(zz)
    w[~C["got"]] = lam * (lam + zz) / sig ** 2
    Hg = (Ag * w[:, None]).T @ Ag / len(mu2)
    null = float(delta @ Hg @ delta); trH = float(np.trace(Hg))
    s_pred = float(np.sqrt(null / trH))
    lvl_risk = TB.risk(phi_m[[0, 1]], sig, Psi[:, [0, 1]], base, C["y"],
                       C["got"], C["thr"])
    print(f"\nrung F3 (gateway offsets): oracle spread sd {np.std(delta):.2f} dB, "
          f"predicted survey-accuracy threshold s* = {s_pred:.2f} dB")

    ss = np.linspace(0.0, 2.5 * s_pred, 12)
    meas_s, pred_s = [], []
    for s_err in ss:
        vals_s = []
        for r in range(16):
            gen = np.random.default_rng(SEED + 101 * r)
            p = phi_m.copy()
            p[cols] = delta + (gen.normal(0, s_err, n_gw) if s_err > 0 else 0.0)
            vals_s.append(TB.risk(p[[0, 1] + cols], sig,
                                  Psi[:, [0, 1] + cols], base, C["y"],
                                  C["got"], C["thr"]))
        meas_s.append(float(np.mean(vals_s)) - lvl_risk)
        pred_s.append(float(s_err ** 2 * trH - null))
    cross = None
    for i in range(len(ss) - 1):
        if meas_s[i] < 0 <= meas_s[i + 1]:
            cross = float(ss[i] + (ss[i + 1] - ss[i]) * (-meas_s[i])
                          / (meas_s[i + 1] - meas_s[i]))
    print(f"  measured crossing at s = "
          f"{cross if cross is not None else float('nan'):.2f} dB "
          f"(predicted {s_pred:.2f} dB)  ({time.time()-t0:.0f}s)")

    out = dict(
        offset=dict(beta=beta, H=H, base_risk=base_risk,
                    cs=cs.tolist(), risk=vals.tolist(),
                    mus=mus.tolist(), taus=taus.tolist(),
                    measured=meas.tolist(), predicted=pred.tolist(),
                    max_abs_err=float(err.max()), rel_err=float(err.max() / scale),
                    sign_agreement=sign_ok,
                    tau_edge=tau_edge, tau_pred=tau_pred,
                    loc_cs=cs_l.tolist(), loc_vals=loc_vals.tolist(),
                    base_loc=base_loc,
                    loc_mus=mus_l.tolist(), loc_taus=[float(t) for t in taus_l],
                    loc_measured=loc.tolist(), loc_predicted=pred_loc.tolist(),
                    loc_sign_agreement=loc_sign,
                    loc_best_c=float(cs_l[np.nanargmin(loc_vals)]),
                    loc_best=float(np.nanmin(loc_vals))),
        gateway=dict(spread_sd=float(np.std(delta)), null=null, trH=trH,
                     s_pred=s_pred, s_measured=cross,
                     s=ss.tolist(), measured=meas_s, predicted=pred_s,
                     level_risk=lvl_risk),
        sigma=sig, n_grid=N_GRID, n_query_loc=int(len(q)))
    with open(os.path.join(D.RESULTS, "exp3_criterion.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> exp3_criterion.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
