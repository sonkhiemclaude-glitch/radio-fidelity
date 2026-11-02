"""Experiment 1 -- the premise.

Estimates, on the real corpus, the four quantities the admission criterion is
made of: the conditional oracle coefficient of every rung, the prior a modeller
would actually hold, the information matrix the comparison happens in, and hence
the rung-by-rung prediction of whether adding that physics helps or hurts.

Nothing here trains anything.  Under the rich-learner assumption the trained
predictor *is* the simulator, so the premise can be read straight off the
likelihood -- which is also why the admission test of Experiment 4 is cheap.

The fit is censored.  A LoRaWAN record contains a link only if a gateway
demodulated it, so least squares over the recorded links estimates the law of
the survivors rather than the physics; this experiment reports both fits and the
gap between them, which is large.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import simulator as S
import tobit as TB
import priors as PR
import plotstyle as PS


def loss_derivatives(mu, sigma, y, got, thr):
    """First and second derivatives of the per-link censored loss with respect
    to the predicted mean -- the gradient the conditional oracle solves against,
    and the weights the information metric uses."""
    g = np.empty(len(mu)); w = np.empty(len(mu))
    g[got] = -(y[got] - mu[got]) / sigma ** 2
    w[got] = 1.0 / sigma ** 2
    z = (thr[~got] - mu[~got]) / sigma
    lam = TB._mills(z)
    g[~got] = lam / sigma
    w[~got] = lam * (lam + z) / sigma ** 2
    return g, w


N_QUAD = 81          # quadrature nodes for a scalar rung's prior
N_MC = 64            # Monte-Carlo draws for a vector rung's prior


def rung(A, mu_lvl, sigma, y, got, thr, prior_mu, prior_tau, gw=None,
         n_gw=None, seed=3108):
    """Conditional oracle and knowledge ratio for one rung, given the simulator
    as it currently stands.

    The oracle is obtained by actually minimising the risk over the rung's
    coefficients; the level below can be wrong by tens of decibels, and a
    quadratic expansion around a point that bad is not informative.  For the
    same reason the knowledge ratio is computed in its *exact* form,

        rho = (E_pi R - R*) / (R(0) - R*),

    by integrating the risk against the prior rather than through the
    information metric.  Under a quadratic risk the two agree identically and
    rho is the ratio of mean-squared errors of Theorem 1; under the censored
    likelihood they do not, and Experiment 3 measures how far apart they get.
    """
    A = np.ascontiguousarray(A)
    if gw is not None:
        beta = TB.fit_offsets(gw, mu_lvl, y, got, thr, sigma, n_gw)
        mu2 = mu_lvl + A @ beta
        w = np.empty(len(mu2))
        w[got] = 1.0 / sigma ** 2
        z = (thr[~got] - mu2[~got]) / sigma
        lam = TB._mills(z)
        w[~got] = lam * (lam + z) / sigma ** 2
        H = (A * w[:, None]).T @ A / len(mu2)
    else:
        beta, H = TB.fit_block(A, mu_lvl, y, got, thr, sigma)

    R = lambda b: TB.risk(np.atleast_1d(b), sigma, A, mu_lvl, y, got, thr)
    r_star = R(beta)
    r_null = R(np.zeros(len(beta)))
    if len(beta) == 1:
        # exact quadrature against the prior density
        nodes = prior_mu + prior_tau * np.linspace(-5, 5, N_QUAD)
        wq = np.exp(-0.5 * ((nodes - prior_mu) / max(prior_tau, 1e-9)) ** 2)
        wq /= wq.sum()
        r_pi = float(sum(wi * R(np.array([c])) for wi, c in zip(wq, nodes)))
        r_rand = R(np.array([prior_mu]))
    else:
        g = np.random.default_rng(seed)
        draws = prior_mu + prior_tau * g.standard_normal((N_MC, len(beta)))
        r_pi = float(np.mean([R(d) for d in draws]))
        r_rand = R(np.full(len(beta), prior_mu))
    denom = r_null - r_star
    m = np.full(len(beta), prior_mu, float)
    t = np.full(len(beta), prior_tau, float)
    q = lambda v: float(v @ H @ v)
    mse = q(m - beta) + float(np.sum(t ** 2 * np.diag(H)))
    null_q = q(beta)
    return dict(beta=beta, H=H,
                r_star=r_star, r_null=r_null, r_pi=r_pi, r_rand=r_rand,
                rho=((r_pi - r_star) / denom if denom > 0 else np.inf),
                rho_quadratic=(mse / null_q if null_q > 0 else np.inf),
                delta_guess=r_pi - r_null,
                delta_randomised=r_rand - r_null,
                admit_guess=bool(r_pi < r_null),
                admit_randomised=bool(r_rand < r_null))


def main():
    t0 = time.time()
    os.makedirs(D.RESULTS, exist_ok=True)
    d = D.load()
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    n_gw = d["n_gw"]
    C = D.candidates(d, geom, cache=os.path.join(D.DATA, "links_cand.npz"))
    Psi = S.features(C["dist"], C["gw"], C["n_cross"], C["depth"], n_gw)
    base = S.free_space(C["dist"])
    print(f"real corpus: {d['n_msg']} messages, {len(d['obs_rssi'])} received "
          f"links, {n_gw} gateways ({d['dropped_gw']} dropped)")
    print(f"candidate corpus: {len(C['dist'])} links from {len(C['msgs'])} "
          f"messages, {C['got'].mean()*100:.2f}% received")
    print(f"geometry: {geom.n_buildings} buildings, "
          f"{geom.occ.mean()*100:.2f}% raster occupancy, "
          f"{len(geom.road_x)} road reference points")
    got = C["got"]
    print(f"candidate links: median distance {np.median(C['dist']):.0f} m, "
          f"median buildings crossed {np.median(C['n_cross']):.0f}, median "
          f"obstruction depth {np.median(C['depth']):.0f} m")
    print(f"received links:  median distance {np.median(C['dist'][got]):.0f} m, "
          f"median buildings crossed {np.median(C['n_cross'][got]):.0f}, median "
          f"obstruction depth {np.median(C['depth'][got]):.0f} m")

    # ---- censored maximum likelihood along the ladder ---------------------
    cache = os.path.join(D.DATA, "tobit_fits.npz")
    z = np.load(cache)
    fits = [(z["phi"][i], float(z["sigma"][i]), float(z["nll"][i]))
            for i in range(S.N_LEVELS)]
    print(f"\ncensored ML along the ladder")
    lad = []
    for lv, (phi, sig, f) in enumerate(fits):
        det = TB.detection_rate(phi, sig, Psi, base, C["thr"])
        lad.append(dict(level=lv, name=S.LEVELS[lv], nll=f, sigma=sig,
                        detection=det, phi=phi.tolist()))
        print(f"  L{lv} {S.LEVELS[lv]:22s} nll {f:.5f}  sigma {sig:5.2f} dB  "
              f"P(detect) {det*100:5.2f}%")
    top = fits[S.N_LEVELS - 1][0]
    print(f"  fully detailed fit: C = {top[0]:.2f} dB at {S.D0:.0f} m, "
          f"n = {2+top[1]:.2f}, gateway offsets sd {top[2:2+n_gw].std():.2f} dB "
          f"(range {np.ptp(top[2:2+n_gw]):.1f}), kappa = {top[2+n_gw]:.3f} "
          f"dB/building, gamma = {top[3+n_gw]:.5f} dB/m")

    # ---- the same ladder fitted the usual way, on received links only -----
    om, og = d["obs_msg"], d["obs_gw"]
    dist_r, ncr_r, dep_r = D.link_geometry(
        d["x"][om], d["y"][om], d["gw_x"], d["gw_y"], og, geom,
        cache=os.path.join(D.DATA, "links_real.npz"))
    Psi_r = S.features(dist_r, og, ncr_r, dep_r, n_gw)
    r0_r = S.free_space(dist_r) - d["obs_rssi"]
    ls = S.oracle_at_level(Psi_r, r0_r, S.N_LEVELS - 1, n_gw)
    resid = r0_r + Psi_r @ ls
    print(f"\nsame ladder by least squares on received links only "
          f"(the usual practice)")
    print(f"  C = {ls[0]:.2f} dB, n = {2+ls[1]:.2f}, gateway sd "
          f"{ls[2:2+n_gw].std():.2f} dB, kappa = {ls[2+n_gw]:.3f}, "
          f"gamma = {ls[3+n_gw]:.5f}, residual sd {resid.std():.2f} dB")
    print(f"  selection bias: clutter offset {top[0]-ls[0]:+.1f} dB, "
          f"exponent {top[1]-ls[1]:+.2f}, scale "
          f"{fits[S.N_LEVELS-1][1]-resid.std():+.1f} dB")

    # ---- data for the selection-bias figure --------------------------------
    edges = np.geomspace(max(50.0, C["dist"].min()), C["dist"].max(), 26)
    mid = np.sqrt(edges[:-1] * edges[1:])
    b = np.digitize(C["dist"], edges) - 1
    ok = (b >= 0) & (b < len(mid))
    med_recv, rate, n_bin = [], [], []
    for i in range(len(mid)):
        m = ok & (b == i)
        n_bin.append(int(m.sum()))
        r = m & C["got"]
        med_recv.append(float(np.median(C["y"][r])) if r.sum() > 20 else np.nan)
        rate.append(float(r.sum() / max(m.sum(), 1)))
    ls_mu = S.free_space(mid) - (ls[0] + ls[1] * (-(-10 * np.log10(mid / S.D0))))
    # the two fitted mean curves at the median gateway, and the reception
    # probability each implies at the dominant spreading factor
    from scipy.stats import norm
    thr7 = S.SENS[7]
    ls_pred = S.free_space(mid) - ls[0] - ls[1] * 10 * np.log10(mid / S.D0)
    tb_pred = S.free_space(mid) - top[0] - top[1] * 10 * np.log10(mid / S.D0)
    sel_fig = dict(
        d=mid.tolist(), n=n_bin, med_recv=med_recv, rate=rate,
        ls_mean=ls_pred.tolist(), tobit_mean=tb_pred.tolist(),
        ls_rate=norm.sf((thr7 - ls_pred) / max(resid.std(), 1e-6)).tolist(),
        tobit_rate=norm.sf((thr7 - tb_pred) / fits[S.N_LEVELS - 1][1]).tolist(),
        thr=float(thr7))

    # ---- rung by rung ------------------------------------------------------
    rungs = [
        ("F1 clutter offset", [0], PR.OFFSET),
        ("F2 exponent", [1], PR.EXPONENT),
        ("F3 gateway offsets", list(range(2, 2 + n_gw)), PR.GATEWAY_UNINFORMED),
        ("F4 crossings", [2 + n_gw], PR.CROSSINGS),
        ("F5 depth", [3 + n_gw], PR.DEPTH),
    ]
    sigma_m = PR.SIGMA["mu"]
    phi_m = np.zeros(S.n_features(n_gw))     # the modeller's simulator so far
    rows = []
    for k, (label, cols, pri) in enumerate(rungs):
        mu_lvl = base + Psi @ phi_m
        is_gw = (len(cols) == n_gw)
        r = rung(Psi[:, cols], mu_lvl, sigma_m, C["y"], C["got"], C["thr"],
                 pri["mu"], pri["tau"],
                 gw=(C["gw"] if is_gw else None), n_gw=n_gw)
        beta = r.pop("beta"); H = r.pop("H")
        r.update(rung=label, k=k + 1, n_params=len(cols),
                 beta=beta.tolist(),
                 beta_summary=(float(beta[0]) if len(cols) == 1
                               else float(np.std(beta))),
                 beta_mean=float(np.mean(beta)),
                 mu=float(pri["mu"]), tau=float(pri["tau"]),
                 H_trace=float(np.trace(H)))
        rows.append(r)
        print(f"\n{label}: {len(cols)} coefficient(s)")
        if len(cols) == 1:
            print(f"  conditional oracle = {beta[0]:+.4f}, prior mu = "
                  f"{pri['mu']:+.3f}, tau = {pri['tau']:.3f}")
        else:
            print(f"  conditional oracle: mean {np.mean(beta):+.2f}, spread sd "
                  f"{np.std(beta):.2f} dB (range {np.ptp(beta):.1f}); "
                  f"prior mu = {pri['mu']:+.1f}, tau = {pri['tau']:.1f}")
        print(f"  knowledge ratio rho = {r['rho']:.4g} "
              f"(quadratic surrogate {r['rho_quadratic']:.4g})  -> "
              f"{'ADMIT' if r['admit_guess'] else 'REJECT'} (guess), "
              f"{'ADMIT' if r['admit_randomised'] else 'REJECT'} (randomised)")
        print(f"  risk: level below {r['r_null']:.4f}, oracle {r['r_star']:.4f}, "
              f"prior {r['r_pi']:.4f}, randomised {r['r_rand']:.4f}")
        # advance the modeller's simulator to this level at the prior mean
        phi_m[cols] = pri["mu"]

    # ---- how good must a site survey be? ----------------------------------
    phi_m2 = np.zeros(S.n_features(n_gw))
    phi_m2[0] = PR.OFFSET["mu"]; phi_m2[1] = PR.EXPONENT["mu"]
    mu_lvl = base + Psi @ phi_m2
    cols = list(range(2, 2 + n_gw))
    A = np.ascontiguousarray(Psi[:, cols])
    beta = TB.fit_offsets(C["gw"], mu_lvl, C["y"], C["got"], C["thr"], sigma_m,
                          n_gw)
    mu2 = mu_lvl + A @ beta
    w = np.empty(len(mu2))
    w[C["got"]] = 1.0 / sigma_m ** 2
    zz = (C["thr"][~C["got"]] - mu2[~C["got"]]) / sigma_m
    lam = TB._mills(zz)
    w[~C["got"]] = lam * (lam + zz) / sigma_m ** 2
    H = (A * w[:, None]).T @ A / len(mu2)
    null = float(beta @ H @ beta); trH = float(np.trace(H))
    s_star = float(np.sqrt(max(null, 0.0) / trH)) if trH > 0 else float("nan")
    print(f"\nF3 gateway offsets: a site survey helps iff its per-gateway error "
          f"is below s* = {s_star:.2f} dB")
    print(f"  (the oracle spread between installations is {np.std(beta):.2f} dB)")

    out = dict(
        n_msg=int(d["n_msg"]), n_links_recv=int(len(d["obs_rssi"])),
        n_gw=int(n_gw), dropped_gw=int(d["dropped_gw"]),
        n_cand=int(len(C["dist"])), n_cand_msg=int(len(C["msgs"])),
        reception_rate=float(C["got"].mean()),
        n_buildings=int(geom.n_buildings), occupancy=float(geom.occ.mean()),
        n_road=int(len(geom.road_x)),
        median_distance=float(np.median(C["dist"])),
        median_crossings=float(np.median(C["n_cross"])),
        median_depth=float(np.median(C["depth"])),
        median_distance_recv=float(np.median(C["dist"][C["got"]])),
        median_crossings_recv=float(np.median(C["n_cross"][C["got"]])),
        median_depth_recv=float(np.median(C["depth"][C["got"]])),
        ladder=lad,
        tobit=dict(C=float(top[0]), n=float(2 + top[1]),
                   gw_sd=float(top[2:2 + n_gw].std()),
                   gw_range=float(np.ptp(top[2:2 + n_gw])),
                   kappa=float(top[2 + n_gw]), gamma=float(top[3 + n_gw]),
                   sigma=float(fits[S.N_LEVELS - 1][1])),
        ls_received=dict(C=float(ls[0]), n=float(2 + ls[1]),
                         gw_sd=float(ls[2:2 + n_gw].std()),
                         kappa=float(ls[2 + n_gw]), gamma=float(ls[3 + n_gw]),
                         sigma=float(resid.std())),
        selection_bias=dict(offset=float(top[0] - ls[0]),
                            exponent=float(top[1] - ls[1]),
                            sigma=float(fits[S.N_LEVELS - 1][1] - resid.std())),
        selection_figure=sel_fig,
        rungs=rows, survey_threshold=s_star,
        gw_spread=float(np.std(beta)), sigma_modeller=sigma_m,
    )
    with open(os.path.join(D.RESULTS, "exp1_premise.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> exp1_premise.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
