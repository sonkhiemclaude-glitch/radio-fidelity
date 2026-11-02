"""Censored-Gaussian (Tobit) fitting of the fidelity ladder.

Why this is not optional.  A LoRaWAN record contains a link only if the gateway
demodulated it, and demodulation is a *tail* event: at a given distance the
links that make it into the record are the ones whose shadowing happened to be
favourable.  Fitting a propagation model to the records alone therefore
estimates the conditional law of the surviving links, not the physics.  On this
corpus that difference is not subtle -- the clutter offset moves by more than
20 dB and the path-loss exponent by more than two -- and a simulator built from
the conditional fit produces fingerprint databases with the wrong sparsity,
because it thinks almost every gateway hears almost every message.

The likelihood.  Every (message, gateway) pair is a candidate link.  Write
mu = free space + Psi phi and let `thr` be the spreading factor's sensitivity.
A received link contributes a Gaussian density at its measured RSSI; an absent
link contributes the probability that its RSSI fell below `thr`:

    -log L = sum_received [ ((y - mu)/s)^2 / 2 + log s ]
             - sum_censored log Phi((thr - mu)/s).

Gradients are analytic, so the 35-parameter fits take seconds rather than
minutes.

The information matrix.  The admission criterion of the paper is stated in the
metric of the risk's own Hessian.  For this likelihood that Hessian in phi is
Psi^T diag(w) Psi / N with

    w = 1/s^2                       for a received link,
    w = lambda(z) (lambda(z) + z)/s^2   for a censored one, z = (thr - mu)/s,

where lambda = phi/Phi is the inverse Mills ratio -- i.e. exactly the
squared-error Gram matrix with each link weighted by how much it actually tells
you.  Setting w = 1 recovers the uncensored case.

The censored weight is positive for every z because lambda(z) > -z for all real
z (the Mills-ratio bound), which is the same statement as the log-concavity of
the normal distribution function.  It is *not* true that lambda(z) > z: lambda
decays to zero in the right tail, so a link far below sensitivity has a small
positive weight, approached as lambda(z)*z from above.  `test_theory.py` checks
both the gradient and this weight against central differences.
"""
from __future__ import annotations
import numpy as np
from scipy import optimize
from scipy.special import log_ndtr, ndtr

LOG2PI = float(np.log(2 * np.pi))


def _mills(z):
    """lambda(z) = phi(z)/Phi(z), the inverse Mills ratio.

    Computed in logs, with the asymptotic expansion lambda(z) ~ -z - 1/z in the
    far left tail where the ratio of two underflowing quantities is not safe.
    """
    z = np.asarray(z, float)
    out = np.empty_like(z)
    far = z < -30.0
    if far.any():
        zf = z[far]
        out[far] = -zf - 1.0 / zf
    if (~far).any():
        zn = z[~far]
        out[~far] = np.exp(-0.5 * zn ** 2 - 0.5 * LOG2PI - log_ndtr(zn))
    return out


def nll_and_grad(theta, Psi, base, y, got, thr):
    """Mean negative log-likelihood and its gradient in (phi, log s)."""
    phi, logs = theta[:-1], theta[-1]
    s = np.exp(logs)
    mu = base + Psi @ phi
    n = len(mu)
    r = (y[got] - mu[got]) / s
    z = (thr[~got] - mu[~got]) / s
    lo = log_ndtr(z)
    f = (np.sum(0.5 * r ** 2 + logs + 0.5 * LOG2PI) - np.sum(lo)) / n
    lam = _mills(z)
    g_mu = np.empty(n)
    g_mu[got] = -r / s
    g_mu[~got] = lam / s
    g_phi = Psi.T @ g_mu / n
    g_logs = (np.sum(-r ** 2 + 1.0) + np.sum(lam * z)) / n
    return f, np.append(g_phi, g_logs)


def fit_newton(Psi, base, y, got, thr, phi0=None, sigma0=15.0,
               outer=8, inner=12, tol=1e-9, sigma_max=60.0):
    """Alternating Newton fit: exact Newton steps in phi at fixed scale, then a
    one-dimensional update of the scale.

    At fixed s the loss is convex in the predicted mean -- the received part is
    quadratic and the censored part has second derivative
    lambda(z)(lambda(z)+z)/s^2 > 0 -- so damped Newton converges in a handful of
    steps and needs no restarts.  This matters: with 33 coefficients and 6e5
    links a quasi-Newton search takes minutes per level and can stall in a poor
    basin, while this takes seconds and cannot.
    """
    n, p = Psi.shape
    phi = np.zeros(p) if phi0 is None else np.asarray(phi0, float).copy()
    s = float(sigma0)
    prev = np.inf
    for it in range(outer):
        for _ in range(inner):
            if p == 0:
                break
            mu = base + Psi @ phi
            g = np.empty(n); w = np.empty(n)
            r = (y[got] - mu[got]) / s
            g[got] = -r / s
            w[got] = 1.0 / s ** 2
            z = (thr[~got] - mu[~got]) / s
            lam = _mills(z)
            g[~got] = lam / s
            w[~got] = lam * (lam + z) / s ** 2
            grad = Psi.T @ g / n
            H = (Psi * w[:, None]).T @ Psi / n
            step = np.linalg.pinv(H, rcond=1e-8) @ grad
            if np.max(np.abs(step)) < 1e-8:
                break
            f0 = risk(phi, s, Psi, base, y, got, thr)
            t = 1.0
            for _ls in range(20):
                cand = phi - t * step
                if risk(cand, s, Psi, base, y, got, thr) <= f0 - 1e-12:
                    break
                t *= 0.5
            phi = phi - t * step
            if np.max(np.abs(step * t)) < 1e-9:
                break
        # scale update: minimise over log s by golden section
        lo, hi = np.log(0.5), np.log(sigma_max)
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
        fa = risk(phi, np.exp(a), Psi, base, y, got, thr)
        fb = risk(phi, np.exp(b), Psi, base, y, got, thr)
        for _ in range(35):
            if fa < fb:
                hi, b, fb = b, a, fa
                a = hi - gr * (hi - lo)
                fa = risk(phi, np.exp(a), Psi, base, y, got, thr)
            else:
                lo, a, fa = a, b, fb
                b = lo + gr * (hi - lo)
                fb = risk(phi, np.exp(b), Psi, base, y, got, thr)
            if hi - lo < 1e-8:
                break
        s = float(np.exp(0.5 * (lo + hi)))
        f = risk(phi, s, Psi, base, y, got, thr)
        if prev - f < tol:
            break
        prev = f
    return phi, s, float(risk(phi, s, Psi, base, y, got, thr))


def fit_block(A, base_off, y, got, thr, sigma, phi0=None, iters=40):
    """Minimise the risk over one block of coefficients with everything else
    frozen and the scale fixed.

    At fixed scale the risk is convex in the predicted mean, so damped Newton
    converges from any start.  This is the *conditional* oracle the admission
    criterion is about: the best the rung can do given the simulator as it
    currently stands, not the best the whole ladder can do.
    """
    n, p = A.shape
    phi = np.zeros(p) if phi0 is None else np.asarray(phi0, float).copy()
    for _ in range(iters):
        mu = base_off + A @ phi
        g = np.empty(n); w = np.empty(n)
        r = (y[got] - mu[got]) / sigma
        g[got] = -r / sigma
        w[got] = 1.0 / sigma ** 2
        z = (thr[~got] - mu[~got]) / sigma
        lam = _mills(z)
        g[~got] = lam / sigma
        w[~got] = lam * (lam + z) / sigma ** 2
        grad = A.T @ g / n
        H = (A * w[:, None]).T @ A / n
        # The centred gateway indicators sum to zero, so the block is rank
        # deficient by one.  The Newton step must be taken on the range space
        # only: keeping the numerically-null direction makes the step explode
        # along a direction that does not change the prediction at all.  A
        # truncated pseudo-inverse does exactly that.
        step = np.linalg.pinv(H, rcond=1e-8) @ grad
        # trust region: the quadratic model is only local, and a step of tens of
        # decibels leaves the region where it means anything
        big = np.max(np.abs(step))
        if big > 8.0:
            step = step * (8.0 / big)
        f0 = risk(phi, sigma, A, base_off, y, got, thr)
        t = 1.0
        for _ls in range(25):
            if risk(phi - t * step, sigma, A, base_off, y, got, thr) <= f0 - 1e-14:
                break
            t *= 0.5
        phi = phi - t * step
        if np.max(np.abs(t * step)) < 1e-9:
            break
    mu = base_off + A @ phi
    w = np.empty(n)
    w[got] = 1.0 / sigma ** 2
    z = (thr[~got] - mu[~got]) / sigma
    lam = _mills(z)
    w[~got] = lam * (lam + z) / sigma ** 2
    H = (A * w[:, None]).T @ A / n
    return phi, H


def fit_offsets(gw, base_off, y, got, thr, sigma, n_gw, iters=40):
    """Conditional oracle for the per-receiver offset rung.

    The rung's features are *centred* indicators, so the shift it applies to a
    link is its receiver's coefficient minus their mean: the block can express
    any mean-zero pattern of offsets and nothing else, the common part being the
    clutter rung's job.  The oracle is therefore the minimiser of the risk over
    the mean-zero subspace, not the unconstrained per-receiver minimiser.

    Within that subspace the loss is still separable across receivers, so a
    Newton step can be computed one receiver at a time and then projected onto
    the subspace.  Projected Newton on a convex objective converges, and each
    iteration costs one pass over the links.
    """
    gw = np.asarray(gw, int)
    order = np.argsort(gw, kind="stable")
    starts = np.searchsorted(gw[order], np.arange(n_gw + 1))
    idxs = [order[starts[g]: starts[g + 1]] for g in range(n_gw)]
    s = np.zeros(n_gw)

    def total_risk(v):
        return risk(v, sigma, np.zeros((0, n_gw)), base_off, y, got, thr) \
            if False else _shift_risk(v, gw, base_off, y, got, thr, sigma)

    f_cur = _shift_risk(s, gw, base_off, y, got, thr, sigma)
    for _ in range(iters):
        grad = np.zeros(n_gw); hess = np.zeros(n_gw)
        mu_all = base_off + s[gw]
        for g in range(n_gw):
            idx = idxs[g]
            if len(idx) == 0:
                hess[g] = 1.0
                continue
            mu = mu_all[idx]; gg = got[idx]
            r = (y[idx][gg] - mu[gg]) / sigma
            z = (thr[idx][~gg] - mu[~gg]) / sigma
            lam = _mills(z)
            grad[g] = (-np.sum(r) + np.sum(lam)) / sigma
            hess[g] = (np.sum(gg) + np.sum(lam * (lam + z))) / sigma ** 2
        # Newton step of the quadratic model restricted to the mean-zero
        # subspace: minimise grad^T d + d^T diag(hess) d / 2 subject to
        # sum(d) = 0, whose solution is d_g = -(grad_g - lambda)/hess_g with
        # lambda the multiplier that enforces the constraint.  Projecting the
        # unconstrained step instead is not a descent direction, because the
        # projection and the diagonal scaling do not commute.
        h = np.maximum(hess, 1e-12)
        lam_c = np.sum(grad / h) / np.sum(1.0 / h)
        step = (grad - lam_c) / h
        step = np.clip(step, -20.0, 20.0)
        t = 1.0
        for _ls in range(25):
            cand = s - t * step
            f_new = _shift_risk(cand, gw, base_off, y, got, thr, sigma)
            if f_new <= f_cur - 1e-14:
                break
            t *= 0.5
        else:
            break
        s, f_cur = s - t * step, f_new
        if np.max(np.abs(t * step)) < 1e-9:
            break
    return s - s.mean()


def _shift_risk(s, gw, base_off, y, got, thr, sigma):
    """Mean censored loss when each link is shifted by its receiver's offset."""
    mu = base_off + np.asarray(s, float)[np.asarray(gw, int)]
    n = len(mu)
    r = (y[got] - mu[got]) / sigma
    z = (thr[~got] - mu[~got]) / sigma
    return float((np.sum(0.5 * r ** 2 + np.log(sigma) + 0.5 * LOG2PI)
                  - np.sum(log_ndtr(z))) / n)


def fit_best(Psi, base, y, got, thr, phi0=None, sigma0=15.0):
    """Alternating Newton to get into the right basin, then a joint quasi-Newton
    polish over (phi, log s).

    Alternating between the coefficients and the scale is fast but stalls when
    the two are strongly coupled -- and they are, because a larger clutter
    offset and a larger residual scale explain the same missing links.  The
    polish step optimises them jointly from a point the alternation has already
    made good, which is both quick and enough.
    """
    p0, s0, f0 = fit_newton(Psi, base, y, got, thr, phi0=phi0, sigma0=sigma0,
                            outer=4, inner=15)
    x0 = np.append(p0, np.log(s0))
    r = optimize.minimize(nll_and_grad, x0, args=(Psi, base, y, got, thr),
                          jac=True, method="L-BFGS-B",
                          options=dict(maxiter=3000, maxfun=6000,
                                       ftol=1e-14, gtol=1e-9))
    if r.fun <= f0:
        return r.x[:-1], float(np.exp(r.x[-1])), float(r.fun)
    return p0, s0, f0


def fit(Psi, base, y, got, thr, x0=None, maxiter=1200, restarts=(8.0, 15.0, 25.0)):
    """Censored maximum likelihood.  Returns (phi, sigma, result).

    The likelihood is not concave in (phi, log s) jointly, and a cold start can
    settle in a poor basin, so the fit is repeated from several scale
    initialisations and the best is kept.
    """
    p = Psi.shape[1]
    starts = []
    if x0 is not None:
        starts.append(np.asarray(x0, float))
    for s0 in restarts:
        z = np.zeros(p + 1); z[-1] = np.log(s0)
        starts.append(z)
    best = None
    for st in starts:
        r = optimize.minimize(nll_and_grad, st, args=(Psi, base, y, got, thr),
                              jac=True, method="L-BFGS-B",
                              options=dict(maxiter=maxiter, maxfun=4 * maxiter,
                                           ftol=1e-12, gtol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best.x[:-1], float(np.exp(best.x[-1])), best


def fit_ladder(Psi, base, y, got, thr, level_cols, n_features):
    """Fit every level of a ladder, warm-starting each from the level below.

    Nested maximum likelihood cannot get worse as columns are added, so a level
    whose fitted likelihood is worse than the level below it has not converged;
    warm starting removes that failure mode, and the caller can assert
    monotonicity as a check.
    """
    out = []
    prev = None
    for lv, cols in enumerate(level_cols):
        if len(cols) == 0:
            empty = np.zeros((len(base), 0))
            _, sig, f = fit_newton(empty, base, y, got, thr)
            phi = np.zeros(n_features)
            out.append((phi, sig, f))
            prev = (phi, sig)
            continue
        phi0 = None if prev is None else prev[0][cols]
        s0 = 15.0 if prev is None else prev[1]
        p, s, f = fit_best(np.ascontiguousarray(Psi[:, cols]), base, y, got,
                           thr, phi0=phi0, sigma0=s0)
        phi = np.zeros(n_features); phi[cols] = p
        out.append((phi, s, f))
        prev = (phi, s)
    return out


def risk(phi, sigma, Psi, base, y, got, thr):
    """Mean negative log-likelihood at a *given* coefficient vector -- the risk
    a simulator with those coefficients actually pays on real data."""
    theta = np.append(np.asarray(phi, float), np.log(sigma))
    return nll_and_grad(theta, Psi, base, y, got, thr)[0]


def link_loss(phi, sigma, Psi, base, y, got, thr):
    """Per-link negative log-likelihood, for paired tests."""
    mu = base + Psi @ np.asarray(phi, float)
    out = np.empty(len(mu))
    r = (y[got] - mu[got]) / sigma
    out[got] = 0.5 * r ** 2 + np.log(sigma) + 0.5 * LOG2PI
    out[~got] = -log_ndtr((thr[~got] - mu[~got]) / sigma)
    return out


def information(phi, sigma, Psi, base, got, thr, cols=None):
    """H = Psi^T diag(w) Psi / N, the Hessian of the risk in phi."""
    mu = base + Psi @ np.asarray(phi, float)
    w = np.empty(len(mu))
    w[got] = 1.0 / sigma ** 2
    z = (thr[~got] - mu[~got]) / sigma
    lam = _mills(z)
    w[~got] = lam * (lam + z) / sigma ** 2
    A = Psi if cols is None else Psi[:, cols]
    return (A * w[:, None]).T @ A / len(mu)


def detection_rate(phi, sigma, Psi, base, thr):
    """Probability that a link clears the demodulator, under the fitted model."""
    mu = base + Psi @ np.asarray(phi, float)
    return float(np.mean(ndtr((mu - thr) / sigma)))
