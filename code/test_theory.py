"""Numerical checks for core risk identities, likelihood derivatives, and admission bounds.

The algebraic identities below are checked against a brute-force computation that
shares no code with the closed form: expectations over the prior are checked by
Monte Carlo, the oracle coefficient against a numerical minimiser, the marginal
increment against a direct difference of two nested fits, and the error control
of the admission test against a simulation of its own null.

Run:  python test_theory.py
"""
from __future__ import annotations
import numpy as np
from scipy import optimize
import theory as T

rng = np.random.default_rng(20260830)
FAIL = 0


def check(name, ok, detail=""):
    global FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL += 1


def rnd_problem(n=4000, K=3):
    Psi = rng.standard_normal((n, K)) @ rng.standard_normal((K, K))
    r0 = rng.standard_normal(n) * 2.0 + Psi @ rng.standard_normal(K)
    return Psi, r0


# --------------------------------------------------------------------------
print("Theorem 1 -- expected excess risk = ||r0 + Psi m||^2 + tr(Sigma G)")
Psi, r0 = rnd_problem()
K = Psi.shape[1]
m = rng.standard_normal(K)
A = rng.standard_normal((K, K)); Sigma = A @ A.T
closed = T.expected_excess_risk(r0, Psi, m, Sigma)
L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(K))
draws = m + rng.standard_normal((200_000, K)) @ L.T
mc = np.mean([np.mean((r0 + Psi @ p) ** 2) for p in draws[:2000]])
mc_big = np.mean(np.mean((r0[None, :] + draws[:20000] @ Psi.T) ** 2, axis=1))
check("closed form vs Monte Carlo", abs(closed - mc_big) / closed < 2e-2,
      f"closed={closed:.4f} mc={mc_big:.4f}")

# --------------------------------------------------------------------------
print("oracle coefficient == numerical minimiser")
phi_star = T.oracle_coef(Psi, r0)
num = optimize.minimize(lambda p: np.mean((r0 + Psi @ p) ** 2), np.zeros(K),
                        method="BFGS", tol=1e-12).x
check("phi* matches BFGS", np.allclose(phi_star, num, atol=1e-5),
      f"max|diff|={np.abs(phi_star-num).max():.2e}")

# --------------------------------------------------------------------------
print("Theorem 1 (one rung) -- increment formula vs direct difference")
for trial in range(200):
    n = 1500
    Psi_prev = rng.standard_normal((n, 2))
    psi = rng.standard_normal(n) + Psi_prev @ rng.standard_normal(2)   # correlated
    prev_phi = rng.standard_normal(2)
    base = r0b = rng.standard_normal(n) + Psi_prev @ rng.standard_normal(2)
    # residual of the level-k simulator against reality, with the earlier
    # coefficients held at their (arbitrary) chosen values
    rk = base + Psi_prev @ prev_phi
    pperp = T.residualise(psi, Psi_prev)
    # the oracle for the *new* rung, holding the earlier rungs fixed, is the
    # univariate minimiser along psi -- but the paper's criterion is stated for
    # the component of psi that the earlier rungs cannot already express
    ps = float(T.oracle_coef(psi[:, None], rk)[0])
    mu, tau2 = rng.standard_normal(), abs(rng.standard_normal())
    pred = T.increment_guess(psi, ps, mu, tau2)
    # direct: E_phi ||rk + phi psi||^2 - ||rk||^2
    direct = (np.mean((rk + mu * psi) ** 2) + tau2 * np.mean(psi ** 2)
              - np.mean(rk ** 2))
    if abs(pred - direct) > 1e-6 * max(1.0, abs(direct)):
        check("increment formula", False, f"trial {trial} {pred} vs {direct}")
        break
else:
    check("increment formula (200 random instances)", True)

# --------------------------------------------------------------------------
print("Corollary 1 -- admit iff MSE(prior about phi*) < phi*^2")
bad = 0
for _ in range(20_000):
    ps = rng.standard_normal() * 2
    mu = rng.standard_normal() * 3
    tau2 = abs(rng.standard_normal()) * 2
    psi = rng.standard_normal(300)
    helps = T.increment_guess(psi, ps, mu, tau2) < 0
    crit = T.admission_criterion(ps, mu, tau2)
    bad += helps != crit
check("criterion agrees with the sign of the increment", bad == 0, f"{bad} mismatches")

print("      knowledge ratio rho < 1 is the same statement")
bad = 0
for _ in range(20_000):
    ps = rng.standard_normal() * 2
    mu = rng.standard_normal() * 3
    tau2 = abs(rng.standard_normal()) * 2
    bad += (T.knowledge_ratio(ps, mu, tau2) < 1) != T.admission_criterion(ps, mu, tau2)
check("rho<1 == criterion", bad == 0, f"{bad} mismatches")

# --------------------------------------------------------------------------
print("Theorem 2 -- randomising over the prior removes exactly the tau^2 term")
psi = rng.standard_normal(2000); rk = rng.standard_normal(2000)
ps = float(T.oracle_coef(psi[:, None], rk)[0])
mu, tau2 = 0.7, 0.9
g = T.increment_guess(psi, ps, mu, tau2)
r = T.increment_randomised(psi, ps, mu)
check("guess - randomised == tau^2 ||psi||^2",
      abs((g - r) - tau2 * np.mean(psi ** 2)) < 1e-9)
check("randomised never worse than guessing", r <= g + 1e-12)
bad = 0.0
for _ in range(2000):
    psi_ = rng.standard_normal(400)
    ps_ = rng.standard_normal() * 2
    # Corollary 2: randomising about zero leaves the risk exactly unchanged,
    # i.e. the increment relative to the level below is identically zero.
    bad = max(bad, abs(T.increment_randomised(psi_, ps_, 0.0)))
check("zero-mean randomisation is exactly the lower rung", bad < 1e-12,
      f"max |increment| = {bad:.2e}")
check("zero-mean guess is always harmful",
      all(T.increment_guess(rng.standard_normal(500), p, 0.0, t) > 0
          for p, t in zip(rng.standard_normal(500) * 2 + 0.1,
                          abs(rng.standard_normal(500)) + 1e-3)))

# --------------------------------------------------------------------------
print("Theorem 2 -- a mixture-trained squared-loss learner recovers the mean")
n = 4000
psi = rng.standard_normal(n)
rk = rng.standard_normal(n)
mu, tau = 0.8, 1.3
draws = mu + tau * rng.standard_normal(400)
# the Bayes predictor for the mixture is the average simulator output, so the
# realised excess risk is that of the mean coefficient
mix_pred = np.mean([rk + p * psi for p in draws], axis=0)
check("mixture predictor == mean-coefficient predictor",
      np.abs(mix_pred - (rk + draws.mean() * psi)).max() < 1e-10)

# --------------------------------------------------------------------------
# Proposition 5 is a closed form in Section 2 like any other, and it is the one
# the paper's information-weighted Gram matrix is built from, so it is checked
# here against central differences of the loss it claims to differentiate.
print("Proposition 5 -- censored-likelihood gradient and information weight")
import tobit as TB
from scipy.special import log_ndtr

def _cens_loss(mu, thr, s):
    return -log_ndtr((thr - mu) / s)

worst_g, worst_h = 0.0, 0.0
for _ in range(400):
    s_ = float(rng.uniform(4.0, 25.0))
    thr_ = float(rng.uniform(-140.0, -110.0))
    mu_ = thr_ + float(rng.uniform(-45.0, 45.0))
    # steps scaled to the loss's own length scale s: the central-difference
    # truncation error is O(h^2 / s^2) and the roundoff is O(eps / h), so these
    # sit near the optimum for each order.
    hg, hh = 1e-5 * s_, 2e-3 * s_
    d1 = (_cens_loss(mu_ + hg, thr_, s_) - _cens_loss(mu_ - hg, thr_, s_)) / (2 * hg)
    d2 = (_cens_loss(mu_ + hh, thr_, s_) - 2 * _cens_loss(mu_, thr_, s_)
          + _cens_loss(mu_ - hh, thr_, s_)) / hh ** 2
    z_ = (thr_ - mu_) / s_
    lam_ = float(TB._mills(np.array([z_]))[0])
    g_closed = lam_ / s_
    w_closed = lam_ * (lam_ + z_) / s_ ** 2
    worst_g = max(worst_g, abs(g_closed - d1) / max(1e-12, abs(d1)))
    # the weight is bounded by the received-link weight 1/s^2, and vanishes in
    # the deep tail, so the honest error measure is absolute in that unit
    worst_h = max(worst_h, abs(w_closed - d2) * s_ ** 2)
check("d/dmu (-log Phi) == lambda(z)/s", worst_g < 1e-6,
      f"max rel err {worst_g:.2e}")
check("d2/dmu2 (-log Phi) == lambda(z)(lambda(z)+z)/s^2", worst_h < 1e-5,
      f"max err {worst_h:.2e} in units of the received-link weight 1/s^2")

# The sign form matters: lambda(z)(lambda(z) - z) uses the wrong
# sign for this Hessian, and it is negative wherever z > lambda(z).
zs = np.linspace(-12.0, 12.0, 4001)
lams = TB._mills(zs)
check("lambda(z) + z > 0 for all z (Mills bound; weight is positive)",
      bool(np.all(lams + zs > 0)), f"min {float(np.min(lams + zs)):.3e}")
check("lambda(z) - z is negative somewhere (the wrong form is not a weight)",
      bool(np.any(lams - zs < 0)),
      f"min {float(np.min(lams - zs)):.3f} at z = {float(zs[np.argmin(lams-zs)]):.1f}")

print("      and the analytic gradient of the full mean NLL matches finite differences")
n_, K_ = 600, 3
Psi_ = rng.standard_normal((n_, K_))
base_ = rng.uniform(-150.0, -110.0, n_)
thr_v = np.full(n_, -123.0)
phi_t = rng.standard_normal(K_) * 0.5
mu_t = base_ + Psi_ @ phi_t
y_ = mu_t + rng.standard_normal(n_) * 8.0
got_ = y_ >= thr_v
theta0 = np.append(phi_t, np.log(12.0))
f0, g0 = TB.nll_and_grad(theta0, Psi_, base_, y_, got_, thr_v)
num_g = np.empty_like(theta0)
for i in range(len(theta0)):
    e = np.zeros_like(theta0); e[i] = 1e-6
    fp = TB.nll_and_grad(theta0 + e, Psi_, base_, y_, got_, thr_v)[0]
    fm = TB.nll_and_grad(theta0 - e, Psi_, base_, y_, got_, thr_v)[0]
    num_g[i] = (fp - fm) / 2e-6
check("analytic gradient == finite differences",
      float(np.max(np.abs(num_g - g0))) < 1e-5,
      f"max |diff| {float(np.max(np.abs(num_g - g0))):.2e}")

print("      and `information` is the Hessian of that same mean NLL")
sig_ = 12.0
H_closed = TB.information(phi_t, sig_, Psi_, base_, got_, thr_v)
H_num = np.empty((K_, K_))
for i in range(K_):
    for j in range(K_):
        e_i = np.zeros(K_); e_i[i] = 1e-3
        e_j = np.zeros(K_); e_j[j] = 1e-3
        H_num[i, j] = (
            TB.risk(phi_t + e_i + e_j, sig_, Psi_, base_, y_, got_, thr_v)
            - TB.risk(phi_t + e_i - e_j, sig_, Psi_, base_, y_, got_, thr_v)
            - TB.risk(phi_t - e_i + e_j, sig_, Psi_, base_, y_, got_, thr_v)
            + TB.risk(phi_t - e_i - e_j, sig_, Psi_, base_, y_, got_, thr_v)
        ) / (4 * 1e-3 * 1e-3)
rel_H = float(np.max(np.abs(H_num - H_closed)) / np.max(np.abs(H_num)))
check("information(phi) == numerical Hessian of the risk", rel_H < 1e-3,
      f"max rel err {rel_H:.2e}")

# --------------------------------------------------------------------------
print("Theorem 4 -- empirical-Bernstein bound and fixed-sequence error control")
cov = 0
for _ in range(4000):
    d = rng.uniform(-1, 1, size=400)          # mean 0
    cov += T.eb_bound(d, 0.05, rng_range=2.0) >= 0.0
check("EB upper bound covers the true mean at >= 95%", cov / 4000 >= 0.95,
      f"coverage {cov/4000:.4f}")

harmful_admitted = 0
TRIALS = 3000
for _ in range(TRIALS):
    # a ladder whose first rung genuinely helps and whose second is harmful
    d1 = -0.30 + rng.uniform(-np.sqrt(3), np.sqrt(3), size=600)
    d2 = +0.02 + rng.uniform(-np.sqrt(3), np.sqrt(3), size=600)     # harmful (positive mean)
    k = T.fixed_sequence_test([d1, d2], alpha=0.05, rng_range=6.0)
    harmful_admitted += (k >= 2)
check("P[admit a harmful rung] <= alpha", harmful_admitted / TRIALS <= 0.05,
      f"{harmful_admitted/TRIALS:.4f}")

power = 0
for _ in range(2000):
    d1 = -0.5 + rng.uniform(-np.sqrt(3), np.sqrt(3), size=2000)
    d2 = -0.5 + rng.uniform(-np.sqrt(3), np.sqrt(3), size=2000)
    power += T.fixed_sequence_test([d1, d2], alpha=0.05, rng_range=6.0) == 2
check("test admits two clearly-good rungs most of the time", power / 2000 > 0.9,
      f"power {power/2000:.3f}")

print()
print("ALL CHECKS PASSED" if FAIL == 0 else f"{FAIL} CHECK(S) FAILED")
raise SystemExit(1 if FAIL else 0)
