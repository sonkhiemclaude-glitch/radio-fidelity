"""Closed forms for the fidelity-admission theory, and the estimators the
experiments use.

The whole theory rests on one structural fact about radio models: in the dB
domain every empirical propagation model is *additive* in its effects, so a
fidelity ladder is literally a sequence of nested linear models

    s_K(z) = s_0(z) + sum_{k<=K} phi_k psi_k(z),

in which `psi_k` is a computable geometric feature (log-distance, a per-gateway
indicator, the number of buildings the line of sight crosses, ...) and `phi_k`
is a coefficient the modeller must *supply* -- from the literature, from a
datasheet, or from a guess.  Omitting effect k is not "not modelling" it; it is
modelling it with the estimate phi_k = 0.  Everything below follows from taking
that seriously.

All inner products are in L2(P_Z) for the real link distribution P_Z, estimated
by the empirical average over a real sample.  `Psi` is therefore always an
(n, K) design matrix evaluated at real links, and `r0` the (n,) residual of the
base simulator against the real measurements.
"""
from __future__ import annotations
import numpy as np

__all__ = [
    "gram", "oracle_coef", "excess_risk", "expected_excess_risk",
    "increment_guess", "increment_randomised", "knowledge_ratio",
    "admission_criterion", "residualise", "eb_bound", "fixed_sequence_test",
]


# --------------------------------------------------------------------------
# Population quantities (estimated on a real sample)
# --------------------------------------------------------------------------
def gram(Psi: np.ndarray) -> np.ndarray:
    """G = E[psi psi^T] under the real link distribution."""
    Psi = np.atleast_2d(np.asarray(Psi, float))
    if Psi.ndim == 1:
        Psi = Psi[:, None]
    return Psi.T @ Psi / len(Psi)


def oracle_coef(Psi: np.ndarray, r0: np.ndarray) -> np.ndarray:
    """phi* = argmin_phi ||r0 + Psi phi||^2 = -G^{-1} E[psi r0].

    The *oracle* coefficient: the value a modeller would use if the real
    measurements were available.  It is not available at simulator-design time;
    the experiments estimate it only in order to test the theory.
    """
    Psi = np.atleast_2d(np.asarray(Psi, float))
    if Psi.ndim == 1:
        Psi = Psi[:, None]
    r0 = np.asarray(r0, float)
    G = gram(Psi)
    b = Psi.T @ r0 / len(r0)
    return -np.linalg.solve(G, b)


def excess_risk(r0: np.ndarray, Psi: np.ndarray | None = None,
                phi: np.ndarray | None = None) -> float:
    """||r0 + Psi phi||^2, the excess squared-error risk of the simulator."""
    r = np.asarray(r0, float).copy()
    if Psi is not None and phi is not None:
        Psi = np.atleast_2d(np.asarray(Psi, float))
        if Psi.ndim == 1:
            Psi = Psi[:, None]
        r = r + Psi @ np.atleast_1d(phi)
    return float(np.mean(r ** 2))


def expected_excess_risk(r0, Psi, m, Sigma) -> float:
    """E_pi ||r0 + Psi phi||^2 for phi ~ pi with mean m and covariance Sigma.

    Theorem 1 (general form): = ||r0 + Psi m||^2 + tr(Sigma G).
    """
    m = np.atleast_1d(np.asarray(m, float))
    Sigma = np.atleast_2d(np.asarray(Sigma, float))
    return excess_risk(r0, Psi, m) + float(np.trace(Sigma @ gram(Psi)))


# --------------------------------------------------------------------------
# The one-rung increment: the paper's Theorem 1 and Corollary 1
# --------------------------------------------------------------------------
def residualise(psi: np.ndarray, Psi_prev: np.ndarray | None) -> np.ndarray:
    """Project psi onto the orthogonal complement of the already-modelled
    features.  With correlated rungs it is this residual, not psi itself, that
    the admission criterion is about."""
    psi = np.asarray(psi, float).ravel()
    if Psi_prev is None or np.size(Psi_prev) == 0:
        return psi
    P = np.atleast_2d(np.asarray(Psi_prev, float))
    if P.ndim == 1:
        P = P[:, None]
    coef, *_ = np.linalg.lstsq(P, psi, rcond=None)
    return psi - P @ coef


def increment_guess(psi_perp: np.ndarray, phi_star: float,
                    mu: float, tau2: float) -> float:
    """E[excess risk at level k+1] - excess risk at level k, when the modeller
    *guesses* a single value phi ~ pi with mean mu and variance tau2
    (Theorem 1):

        ||psi||^2 [ (mu - phi*)^2 + tau^2 - phi*^2 ].
    """
    n2 = float(np.mean(np.asarray(psi_perp, float) ** 2))
    return n2 * ((mu - phi_star) ** 2 + tau2 - phi_star ** 2)


def increment_randomised(psi_perp: np.ndarray, phi_star: float,
                         mu: float) -> float:
    """Same increment when the modeller *randomises* over pi and the learner
    fits the mixture (Theorem 2): the tau^2 penalty disappears, because a
    squared-loss learner trained on the mixture recovers the prior mean.

        ||psi||^2 [ (mu - phi*)^2 - phi*^2 ].
    """
    n2 = float(np.mean(np.asarray(psi_perp, float) ** 2))
    return n2 * ((mu - phi_star) ** 2 - phi_star ** 2)


def knowledge_ratio(phi_star: float, mu: float, tau2: float) -> float:
    """rho = MSE(pi about phi*) / phi*^2.  Corollary 1: admit iff rho < 1."""
    if phi_star == 0:
        return np.inf
    return ((mu - phi_star) ** 2 + tau2) / phi_star ** 2


def admission_criterion(phi_star: float, mu: float, tau2: float,
                        randomised: bool = False) -> bool:
    """True iff the rung is predicted to reduce expected real risk."""
    t = 0.0 if randomised else tau2
    return ((mu - phi_star) ** 2 + t) < phi_star ** 2


# --------------------------------------------------------------------------
# The admission test (Theorem 4)
# --------------------------------------------------------------------------
def eb_bound(d: np.ndarray, alpha: float, rng_range: float | None = None) -> float:
    """One-sided empirical-Bernstein upper confidence bound on E[d] from the
    paired loss differences `d`, valid with probability >= 1 - alpha
    (Maurer & Pontil 2009).  `rng_range` is the a-priori range of d; if None it
    is taken from the sample, which makes the bound heuristic rather than
    certified -- the experiments pass an explicit range.
    """
    d = np.asarray(d, float)
    if d.ndim != 1 or not np.all(np.isfinite(d)):
        raise ValueError("Expected a finite one-dimensional sample")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    n = len(d)
    if n < 2:
        return np.inf
    R = float(rng_range) if rng_range is not None else float(d.max() - d.min())
    if R < 0 or (rng_range is not None and np.ptp(d) > R + 1e-10):
        raise ValueError("rng_range must bound the width of paired differences")
    v = float(np.var(d, ddof=1))
    L = np.log(2.0 / alpha)
    return float(d.mean() + np.sqrt(2.0 * v * L / n) + 7.0 * R * L / (3.0 * (n - 1)))


def fixed_sequence_test(diffs: list[np.ndarray], alpha: float = 0.05,
                        rng_range: float | None = None) -> int:
    """Walk the ladder from the bottom, admitting rung k+1 only while the
    one-sided upper confidence bound on E[loss_{k+1} - loss_k] is < 0.  Because
    the rungs are tested in a pre-specified order and the walk stops at the
    first failure, no multiplicity correction is needed and the family-wise
    probability of admitting a harmful rung is at most alpha.

    Returns the index of the highest admitted level (0 = base simulator only).
    """
    k = 0
    for d in diffs:
        if eb_bound(d, alpha, rng_range) < 0.0:
            k += 1
        else:
            break
    return k
