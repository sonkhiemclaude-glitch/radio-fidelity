"""A radio simulator with an explicit fidelity ladder.

Every rung adds one physical effect, and with it one coefficient the modeller
has to supply.  In the dB domain the effects are additive, so the simulator at
level K is exactly

    RSSI_K(p, g) = FS(p, g) + sum_{k<=K} phi_k psi_k(p, g)          (+ shadowing)

with `psi_k` computable from geometry alone.  That structure is what makes the
admission theory exact rather than approximate: choosing to model effect k is
choosing an estimate of phi_k, and choosing not to model it is choosing zero.

Ladder
------
F0  free space                    no coefficients
F1  + urban clutter offset        C (dB at d0)          psi1 = -1
F2  + excess path-loss exponent   n - 2                 psi2 = -10 log10(d/d0)
F3  + per-gateway offsets         delta_g (G-vector)    psi3 = 1{gateway = g}
F4  + building-crossing count     kappa (dB/building)   psi4 = -n_cross
F5  + obstruction depth           gamma (dB/m)          psi5 = -depth

Every level also carries a residual scale s and, when the simulator is *sampled*
rather than evaluated, a spatially correlated shadowing field of that scale.
The scale is a nuisance rather than a rung: it is present whether or not any
given geometric effect is modelled, so it is not something a modeller can choose
to leave out.

The order is the order in which detail accumulates, from a model that knows
only the frequency to one that traces the line of sight through the city.  Two
choices in it are deliberate.  The clutter offset is its own rung, below the
per-gateway offsets, so that the gateway rung is about the *spread* between
installations and not about the average excess loss of a city -- otherwise the
two are confounded and neither is interpretable.  And the per-gateway rung sits
where it does because it is the effect the guest editors' own measurement study
found to dominate (antenna location and orientation), while being the effect
whose coefficients a modeller is least likely to know.
"""
from __future__ import annotations
import os
import numpy as np

C_LIGHT = 299_792_458.0
F_HZ = 868.1e6                 # EU868 LoRaWAN uplink centre
P_TX_DBM = 14.0                # ERP ceiling in EU868, and the campaign's setting
D0 = 1000.0                    # reference distance, near the median link length

# LoRa demodulator sensitivity by spreading factor at 125 kHz (SX1276 datasheet)
SENS = {7: -123.0, 8: -126.0, 9: -129.0, 10: -132.0, 11: -134.5, 12: -137.0}

LEVELS = ["F0 free space", "F1 +clutter offset", "F2 +exponent",
          "F3 +gateway offsets", "F4 +buildings crossed",
          "F5 +obstruction depth"]
N_LEVELS = len(LEVELS)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
class Geometry:
    """Building occupancy raster with line-of-sight integrals."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        ny, nx = z["occ_shape"]
        self.occ = np.unpackbits(z["occ"])[: int(ny) * int(nx)].astype(bool)
        self.occ = self.occ.reshape(int(ny), int(nx))
        self.x_lo = float(z["x_lo"]); self.y_lo = float(z["y_lo"])
        self.res = float(z["res"])
        self.road_x = z["road_x"]; self.road_y = z["road_y"]
        self.n_buildings = int(z["n_buildings"])
        self.keep_gw = z["keep_gw"]
        self.ny, self.nx = self.occ.shape

    def los(self, x1, y1, x2, y2, chunk: int = 200_000):
        """Line-of-sight integrals for a batch of links.

        Returns (n_cross, depth_m): the number of separate building runs the
        segment enters, and the total length of segment inside buildings.  The
        segment is sampled at the raster pitch, so both are exact up to one
        cell; short links are sampled at least 16 times.
        """
        x1 = np.asarray(x1, float); y1 = np.asarray(y1, float)
        x2 = np.asarray(x2, float); y2 = np.asarray(y2, float)
        n = len(x1)
        n_cross = np.zeros(n, np.int32); depth = np.zeros(n, np.float64)
        d = np.hypot(x2 - x1, y2 - y1)
        # bucket by required sample count so each bucket is one dense array
        need = np.maximum(np.ceil(d / self.res).astype(np.int64), 16)
        buckets = 1 << np.ceil(np.log2(need)).astype(np.int64)
        for S in np.unique(buckets):
            idx = np.flatnonzero(buckets == S)
            for a in range(0, len(idx), max(1, chunk // int(S))):
                b = idx[a: a + max(1, chunk // int(S))]
                t = (np.arange(S, dtype=np.float64) + 0.5) / S     # (S,)
                xs = x1[b, None] + (x2[b] - x1[b])[:, None] * t
                ys = y1[b, None] + (y2[b] - y1[b])[:, None] * t
                i = ((xs - self.x_lo) / self.res).astype(np.int64)
                j = ((ys - self.y_lo) / self.res).astype(np.int64)
                ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
                hit = np.zeros(i.shape, bool)
                hit[ok] = self.occ[j[ok], i[ok]]
                step = (d[b] / S)[:, None]
                depth[b] = (hit * step).sum(1)
                # a "crossing" is a 0 -> 1 transition along the ray
                first = hit[:, :1]
                trans = hit[:, 1:] & ~hit[:, :-1]
                n_cross[b] = first.sum(1) + trans.sum(1)
        return n_cross, depth


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def free_space(d_m: np.ndarray) -> np.ndarray:
    """Friis received power in dBm for an isotropic 14 dBm ERP transmitter."""
    d = np.maximum(np.asarray(d_m, float), 1.0)
    fspl = 20 * np.log10(d) + 20 * np.log10(F_HZ) - 147.55
    return P_TX_DBM - fspl


def n_features(n_gw: int) -> int:
    return 2 + n_gw + 2


def features(d_m, gw, n_cross, depth, n_gw):
    """The design matrix [psi_1 ... psi_K] for one batch of links.

    Column layout: 0 = clutter offset, 1 = exponent, 2..1+n_gw = gateway
    indicators, 2+n_gw = -crossings, 3+n_gw = -depth.  Every column is written
    so that a *positive* coefficient means *more loss*.

    The gateway indicators are centred (1{gw = g} - 1/G) so that they sum to
    zero across gateways and are therefore orthogonal to the clutter offset.
    Without that the two rungs are collinear, the ladder is not identified, and
    "how much does knowing the gateways buy" has no answer.  With it, delta_g
    is a deviation from the average installation, which is exactly the quantity
    a site survey measures.
    """
    d = np.maximum(np.asarray(d_m, float), 1.0)
    n = len(d)
    Psi = np.zeros((n, n_features(n_gw)))
    Psi[:, 0] = -1.0
    Psi[:, 1] = -10 * np.log10(d / D0)
    Psi[:, 2: 2 + n_gw] = -1.0 / n_gw
    Psi[np.arange(n), 2 + np.asarray(gw, int)] += 1.0
    Psi[:, 2 + n_gw] = -np.asarray(n_cross, float)
    Psi[:, 3 + n_gw] = -np.asarray(depth, float)
    return Psi


def level_columns(level: int, n_gw: int) -> np.ndarray:
    """Which columns of `features` are active at a given fidelity level."""
    cols = []
    if level >= 1:
        cols.append(0)
    if level >= 2:
        cols.append(1)
    if level >= 3:
        cols += list(range(2, 2 + n_gw))
    if level >= 4:
        cols.append(2 + n_gw)
    if level >= 5:
        cols.append(3 + n_gw)
    return np.array(cols, int)


def oracle_at_level(Psi, r0, level: int, n_gw: int, ridge: float = 1e-9):
    """Coefficients a modeller would use at this level if the real
    measurements were available.  Refitted per level: the best two-parameter
    model is not the two-parameter restriction of the best five-parameter one.
    """
    cols = level_columns(level, n_gw)
    phi = np.zeros(Psi.shape[1])
    if len(cols) == 0:
        return phi
    A = Psi[:, cols]
    sol, *_ = np.linalg.lstsq(A, -np.asarray(r0, float), rcond=None)
    phi[cols] = sol
    return phi


class ShadowField:
    """Gudmundson-style log-normal shadowing: a zero-mean Gaussian field on the
    plane with exponential spatial correlation, drawn independently per gateway
    and generated by FFT so that a whole city is one array."""

    def __init__(self, x_lo, y_lo, x_hi, y_hi, pitch, sigma, ell, n_gw, seed=0):
        self.x_lo, self.y_lo, self.pitch = x_lo, y_lo, pitch
        nx = int(np.ceil((x_hi - x_lo) / pitch)) + 1
        ny = int(np.ceil((y_hi - y_lo) / pitch)) + 1
        rng = np.random.default_rng(seed)
        # white noise smoothed by an exponential kernel, renormalised to sigma
        kx = np.fft.fftfreq(nx, d=pitch)[None, :]
        ky = np.fft.fftfreq(ny, d=pitch)[:, None]
        k2 = (2 * np.pi) ** 2 * (kx ** 2 + ky ** 2)
        # spectral density of the exponential covariance exp(-r/ell) in 2-D
        S = (2 * np.pi * ell ** 2) / (1 + (ell ** 2) * k2) ** 1.5
        amp = np.sqrt(np.maximum(S, 0))
        fields = []
        for _ in range(n_gw):
            w = rng.standard_normal((ny, nx))
            f = np.real(np.fft.ifft2(np.fft.fft2(w) * amp))
            f *= sigma / (f.std() + 1e-12)
            fields.append(f)
        self.f = np.stack(fields)          # (G, ny, nx)
        self.nx, self.ny = nx, ny

    def __call__(self, x, y, gw):
        i = np.clip(((np.asarray(x) - self.x_lo) / self.pitch).astype(int), 0, self.nx - 1)
        j = np.clip(((np.asarray(y) - self.y_lo) / self.pitch).astype(int), 0, self.ny - 1)
        return self.f[np.asarray(gw, int), j, i]


def rssi(level: int, phi: np.ndarray, d_m, gw, n_cross, depth, n_gw,
         shadow: np.ndarray | None = None) -> np.ndarray:
    """Simulated RSSI at a given fidelity level.

    `phi` is the *full* coefficient vector in the layout of `features`; only
    the columns active at `level` are used, which is exactly the statement that
    omitting an effect means setting its coefficient to zero.  `shadow`, when
    given, is a draw from the residual field -- used to *sample* the simulator
    when a fingerprint database is being built, and omitted when the mean
    function is being scored.
    """
    phi = np.asarray(phi, float)
    d = np.maximum(np.asarray(d_m, float), 1.0)
    out = free_space(d)
    # evaluated term by term rather than through the design matrix: at 3e5
    # links the matrix is 100 MB and is rebuilt on every call, which is the
    # difference between an experiment that runs and one that is killed
    if level >= 1:
        out -= phi[0]
    if level >= 2:
        out += phi[1] * (-10.0 * np.log10(d / D0))
    if level >= 3:
        out += phi[2 + np.asarray(gw, int)] - phi[2: 2 + n_gw].mean()
    if level >= 4:
        out += phi[2 + n_gw] * (-np.asarray(n_cross, float))
    if level >= 5:
        out += phi[3 + n_gw] * (-np.asarray(depth, float))
    if shadow is not None:
        out += shadow
    return out
