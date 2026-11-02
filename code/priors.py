"""What a modeller knows about each coefficient before measuring anything.

These priors are the load-bearing input of the whole study, so they are set
from published ranges rather than from the data, and each one is sourced in the
comment beside it.  Two regimes are carried throughout:

`uninformed`  what a modeller has who builds a simulator from the literature
              and a map, with no site survey: a defensible central value for
              the effects that generalise, and no location information at all
              for the effects that are site-specific;
`informed`    what the same modeller has after a site survey that measures the
              gateway installations -- the situation the guest editors' study
              creates when it *optimises* antenna position and orientation.

The theory's prediction is that the two regimes disagree about the same rung,
which is the sharpest available test of it.
"""
from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------
# F1: urban clutter offset C (dB of excess loss over free space at d0 = 1 km).
#   Okumura-Hata for a large city at 868 MHz with a 30 m base and a 1.5 m
#   terminal gives about 35 dB more loss than free space at 1 km; the same
#   comparison for "suburban" and "medium city" gives roughly 25-30 dB.  A
#   modeller choosing an urban environment class therefore has a well-centred
#   value and a spread of a few dB.
OFFSET = dict(mu=30.0, tau=6.0)

# --------------------------------------------------------------------------
# F2: excess path-loss exponent, phi = n - 2
#   Urban and suburban 868 MHz LPWAN measurement campaigns report n in roughly
#   2.3-3.5 (Petajajarvi et al. 2015; Callebaut & Van der Perre 2019; the
#   Antwerp campaign's own companion analysis).  A modeller picking a single
#   urban value would take n ~ 3.
EXPONENT = dict(mu=1.0, tau=0.4)

# --------------------------------------------------------------------------
# F3: per-gateway offset delta_g (dB), absorbing antenna gain, height, feeder
#   loss and mounting.  Without a site survey a modeller has no per-gateway
#   information whatsoever, which is a *zero-mean* prior; its width is the
#   spread of plausible installations, ~5 dB.  With a survey, the mean is the
#   surveyed value and the width shrinks to the survey's own error.
GATEWAY_UNINFORMED = dict(mu=0.0, tau=5.0)
GATEWAY_INFORMED = dict(tau=2.0)      # mu supplied per gateway by the survey

# --------------------------------------------------------------------------
# F4: loss per building crossed, kappa (dB/building).  ITU-R P.1411 and urban
#   microcell measurements put the first-building obstruction loss at roughly
#   10-20 dB at UHF, with later buildings contributing less.  In a dense
#   European city a 1 km line of sight crosses of the order of fifteen
#   buildings, and multi-screen diffraction theory (Walfisch-Bertoni, and the
#   COST 231 model built on it) makes the *incremental* loss of each further
#   building row small.  A modeller can therefore reason their way to a few dB,
#   or transplant the single-building penetration figure; both happen, so both
#   are carried.  The first is the declared prior, the second a documented
#   alternative reported in the sensitivity analysis.
CROSSINGS = dict(mu=3.0, tau=2.0)
CROSSINGS_TRANSPLANT = dict(mu=12.0, tau=5.0)   # the single-building figure
CROSSINGS_SMALL = dict(mu=0.5, tau=0.4)         # a modeller who discounts hard

# --------------------------------------------------------------------------
# F5: loss per metre of building material, gamma (dB/m).  Reported through-
#   building attenuation at UHF spans about 0.2-1.5 dB/m depending on
#   construction.
DEPTH = dict(mu=0.6, tau=0.35)
DEPTH_SMALL = dict(mu=0.05, tau=0.04)           # a modeller who discounts hard

# --------------------------------------------------------------------------
# F6: log-normal shadowing.  Urban sigma is usually quoted at 4-8 dB with a
#   decorrelation distance of tens to a couple of hundred metres.
SHADOW = dict(sigma=6.0, sigma_tau=1.5, ell=100.0)


def scalar_prior(name: str, regime: str = "uninformed"):
    if name == "exponent":
        return EXPONENT["mu"], EXPONENT["tau"]
    if name == "crossings":
        return CROSSINGS["mu"], CROSSINGS["tau"]
    if name == "depth":
        return DEPTH["mu"], DEPTH["tau"]
    raise KeyError(name)

# --------------------------------------------------------------------------
# The residual scale s (dB).  Every level of the ladder needs one -- it is the
# spread the mean function does not explain -- and it is a nuisance rather than
# a rung, because it is present whether or not any given effect is modelled.
# Urban log-normal shadowing is usually quoted at 4-8 dB, so a modeller reading
# the literature would take about 8.
SIGMA = dict(mu=8.0, tau=3.0)

# Spatial decorrelation distance of the shadowing field, used when the
# simulator is *sampled* to build a fingerprint database (tens to a couple of
# hundred metres in urban macrocells).
SHADOW_ELL = 100.0
