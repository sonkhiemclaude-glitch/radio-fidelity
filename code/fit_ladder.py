"""Fit the ladder by censored maximum likelihood and cache the result.

Run once; every experiment reads `data/tobit_fits.npz`.  Levels are fitted
bottom-up and warm-started from the level below, and the fit of each level is
checked against the level below: nested maximum likelihood cannot get worse when
columns are added, so a level whose likelihood is worse has not converged and is
retried from a longer schedule.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import simulator as S
import tobit as TB
import priors as PR


def main():
    t0 = time.time()
    d = D.load(); n_gw = d["n_gw"]
    geom = S.Geometry(os.path.join(D.DATA, "geometry.npz"))
    C = D.candidates(d, geom, cache=os.path.join(D.DATA, "links_cand.npz"))
    Psi = S.features(C["dist"], C["gw"], C["n_cross"], C["depth"], n_gw)
    base = S.free_space(C["dist"])
    print(f"{len(C['dist'])} candidate links, {C['got'].mean()*100:.2f}% received "
          f"({time.time()-t0:.0f}s)", flush=True)

    phis, sigs, nlls = [], [], []
    prev_phi, prev_sig, prev_nll = None, 15.0, np.inf
    for lv in range(S.N_LEVELS):
        cols = S.level_columns(lv, n_gw)
        if len(cols) == 0:
            # F0 has no coefficients at all, so there is nothing to fit: its
            # maximum-likelihood scale is degenerate (the model is so biased
            # that the likelihood prefers to call everything noise), and the
            # meaningful comparison uses the modeller's own scale.
            full = np.zeros(S.n_features(n_gw))
            sig = PR.SIGMA["mu"]
            f = TB.risk(np.zeros(0), sig, np.zeros((len(base), 0)), base,
                        C["y"], C["got"], C["thr"])
            phis.append(full); sigs.append(sig); nlls.append(f)
            prev_phi, prev_sig, prev_nll = full, 15.0, f
            det = TB.detection_rate(full, sig, Psi, base, C["thr"])
            print(f"L0 {S.LEVELS[0]:22s} nll={f:.5f} sigma={sig:6.2f} "
                  f"det={det*100:5.2f}% (scale fixed at the modeller's prior "
                  f"mean)  ({time.time()-t0:.0f}s)", flush=True)
            continue
        A = np.ascontiguousarray(Psi[:, cols])
        p0 = None if prev_phi is None else prev_phi[cols]
        phi, sig, f = TB.fit_best(A, base, C["y"], C["got"], C["thr"],
                                  phi0=p0, sigma0=prev_sig)
        if f > prev_nll + 1e-9:
            print(f"  L{lv} warm start gave {f:.5f} > {prev_nll:.5f}; "
                  f"retrying cold", flush=True)
            alt = TB.fit_best(A, base, C["y"], C["got"], C["thr"],
                              phi0=None, sigma0=15.0)
            if alt[2] < f:
                phi, sig, f = alt
        full = np.zeros(S.n_features(n_gw))
        full[cols] = phi
        det = TB.detection_rate(full, sig, Psi, base, C["thr"])
        phis.append(full); sigs.append(sig); nlls.append(f)
        prev_phi, prev_sig, prev_nll = full, sig, f
        print(f"L{lv} {S.LEVELS[lv]:22s} nll={f:.5f} sigma={sig:6.2f} "
              f"det={det*100:5.2f}% C={full[0]:6.2f} n={2+full[1]:5.2f} "
              f"gwsd={full[2:2+n_gw].std():5.2f} kap={full[2+n_gw]:+.4f} "
              f"gam={full[3+n_gw]:+.5f}  ({time.time()-t0:.0f}s)", flush=True)

    np.savez(os.path.join(D.DATA, "tobit_fits.npz"),
             phi=np.array(phis), sigma=np.array(sigs), nll=np.array(nlls))
    print(f"-> tobit_fits.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
