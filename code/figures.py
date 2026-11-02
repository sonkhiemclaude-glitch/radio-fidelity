"""Every figure in the manuscript, from the experiments' JSON output."""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import simulator as S
import plotstyle as PS
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

PS.use_paper_style()
P = PS.PALETTE



def legend_below(ax, ncol=3, fig=None, y=-0.30, fontsize=6.4):
    """Legend outside the axes, centred beneath it.

    Elsevier figures are small and a legend inside the data area competes with
    it; below the plot the reader's eye finishes the curve first and reads the
    key second."""
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=ncol, fontsize=fontsize, frameon=False,
              borderaxespad=0.0, handlelength=1.6, columnspacing=1.2)


def figure_legend_below(fig, ax, ncol=4, fontsize=6.4, y=0.0):
    """One key for a whole multi-panel figure, under the panels."""
    h, l = ax.get_legend_handles_labels()
    if not h:
        return
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
               fontsize=fontsize, frameon=False, handlelength=1.6,
               columnspacing=1.3)


def load(name):
    p = os.path.join(D.RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_selection(e1):
    f = e1.get("selection_figure")
    if f is None:
        raise RuntimeError("exp1 has no selection_figure block")
    d = np.array(f["d"])
    fig, ax = plt.subplots(1, 2, figsize=(PS.COL2, 2.3))
    a = ax[0]
    a.plot(d, f["med_recv"], "o", color=P[0], ms=3.2,
           label="median received power")
    a.plot(d, f["ls_mean"], "--", color=P[1], label="fit on received links")
    a.plot(d, f["tobit_mean"], "-", color=P[2], label="censored fit")
    a.axhline(f["thr"], color="k", lw=.8, ls=":")
    a.text(d[0], f["thr"] + 1.5, "SF7 sensitivity", fontsize=6)
    a.set_xscale("log")
    a.set_xlabel("link distance (m)")
    a.set_ylabel("received power (dBm)")
    legend_below(a, ncol=1, y=-0.34)
    a = ax[1]
    a.plot(d, f["rate"], "o", color=P[0], ms=3.2, label="measured")
    a.plot(d, f["ls_rate"], "--", color=P[1], label="fit on received links")
    a.plot(d, f["tobit_rate"], "-", color=P[2], label="censored fit")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("link distance (m)")
    a.set_ylabel("probability a gateway hears the message")
    legend_below(a, ncol=1, y=-0.34)
    PS.save(fig, os.path.join(D.FIGS, "fig_selection"))


def fig_ladder(e2):
    lv = list(range(len(e2["levels"])))
    tags = [n.split()[0] for n in e2["levels"]]

    def pick(regime, mode):
        out = []
        for l in lv:
            for r in e2["rows"]:
                if (r["regime"] == regime and r["mode"] == mode
                        and r["level"] == l):
                    out.append(r)
        return out

    fig, ax = plt.subplots(1, 2, figsize=(PS.COL2, 2.5))
    for i, (key, lab) in enumerate((("nll", "censored log-loss per link"),
                                    ("err50", "median positioning error (m)"))):
        a = ax[i]
        for j, (regime, mode, name, ls) in enumerate((
                ("uninformed", "oracle", "oracle coefficients", "-"),
                ("uninformed", "guess", "literature prior, one draw", "--"),
                ("uninformed", "randomised", "literature prior, randomised", "-."),
                ("informed", "guess", "with a site survey", ":"))):
            rows = pick(regime, mode)
            v = [r[key] for r in rows]
            a.plot(lv, v, color=P[j], linestyle=ls, marker=PS.MARKERS[j],
                   label=name)
            if mode == "guess":
                # standard error of the mean curve, not the spread of the
                # draws: the draws are paired across levels, so the spread
                # within a level says nothing about the increment between two
                sem = [r[key + "_sd"] / np.sqrt(max(r["reps"], 1)) for r in rows]
                a.fill_between(lv, np.array(v) - np.array(sem),
                               np.array(v) + np.array(sem), color=P[j],
                               alpha=.18, lw=0)
        if key == "err50":
            a.axhline(e2["ceiling"]["err50"], color="k", lw=.8, ls=(0, (1, 1)))
            a.text(0.02, e2["ceiling"]["err50"] * 1.06,
                   "real-data ceiling", transform=a.get_yaxis_transform(),
                   fontsize=6.5)
        a.set_xticks(lv); a.set_xticklabels(tags)
        a.set_xlabel("simulator fidelity level")
        a.set_ylabel(lab)
        a.set_yscale("log")
    legend_below(ax[0], ncol=2, y=-0.22)
    PS.save(fig, os.path.join(D.FIGS, "fig_ladder"))


def fig_criterion(e3):
    off = e3["offset"]
    mus = np.array(off["mus"]); taus = np.array(off["taus"])
    meas = np.array(off["measured"]); pred = np.array(off["predicted"])
    beta = off["beta"]
    fig, ax = plt.subplots(1, 2, figsize=(PS.COL2, 2.45))

    a = ax[0]
    lim = np.abs(meas).max()
    im = a.pcolormesh(mus, taus, meas, cmap="RdBu_r", vmin=-lim, vmax=lim,
                      shading="nearest")
    a.contour(mus, taus, meas, levels=[0.0], colors="k", linewidths=1.1)
    th = np.linspace(0, np.pi, 200)
    a.plot(beta + beta * np.cos(th), beta * np.sin(th), color=P[2], lw=1.4,
           ls="--")
    a.plot([beta], [0], marker="*", color=P[2], ms=7, ls="none")
    a.set_xlabel(r"prior mean $m$ (dB)")
    a.set_ylabel(r"prior width $\tau$ (dB)")
    a.set_title("measured change in risk", fontsize=8)
    a.set_xlim(mus.min(), mus.max()); a.set_ylim(taus.min(), taus.max())
    fig.colorbar(im, ax=a, pad=.02, fraction=.046)

    a = ax[1]
    a.plot(pred.ravel(), meas.ravel(), "o", ms=2.6, color=P[0], alpha=.75)
    lo = min(pred.min(), meas.min()); hi = max(pred.max(), meas.max())
    a.plot([lo, hi], [lo, hi], "k-", lw=.8)
    a.set_xlabel("predicted change (quadratic surrogate)")
    a.set_ylabel("measured change")
    a.set_title(f"sign agreement {100*off['sign_agreement']:.0f}%", fontsize=8)
    PS.save(fig, os.path.join(D.FIGS, "fig_criterion"))


def fig_survey(e3):
    gw = e3["gateway"]
    s = np.array(gw["s"])
    fig, a = plt.subplots(figsize=(PS.COL1, 1.95))
    a.plot(s, gw["measured"], "-o", color=P[0], label="measured")
    a.plot(s, gw["predicted"], "--", color=P[1], label="predicted")
    a.axhline(0, color="k", lw=.7)
    a.axvline(gw["s_pred"], color=P[2], lw=1.0, ls=":")
    a.text(gw["s_pred"], a.get_ylim()[1] * .72,
           rf"  $s^\star={gw['s_pred']:.1f}$ dB", fontsize=6.6, color=P[2])
    a.set_xlabel("site-survey error per gateway (dB)")
    a.set_ylabel("change in risk")
    legend_below(a, ncol=2, y=-0.24)
    PS.save(fig, os.path.join(D.FIGS, "fig_survey"))


def fig_test(e4):
    rows = e4["rows"]
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(PS.COL2, 2.2))
    a = ax[0]
    a.plot(n, [r["test_mean_level"] for r in rows], "-o", color=P[0],
           label="admission test")
    a.plot(n, [r["argmin_mean_level"] for r in rows], "--s", color=P[1],
           label="empirical argmin")
    a.axhline(e4["best_level"], color="k", lw=.8, ls=":")
    a.text(n[0], e4["best_level"] + .12, "true argmin", fontsize=6.5)
    a.set_xscale("log"); a.set_xlabel("real link measurements $n$")
    a.set_ylabel("selected level")
    a = ax[1]
    a.plot(n, [100 * r["test_harmful"] for r in rows], "-o", color=P[0],
           label="admission test")
    a.plot(n, [100 * r["argmin_harmful"] for r in rows], "--s", color=P[1],
           label="empirical argmin")
    a.axhline(100 * e4["alpha"], color="k", lw=.8, ls=":")
    a.text(n[0], 100 * e4["alpha"] * 1.15, r"$\alpha$", fontsize=7)
    a.set_xscale("log"); a.set_xlabel("real link measurements $n$")
    a.set_ylabel("harmful level admitted (\\%)")
    figure_legend_below(fig, ax[0], ncol=2, y=-0.04)
    PS.save(fig, os.path.join(D.FIGS, "fig_test"))


def fig_unlabelled(e5):
    rows = e5["rows"]
    lv = [r["level"] for r in rows]
    tags = [r["name"].split()[0] for r in rows]
    keys = [("loc_err", "positioning error (m)", True),
            ("rssi_risk", "censored log-loss", True),
            ("w1", r"$W_1$ to measured RSSI", False),
            ("mmd2", "MMD$^2$ of fingerprints", False),
            ("domain_auc", "domain-classifier AUC", False),
            ("det_gap", "detection-rate gap", False)]
    loc_best = int(np.nanargmin([r["loc_err"] for r in rows]))
    risk_best = int(np.nanargmin([r["rssi_risk"] for r in rows]))
    fig, ax = plt.subplots(2, 3, figsize=(PS.COL2, 3.2), sharex=True)
    for k, (key, lab, truth) in enumerate(keys):
        a = ax[k // 3][k % 3]
        v = [r[key] for r in rows]
        a.plot(lv, v, "-o", color=P[0] if truth else P[1])
        j = int(np.nanargmin(v))
        a.plot([lv[j]], [v[j]], marker="*", ms=9,
               color=P[2] if truth else P[3], ls="none")
        if not truth:
            # where the two ground truths put their optimum, for comparison
            a.axvline(loc_best, color=P[2], lw=.8, ls="--", alpha=.7)
            a.axvline(risk_best, color=P[0], lw=.8, ls=":", alpha=.7)
        a.set_ylabel(lab, fontsize=6.8)
        a.set_xticks(lv); a.set_xticklabels(tags, fontsize=6)
        if truth:
            a.set_title("ground truth", fontsize=7, color=P[0])
        else:
            a.set_title("unlabelled proxy", fontsize=7, color=P[1])
    ax[0][0].text(0.03, 0.06, "dashed: best for positioning\ndotted: best for likelihood",
                  transform=ax[0][0].transAxes, fontsize=5.6)
    PS.save(fig, os.path.join(D.FIGS, "fig_unlabelled"))


def fig_foundation(e6):
    rows = e6["rows"]
    lv = [r["level"] for r in rows]
    tags = [r["name"].split()[0] for r in rows]
    fig, a = plt.subplots(figsize=(PS.COL1 * 1.35, 2.1))
    for j, n in enumerate(e6["n_probe"]):
        v = [r["err"][str(n)] for r in rows]
        a.plot(lv, v, color=P[j % len(P)],
               linestyle=PS.LINES[j % len(PS.LINES)],
               marker=PS.MARKERS[j % len(PS.MARKERS)],
               label=("zero-shot" if n == 0 else f"$n={n}$"))
    a.set_xticks(lv); a.set_xticklabels(tags)
    a.set_xlabel("simulator level the encoder was pretrained on")
    a.set_ylabel("median positioning error (m)")
    a.set_yscale("log")
    legend_below(a, ncol=3, y=-0.26)
    PS.save(fig, os.path.join(D.FIGS, "fig_foundation"))


def fig_mechanism(e10):
    """Is the top rungs' downstream advantage fidelity or compensation?"""
    rows = e10["rows"]
    sc = np.array([r["scale"] for r in rows])
    err = np.array([r["err"] for r in rows])        # (scale, level)
    den = np.array([r["density"] for r in rows])
    real = e10["real_density"]
    nz = sc > 0
    fig, ax = plt.subplots(1, 2, figsize=(PS.COL2, 2.3))

    a = ax[0]
    for lv in (2, 3, 4, 5):
        a.plot(sc[nz], err[nz, lv], marker=PS.MARKERS[lv % len(PS.MARKERS)],
               ms=3.2, color=P[lv % len(P)],
               linestyle=PS.LINES[lv % len(PS.LINES)], label=f"F{lv}")
    a.axhline(err[-1, 3], color="k", lw=.8, ls=":")
    a.text(sc[nz].min(), err[-1, 3] * 1.05,
           "rungs switched off entirely", fontsize=5.8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("geometric priors, scaled towards the fitted values")
    a.set_ylabel("median positioning error (m)")
    a.invert_xaxis()
    legend_below(a, ncol=4, y=-0.30)

    a = ax[1]
    mm = np.abs(np.log((den + 1e-3) / real))
    for lv in range(err.shape[1]):
        a.plot(mm[:, lv], err[:, lv], ls="none",
               marker=PS.MARKERS[lv % len(PS.MARKERS)], ms=3.4,
               color=P[lv % len(P)], label=f"F{lv}")
    a.set_xlabel("sparsity mismatch  $|\\log(\\mathrm{sim}/\\mathrm{real})|$")
    a.set_ylabel("median positioning error (m)")
    a.set_yscale("log")
    a.text(0.03, 0.05,
           f"Spearman {e10['mismatch']['rho']:.2f} over {e10['mismatch']['n']} cells\n"
           f"within a level, {e10['mismatch']['within_level_mean']:+.2f}",
           transform=a.transAxes, fontsize=6)
    legend_below(a, ncol=6, y=-0.30)
    PS.save(fig, os.path.join(D.FIGS, "fig_mechanism"))


def fig_family(e5):
    """The unlabelled proxies over a family of simulators rather than six."""
    fam = e5.get("family")
    fc = e5.get("family_corr")
    if not fam or not fc:
        raise RuntimeError("exp5 has no family arm")
    keys = [("w1", r"$W_1$"), ("mmd2", "MMD$^2$"),
            ("domain_auc", "domain AUC"), ("det_gap", "reception gap")]
    fig, ax = plt.subplots(1, 2, figsize=(PS.COL2, 2.2))
    a = ax[0]
    y = np.arange(len(keys))
    for j, (k, lab) in enumerate(keys):
        for off, tgt, col, mk in ((-0.16, "rssi_risk", P[0], "o"),
                                  (0.16, "loc_err", P[2], "s")):
            v = fc[k][tgt]
            a.plot([v["lo"], v["hi"]], [j + off] * 2, color=col, lw=1.6)
            a.plot([v["rho"]], [j + off], marker=mk, ms=4, color=col, ls="none")
    a.axvline(0, color="k", lw=.8, ls=":")
    a.set_yticks(y); a.set_yticklabels([l for _, l in keys], fontsize=6.5)
    a.set_xlabel("Spearman correlation with the truth")
    a.set_xlim(-1.05, 1.05)
    a.plot([], [], "o", color=P[0], label=r"vs.\ link risk")
    a.plot([], [], "s", color=P[2], label=r"vs.\ positioning error")
    legend_below(a, ncol=2, y=-0.24)
    a.set_title(f"{e5['n_family']} simulators, bootstrap 95\\%", fontsize=7)

    a = ax[1]
    lv = np.array([r["level"] for r in fam])
    x = np.array([r["det_gap"] for r in fam])
    yy = np.array([r["loc_err"] for r in fam])
    for l in np.unique(lv):
        m = lv == l
        a.plot(x[m], yy[m], ls="none", marker=PS.MARKERS[l % len(PS.MARKERS)],
               ms=3.4, color=P[l % len(P)], label=f"F{l}")
    a.set_xlabel("reception-rate mismatch")
    a.set_ylabel("median positioning error (m)")
    a.set_yscale("log")
    legend_below(a, ncol=6, y=-0.24)
    PS.save(fig, os.path.join(D.FIGS, "fig_family"))


def fig_indoor(e7):
    lv = list(range(len(e7["levels"])))
    tags = [n.split()[0] for n in e7["levels"]]
    fig, ax = plt.subplots(1, 2, figsize=(PS.COL2, 2.2))
    for i, (key, lab) in enumerate((("rssi_rmse", "received-power RMSE (dB)"),
                                    ("err50", "median positioning error (m)"))):
        a = ax[i]
        for j, mode in enumerate(("oracle", "guess", "randomised")):
            rows = [r for r in e7["rows"] if r["mode"] == mode]
            rows.sort(key=lambda r: r["level"])
            a.plot(lv, [r[key] for r in rows], color=P[j],
                   linestyle=PS.LINES[j], marker=PS.MARKERS[j], label=mode)
        if key == "err50":
            a.axhline(e7["ceiling"]["err50"], color="k", lw=.8, ls=(0, (1, 1)))
        a.set_xticks(lv); a.set_xticklabels(tags, fontsize=6.4)
        a.set_xlabel("indoor fidelity level")
        a.set_ylabel(lab)
    figure_legend_below(fig, ax[0], ncol=3, y=-0.03)
    PS.save(fig, os.path.join(D.FIGS, "fig_indoor"))


def fig_premise(e1):
    rows = e1["rungs"]
    tags = [r["rung"].split()[0] for r in rows]
    rho = [r["rho"] for r in rows]
    fig, a = plt.subplots(figsize=(PS.COL1, 1.9))
    a.bar(tags, rho, color=[P[2] if v < 1 else P[1] for v in rho], width=.6)
    a.axhline(1.0, color="k", lw=.9, ls="--")
    a.set_yscale("log")
    a.set_ylabel(r"knowledge ratio $\rho$")
    a.text(0.02, 0.90, r"$\rho<1$: model it", transform=a.transAxes, fontsize=6.8)
    a.text(0.02, 0.78, r"$\rho>1$: do not", transform=a.transAxes, fontsize=6.8)
    PS.save(fig, os.path.join(D.FIGS, "fig_premise"))


def main():
    os.makedirs(D.FIGS, exist_ok=True)
    for name, fn in (("exp1_premise.json", fig_premise),
                     ("exp1_premise.json", fig_selection),
                     ("exp2_ladder.json", fig_ladder),
                     ("exp3_criterion.json", fig_criterion),
                     ("exp3_criterion.json", fig_survey),
                     ("exp4_admission.json", fig_test),
                     ("exp5_unlabelled.json", fig_unlabelled),
                     ("exp5_unlabelled.json", fig_family),
                     ("exp6_foundation.json", fig_foundation),
                     ("exp10_mechanism.json", fig_mechanism),
                     ("exp7_indoor.json", fig_indoor)):
        e = load(name)
        if e is None:
            print(f"  skip {fn.__name__}: {name} missing")
            continue
        try:
            fn(e)
        except Exception as exc:
            print(f"  FAILED {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
