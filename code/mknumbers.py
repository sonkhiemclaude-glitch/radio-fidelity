"""Emit `paper/numbers.tex`: every number the manuscript quotes, as a macro.

The manuscript never hard-codes a figure.  Each one is a macro defined here from
the experiments' own JSON output, so the text cannot drift from the artefact --
if an experiment is re-run and a number moves, the paper moves with it.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
import simulator as S
import priors as PR

OUT = os.path.join(os.path.abspath(os.environ.get("PHYSICS_PAPER_DIR", os.path.join(os.path.dirname(__file__), "..", "paper"))), "numbers.tex")
LINES: list[str] = []


def load(name):
    p = os.path.join(D.RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


_SEEN: set[str] = set()


def M(name, value, fmt="{}"):
    """First definition wins.  Several experiments legitimately report the same
    quantity, and a LaTeX macro may only be defined once."""
    if name in _SEEN:
        return
    _SEEN.add(name)
    LINES.append(rf"\newcommand{{\{name}}}{{{fmt.format(value)}}}")


def num(name, value, prec=2, comma=False):
    if comma:
        M(name, f"{value:,.0f}".replace(",", "{,}"))
    else:
        M(name, f"{value:.{prec}f}")


def main():
    e1 = load("exp1_premise.json")
    e2 = load("exp2_ladder.json")
    e3 = load("exp3_criterion.json")
    e4 = load("exp4_admission.json")
    e5 = load("exp5_unlabelled.json")
    e6 = load("exp6_foundation.json")
    e7 = load("exp7_indoor.json")
    e8 = load("exp8_robustness.json")
    e9 = load("exp9_stability.json")
    e10 = load("exp10_mechanism.json")

    # ---------------- corpus and geometry ----------------------------------
    if e1:
        num("nMsg", e1["n_msg"], comma=True)
        num("nLinksRecv", e1["n_links_recv"], comma=True)
        num("nGw", e1["n_gw"], 0)
        num("nCand", e1["n_cand"], comma=True)
        num("nCandMsg", e1["n_cand_msg"], comma=True)
        num("recvPct", 100 * e1["reception_rate"], 2)
        num("recvComplPct", 100 - 100 * e1["reception_rate"], 2)
        num("nBuildings", e1["n_buildings"], comma=True)
        num("occupancyPct", 100 * e1["occupancy"], 1)
        num("nRoad", e1["n_road"], comma=True)
        num("medDist", e1["median_distance"], 0)
        num("medCross", e1["median_crossings"], 0)
        num("medDepth", e1["median_depth"], 0)
        num("medDistRecv", e1["median_distance_recv"], 0)
        num("medCrossRecv", e1["median_crossings_recv"], 0)
        num("medDepthRecv", e1["median_depth_recv"], 0)
        # the spurious loss the top rung's literature prior adds to a median
        # received link, which is the number "hundreds of decibels" stood for
        num("spuriousLoss", 0.6 * e1["median_depth_recv"], 0)
        num("spuriousLossCross", 3.0 * e1["median_crossings_recv"], 0)
        t = e1["tobit"]; l = e1["ls_received"]; b = e1["selection_bias"]
        num("tobitC", t["C"], 1); num("tobitN", t["n"], 2)
        num("tobitGwSd", t["gw_sd"], 2); num("tobitGwRange", t["gw_range"], 1)
        num("kappaStar", t["kappa"], 3); num("gammaStar", t["gamma"], 5)
        num("tobitSigma", t["sigma"], 1)
        num("lsC", l["C"], 1); num("lsN", l["n"], 2)
        num("lsKappa", l["kappa"], 3); num("lsSigma", l["sigma"], 1)
        num("selBiasOffset", b["offset"], 1)
        num("selBiasExp", b["exponent"], 2)
        num("surveyThresh", e1["survey_threshold"], 2)
        num("gwSpread", e1["gw_spread"], 2)
        num("sigmaModeller", e1["sigma_modeller"], 0)
        # LaTeX command names may only contain letters, so F1 becomes Fone
        WORD = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}
        ratios = []
        for r in e1["rungs"]:
            tag = "F" + WORD[r["rung"].split()[0][1]]
            num(f"rho{tag}", r["rho"], 3 if r["rho"] < 100 else 0)
            num(f"rhoQ{tag}", r["rho_quadratic"], 3 if r["rho_quadratic"] < 100 else 0)
            M(f"admit{tag}", "admit" if r["admit_guess"] else "reject")
            num(f"beta{tag}", r["beta_summary"], 3)
            if np.isfinite(r["rho"]) and r["rho"] > 0 and np.isfinite(r["rho_quadratic"]):
                ratios.append(max(r["rho_quadratic"] / r["rho"],
                                  r["rho"] / r["rho_quadratic"]))
        if ratios:
            num("surrogateRatio", max(ratios), 1)
        agree = all((r["rho"] < 1) == (r["rho_quadratic"] < 1) for r in e1["rungs"])
        M("surrogateAgree", "every" if agree else "not every")
        det = [x["detection"] for x in e1["ladder"]]
        num("detTop", 100 * det[-1], 2)

    # ---------------- ladder ------------------------------------------------
    if e2:
        def pick(regime, mode, level):
            for r in e2["rows"]:
                if (r["regime"] == regime and r["mode"] == mode
                        and r["level"] == level):
                    return r
            return None
        g = [pick("uninformed", "guess", lv) for lv in range(len(e2["levels"]))]
        o = [pick("uninformed", "oracle", lv) for lv in range(len(e2["levels"]))]
        inf = [pick("informed", "guess", lv) for lv in range(len(e2["levels"]))]
        nll = [x["nll"] for x in g]
        best = int(np.argmin(nll))
        num("minLevel", best, 0)
        M("minLevelName", "at " + e2["levels"][best].split()[0])
        num("nllBest", nll[best], 4)
        num("nllTop", nll[-1], 4)
        num("nllBase", nll[0], 4)
        num("nllPenaltyPct", 100 * (nll[-1] - nll[best]) / abs(nll[best]), 1)
        M("nllPenalty", f"{nll[-1] / nll[best]:.0f}$\\times$")
        err = [x["err50"] for x in g]
        bestl = int(np.nanargmin(err))
        num("locMinLevel", bestl, 0)
        M("locMinLevelName", e2["levels"][bestl].split()[0])
        num("locBest", err[bestl], 0)
        num("locTop", err[-1], 0)
        num("locBase", err[0], 0)
        num("locCeiling", e2["ceiling"]["err50"], 0)
        errI = [x["err50"] for x in inf]
        num("locInformedBest", np.nanmin(errI), 0)
        num("surveyBefore", g[3]["nll"], 3)
        num("surveyAfter", inf[3]["nll"], 3)
        num("surveyMinLevel", int(np.argmin([x["nll"] for x in inf])), 0)
        oerr = [x["err50"] for x in o]
        num("locOracleBest", np.nanmin(oerr), 0)
        M("locOracleLevelName", e2["levels"][int(np.nanargmin(oerr))].split()[0])
        num("divergenceFactor", err[best] / err[bestl], 1)
        onll = [x["nll"] for x in o]
        num("nllOracleTop", onll[-1], 4)
        num("nQueryLoc", e2["n_query"], comma=True)
        num("nRefLoc", e2["n_ref"], comma=True)
        num("nRep", e2["n_rep"], 0)
        pr = [p for p in e2.get("pairs", []) if p["regime"] == "uninformed"]
        for p in pr:
            w = ["one", "two", "three", "four", "five"][p["rung"] - 1]
            num(f"pair{w}Delta", p["nll_delta"], 3)
            num(f"pair{w}Abs", abs(p["nll_delta"]), 3)
            num(f"pair{w}Lo", p["nll_lo"], 3)
            num(f"pair{w}Hi", p["nll_hi"], 3)
            M(f"pair{w}P", "<10^{-4}" if p["nll_p_holm"] < 1e-4
              else f"{p['nll_p_holm']:.4f}")
            M(f"pair{w}Sig", "significant" if p["nll_reject"] else "not significant")
            num(f"pair{w}LocDelta", p["err_delta"], 0)
            num(f"pair{w}LocAbs", abs(p["err_delta"]), 0)
            num(f"pair{w}LocLo", p["err_lo"], 0)
            num(f"pair{w}LocHi", p["err_hi"], 0)
            M(f"pair{w}LocP", "<10^{-4}" if p["err_p_holm"] < 1e-4
              else f"{p['err_p_holm']:.4f}")
            M(f"pair{w}LocSig",
              "significant" if p["err_reject"] else "not significant")
        if pr:
            num("nPairsSig", sum(p["nll_reject"] for p in pr), 0)
            num("nPairs", len(pr), 0)

    # ---------------- criterion --------------------------------------------
    if e3:
        off = e3["offset"]
        num("critBeta", off["beta"], 2)
        # the two distances the theory section quotes, so they cannot drift
        num("critOriginGap", abs(off["beta"]), 0)
        num("critPriorGap", abs(off["beta"] - PR.OFFSET["mu"]), 0)
        num("critSign", 100 * off["sign_agreement"], 1)
        num("critRelErr", 100 * off["rel_err"], 2)
        num("critTauMeasured", max(off["tau_edge"]), 2)
        num("critTauPred", abs(off["beta"]), 2)
        num("critTauRatio", max(off["tau_edge"]) / abs(off["beta"]), 2)
        num("critLocSign", 100 * off["loc_sign_agreement"], 1)
        num("critLocBestC", off["loc_best_c"], 1)
        num("critLocBest", off["loc_best"], 0)
        gw = e3["gateway"]
        num("survPred", gw["s_pred"], 2)
        if gw["s_measured"] is not None:
            num("survMeas", gw["s_measured"], 2)
        num("survSpread", gw["spread_sd"], 2)
        num("survRatio", gw["s_pred"] / gw["spread_sd"], 2)
        num("survPhysRatio", gw["s_pred"] / float(np.load(os.path.join(D.DATA, "tobit_fits.npz"))["phi"][5][2:2 + e1["n_gw"]].std()) if e1 else 0.0, 2)

    # ---------------- admission test ---------------------------------------
    if e4:
        num("testBest", e4["best_level"], 0)
        rows = e4["rows"]
        ok = [r for r in rows if r["test_exact"] >= 0.9]
        num("nTestN", ok[0]["n"] if ok else rows[-1]["n"], comma=True)
        num("testExactPct", 100 * (ok[0]["test_exact"] if ok else 0.0), 1)
        full = [r for r in rows if r["test_exact"] >= 0.999]
        num("nTestNfull", full[0]["n"] if full else rows[-1]["n"], comma=True)
        num("testHarmfulMax", 100 * max(r["test_harmful"] for r in rows), 2)
        num("argminHarmfulMax", 100 * max(r["argmin_harmful"] for r in rows), 1)
        small = rows[0]
        num("testSmallN", small["n"], 0)
        num("testSmallExact", 100 * small["test_exact"], 1)
        num("argminSmallExact", 100 * small["argmin_exact"], 1)
        num("alphaLevel", 100 * e4["alpha"], 0)
        # cost comparison, from the artefact's own timings
        if e2 and "runtime_s" in e2 and "runtime_s" in e4:
            num("sweepMinutes", e2["runtime_s"] / 60.0, 0)
            num("testMillis", 1000 * e4["runtime_s"] / max(e4["n_campaigns"], 1), 1)

    # ---------------- unlabelled -------------------------------------------
    if e5:
        w = e5["witness"]
        num("witnessWone", w["w1_between"], 6)
        num("witnessRatio", w["ratio"], 2)
        c = e5["corr"]
        for k, tag in (("w1", "Wone"), ("mmd2", "Mmd"),
                       ("domain_auc", "Auc"), ("det_gap", "Det")):
            num(f"corr{tag}Risk", c[k]["vs_risk"][0], 2)
            num(f"corr{tag}Loc", c[k]["vs_loc"][0], 2)
        num("corrAbsMax", max(abs(c[k]["vs_loc"][0]) for k in c), 2)
        num("corrAbsMaxRisk", max(abs(c[k]["vs_risk"][0]) for k in c), 2)
        for k, tag in (("w1", "Wone"), ("mmd2", "Mmd"),
                       ("domain_auc", "Auc"), ("det_gap", "Det")):
            num(f"corr{tag}LocP", c[k]["vs_loc"][1], 4)
            num(f"corr{tag}RiskP", c[k]["vs_risk"][1], 4)
        num("detReal", 100 * e5["det_real"], 2)
        fc = e5.get("family_corr")
        if fc:
            num("famN", e5["n_family"], 0)
            num("famDraws", e5["n_draw_family"], 0)
            for k, tag in (("w1", "Wone"), ("mmd2", "Mmd"),
                           ("domain_auc", "Auc"), ("det_gap", "Det")):
                for tgt, ttag in (("rssi_risk", "Risk"), ("loc_err", "Loc")):
                    v = fc[k][tgt]
                    num(f"fam{tag}{ttag}", v["rho"], 2)
                    num(f"fam{tag}{ttag}Lo", v["lo"], 2)
                    num(f"fam{tag}{ttag}Hi", v["hi"], 2)
                    num(f"fam{tag}{ttag}Win", v["within_mean"], 2)
            # how many of the four proxies have a risk interval covering zero
            n0 = sum(1 for k in fc
                     if fc[k]["rssi_risk"]["lo"] <= 0 <= fc[k]["rssi_risk"]["hi"])
            num("famRiskNull", n0, 0)
            num("famLocMin", min(fc[k]["loc_err"]["rho"]
                                 for k in ("mmd2", "domain_auc", "det_gap")), 2)
            num("famRiskWinMax", max(fc[k]["rssi_risk"]["within_mean"]
                                     for k in fc), 2)
            M("famRiskWinBest", {"w1": "the Wasserstein distance",
                                 "mmd2": "the maximum-mean discrepancy",
                                 "domain_auc": "the domain-classifier score",
                                 "det_gap": "the reception-rate mismatch"}[
                max(fc, key=lambda k: fc[k]["rssi_risk"]["within_mean"])])

    # ---------------- foundation-model arm ---------------------------------
    if e6:
        sp = e6["level_spread"]
        num("fmSpreadZero", sp["0"], 2)
        keys = sorted((int(k) for k in sp), reverse=False)
        num("fmSpreadMax", sp[str(keys[-1])], 2)
        num("fmProbeMax", keys[-1], 0)
        num("fmBestZero", e6["best_level"]["0"], 0)
        num("fmDraws", e6["n_draw"], 0)
        # where a plain real-data baseline overtakes the pretrained encoder
        ro = e6["real_only"]
        best_pre = {k: min(r["err"][k] for r in e6["rows"]) for k in ro}
        over = [int(k) for k in sorted(ro, key=lambda x: int(x))
                if ro[k] < best_pre[k]]
        num("fmCrossover", over[0] if over else -1, 0)
        num("fmRealOnlyBest", min(ro.values()), 0)
        num("fmZeroBest", min(r["err"]["0"] for r in e6["rows"]), 0)
        M("fmZeroBestLevel",
          "F" + str(int(np.argmin([r["err"]["0"] for r in e6["rows"]]))))
        for k in keys:
            if k and sp[str(k)] < 1.10:
                num("fmNeutralN", k, 0)
                break
        else:
            num("fmNeutralN", keys[-1], 0)
        if "n_sim_free" in e6:
            ns = e6["n_sim_free"]
            num("fmSimTop", max(ns), comma=True)
            num("fmSimBottom", min(ns), comma=True)
            num("fmSimRatio", max(ns) / max(min(ns), 1), 1)
            num("fmCapEqual", e6["cap_equal"], comma=True)
            num("fmOrderAgree", e6["order_agree"], 2)
            re_ = e6["rows_equal"]
            M("fmZeroBestLevelEqual",
              "F" + str(int(np.argmin([r["err"]["0"] for r in re_]))))
            num("fmZeroBestEqual", min(r["err"]["0"] for r in re_), 0)
            num("fmSpreadZeroEqual", e6["level_spread_equal"]["0"], 2)

    # ---------------- mechanism ---------------------------------------------
    if e10:
        rows = e10["rows"]
        base = rows[0]; nz = [r for r in rows if r["scale"] > 0]
        num("mechScales", len(rows), 0)
        num("mechScaleMin", min(r["scale"] for r in nz), 2)
        num("mechTopBase", base["err"][-1], 0)
        num("mechTopSmall", nz[-1]["err"][-1], 0)
        num("mechTopZero", rows[-1]["err"][-1], 0)
        num("mechDecay", nz[-1]["err"][-1] / base["err"][-1], 1)
        M("mechArgminSet", ", ".join(sorted({f"F{r['argmin']}" for r in nz})))
        M("mechArgminZero", f"F{rows[-1]['argmin']}")
        num("mechRho", e10["mismatch"]["rho"], 2)
        M("mechRhoP", f"{e10['mismatch']['p']:.0e}".replace("e-", r"\times10^{-")
          + "}")
        num("mechN", e10["mismatch"]["n"], 0)
        num("mechWithin", e10["mismatch"]["within_level_mean"], 2)
        num("realSupport", e10["real_density"], 2)
        num("mechRep", e10["n_rep"], 0)

    # ---------------- indoor ------------------------------------------------
    if e7:
        num("nUjiPt", e7["n_pt"], comma=True)
        num("nUjiAp", e7["n_ap"], 0)
        num("nUjiLinks", e7["n_links"], comma=True)
        g = [r for r in e7["rows"] if r["mode"] == "guess"]
        err = [r["err50"] for r in sorted(g, key=lambda r: r["level"])]
        # `err` is positioning error, so its argmin is the *localization* level;
        # the link-risk level is taken from `rm` below.  Keep the two names
        # apart: an earlier version called this one `ujiMinLevel`, which read
        # as though it were the link-risk minimum.
        num("ujiLocMinLevel", int(np.nanargmin(err)), 0)
        M("ujiLocMinLevelName", e7["levels"][int(np.nanargmin(err))].split()[0])
        num("ujiBest", np.nanmin(err), 1)
        num("ujiTop", err[-1], 1)
        num("ujiCeiling", e7["ceiling"]["err50"], 1)
        num("ujiApSd", e7["oracle"]["ap_sd"], 1)
        num("ujiFloorLoss", e7["oracle"]["floor"], 1)
        num("ujiGap", np.nanmin(err) / e7["ceiling"]["err50"], 1)
        rm = [r["rssi_rmse"] for r in sorted(g, key=lambda r: r["level"])]
        num("ujiMinLevel", int(np.nanargmin(rm)), 0)
        M("ujiMinLevelName", e7["levels"][int(np.nanargmin(rm))].split()[0])
        num("ujiRmseMin", np.nanmin(rm), 2)

    # ---------------- robustness -------------------------------------------
    if e8:
        v = e8["loc_variants"]
        args = sorted({r["argmin"] for r in v.values()})
        M("robustArgmins", ", ".join(f"F{a}" for a in args))
        num("robustNVariants", len(v), 0)
        num("leakRandom", e8["leakage"]["random"]["err50"], 0)
        num("leakBlocked", e8["leakage"]["blocked"]["err50"], 0)
        num("leakInflation", e8["leakage"]["inflation"], 2)
        ls = e8["link_scale"]
        M("scaleArgmins", ", ".join(f"F{a}" for a in
                                     sorted({ls[k]["argmin"] for k in ls})))

    # ---------------- subsample stability ----------------------------------
    if e9:
        sm = e9["summary"]
        num("bootB", e9["B"], 0)
        num("bootBlocks", e9["n_block"], 0)
        for k, tag in (("C", "C"), ("n", "N"), ("gw_sd", "GwSd"),
                       ("kappa", "Kappa"), ("gamma", "Gamma"),
                       ("sigma", "Sigma")):
            num(f"boot{tag}Sd", sm[k]["sd"], 4 if k in ("kappa", "gamma") else 2)
            num(f"boot{tag}Lo", sm[k]["lo"], 4 if k in ("kappa", "gamma") else 2)
            num(f"boot{tag}Hi", sm[k]["hi"], 4 if k in ("kappa", "gamma") else 2)

    # ---------------- robustness additions ---------------------------------
    if e8 and "clip" in e8:
        cl = e8["clip"]
        M("clipArgmins", ", ".join(f"F{a}" for a in
                                    sorted({cl[k]["argmin"] for k in cl})))
        M("clipTestLevels", ", ".join(f"F{cl[k]['mean_level']:.0f}" for k in cl))
        num("nClips", len(cl), 0)
        num("clipHarmfulMax", 100 * max(cl[k]["harmful"] for k in cl), 1)
    if e8 and "gateway_stability" in e8:
        gs = e8["gateway_stability"]
        num("nUnstableGw", gs["n_unstable"], 0)
        num("stableArgmin", gs["stable"]["argmin"], 0)
        num("allArgmin", gs["all"]["argmin"], 0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("%% Generated by code/mknumbers.py -- do not edit.\n")
        f.write("\n".join(sorted(set(LINES))) + "\n")
    print(f"-> {OUT}  ({len(set(LINES))} macros)")


if __name__ == "__main__":
    main()
