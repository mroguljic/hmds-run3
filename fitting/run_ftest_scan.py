#!/usr/bin/env python3
"""
Sequential F-test order scan for the HMDS SR rhalphabet transfer function.

Phase 1 (not related to CMS detector geometry :)) is the MC transfer factor order.
Fit purely to the QCD MC pass/fail templates via card_utils.fit_mc_tf().
The order is chosen by an F-test on that MC-only chi2, exactly analogous to
Phase 2 but comparing the Bernstein fit to the raw MC ratio instead of a
Combine GoodnessOfFit statistic to data.

Phase 2 (data residual order) fits to data: MC order is fixed at the
Phase 1 best order, and each candidate residual order is F-tested against
data by building a full workspace and running run_ftest.py's Combine
commands (GoodnessOfFit toys). Workspaces are cached on disk.

Usable standalone:
  python run_ftest_scan.py --tag 20260721 --year 2024
or via the driver, which imports scan() directly:
  python run_fit.py --tag 20260721 --year 2024 --stages ftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import rhalphalib as rl
from scipy.stats import f as f_dist

from fit_common import FITTING_DIR, dc_dir, n_fit_bins, run, ws_name

sys.path.insert(0, str(FITTING_DIR))
import run_ftest  # noqa: E402
from card_utils import fit_mc_tf, mc_tf_chi2, raw_pass_fail_ratio  # noqa: E402
from plot_ftest import calculate_f_statistic, get_chi2_values  # noqa: E402


def load_mc_context(cfg):
    """
    Build the (pt, rho) grid + raw MC pass/fail ratio needed by fit_mc_tf(),
    exactly as make_datacards.py / plot_mctf_scan.py do. Only supports a
    single category/region, matching make_datacards.py's single global
    --mc-*-order CLI knob.
    """
    with (FITTING_DIR / cfg["setup"]).open() as f:
        config = json.load(f)

    root_file_name = config.get("root_filename", "signalregion.root").replace("{year}", cfg["year"])
    infile_path = Path(cfg["indir"]) / root_file_name

    qcd_tf_proc = config.get("qcd_proc", "QCD")
    pt_min_scale = config.get("pt_min_scale", 450.0)
    regions_to_fit = config.get("regions_to_fit", [""])
    rho_min = config.get("rho_scaling_min", -6.0)
    rho_max = config.get("rho_scaling_max", -2.1)

    msd_cfg = config["observable"]
    msdbins = np.linspace(msd_cfg["min"], msd_cfg["max"], msd_cfg["nbins"] + 1)
    msd = rl.Observable(msd_cfg["name"], msdbins)

    cats_cfg = config["categories"]
    cats = [c for c in cats_cfg if c != "mucr"]
    if len(cats) != 1 or len(regions_to_fit) != 1:
        raise SystemExit(
            f"[ftest_scan] mc_scan_dim only supports a single category/region "
            f"(got cats={cats}, regions_to_fit={regions_to_fit}); "
            "make_datacards.py's --mc-*-order is a single global order anyway."
        )
    cat, reg = cats[0], regions_to_fit[0]

    ptbins = np.array(cats_cfg[cat].get("bins_pt", cats_cfg[cat].get("bins")), dtype=float)
    npt = len(ptbins) - 1
    ptpts, msdpts = np.meshgrid(
        ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing="ij"
    )
    rhopts = 2 * np.log(msdpts / ptpts)
    ptscaled = (ptpts - pt_min_scale) / (1200.0 - pt_min_scale)
    rhoscaled = (rhopts - rho_min) / (rho_max - rho_min)
    validbins = (rhoscaled >= 0.0) & (rhoscaled <= 1.0)
    rhoscaled[~validbins] = 1

    ratio, err = raw_pass_fail_ratio(infile_path, qcd_tf_proc, reg, cat, msd, npt)

    return {
        "infile_path": infile_path, "qcd_tf_proc": qcd_tf_proc, "cat": cat, "reg": reg,
        "year": cfg["year"], "msd": msd, "npt": npt, "ptscaled": ptscaled,
        "rhoscaled": rhoscaled, "validbins": validbins, "ratio": ratio, "err": err,
        "nbins": int(validbins.sum()),
    }


def mc_chi2_cached(mc_ctx, order, cache):
    if order not in cache:
        tf_MCtempl, _qcdfit, qcdeff, _fitted = fit_mc_tf(
            mc_ctx["infile_path"], mc_ctx["qcd_tf_proc"], mc_ctx["cat"], mc_ctx["reg"],
            mc_ctx["year"], mc_ctx["msd"], mc_ctx["npt"], mc_ctx["ptscaled"],
            mc_ctx["rhoscaled"], mc_ctx["validbins"], order,
        )
        chi2, _nvalid = mc_tf_chi2(
            tf_MCtempl, qcdeff, mc_ctx["ptscaled"], mc_ctx["rhoscaled"],
            mc_ctx["validbins"], mc_ctx["ratio"], mc_ctx["err"],
        )
        cache[order] = chi2
    return cache[order]


def mc_ftest(o_null, o_alt, mc_ctx, cache):
    chi2_n = mc_chi2_cached(mc_ctx, o_null, cache)
    chi2_a = mc_chi2_cached(mc_ctx, o_alt, cache)
    p1 = (o_null[0] + 1) * (o_null[1] + 1)
    p2 = (o_alt[0] + 1) * (o_alt[1] + 1)
    nbins = mc_ctx["nbins"]
    F = calculate_f_statistic(chi2_n, chi2_a, p1, p2, nbins)
    pval = float(f_dist.sf(F, p2 - p1, nbins - p2)) if np.isfinite(F) and F >= 0 else float("nan")
    print(
        f"  --> null({o_null}) vs alt({o_alt}): p1={p1} p2={p2} "
        f"chi2_null={chi2_n:.3f} chi2_alt={chi2_a:.3f} F={F:.3f}  p-value={pval:.3f}",
        flush=True,
    )
    return pval


def mc_scan_dim(mc_ctx, base, idx, cap, label, cfg, cache):
    orders = list(base)
    while orders[idx] < cap:
        nxt = list(orders)
        nxt[idx] += 1
        print(f"\n[{label}] test order[{idx}] {orders[idx]} -> {nxt[idx]}", flush=True)
        try:
            pval = mc_ftest(tuple(orders), tuple(nxt), mc_ctx, cache)
        except RuntimeError as e:
            print(f"[{label}] fit failed ({e}); stopping.", flush=True)
            break
        if np.isnan(pval):
            print(f"[{label}] NaN p-value; stopping.", flush=True)
            break
        if pval < cfg["pthresh"]:
            print(f"[{label}] significant (p={pval:.3f}) -> adopt {nxt[idx]}", flush=True)
            orders = nxt
        else:
            print(f"[{label}] not significant (p={pval:.3f}) -> keep {orders[idx]}", flush=True)
            break
    return tuple(orders)


def build_ws(o, cfg):
    p = cfg["dc"] / ws_name(o)
    if p.exists():
        print(f"  [cache] {p.name}", flush=True)
        return p
    run(
        f"python make_datacards.py --year {cfg['year']} --tag {cfg['tag']} "
        f"--indir {cfg['indir']} --outdir {cfg['outdir']} --analysis sr "
        f"--mc-pt-order {o[0]} --mc-rho-order {o[1]} "
        f"--res-pt-order {o[2]} --res-rho-order {o[3]} > /dev/null 2>&1",
        cwd=FITTING_DIR,
        check=False,
        prefix="ftest_scan",
    )
    run("./build.sh > /dev/null 2>&1", cwd=cfg["dc"], check=False, prefix="ftest_scan")
    run(f"cp workspace.root {p.name}", cwd=cfg["dc"], check=False, prefix="ftest_scan")
    return p


def param_count(o_null, o_alt):
    # The MC-TF orders differ so we count the MC params
    if o_null[:2] != o_alt[:2]:
        return (o_null[0] + 1) * (o_null[1] + 1), (o_alt[0] + 1) * (o_alt[1] + 1)
    # The MC-TF orders match so we count the residual correction params
    return (o_null[2] + 1) * (o_null[3] + 1), (o_alt[2] + 1) * (o_alt[3] + 1)


def ftest(o_null, o_alt, cfg):
    wn, wa = build_ws(o_null, cfg), build_ws(o_alt, cfg)
    seed = cfg["seed"]
    tag = f"mc{o_null[0]}{o_null[1]}res{o_null[2]}{o_null[3]}_vs_mc{o_alt[0]}{o_alt[1]}res{o_alt[2]}{o_alt[3]}"
    cwd = os.getcwd()
    os.chdir(cfg["dc"])
    try:
        run_ftest.main(wn.name, wa.name, cfg["ntoys"], seed, tag)
        obs_n = get_chi2_values(f"higgsCombine_Observed_Null_{tag}.GoodnessOfFit.mH120.root")
        obs_a = get_chi2_values(f"higgsCombine_Observed_Alt_{tag}.GoodnessOfFit.mH120.root")
        tn = get_chi2_values(f"higgsCombine_Toys_Null_{tag}.GoodnessOfFit.mH120.{seed}.root")
        ta = get_chi2_values(f"higgsCombine_Toys_Alt_{tag}.GoodnessOfFit.mH120.{seed}.root")
    finally:
        os.chdir(cwd)
    if obs_n is None or obs_a is None or tn is None or ta is None:
        return float("nan"), float("nan")
    p1, p2 = param_count(o_null, o_alt)
    nbins = cfg["nbins"]
    f_obs = calculate_f_statistic(obs_n[0], obs_a[0], p1, p2, nbins)
    n = min(len(tn), len(ta))
    f_toys = calculate_f_statistic(tn[:n], ta[:n], p1, p2, nbins)
    valid = np.isfinite(f_toys)
    pval = float(np.mean(f_toys[valid] >= f_obs)) if valid.any() else float("nan")
    print(
        f"  --> null({o_null}) vs alt({o_alt}): p1={p1} p2={p2} "
        f"F_obs={f_obs:.3f}  p-value={pval:.3f}  ({int(valid.sum())} valid toys)",
        flush=True,
    )
    return pval, f_obs


def scan_dim(base, idx, cap, label, cfg):
    orders = list(base)
    while orders[idx] < cap:
        nxt = list(orders)
        nxt[idx] += 1
        print(f"\n[{label}] test order[{idx}] {orders[idx]} -> {nxt[idx]}", flush=True)
        pval, _ = ftest(tuple(orders), tuple(nxt), cfg)
        if np.isnan(pval):
            print(f"[{label}] NaN p-value; stopping.", flush=True)
            break
        if pval < cfg["pthresh"]:
            print(f"[{label}] significant (p={pval:.3f}) -> adopt {nxt[idx]}", flush=True)
            orders = nxt
        else:
            print(f"[{label}] not significant (p={pval:.3f}) -> keep {orders[idx]}", flush=True)
            break
    return tuple(orders)


def scan(tag, year, model="srModel", outdir="results", indir=None,
         setup="setup_sr.json", seed=123456, ntoys=100, pthresh=0.05, nbins=None):
    """
    Run the two-phase F-test scan and return the winning order as a 4-tuple
    (mc_pt, mc_rho, res_pt, res_rho). Called by run_fit.py's ftest stage.
    """
    cfg = {
        "dc": dc_dir(outdir, tag, year, model),
        "tag": tag,
        "year": year,
        "outdir": outdir,
        "indir": indir if indir else f"results/{tag}",
        "setup": setup,
        "seed": seed,
        "ntoys": ntoys,
        "pthresh": pthresh,
        "nbins": nbins if nbins else n_fit_bins(FITTING_DIR / setup),
    }
    print(f"[ftest_scan] datacards: {cfg['dc']}", flush=True)
    print(f"[ftest_scan] templates: {cfg['indir']}", flush=True)
    print(f"[ftest_scan] nbins={cfg['nbins']} (from {setup}, rho-masked bins excluded), "
          f"seed={seed}, ntoys={ntoys}, p<{pthresh}", flush=True)

    print("=" * 70, flush=True)
    print("PHASE 1: MC transfer factor scan (fit to MC only - no data, no Combine)", flush=True)
    print("=" * 70, flush=True)
    mc_ctx = load_mc_context(cfg)
    mc_cache = {}
    print(f"[ftest_scan] MC-TF fit grid: {mc_ctx['cat']} {mc_ctx['reg'] or '(inclusive)'}, "
          f"nbins={mc_ctx['nbins']} (pt x msd, rho-masked bins excluded)", flush=True)

    #First scan pT axis by assuming some reasonable (order=1) rho dependence
    o = mc_scan_dim(mc_ctx, (0, 1), idx=0, cap=2, label="MC-pt@rho1", cfg=cfg, cache=mc_cache) # cap 2 because we have three pT bins

    # Then scan rho, taking optimal pT value
    o = (o[0], 0)
    o = mc_scan_dim(mc_ctx, o, idx=1, cap=3, label="MC-rho", cfg=cfg, cache=mc_cache)
    if o[1] != 1 and o[0] < 2:
        # If best rho order!=1, do we want to rerun pT tuning?
        print("\n[MC-pt confirm] re-check pt at final rho", flush=True)
        o = mc_scan_dim(mc_ctx, o, idx=0, cap=2, label="MC-pt-confirm", cfg=cfg, cache=mc_cache)
    mc_final = o
    print(f"\n>>> MC factor converged at (pt,rho)=({mc_final[0]},{mc_final[1]})", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"PHASE 2: data-residual scan (MC factor fixed at {mc_final[:2]})", flush=True)
    print("=" * 70, flush=True)
    o = (mc_final[0], mc_final[1], 0, 0)
    o = scan_dim(o, idx=2, cap=2, label="RES-pt", cfg=cfg)
    o = (o[0], o[1], o[2], 0)
    o = scan_dim(o, idx=3, cap=3, label="RES-rho", cfg=cfg)
    print(
        f"\n>>> FINAL MODEL: mc=({o[0]},{o[1]}) res=({o[2]},{o[3]})  "
        f"workspace: {ws_name(o)}",
        flush=True,
    )
    return o


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--year", required=True)
    p.add_argument("--model", default="srModel")
    p.add_argument("--outdir", default="results")
    p.add_argument("--indir", default=None, help="template dir (default results/<tag>)")
    p.add_argument("--setup", default="setup_sr.json", help="used to derive the bin count")
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--ntoys", type=int, default=100, help="toys per F-test comparison")
    p.add_argument("--pthresh", type=float, default=0.05)
    p.add_argument("--nbins", type=int, default=None,
                   help="override the bin count derived from --setup")
    a = p.parse_args()
    scan(tag=a.tag, year=a.year, model=a.model, outdir=a.outdir, indir=a.indir,
         setup=a.setup, seed=a.seed, ntoys=a.ntoys, pthresh=a.pthresh, nbins=a.nbins)


if __name__ == "__main__":
    main()
