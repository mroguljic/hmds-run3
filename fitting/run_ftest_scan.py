#!/usr/bin/env python3
"""
Sequential F-test order scan for the HMDS SR rhalphabet transfer function.

Scans the MC transfer factor order (--mc-pt-order/--mc-rho-order) to convergence,
fixes it, then scans the data-residual order (--res-pt-order/--res-rho-order),
stopping in each direction when the next higher-order model gives no significant
improvement (F-test p > threshold). Builds each workspace on demand (cached),
runs run_ftest.py's Combine commands, and computes the empirical F-test p-value
from the toys. Prints a step-by-step log and the final chosen model.

Usable standalone:
  python run_ftest_scan.py --tag 20260721 --year 2024
or via the driver, which imports scan() directly:
  python run_fit.py --tag 20260721 --year 2024 --stages ftest
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from fit_common import FITTING_DIR, dc_dir, n_fit_bins, run, ws_name

sys.path.insert(0, str(FITTING_DIR))
import run_ftest  # noqa: E402
from plot_ftest import calculate_f_statistic, get_chi2_values  # noqa: E402


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
    print("PHASE 1: MC transfer factor scan (data residual fixed at 0,0)", flush=True)
    print("=" * 70, flush=True)
    #First scan pT axis by assuming some reasonable (order=1) rho dependence
    o = scan_dim((0, 1, 0, 0), idx=0, cap=2, label="MC-pt@rho1", cfg=cfg) # cap 2 because we have three pT bins

    # Then scan rho, taking optimal pT value
    o = (o[0], 0, 0, 0)
    o = scan_dim(o, idx=1, cap=3, label="MC-rho", cfg=cfg)
    if o[1] != 1 and o[0] < 2:
        # If best rho order!=1, do we want to rerun pT tuning?
        print("\n[MC-pt confirm] re-check pt at final rho", flush=True)
        o = scan_dim(o, idx=0, cap=2, label="MC-pt-confirm", cfg=cfg)
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
