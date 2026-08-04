#!/usr/bin/env python3
"""
Unified fit-diagnostic plotter for the HMDS signal region.

Produces, from a single run, all of:
  1. MC transfer factor  - raw QCD pass/fail ratio (points) vs the Bernstein
     tf_MCtempl fit (curve), per pt bin.
  2. Data residual       - tf_dataResidual evaluated at its POSTFIT coefficients
     (prefit == 1 shown as reference), per pt bin.
  3. Final transfer factor - qcdeff * tf_MCtempl * tf_dataResidual (postfit): the
     effective fail->pass multiplier, per pt bin.
  4./5. Prefit / postfit stacked shapes per channel.

The MC-TF Bernstein coefficients come from the build-time fit stored in
`initial_vals/initial_vals_{cat}_{reg}.json`; the residual (and the postfit
shapes) come from a FitDiagnostics run on the chosen workspace. Orders must
match the workspace being diagnosed (default = F-test final model mc=(2,0)
res=(1,0)).

Usage:
    python plot_fit.py --tag 20260721 --year 2024 \
        --fitdiag fitDiagnostics_final.root \
        --mc-pt-order 2 --mc-rho-order 0 --res-pt-order 1 --res-rho-order 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import rhalphalib as rl
import ROOT
import uproot

from card_utils import get_template
from shape_utils import FIT_DIRS, draw_channel

ROOT.gROOT.SetBatch(True)

PT_LO, PT_HI = None, None  # set from config


def build_grid(ptbins, msdbins, pt_min_scale, rho_min, rho_max):
    """Reproduce the (pt, msd) evaluation grid used by make_datacards."""
    ptpts, msdpts = np.meshgrid(
        ptbins[:-1] + 0.3 * np.diff(ptbins),
        msdbins[:-1] + 0.5 * np.diff(msdbins),
        indexing="ij",
    )
    rhopts = 2 * np.log(msdpts / ptpts)
    ptscaled = (ptpts - pt_min_scale) / (1200.0 - pt_min_scale)
    rhoscaled = (rhopts - rho_min) / (rho_max - rho_min)
    valid = (rhoscaled >= 0.0) & (rhoscaled <= 1.0)
    rhoscaled = np.where(valid, rhoscaled, 1.0)
    return ptpts, msdpts, ptscaled, rhoscaled, valid


def read_postfit_params(fitdiag, name_substr):
    """Return {param_name: value} for postfit (fit_s) params containing substr."""
    f = ROOT.TFile.Open(str(fitdiag))
    fit_s = f.Get("fit_s")
    if not fit_s:
        raise RuntimeError(f"No fit_s RooFitResult in {fitdiag}")
    fp = fit_s.floatParsFinal()
    out = {}
    for p in fp.contentsString().split(","):
        if name_substr in p:
            out[p] = fp.find(p).getVal()
    f.Close()
    return out


def coeffs_from_postfit(fitdiag, poly_name, shape):
    """Assemble a coefficient array of `shape` from postfit params of poly_name.

    Params are named <poly>_pt_par{i}_rho_par{j} (rhalphalib BasisPoly convention).
    """
    vals = read_postfit_params(fitdiag, poly_name)
    coeffs = np.ones(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            key = f"{poly_name}_pt_par{i}_rho_par{j}"
            if key not in vals:
                raise KeyError(f"Missing postfit param {key} in fit result")
            coeffs[i, j] = vals[key]
    return coeffs


def eval_poly(name, order, coeffs, ptscaled, rhoscaled, limits):
    poly = rl.BasisPoly(
        name, order, ["pt", "rho"], basis="Bernstein", init_params=coeffs, limits=limits
    )
    return poly(ptscaled, rhoscaled, nominal=True)


def plot_transfer_factors(args, cfg, outdir):
    global PT_LO, PT_HI
    cat = "sr"
    reg = ""  # regions_to_fit == [""]
    qcd_proc = cfg.get("qcd_proc", "QCD")
    pt_min_scale = cfg.get("pt_min_scale", 450.0)
    rho_min = cfg.get("rho_scaling_min", -6.0)
    rho_max = cfg.get("rho_scaling_max", -2.1)

    msd_cfg = cfg["observable"]
    msdbins = np.linspace(msd_cfg["min"], msd_cfg["max"], msd_cfg["nbins"] + 1)
    ptbins = np.array(cfg["categories"][cat]["bins"], dtype=float)
    npt = len(ptbins) - 1
    PT_LO, PT_HI = ptbins[0], ptbins[-1]

    # rhalphalib Observable for template reading
    obs = rl.Observable(msd_cfg["name"], msdbins)
    msd_centers = 0.5 * (msdbins[:-1] + msdbins[1:])

    infile = Path(args.indir) / cfg["root_filename"].replace("{year}", args.year)

    # ---- raw QCD pass/fail per pt bin + inclusive qcdeff ----
    raw_ratio = np.full((npt, len(msd_centers)), np.nan)
    tot_pass, tot_fail = 0.0, 0.0
    for ptbin in range(npt):
        failT = get_template(infile, qcd_proc, "fail_", ptbin + 1, cat, obs, "nominal")[0]
        passT = get_template(infile, qcd_proc, f"pass_{reg}_", ptbin + 1, cat, obs, "nominal")[0]
        tot_pass += passT.sum()
        tot_fail += failT.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_ratio[ptbin] = np.where(failT > 0, passT / failT, np.nan)
    qcdeff = tot_pass / tot_fail
    print(f"Inclusive QCD pass/fail (qcdeff) = {qcdeff:.4f}")

    # ---- grid + Bernstein surfaces ----
    _, _, ptscaled, rhoscaled, valid = build_grid(ptbins, msdbins, pt_min_scale, rho_min, rho_max)

    mc_order = (args.mc_pt_order, args.mc_rho_order)
    res_order = (args.res_pt_order, args.res_rho_order)

    # MC-TF coefficients (build-time fit)
    initf = (
        Path(args.outdir) / args.tag / args.year / "initial_vals"
        / f"initial_vals_{cat}_{reg}.json"
    )
    mc_coeffs = np.array(json.load(initf.open())["initial_vals"], dtype=float)
    if mc_coeffs.shape != (mc_order[0] + 1, mc_order[1] + 1):
        raise ValueError(
            f"initial_vals shape {mc_coeffs.shape} != MC order {mc_order}; "
            "rebuild the final model or pass matching --mc-*-order."
        )
    mctf = qcdeff * eval_poly(
        f"tf_MCtempl_{cat}{reg}{args.year}", mc_order, mc_coeffs, ptscaled, rhoscaled, (0, 10)
    )

    # Residual coefficients (postfit from FitDiagnostics)
    res_coeffs = coeffs_from_postfit(
        args.fitdiag_path, f"tf_dataResidual_{args.year}{cat}{reg}", (res_order[0] + 1, res_order[1] + 1)
    )
    residual = eval_poly(
        f"tf_dataResidual_{args.year}{cat}{reg}", res_order, res_coeffs, ptscaled, rhoscaled, (0, 20)
    )

    final_tf = mctf * residual

    # ---- draw: 3 rows (MC TF, residual, final TF) x npt columns ----
    plt.style.use(hep.style.CMS)
    fig, axes = plt.subplots(3, npt, figsize=(5 * npt, 12), sharex=True)
    if npt == 1:
        axes = axes.reshape(3, 1)
    row_titles = ["MC transfer factor", "Data residual (postfit)", "Final TF = qcdeff·MC·resid"]

    for ptbin in range(npt):
        m = valid[ptbin]
        pt_lbl = f"{ptbins[ptbin]:.0f}–{ptbins[ptbin + 1]:.0f} GeV"

        # Row 0: MC TF
        ax = axes[0, ptbin]
        ax.plot(msd_centers[m], raw_ratio[ptbin][m], "o", color="black", ms=5,
                label="QCD MC pass/fail")
        ax.plot(msd_centers[m], mctf[ptbin][m], "-", color="#c0392b", lw=2.2,
                label="Bernstein fit")
        ax.set_title(f"$p_T$ {pt_lbl}", fontsize=13)
        ax.legend(fontsize=9)

        # Row 1: residual
        ax = axes[1, ptbin]
        ax.axhline(1.0, color="gray", ls="--", lw=1.2, label="prefit = 1")
        ax.plot(msd_centers[m], residual[ptbin][m], "-", color="#2471a3", lw=2.2,
                label="postfit residual")
        ax.legend(fontsize=9)

        # Row 2: final TF
        ax = axes[2, ptbin]
        ax.plot(msd_centers[m], final_tf[ptbin][m], "-", color="#1e8449", lw=2.2,
                label="final TF")
        ax.plot(msd_centers[m], mctf[ptbin][m], ":", color="#c0392b", lw=1.6,
                label="MC-only")
        ax.set_xlabel(r"$m_{sd}$ [GeV]")
        ax.legend(fontsize=9)

    for r in range(3):
        axes[r, 0].set_ylabel(row_titles[r], fontsize=11)

    hep.cms.text("WiP", ax=axes[0, 0], fontsize=13)
    fig.suptitle(
        f"HMDS SR transfer factor  |  MC order {mc_order}, residual order {res_order}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = outdir / "transfer_factors.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

    # ---- companion: TF vs pt (shows the pt shape the rho-flat model captures) ----
    pt_centers = ptbins[:-1] + 0.5 * np.diff(ptbins)
    # value at the middle msd bin of each pt row (flat in rho so any valid bin works)
    def midval(surf):
        out = []
        for ptbin in range(npt):
            mm = np.where(valid[ptbin])[0]
            out.append(surf[ptbin][mm[len(mm) // 2]] if len(mm) else np.nan)
        return np.array(out)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pt_centers, midval(mctf), "o-", color="#c0392b", label="MC TF")
    ax.plot(pt_centers, midval(residual), "s--", color="#2471a3", label="residual (postfit)")
    ax.plot(pt_centers, midval(final_tf), "D-", color="#1e8449", label="final TF")
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel("Transfer factor")
    ax.legend(fontsize=11)
    hep.cms.label("WiP", data=True, year=args.year, ax=ax)
    out = outdir / "transfer_factors_vs_pt.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_prefit_postfit(args, outdir):
    if not Path(args.fitdiag_path).exists():
        print(f"  [skip prefit/postfit] {args.fitdiag_path} not found")
        return
    f = uproot.open(args.fitdiag_path)
    channels = [f"ptbin{i}sr{reg}{args.year}" for i in range(args.npt) for reg in ("pass", "fail")]
    for fit in ("prefit", "postfit"):
        fitdir = FIT_DIRS[fit]
        for ch in channels:
            draw_channel(f, fitdir, ch, args.year, outdir / f"{ch}_{fit}.png", f"{fit}  |  {ch}")


def main():
    p = argparse.ArgumentParser(description="Unified TF + prefit/postfit plots for the HMDS SR.")
    p.add_argument("--tag", required=True)
    p.add_argument("--year", required=True)
    p.add_argument("--indir", default=None, help="dir with fitting_{year}_sr_msd.root (default results/<tag>)")
    p.add_argument("--outdir", default="results", help="base results dir (holds <tag>/<year>/...)")
    p.add_argument("--model", default="srModel")
    p.add_argument("--fitdiag", default="fitDiagnostics_final.root",
                   help="FitDiagnostics file inside the datacard dir")
    p.add_argument("--npt", type=int, default=3)
    p.add_argument("--mc-pt-order", type=int, default=2)
    p.add_argument("--mc-rho-order", type=int, default=0)
    p.add_argument("--res-pt-order", type=int, default=1)
    p.add_argument("--res-rho-order", type=int, default=0)
    p.add_argument("--which", nargs="+", default=["tf", "prefitpostfit"],
                   choices=["tf", "prefitpostfit"])
    args = p.parse_args()

    if args.indir is None:
        args.indir = f"results/{args.tag}"

    dcdir = Path(args.outdir) / args.tag / args.year / "datacards" / f"{args.model}_{args.year}"
    args.fitdiag_path = dcdir / args.fitdiag

    with Path("setup_sr.json").open() as fh:
        cfg = json.load(fh)

    outdir = dcdir / "fit_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    if "tf" in args.which:
        plot_transfer_factors(args, cfg, outdir)
    if "prefitpostfit" in args.which:
        plot_prefit_postfit(args, outdir)

    print(f"\nAll plots in {outdir}")


if __name__ == "__main__":
    hep.style.use("CMS")
    main()
