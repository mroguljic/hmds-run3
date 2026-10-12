#!/usr/bin/env python3
"""
MC-transfer-factor order scan: fits tf_MCtempl to the QCD MC pass/fail ratio
at several candidate (mc_pt, mc_rho) orders and overlays them against the raw
MC ratio, per pt bin - a visual complement to the F-test's numeric decision.

Usage:
    python plot_mctf_scan.py --tag 20260721 --year 2024 \
        --orders 0,0;1,0;1,1;2,0;2,1
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

from card_utils import fit_mc_tf, mc_tf_chi2, raw_pass_fail_ratio

FITTING_DIR = Path(__file__).resolve().parent


def parse_orders(s):
    orders = []
    for chunk in s.split(";"):
        pt, rho = (int(x) for x in chunk.split(","))
        orders.append((pt, rho))
    return orders


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--year", required=True)
    p.add_argument("--indir", default=None, help="template dir (default results/<tag>)")
    p.add_argument("--outdir", default="results")
    p.add_argument("--setup", default="setup_sr.json")
    p.add_argument("--orders", default="0,0;1,0;1,1;2,0;2,1",
                    help="semicolon-separated 'mc_pt,mc_rho' list, e.g. '0,0;1,0;2,0'")
    p.add_argument("--ylim", default=None,
                    help="'ymin,ymax' for the pass/fail axis (default: autoscale, top=0.4)")
    p.add_argument("--outname", default="MCTF_scan", help="output file stem")
    a = p.parse_args()

    with (FITTING_DIR / a.setup).open() as f:
        config = json.load(f)

    root_file_name = config.get("root_filename", "signalregion.root").replace("{year}", a.year)
    indir = a.indir if a.indir else f"results/{a.tag}"
    infile_path = Path(indir) / root_file_name

    qcd_tf_proc = config.get("qcd_proc", "QCD")
    pt_min_scale = config.get("pt_min_scale", 450.0)
    regions_to_fit = config.get("regions_to_fit", [""])
    rho_scaling_min = config.get("rho_scaling_min", -6.0)
    rho_scaling_max = config.get("rho_scaling_max", -2.1)

    msd_cfg = config["observable"]
    msdbins = np.linspace(msd_cfg["min"], msd_cfg["max"], msd_cfg["nbins"] + 1)
    msd_centers = 0.5 * (msdbins[:-1] + msdbins[1:])
    msd = rl.Observable(msd_cfg["name"], msdbins)

    cats_cfg = config["categories"]
    cats = [c for c in cats_cfg if c != "mucr"]

    orders = parse_orders(a.orders)

    outdir = Path(a.outdir) / a.tag / a.year / "plots" / "MCTF_scan"
    outdir.mkdir(parents=True, exist_ok=True)

    plt.style.use(hep.style.CMS)
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(orders)))

    for cat in cats:
        ptbins = np.array(cats_cfg[cat].get("bins_pt", cats_cfg[cat].get("bins")))
        npt = len(ptbins) - 1

        ptpts, msdpts = np.meshgrid(
            ptbins[:-1] + 0.3 * np.diff(ptbins),
            msdbins[:-1] + 0.5 * np.diff(msdbins),
            indexing="ij",
        )
        rhopts = 2 * np.log(msdpts / ptpts)
        ptscaled = (ptpts - pt_min_scale) / (1200.0 - pt_min_scale)
        rhoscaled = (rhopts - rho_scaling_min) / (rho_scaling_max - rho_scaling_min)
        validbins = (rhoscaled >= 0.0) & (rhoscaled <= 1.0)
        rhoscaled[~validbins] = 1

        for reg in regions_to_fit:
            ratio, err = raw_pass_fail_ratio(infile_path, qcd_tf_proc, reg, cat, msd, npt)

            print(f"\n=== {cat} {reg or '(inclusive)'} : MC-TF order scan ===")
            curves = []
            for order in orders:
                tf_MCtempl, qcdfit, qcdeff, _ = fit_mc_tf(
                    infile_path, qcd_tf_proc, cat, reg, a.year, msd, npt,
                    ptscaled, rhoscaled, validbins, order,
                )
                mctf = qcdeff * tf_MCtempl(ptscaled, rhoscaled, nominal=True)
                chi2, nvalid = mc_tf_chi2(tf_MCtempl, qcdeff, ptscaled, rhoscaled, validbins, ratio, err)
                nparam = (order[0] + 1) * (order[1] + 1)
                ndof = nvalid - nparam
                status = qcdfit.status()
                print(f"  order={order}  status={status}  chi2/ndof={chi2:.1f}/{ndof}"
                      f" = {chi2 / ndof if ndof > 0 else float('nan'):.2f}")
                curves.append((order, mctf, chi2, ndof, status))

            fig, axes = plt.subplots(1, npt, figsize=(5.2 * npt, 4.6), sharey=True)
            if npt == 1:
                axes = [axes]

            for ptbin in range(npt):
                m = validbins[ptbin]
                ax = axes[ptbin]
                ax.errorbar(
                    msd_centers[m], ratio[ptbin][m], yerr=err[ptbin][m],
                    fmt="o", color="black", ms=4, lw=1, label="QCD MC pass/fail",
                    zorder=10,
                )
                for (order, mctf, chi2, ndof, status), color in zip(curves, colors):
                    rchi2 = chi2 / ndof if ndof > 0 else float("nan")
                    lbl = f"mc=({order[0]},{order[1]})  " + (
                        f"$\\chi^2$/ndof={rchi2:.1f}" if np.isfinite(rchi2) else "fit failed"
                    )
                    ls = "-" if status == 0 else "--"
                    ax.plot(msd_centers[m], mctf[ptbin][m], ls, color=color, lw=2, label=lbl)
                ax.set_title(f"$p_T$ {ptbins[ptbin]:.0f}–{ptbins[ptbin + 1]:.0f} GeV", fontsize=12)
                ax.set_xlabel(r"$m_{sd}$ [GeV]")
                if a.ylim:
                    ax.set_ylim(*(float(x) for x in a.ylim.split(",")))
                else:
                    ax.set_ylim(top=0.4)
                ax.legend(fontsize=7, loc="best", ncol=2)

            axes[0].set_ylabel("Pass / Fail")
            hep.cms.text("WiP", ax=axes[0], fontsize=13)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                out = outdir / f"{a.outname}.{ext}"
                fig.savefig(out, dpi=120, bbox_inches="tight")
                print(f"  saved {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()
