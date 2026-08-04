#!/usr/bin/env python3
"""
Goodness-of-fit toy distribution plotter.

Reads the saturated-GoF ROOT files produced by run_gof_final.sh (one observed
value + N toys) and draws the toy test-statistic distribution with the observed
value marked, reporting the p-value = fraction of toys with GoF >= observed.

Usage:
    python plot_gof.py --dir results/20260721/2024/datacards/srModel_2024 \
        --tag gof_final --seed 123456
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot


def get_limit(filename):
    if not Path(filename).exists():
        print(f"Error: missing {filename}")
        return None
    with uproot.open(filename) as f:
        return f["limit"]["limit"].array(library="np")


def main(args):
    d = Path(args.dir)
    obs = get_limit(d / f"higgsCombine_{args.tag}_Observed.GoodnessOfFit.mH120.root")
    toys = get_limit(d / f"higgsCombine_{args.tag}_Toys.GoodnessOfFit.mH120.{args.seed}.root")
    if obs is None or toys is None:
        raise SystemExit("Could not load GoF inputs.")

    obs_val = float(obs[0])
    toys = np.asarray(toys, dtype=float)
    toys = toys[np.isfinite(toys)]
    n = len(toys)
    n_above = int(np.sum(toys >= obs_val))
    pval = n_above / n if n > 0 else float("nan")

    print(f"Observed GoF (saturated): {obs_val:.2f}")
    print(f"Toys: {n}   mean={toys.mean():.2f}  std={toys.std():.2f}")
    print(f"p-value (fraction toys >= observed): {pval:.3f}  ({n_above}/{n})")

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(10, 8))

    lo = min(toys.min(), obs_val)
    hi = max(toys.max(), obs_val)
    pad = 0.05 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 30)

    ax.hist(toys, bins=bins, histtype="stepfilled", color="#6a8caf",
            alpha=0.6, edgecolor="#31506e", linewidth=1.4,
            label=f"Toys ($N={n}$)")
    ymax = ax.get_ylim()[1] * 1.15
    ax.vlines(obs_val, 0, ymax, color="red", linewidth=2.5,
              label=f"Observed = {obs_val:.1f}\n$p = {pval:.3f}$")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Saturated GoF test statistic")
    ax.set_ylabel("Pseudo-experiments")
    ax.legend(loc="upper right", fontsize=14, frameon=True)
    ax.text(0.04, 0.92, args.label, transform=ax.transAxes,
            fontsize=12, va="top", fontweight="bold")
    hep.cms.label("WiP", data=True, ax=ax)

    out = d / f"gof_{args.tag}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--tag", default="gof_final")
    p.add_argument("--seed", default=123456, type=int)
    p.add_argument("--label", default="HMDS SR", help="annotation drawn on the plot")
    main(p.parse_args())
