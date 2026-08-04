"""
Prefit / postfit plotting methods for the HMDS signal region.
Imported by plot_fit.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from hbb.common_vars import LUMI

# Background processes to stack (bottom -> top), with display styling. Only those
# actually present in the shapes directory are drawn.
BKG_STYLE = [
    ("qcd", "QCD", "#9e9e9e"),
    ("Wjets", r"W$\to qq$", "#8fbc8f"),
    ("Zjetslight", r"Z$\to qq$ (light)", "#4c9a8a"),
    ("Zjetsc", r"Z$\to qq$ (c)", "#2a7f7f"),
    ("Zjetsbb", r"Z$\to qq$ (bb)", "#1f6f6f"),
    ("VV", "VV", "#6a8caf"),
    ("ttbar", r"$t\bar{t}$", "#d98c5f"),
]
SIGNAL = "HMDS"

FIT_DIRS = {"prefit": "shapes_prefit", "postfit": "shapes_fit_s"}


def read_th1(f, path):
    """Return (values, edges, errors) for a TH1 at `path`, or None if absent."""
    try:
        h = f[path]
    except (KeyError, uproot.KeyInFileError):
        return None
    vals, edges = h.to_numpy()
    try:
        errs = h.errors()
    except Exception:  # noqa: BLE001
        errs = np.zeros_like(vals)
    return vals, edges, errs


def read_data_graph(f, path):
    """Return (x, y, eylow, eyhigh) for the data TGraphAsymmErrors, or None."""
    try:
        g = f[path]
    except (KeyError, uproot.KeyInFileError):
        return None
    x = np.asarray(g.member("fX"), dtype=float)
    y = np.asarray(g.member("fY"), dtype=float)
    eyl = np.asarray(g.member("fEYlow"), dtype=float)
    eyh = np.asarray(g.member("fEYhigh"), dtype=float)
    return x, y, eyl, eyh


def draw_channel(f, fitdir, channel, year, out_png, title):
    centers_edges = None

    # Stack backgrounds
    stack_vals = []
    stack_labels = []
    stack_colors = []
    for key, label, color in BKG_STYLE:
        res = read_th1(f, f"{fitdir}/{channel}/{key}")
        if res is None:
            continue
        vals, edges, _ = res
        if vals.sum() <= 0:
            continue
        centers_edges = edges
        stack_vals.append(vals)
        stack_labels.append(label)
        stack_colors.append(color)

    total = read_th1(f, f"{fitdir}/{channel}/total_background")
    data = read_data_graph(f, f"{fitdir}/{channel}/data")
    sig = read_th1(f, f"{fitdir}/{channel}/{SIGNAL}")

    if centers_edges is None and total is not None:
        centers_edges = total[1]
    if centers_edges is None:
        print(f"  [skip] no templates for {channel} in {fitdir}")
        return False

    edges = centers_edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)

    # NB: Combine's FitDiagnostics writes shapes_* as a DENSITY (events per unit of the
    # observable), not as events per bin. We plot that density as-is, so with the 7 GeV
    # msd bins used here these yields sit a factor 7 below the template yields in
    # fitting_{year}_sr_msd.root and below python/plot_histos.py

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True
    )
    hep.cms.label("WiP", data=True, lumi=f"{LUMI[year] / 1000.0:.1f}", year=year, ax=ax)

    # Stacked backgrounds
    if stack_vals:
        ax.hist(
            [centers] * len(stack_vals),
            bins=edges,
            weights=stack_vals,
            stacked=True,
            histtype="stepfilled",
            label=stack_labels,
            color=stack_colors,
            edgecolor="none",
        )

    total_bkg = total[0] if total is not None else np.sum(stack_vals, axis=0)
    total_err = total[2] if total is not None else np.zeros_like(total_bkg)

    # Background uncertainty band
    ax.bar(
        centers, 2 * total_err, width=width, bottom=total_bkg - total_err,
        color="none", hatch="/////", edgecolor="gray", linewidth=0, label="Bkg unc.",
    )

    # Signal overlay (line)
    if sig is not None and sig[0].sum() > 0:
        ax.step(edges, np.append(sig[0], sig[0][-1]), where="post",
                color="red", linewidth=1.8, label=f"{SIGNAL}")

    # Data
    if data is not None:
        x, y, eyl, eyh = data
        ax.errorbar(x, y, yerr=[eyl, eyh], fmt="ko", markersize=4, label="Data", zorder=5)

    ax.set_ylabel("Events / GeV")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, fontsize=9, loc="upper right")
    # In-axes label (below the CMS header, above the stack) to avoid colliding
    # with the hep.cms.label() header text.
    ax.text(0.04, 0.90, title, transform=ax.transAxes, fontsize=10, va="top", fontweight="bold")

    # Ratio panel: data / total background
    if data is not None:
        x, y, eyl, eyh = data
        # match data points to bin centers by index
        tb = total_bkg[: len(y)]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(tb > 0, y / tb, np.nan)
            r_lo = np.where(tb > 0, eyl / tb, 0)
            r_hi = np.where(tb > 0, eyh / tb, 0)
        rax.errorbar(x, ratio, yerr=[r_lo, r_hi], fmt="ko", markersize=4)
    # band on ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(total_bkg > 0, total_err / total_bkg, 0)
    rax.bar(centers, 2 * rel, width=width, bottom=1 - rel, color="none",
            hatch="/////", edgecolor="gray", linewidth=0)
    rax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    rax.set_ylim(0, 2)
    rax.set_ylabel("Data / Bkg")
    rax.set_xlabel(r"Jet 0 $m_{sd}$ [GeV]")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_png}")
    return True
