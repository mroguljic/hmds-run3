#!/usr/bin/env python3
"""
Scan (FatJet0_ParTPQCD, FatJet0_nSV) working points and rank them by S/sqrt(B),
pT-inclusive, using the histograms produced by make_histos.py
(histograms_tagger_nsv_{year}_{region}.pkl).

For a cut of the form "TQCD < t" and "nSV > n", this computes S and B summed
over all bins that satisfy both cuts simultaneously (integrated over a msd
window and over all pT), for every (t, n) grid point defined by the
histogram's own bin edges, then reports/plots Z = S/sqrt(B).

For the top --top-n candidates by Z, it also renders the full FatJet0_msd
spectrum (Signal + backgrounds + Data) at that specific working point, so a
high Z driven by a near-empty, low-MC-statistics background bin can be told
apart from a WP with a genuinely well-populated background.

NOTE ON BLINDING: this script never reloads raw events - it only re-projects
the histograms make_histos.py already filled. So the 'data' spectra shown
here are exactly as blinded as make_histos.py was run with (--prescale); if
that pkl was made without --prescale, Data here is fully unblinded.

Example usage:
python python/optimize_wp.py --year 2024 --region signal-all \
    --indir histograms/26Jul10 --outdir plots/26Jul10/wp_scan
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml
from plotting import ratio_plot

from hbb.common_vars import LUMI

hep.style.use("CMS")


def load_tagger_nsv_hists(indir, year, region):
    pkl_path = Path(indir) / f"histograms_tagger_nsv_{year}_{region}.pkl"
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def significance_grid(sig_vals2d, bkg_vals2d, min_bkg):
    """
    sig_vals2d / bkg_vals2d: 2D arrays (partqcd1, nsv1) of weighted yields.
    Returns S_pass, B_pass, Z, each shape (n_tqcd_bins, n_nsv_bins), where
    entry [i, j] is the yield/significance for the cut
    TQCD < tqcd_edges[i + 1]  AND  nSV > j.
    """

    def cum_pass(values2d):
        c = np.cumsum(np.cumsum(values2d, axis=0), axis=1)
        row_total = c[:, -1][:, None]
        return row_total - c

    s_pass = cum_pass(sig_vals2d)
    b_pass = cum_pass(bkg_vals2d)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(b_pass > min_bkg, s_pass / np.sqrt(np.where(b_pass > 0, b_pass, np.nan)), 0.0)
    return s_pass, b_pass, np.nan_to_num(z, nan=0.0)


def select_mass_window(h, mass_lo, mass_hi):
    """Selects the 'inclusive' category and sums msd1/genflavor within [mass_lo, mass_hi)."""
    return h[
        {
            "category": "inclusive",
            "msd1": slice(hist.loc(mass_lo), hist.loc(mass_hi), sum),
        }
    ]


def best_point(s_pass, b_pass, z, tqcd_edges):
    t_idx, n_idx = np.unravel_index(np.argmax(z), z.shape)
    return {
        "tqcd_cut": float(tqcd_edges[t_idx + 1]),
        "nsv_cut": int(n_idx),
        "S": float(s_pass[t_idx, n_idx]),
        "B": float(b_pass[t_idx, n_idx]),
        "Z": float(z[t_idx, n_idx]),
    }


def top_n_points(s_pass, b_pass, z, tqcd_edges, n):
    """Returns up to n (t_idx, n_idx) grid points with Z > 0, ranked by Z descending."""
    flat_order = np.argsort(z, axis=None)[::-1]
    points = []
    for flat_idx in flat_order:
        if len(points) >= n:
            break
        t_idx, n_idx = np.unravel_index(flat_idx, z.shape)
        if z[t_idx, n_idx] <= 0:
            break
        points.append(
            {
                "rank": len(points) + 1,
                "t_idx": int(t_idx),
                "n_idx": int(n_idx),
                "tqcd_cut": float(tqcd_edges[t_idx + 1]),
                "nsv_cut": int(n_idx),
                "S": float(s_pass[t_idx, n_idx]),
                "B": float(b_pass[t_idx, n_idx]),
                "Z": float(z[t_idx, n_idx]),
            }
        )
    return points


def plot_significance(z, tqcd_edges, nsv_edges, title, output_name, best):
    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(tqcd_edges[1:], nsv_edges[:-1], z.T, cmap="viridis", shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$S/\sqrt{B}$")
    ax.set_xlabel("ParT QCD cut (TQCD < x)")
    ax.set_ylabel("nSV cut (nSV > y)")
    ax.set_title(title, pad=15)

    ax.plot(
        best["tqcd_cut"], best["nsv_cut"], marker="*", color="red", markersize=18, markeredgecolor="black"
    )
    ax.annotate(
        f"best: TQCD<{best['tqcd_cut']:.2f}, nSV>{best['nsv_cut']}\nZ={best['Z']:.2f}",
        xy=(best["tqcd_cut"], best["nsv_cut"]),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
        fontsize=11,
        fontweight="bold",
    )

    fig.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_msd_spectrum(h, t_idx, n_idx):
    """Full-range FatJet0_msd spectrum, 'inclusive' category, at a specific (TQCD, nSV) cut.

    Note: deliberately NOT using hist's dict-slice-then-.project() here - that combination
    (narrowing partqcd1/nsv1 via a dict slice alongside a string-selected category, then
    projecting onto msd1) was observed to silently ignore the narrowed sub-range and return
    the full, uncut sum instead (verified: sliced_hist.sum() gives the correct small value,
    but sliced_hist.project('msd1').sum() gives the full unsliced total). Summing the raw
    .values() array with numpy - the same approach significance_grid() uses - sidesteps it.
    """
    hh = h[{"category": "inclusive"}]  # axes now: partqcd1, nsv1, msd1, pt1, genflavor
    values = hh.values()
    msd_values = values[: t_idx + 1, n_idx + 1 :, :, :, :].sum(axis=(0, 1, 3, 4))
    out = hist.Hist(hh.axes["msd1"])
    out.view()[:] = msd_values
    return out


def plot_wp_mass_spectrum(hists, point, mass_lo, mass_hi, style, year, region, outdir):
    histograms_to_plot = {}
    for process, h in hists.items():
        h_proj = build_msd_spectrum(h, point["t_idx"], point["n_idx"])
        if process == "data":
            # Blind the signal mass window, same convention as the other Data/MC plots -
            # on top of whatever --prescale blinding make_histos.py already applied.
            edges = h_proj.axes[0].edges
            mask = (edges[:-1] >= mass_lo) & (edges[:-1] < mass_hi)
            data_val = h_proj.values()
            data_val[mask] = 0
            h_proj.values()[:] = data_val
        histograms_to_plot[process] = h_proj

    legend_title = (
        f"#{point['rank']}: TQCD<{point['tqcd_cut']:.3f}, nSV>{point['nsv_cut']}\n"
        f"S={point['S']:.2f}, B={point['B']:.2f}, "
        + r"S/$\sqrt{B}$"
        + f"={point['Z']:.2f}"
    )
    fig, (ax, _rax) = ratio_plot(
        histograms_to_plot,
        sigs=["Signal"],
        bkgs=["zjets", "wjets", "other", "top"],
        onto="qcd",
        style=style,
        sort_by_yield=True,
        legend_title=legend_title,
    )
    luminosity = LUMI[year] / 1000.0
    hep.cms.label(
        "WiP",
        data=True,
        ax=ax,
        lumi=luminosity,
        lumi_format="{:0.1f}",
        com=13.6,
        year=year,
    )

    output_name = (
        Path(outdir)
        / f"{year}_{region}_wp_scan_rank{point['rank']:02d}_"
        f"TQCD{point['tqcd_cut']:.3f}_nSV{point['nsv_cut']}.png"
    )
    fig.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_name


def main(args):
    hists = load_tagger_nsv_hists(args.indir, args.year, args.region)
    if "Signal" not in hists:
        raise SystemExit("No 'Signal' histogram found - run make_histos.py with the Signal skim first.")

    bkg_keys = [k for k in hists if k not in ("Signal", "data")]
    if not bkg_keys:
        raise SystemExit("No background histograms found.")

    print(
        f"Integrating S/B over FatJet0 msd in [{args.mass_lo:g}, {args.mass_hi:g}) GeV, "
        "inclusive in pT"
    )
    h_sig = select_mass_window(hists["Signal"], args.mass_lo, args.mass_hi).project("partqcd1", "nsv1")
    h_bkg = sum(
        select_mass_window(hists[k], args.mass_lo, args.mass_hi).project("partqcd1", "nsv1")
        for k in bkg_keys
    )

    tqcd_edges = hists["Signal"].axes["partqcd1"].edges
    nsv_edges = hists["Signal"].axes["nsv1"].edges

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"mass{args.mass_lo:g}to{args.mass_hi:g}"

    sig2d = h_sig.values()
    bkg2d = h_bkg.values()
    s_pass, b_pass, z = significance_grid(sig2d, bkg2d, args.min_bkg)

    if z.max() <= 0:
        raise SystemExit("No valid WP found (zero signal or background everywhere).")

    best = best_point(s_pass, b_pass, z, tqcd_edges)
    plot_significance(
        z,
        tqcd_edges,
        nsv_edges,
        f"Inclusive in $p_T$, {args.mass_lo:g}<msd<{args.mass_hi:g} GeV, {args.year}",
        output_dir / f"{args.year}_{args.region}_wp_scan_{tag}_inclusive.png",
        best,
    )
    print(
        f"[best] TQCD < {best['tqcd_cut']:.3f}, nSV > {best['nsv_cut']} "
        f"-> S={best['S']:.3f}, B={best['B']:.3f}, S/sqrt(B)={best['Z']:.3f}"
    )

    top_points = top_n_points(s_pass, b_pass, z, tqcd_edges, args.top_n)
    print(f"\nTop {len(top_points)} candidate working points by S/sqrt(B):")

    style_path = Path(__file__).with_name("style_hbb.yaml")
    with style_path.open() as f:
        style = yaml.safe_load(f)

    for point in top_points:
        print(
            f"  #{point['rank']:>2}: TQCD < {point['tqcd_cut']:.3f}, nSV > {point['nsv_cut']} "
            f"-> S={point['S']:.3f}, B={point['B']:.3f}, S/sqrt(B)={point['Z']:.3f}"
        )
        output_name = plot_wp_mass_spectrum(
            hists, point, args.mass_lo, args.mass_hi, style, args.year, args.region, output_dir
        )
        point["mass_spectrum_plot"] = str(output_name)

    summary_path = output_dir / f"{args.year}_{args.region}_wp_scan_{tag}_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {"mass_window": [args.mass_lo, args.mass_hi], "best": best, "top_points": top_points},
            f,
            indent=2,
        )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan ParT-QCD/nSV working points by S/sqrt(B), inclusive in pT."
    )
    parser.add_argument(
        "--year", type=str, required=True, choices=["2022", "2022EE", "2023", "2023BPix", "2024"]
    )
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument(
        "--indir", type=str, required=True, help="Directory with histograms_tagger_nsv_*.pkl"
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--mass-lo",
        type=float,
        default=100.0,
        help="Lower edge (GeV) of the FatJet0 msd window to integrate S/B over.",
    )
    parser.add_argument(
        "--mass-hi",
        type=float,
        default=140.0,
        help="Upper edge (GeV) of the FatJet0 msd window to integrate S/B over.",
    )
    parser.add_argument(
        "--min-bkg",
        type=float,
        default=0.0,
        help="Minimum weighted background yield required for a WP to be considered, to avoid "
        "S/sqrt(B) blowing up in near-zero-background, low-MC-statistics corners.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top-ranked working points to render a full msd spectrum plot for.",
    )
    args = parser.parse_args()
    main(args)
