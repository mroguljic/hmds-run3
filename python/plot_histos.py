#!/usr/bin/env python3
"""
Unified plotting script for the Hbb analysis.

This script serves as a central manager for creating various types of plots
from the histogram `.pkl` files produced by `make_histos.py`. It can generate
five main types of plots, selectable via the `--plot-type` argument:

1.  `process`: Standard stacked data vs. Monte Carlo plots, with samples
    grouped by their physics process (e.g., Top, W+jets, Z+jets).

2.  `flavor`: Detailed stacked plots where the W+jets and Z+jets backgrounds
    are further broken down by their generator-level quark flavor (b-jet,
    c-jet, light-jet).

3.  `qcd_shape`: A diagnostic plot comparing the normalized shapes of the
    QCD MC distribution in the 'pass' and 'fail' regions to validate
    background estimation techniques.

4.  `nsv`: Data/MC comparisons of the `FatJet0_nSV` distribution.

5.  `nsv2d`: Per-process 2D heatmaps of `FatJet0_ParTPQCD` vs. `FatJet0_nSV`.

6.  `tqcd`: Data/MC comparisons of the `FatJet0_ParTPQCD` (TQCD) distribution, inclusive
    (no nSV or TQCD selection applied).

`--plot-tagger` (with `--tagger-var`) is a separate, orthogonal switch: standalone
shape-only (unit-area) overlays of every process for a single tagger-score axis, inclusive
category only. `--tagger-var` picks the axis: `partqcd` (`FatJet0_ParTPQCD`, the previous
default) or `pnettxbb` (`FatJet0_pnetTXbb`, the PNet Xbb-vs-QCD score used by one of the
signal triggers).

Example usage:
# To plot stacked by process for a single year
python python/plot_histos.py --year 2022EE --region signal-all --indir histograms/25Aug27 --outdir plots --plot-type process

# To plot with flavor breakdown for multiple years combined
python python/plot_histos.py --year 2022EE 2023 --region signal-all --indir histograms/25Aug27 --outdir plots --plot-type flavor

# To plot the QCD shape comparison
python python/plot_histos.py --year 2022EE --region signal-all --indir histograms/25Aug27 --outdir plots --plot-type qcd_shape --norm-type density

# To scan nSV / ParT QCD working points for signal vs. background
python python/plot_histos.py --year 2024 --region signal-all --indir histograms/25Aug27 --outdir plots --plot-type nsv
python python/plot_histos.py --year 2024 --region signal-all --indir histograms/25Aug27 --outdir plots --plot-type nsv2d
python python/plot_histos.py --year 2024 --region signal-all --indir histograms/25Aug27 --outdir plots --plot-type tqcd

# To plot shape-only (unit-area) overlays of the PNet Xbb-vs-QCD score
python python/plot_histos.py --year 2024 --region signal-all --indir histograms/25Aug27 --outdir plots --plot-tagger --tagger-var pnettxbb
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml
from matplotlib.colors import LogNorm
from plotting import ratio_plot

from hbb.common_vars import LUMI

hep.style.use("CMS")

# --- Globals for Plotting Logic ---
process_grouping = {
    "QCD": ["qcd"],
    "Z->qq": ["zjets"],
    "W->qq": ["wjets"],
    "Top": ["tt"],
    "Other": ["diboson"],
    "HMDS": ["Signal"],
}


flavor_map = {3: "b-jet", 2: "c-jet", 1: "light-jet"}

# Config for the standalone shape-only (--plot-tagger) plots: each entry is a single 1D
# tagger-score axis, plotted per-process, normalized to unit area, no ratio panel.
TAGGER_SHAPE_VARS = {
    "partqcd": {
        "axis": "partqcd1",
        "pkl": "histograms_tagger",
        "xlabel": "Jet 0 ParT QCD",
        "tag": "tagger_shape",
    },
    "pnettxbb": {
        "axis": "pnettxbb1",
        "pkl": "histograms_pnettxbb",
        "xlabel": "Jet 0 PNet Xbb (vs QCD)",
        "tag": "pnettxbb_shape",
    },
}


def validate_hist_schema(hists, expected_axes):
    if isinstance(expected_axes, str):
        expected_axes = [expected_axes, "pt1", "category", "genflavor"]
    for process, histogram in hists.items():
        axis_names = [axis.name for axis in histogram.axes]
        if axis_names != expected_axes:
            raise ValueError(
                f"Histogram schema mismatch for '{process}'. "
                f"Expected axes {expected_axes}, got {axis_names}."
            )


# Raw process names in the histograms that aren't top-level keys in style_hbb.yaml
# (they're only defined there via a grouped entry's 'contains' list).
PROCESS_STYLE_FALLBACK = {"tt": "top", "diboson": "other"}


def _process_color(process, style):
    """Looks up a process's plot color from style_hbb.yaml, falling back to its
    grouped entry (e.g. 'tt' -> 'top') and finally to None (matplotlib auto-cycle)."""
    key = process if process in style else PROCESS_STYLE_FALLBACK.get(process)
    return style.get(key, {}).get("color") if key else None


def plot_tagger_shapes(
    hists, category, year_str, outdir, region, axis_name, xlabel, output_tag, style, processes=None
):
    """Standalone shape-only (unit-area) plotting for a single 1D tagger-score axis.

    `processes`, if given, restricts and orders which processes are drawn (e.g.
    ["Signal", "qcd", "data"]) instead of plotting every process in `hists`.
    """
    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes["pt1"]
    plot_order = processes if processes is not None else list(hists.keys())

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"  Processing pt bin: {pt_low} - {pt_high}")

        fig, ax = plt.subplots(figsize=(10, 8))
        n_plotted = 0

        for process in plot_order:
            if process not in hists:
                print(f"  Skipping '{process}': not found in loaded histograms.")
                continue
            h = hists[process]
            h_proj = h[:, i_start, category, :].project(axis_name)
            total = h_proj.sum()
            if total <= 0:
                continue

            h_shape = h_proj / total
            hep.histplot(
                h_shape,
                ax=ax,
                label=process,
                histtype="step",
                yerr=False,
                color=_process_color(process, style),
            )
            n_plotted += 1

        if n_plotted == 0:
            plt.close(fig)
            print("  Skipping pt bin due to zero events in all processes.")
            continue

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized to unity")
        ax.grid(True)
        ax.legend(
            title=f"{category.capitalize()}, {pt_low:g} < $p_T$ < {pt_high:g} GeV",
            prop={"size": 11},
            title_fontsize=12,
            loc="best",
        )

        luminosity = sum(LUMI[y] / 1000.0 for y in year_str.split("-") if y != "all-years")
        hep.cms.label(
            "WiP",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
            loc=0,
        )

        output_name = (
            f"{outdir}/{year_str}_{region}_{category}_{output_tag}_ptbin{pt_low}_{pt_high}.png"
        )
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_nsv_distributions(hists, category, year_str, outdir, region, style):
    """Plots data/MC comparisons for the nSV observable in each pt bin."""
    validate_hist_schema(hists, ["nsv1", "pt1", "category", "genflavor"])

    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"  Processing pt bin: {pt_low} - {pt_high}")

        histograms_to_plot = {}
        for process, h in hists.items():
            h_proj = h[:, i_start, category, :].project("nsv1")
            histograms_to_plot[process] = h_proj

        legend_title = f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"
        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=["Signal"],
            bkgs=["zjets", "wjets", "other", "top"],
            onto="qcd",
            style=style,
            sort_by_yield=True,
            legend_title=legend_title,
            ylabel="Events / bin",
        )

        luminosity = sum(LUMI[y] / 1000.0 for y in year_str.split("-") if y != "all")
        hep.cms.label(
            "WiP",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
        )

        output_name = f"{outdir}/{year_str}_{region}_{category}_nsv_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_tqcd_distributions(hists, category, year_str, outdir, region, style):
    """Plots data/MC comparisons for the FatJet0_ParTPQCD observable in each pt bin."""
    validate_hist_schema(hists, "partqcd1")

    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"  Processing pt bin: {pt_low} - {pt_high}")

        histograms_to_plot = {}
        for process, h in hists.items():
            h_proj = h[:, i_start, category, :].project("partqcd1")
            histograms_to_plot[process] = h_proj

        legend_title = f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"
        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=["Signal"],
            bkgs=["zjets", "wjets", "other", "top"],
            onto="qcd",
            style=style,
            sort_by_yield=True,
            legend_title=legend_title,
            ylabel="Events / bin",
        )

        luminosity = sum(LUMI[y] / 1000.0 for y in year_str.split("-") if y != "all")
        hep.cms.label(
            "WiP",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
        )

        output_name = f"{outdir}/{year_str}_{region}_{category}_tqcd_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_parTqcd_vs_nsv(hists, category, year_str, outdir, region):
    """Plots 2D ParT QCD vs nSV heatmaps for each process in each pt bin."""
    validate_hist_schema(hists, ["partqcd1", "nsv1", "msd1", "pt1", "category", "genflavor"])

    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes["pt1"]

    for process, h in hists.items():
        for i in range(len(pt_axis.edges) - 1):
            pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
            i_start = pt_axis.index(pt_low)
            print(f"  Processing {process}, pt bin: {pt_low} - {pt_high}")

            h_2d = h[{"pt1": i_start, "category": category}].project("partqcd1", "nsv1")
            if h_2d.sum() <= 0:
                continue

            values = h_2d.values().T
            values = values / np.sum(values)
            xedges = h_2d.axes[0].edges
            yedges = h_2d.axes[1].edges

            positive = values[values > 0]
            if positive.size == 0:
                continue

            vmin = max(np.min(positive), 1e-6)
            vmax = np.max(positive)
            values_to_plot = np.where(values > 0, values, np.nan)

            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = plt.cm.viridis.copy()
            cmap.set_bad("white")
            mesh = ax.pcolormesh(
                xedges,
                yedges,
                values_to_plot,
                cmap=cmap,
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label("Event fraction (log scale)")

            ax.set_xlabel(h_2d.axes[0].label)
            ax.set_ylabel(h_2d.axes[1].label)
            ax.set_title(
                f"{process} | {category.capitalize()} | {pt_low:g} < $p_T$ < {pt_high:g} GeV",
                pad=50,
            )
            ax.grid(False)

            hep.cms.label(
                "WiP",
                data=True,
                ax=ax,
                com=13.6,
                year=year_str,
                loc=0,
            )

            fig.subplots_adjust(top=0.80)

            output_name = (
                f"{outdir}/{year_str}_{region}_{category}_{process}_parTQCD_vs_nSV_ptbin{pt_low}_{pt_high}.png"
            )
            fig.savefig(output_name, dpi=300, bbox_inches="tight")
            plt.close(fig)


# --- Function 1: Plotting Stacked by Process ---
def plot_by_process(hists, category, year_str, outdir, region, style):
    """Plots a stacked histogram for each pt bin, with grouping handled by the style file."""

    validate_hist_schema(hists, "msd1")

    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"  Processing pt bin: {pt_low} - {pt_high}")

        # Project all raw histograms to 1D for this pt bin
        histograms_to_plot = {}
        for process, h in hists.items():
            h_proj = h[:, i_start, category, :].project("msd1")
            histograms_to_plot[process] = h_proj

        # Define the lists of signals and backgrounds using the final group names
        # These names must have a corresponding entry in the style file with a 'contains' key
        bkg_order = ["zjets", "wjets", "other", "top"]
        signals = ["Signal"]

        legend_title = f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"
        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=signals,
            bkgs=bkg_order,
            onto="qcd",
            style=style,
            sort_by_yield=True,
            legend_title=legend_title,
        )
        luminosity = sum(LUMI[y] / 1000.0 for y in year_str.split("-") if y != "all")
        hep.cms.label(
            "WiP",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
        )

        output_name = f"{outdir}/{year_str}_{region}_{category}_process_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


# --- Function 2: Plotting Stacked by Flavor ---
def plot_by_flavor(hists, category, year_str, outdir, region, style):
    """Plots a stacked histogram for each pt bin, splitting W/Z jets by flavor."""
    validate_hist_schema(hists, "msd1")
    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"  Processing pt bin: {pt_low} - {pt_high}")

        histograms_to_plot = {}
        for process, h in hists.items():
            if process in ["wjets", "zjets"]:
                h_2d = h[:, i_start, category, :]
                for flavor_code, flavor_name in flavor_map.items():
                    new_key = f"{process}_{flavor_name}"
                    histograms_to_plot[new_key] = h_2d[:, hist.loc(flavor_code)]
            else:
                h_proj = h[:, i_start, category, :].project("msd1")
                histograms_to_plot[process] = h_proj

        bkg_order = [
            "other",
            "top",
            "wjets_light-jet",
            "wjets_c-jet",
            "zjets_light-jet",
            "zjets_c-jet",
            "zjets_b-jet",
        ]

        legend_title = f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"

        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=[],
            bkgs=bkg_order,
            onto="qcd",
            style=style,
            sort_by_yield=True,
            legend_title=legend_title,
        )

        luminosity = sum(LUMI[y] / 1000.0 for y in year_str.split("-") if y != "all-years")
        hep.cms.label(
            "WiP",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
            loc=0,
        )

        output_name = f"{outdir}/{year_str}_{region}_{category}_flavor_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


# --- Function 3: QCD Pass/Fail Shape Comparison ---
def plot_qcd_shapes(hists, year_str, outdir, region, norm_type):
    """For each pt bin, plots the normalized 'pass' and 'fail' distributions for the QCD sample."""
    validate_hist_schema(hists, "msd1")
    if "qcd" not in hists:
        print("No 'qcd' histogram found in the input file. Exiting.")
        return
    h_qcd = hists["qcd"]
    pt_axis = h_qcd.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"Processing pt bin: {pt_low} - {pt_high}")

        h_pass = h_qcd[:, i_start, "pass", :].project("msd1")
        h_fail = h_qcd[:, i_start, "fail", :].project("msd1")

        if h_pass.sum() == 0 or h_fail.sum() == 0:
            print("  Skipping pt bin due to zero events in pass or fail.")
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        if norm_type == "shape":
            hep.histplot(
                h_fail,
                ax=ax,
                label="QCD MC fail",
                color="blue",
                histtype="errorbar",
                yerr=True,
                density=True,
            )
            hep.histplot(
                h_pass,
                ax=ax,
                label="QCD MC pass",
                color="black",
                histtype="errorbar",
                yerr=True,
                density=True,
            )
            ylabel = "Probability Density"

        elif norm_type == "density":
            bin_width = h_pass.axes[0].widths[0]
            pass_yield = h_pass.sum()
            fail_yield = h_fail.sum()
            h_fail_scaled = h_fail * (pass_yield / fail_yield)
            h_pass_toplot = h_pass / bin_width
            h_fail_toplot = h_fail_scaled / bin_width
            ylabel = f"Events / {bin_width:g} GeV"
            hep.histplot(
                h_fail_toplot,
                ax=ax,
                label="QCD MC fail",
                color="blue",
                histtype="errorbar",
                yerr=True,
            )
            hep.histplot(
                h_pass_toplot,
                ax=ax,
                label="QCD MC pass",
                color="black",
                histtype="errorbar",
                yerr=True,
            )

        ax.set_xlabel("Jet $m_{sd}$ [GeV]")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        hep.cms.label("WiP", data=False, ax=ax, com=13.6, year=year_str)

        ax.legend(
            title=f"{pt_low:g} < $p_T$ < {pt_high:g} GeV",
            prop={"size": 14},
            title_fontsize=16,
            loc="upper right",
        )

        output_name = f"{outdir}/{year_str}_{region}_qcd_{norm_type}_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {output_name}")
        plt.close(fig)


# --- Main Function: The Control Center ---
def main(args):
    histograms = {}
    year_str = "all-years" if len(args.year) > 3 else "-".join(args.year)

    for year in args.year:
        if args.plot_tagger:
            pkl_name = f"{TAGGER_SHAPE_VARS[args.tagger_var]['pkl']}_{year}_{args.region}.pkl"
        elif args.plot_type == "nsv":
            pkl_name = f"histograms_nsv_{year}_{args.region}.pkl"
        elif args.plot_type == "nsv2d":
            pkl_name = f"histograms_tagger_nsv_{year}_{args.region}.pkl"
        elif args.plot_type == "tqcd":
            pkl_name = f"histograms_tagger_{year}_{args.region}.pkl"
        else:
            pkl_name = f"histograms_{year}_{args.region}.pkl"
        pkl_path = Path(args.indir) / pkl_name
        if not pkl_path.exists():
            print(f"Error: File not found at {pkl_path}. Skipping.")
            continue
        with pkl_path.open("rb") as f:
            histograms_tmp = pickle.load(f)
            for process, h in histograms_tmp.items():
                if process in histograms:
                    histograms[process] += h
                else:
                    histograms[process] = h

    if not histograms:
        print("No histograms were loaded. Exiting.")
        return

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_path = Path(__file__).with_name("style_hbb.yaml")
    with style_path.open() as f:
        style = yaml.safe_load(f)

    if args.plot_tagger:
        cfg = TAGGER_SHAPE_VARS[args.tagger_var]
        validate_hist_schema(histograms, [cfg["axis"], "pt1", "category", "genflavor"])
        for category in ["inclusive"]:
            print(
                f"Plotting {args.tagger_var} shapes for category: {category}, "
                f"year: {year_str}..."
            )
            plot_tagger_shapes(
                histograms,
                category,
                year_str,
                args.outdir,
                args.region,
                cfg["axis"],
                cfg["xlabel"],
                cfg["tag"],
                style,
                processes=args.tagger_processes,
            )
        return

    # Call the correct plotting function based on --plot-type
    if args.plot_type == "process":
        for category in ["pass", "fail", "nsv_pass", "nsv_fail"]:
            print(f"Plotting histograms by process for category: {category}, year: {year_str}...")
            plot_by_process(histograms, category, year_str, args.outdir, args.region, style)
    elif args.plot_type == "flavor":
        for category in ["pass", "fail"]:
            print(f"Plotting histograms by flavor for category: {category}, year: {year_str}...")
            plot_by_flavor(histograms, category, year_str, args.outdir, args.region, style)
    elif args.plot_type == "qcd_shape":
        print(f"Plotting QCD pass/fail shapes for year: {year_str}...")
        plot_qcd_shapes(histograms, year_str, args.outdir, args.region, args.norm_type)
    elif args.plot_type == "nsv":
        for category in ["inclusive"]:
            print(f"Plotting nSV data/MC for category: {category}, year: {year_str}...")
            plot_nsv_distributions(histograms, category, year_str, args.outdir, args.region, style)
    elif args.plot_type == "nsv2d":
        for category in ["inclusive"]:
            print(f"Plotting ParT QCD vs nSV for category: {category}, year: {year_str}...")
            plot_parTqcd_vs_nsv(histograms, category, year_str, args.outdir, args.region)
    elif args.plot_type == "tqcd":
        for category in ["inclusive"]:
            print(f"Plotting TQCD data/MC for category: {category}, year: {year_str}...")
            plot_tqcd_distributions(histograms, category, year_str, args.outdir, args.region, style)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified plotting script for Hbb analysis.")
    parser.add_argument(
        "--year",
        help="List of years",
        type=str,
        required=True,
        nargs="+",
        choices=["2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument("--indir", help="Input directory for .pkl files", type=str, required=True)
    parser.add_argument("--outdir", help="Output directory for plots", type=str, required=True)
    parser.add_argument("--region", help="Analysis region", type=str, required=True)
    parser.add_argument(
        "--plot-type",
        help="Type of plot to produce",
        type=str,
        default="process",
        choices=["process", "flavor", "qcd_shape", "nsv", "nsv2d", "tqcd"],
    )
    parser.add_argument(
        "--norm-type",
        help="Normalization for QCD shape plot ('shape' or 'density')",
        type=str,
        default="shape",
        choices=["shape", "density"],
    )
    parser.add_argument(
        "--plot-tagger",
        help="Plot standalone shape-normalized (unit-area) tagger-score distributions; "
        "which variable is selected via --tagger-var",
        action="store_true",
    )
    parser.add_argument(
        "--tagger-var",
        help="Which tagger variable to plot shapes for when --plot-tagger is set: "
        "'partqcd' (FatJet0_ParTPQCD, previous default) or 'pnettxbb' (FatJet0_pnetTXbb, "
        "the PNet Xbb-vs-QCD score).",
        type=str,
        default="partqcd",
        choices=list(TAGGER_SHAPE_VARS.keys()),
    )
    parser.add_argument(
        "--tagger-processes",
        help="Restrict --plot-tagger to only these processes (also sets legend/draw order), "
        "e.g. 'Signal qcd data'. Defaults to all processes found in the histograms.",
        type=str,
        nargs="+",
        default=None,
    )
    args = parser.parse_args()
    main(args)
