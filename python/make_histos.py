#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import hist
import numpy as np
from common import common_mc, data_by_year

from hbb import utils

# Define the possible ptbins
ptbins = np.array([300, 450, 500, 550, 600, 675, 800, 1200])

# Define the histogram axes
axis_to_histaxis = {
    "pt1": hist.axis.Variable(ptbins, name="pt1", label=r"Jet 0 $p_{T}$ [GeV]"),
    "pt2": hist.axis.Variable(ptbins, name="pt2", label=r"Jet 1 $p_{T}$ [GeV]"),
    "msd1": hist.axis.Regular(23, 40, 201, name="msd1", label="Jet 0 $m_{sd}$ [GeV]"),
    "mass1": hist.axis.Regular(30, 0, 200, name="mass1", label="Jet 0 PNet mass [GeV]"),
    "nsv1": hist.axis.Regular(16, -0.5, 15.5, name="nsv1", label="Jet 0 nSV"),
    "partqcd1": hist.axis.Regular(40, 0.0, 1.0, name="partqcd1", label="Jet 0 ParT QCD"),
    "category": hist.axis.StrCategory([], name="category", label="Category", growth=True),
    "genflavor": hist.axis.IntCategory([0, 1, 2, 3], name="genflavor", label="Gen Flavor"),
}

# add more as needed
axis_to_column = {
    "pt1": "FatJet0_pt",
    "pt2": "FatJet1_pt",
    "msd1": "FatJet0_msd",
    "mass1": "FatJet0_pnetMass",
    "nsv1": "FatJet0_nSV",
    "partqcd1": "FatJet0_ParTPQCD",
    "category": "category",
    "genflavor": "GenFlavor",
}


def fill_ptbinned_histogram(h, events, axis):
    """
    Fills a histogram with events from a single dataset.
    """
    for _process_name, data in events.items():
        weight_val = data["finalWeight"].astype(float)
        var = data[axis_to_column[axis]]

        isRealData = "GenFlavor" not in data.columns
        genflavordata = (
            data["GenFlavor"].astype(int) if not isRealData else np.zeros_like(var, dtype=int)
        )

        # Event selection
        TQCD = data["FatJet0_ParTPQCD"]
        msd = data["FatJet0_msd"]
        pt = data["FatJet0_pt"]
        nsv = data["FatJet0_nSV"]
        pre_selection = (msd > 40) & (msd < 200) & (pt > 300) & (pt < 1200)
        selection_dict = {
            "pass": pre_selection & ((TQCD < 0.075) & (nsv > 6)),
            "fail": pre_selection & ((TQCD > 0.075) & (TQCD < 0.6) & (nsv > 6)),
            "nsv_pass": pre_selection & ((nsv > 6) & (TQCD < 0.6)),
            "nsv_fail": pre_selection & ((nsv <= 6) & (TQCD < 0.6)),
            "inclusive": pre_selection,  # No tagger/nSV cut — use for WP scans
        }

        # Fill histograms
        for category, selection in selection_dict.items():
            h.fill(
                var[selection],
                pt[selection],
                category=category,
                genflavor=genflavordata[selection],
                weight=weight_val[selection],
            )
    return h


def fill_tagger_nsv_histogram(h, events):
    """
    Fills a 2D ParT QCD vs nSV histogram with events from a single dataset.
    """
    for _process_name, data in events.items():
        weight_val = data["finalWeight"].astype(float)
        tqcd = data["FatJet0_ParTPQCD"]
        nsv = data["FatJet0_nSV"]

        isRealData = "GenFlavor" not in data.columns
        genflavordata = (
            data["GenFlavor"].astype(int) if not isRealData else np.zeros_like(tqcd, dtype=int)
        )

        # Event selection
        msd = data["FatJet0_msd"]
        pt = data["FatJet0_pt"]
        pre_selection = (msd > 40) & (msd < 200) & (pt > 300) & (pt < 1200)
        selection_dict = {
            "pass": pre_selection & (tqcd < 0.6),
            "fail": pre_selection & (tqcd > 0.6),
            "inclusive": pre_selection,
        }

        for category, selection in selection_dict.items():
            h.fill(
                tqcd[selection],
                nsv[selection],
                msd[selection],
                pt[selection],
                category=category,
                genflavor=genflavordata[selection],
                weight=weight_val[selection],
            )
    return h


def main(args):
    year = args.year
    region = args.region

    # Backgrounds/data (standard v15 NanoAOD) and Signal (v15_private PFNanoAOD) were skimmed
    # into separate tags/directories, so each process is loaded from its own base directory.
    MAIN_DIR = "/eos/uscms/store/user/roguljic/lpchmdsrun3/"
    bkg_data_dir = Path(MAIN_DIR) / args.bkg_tag / year
    sig_data_dir = Path(MAIN_DIR) / args.sig_tag / year

    load_columns_mc = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_nSV",
        "FatJet0_pnetTXbb",
        "FatJet0_ParTPQCD",
        "GenFlavor",
    ]
    load_columns_data = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_nSV",
        "FatJet0_pnetTXbb",
        "FatJet0_ParTPQCD",
    ]
    filters = None

    histograms = {}
    histograms_nsv = {}
    tagger_histograms = {}
    tagger_nsv_histograms = {}
    samples = {
        **common_mc,
        "data": data_by_year[year],
    }

    for process, datasets in samples.items():
        load_columns = load_columns_data if process == "data" else load_columns_mc
        data_dir = sig_data_dir if process == "Signal" else bkg_data_dir
        print(f"Processing {process} for year {year}...")

        h = hist.Hist(
            axis_to_histaxis["msd1"],
            axis_to_histaxis["pt1"],
            axis_to_histaxis["category"],
            axis_to_histaxis["genflavor"],
        )
        h_nsv = hist.Hist(
            axis_to_histaxis["nsv1"],
            axis_to_histaxis["pt1"],
            axis_to_histaxis["category"],
            axis_to_histaxis["genflavor"],
        )
        h_tagger = hist.Hist(
            axis_to_histaxis["partqcd1"],
            axis_to_histaxis["pt1"],
            axis_to_histaxis["category"],
            axis_to_histaxis["genflavor"],
        )
        h_tagger_nsv = hist.Hist(
            axis_to_histaxis["partqcd1"],
            axis_to_histaxis["nsv1"],
            axis_to_histaxis["msd1"],
            axis_to_histaxis["pt1"],
            axis_to_histaxis["category"],
            axis_to_histaxis["genflavor"],
        )

        # Loop through each dataset within the process
        for dataset in datasets:
            events = utils.load_samples(
                data_dir,
                {process: [dataset]},  # Pass a list with a single dataset
                columns=load_columns,
                region=region,
                filters=filters,
                prescale=args.prescale,
            )

            if not events:
                print(f"No events found for dataset {dataset} in year {year}. Skipping.")
                continue

            h = fill_ptbinned_histogram(h, events, "msd1")
            h_nsv = fill_ptbinned_histogram(h_nsv, events, "nsv1")
            h_tagger = fill_ptbinned_histogram(h_tagger, events, "partqcd1")
            h_tagger_nsv = fill_tagger_nsv_histogram(h_tagger_nsv, events)

        if h.sum() == 0:
            print(
                f"WARNING: No events were found for the entire '{process}' process group. Skipping."
            )
            continue
        histograms[process] = h
        if h_nsv.sum() > 0:
            histograms_nsv[process] = h_nsv
        if h_tagger.sum() > 0:
            tagger_histograms[process] = h_tagger
        if h_tagger_nsv.sum() > 0:
            tagger_nsv_histograms[process] = h_tagger_nsv

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        f"histograms_{year}_{region}.pkl": histograms,
        f"histograms_nsv_{year}_{region}.pkl": histograms_nsv,
        f"histograms_tagger_{year}_{region}.pkl": tagger_histograms,
        f"histograms_tagger_nsv_{year}_{region}.pkl": tagger_nsv_histograms,
    }
    for name, hists in outputs.items():
        output_file = output_dir / name
        with output_file.open("wb") as f:
            pickle.dump(hists, f)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make histograms for a given year.")
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument(
        "--region",
        help="region",
        type=str,
        required=True,
        choices=[
            "signal-all",
            "signal-ggf",
            "signal-vh",
            "signal-vbf",
            "control-tt",
            "control-zgamma",
        ],
    )
    parser.add_argument(
        "--outdir", help="Output directory to save histograms.", type=str, default="histograms"
    )
    parser.add_argument(
        "--bkg-tag",
        help="Condor tag (under /eos/uscms/store/user/roguljic/lpchmdsrun3/) holding background/data skims (v15).",
        type=str,
        default="260706_v15",
    )
    parser.add_argument(
        "--sig-tag",
        help="Condor tag holding the Signal skim (v15_private).",
        type=str,
        default="260706_v15_private",
    )
    parser.add_argument(
        "--prescale",
        help="If >1, blind data to a fixed 1/prescale subset via event %% prescale == 0. "
        "MC is never prescaled. Requires 'event' to be saved in the skim. Defaults to 10 "
        "(the current blinding policy); pass --prescale 1 for a full-lumi, unblinded run.",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    main(args)
