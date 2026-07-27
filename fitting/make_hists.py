#!/usr/bin/env python3
"""
Histogram Maker - Fully Configuration-Driven
Supports: VBF Hbb Analysis, ZGamma Validation Region

Author(s): Gabi Hamilton, Lara Zygala, Cristina Mantilla

Adapted by Matej Roguljic for Higgs-mediated dark shower search
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path

import hist
import numpy as np
import uproot

from hbb import utils

# --- REGION DIRECTORY MAPPING ---
# Maps the keys in setup.json to the actual directory names on EOS
REGION_MAP = {
    "zgcr": "control-zgamma",
    "mucr": "control-tt",
    "zmmcr": "control-zmumu",
    "vh": "signal-vh",
    "vbf": "signal-vbf",
    "ggf": "signal-ggf",
    "sr": "signal-all",
}


def fill_binned_histogram(
    h, events, region_key, setup, bin_branch="FatJet0_pt", weight_syst="nominal"
):
    for process_name, data in events.items():
        is_data = "data" in process_name.lower()

        # --- 1. WEIGHTING LOGIC ---
        if not is_data and weight_syst != "nominal" and weight_syst in data.columns:
            # Weight-based systematic (pileup, ISR/FSR, ...): scale the nominal
            # finalWeight by the per-event variation ratio (syst weight / nominal
            # weight). Both columns are xsec-normalized, so the ratio is the pure
            # per-event shift; multiplying finalWeight by it cancels sum_genweights
            # and preserves the 1/prescale blinding factor that finalWeight already
            # carries (dividing the raw syst column by sum_genWeight would drop it).
            nom_w = data["weight"].astype(float)
            syst_w = data[weight_syst].astype(float)
            ratio = (syst_w / nom_w).where(nom_w != 0, other=1.0)
            weight_val = data["finalWeight"].astype(float) * ratio
        else:
            # load_samples already calculated finalWeight (weight / sum_genWeight)
            weight_val = data["finalWeight"].astype(float)

        # --- 2. VARIABLE EXTRACTION ---
        var_col = setup["observable"]["branch_name"]
        pt = data["FatJet0_pt"]
        msd = data["FatJet0_msd"]

        # Extract the dynamic binning variable
        bin_data = data[bin_branch]

        # Robust MET extraction from the parquet record
        if "MET" in data.columns:
            met_pt = data["MET"].pt if hasattr(data["MET"], "pt") else data["MET"]
        else:
            met_pt = np.zeros(len(data))

        dphi = np.nan
        if "Photon0_phi" in data.columns and "FatJet0_phi" in data.columns:
            dphi_raw = np.abs(data["Photon0_phi"] - data["FatJet0_phi"])
            dphi = np.where(dphi_raw > np.pi, 2 * np.pi - dphi_raw, dphi_raw)

        var_series = dphi if var_col == "delta_phi_photon_jet" else data[var_col]

        is_mc = "GenFlavor" in data.columns
        genflavordata = (
            data["GenFlavor"].astype(np.int8) if is_mc else np.zeros(len(data), dtype=np.int8)
        )

        # --- 3. SELECTION LOGIC ---
        working_point = setup.get("working_point", 0.82)
        obs_min, obs_max = setup["observable"]["min"], setup["observable"]["max"]
        basic_cuts = (msd > obs_min) & (msd < obs_max)

        # Pull the pt cut directly from the JSON
        pt_min = setup.get("pt_min_scale", 450.0)

        actual_reg_name = REGION_MAP.get(region_key, region_key)

        if "zgamma" in actual_reg_name:
            # Specific Z-Gamma logic from Gabi's script
            trigger = data["Photon200"] | data["Photon110EB_TightID_TightIso"]
            topo_cuts = (dphi > 2.2) & (met_pt < 50) & (data["Photon0_pt"] > 120)
            pre_selection = basic_cuts & topo_cuts & trigger & (pt > pt_min)
        elif "zmumu" in actual_reg_name:
            # Z(mumu) CR: observable is mll, bin variable is dimuon pair pt
            basic_cuts = (var_series > obs_min) & (var_series < obs_max)
            pre_selection = basic_cuts & (bin_data > pt_min)
        else:
            # Lara's Signal Region logic
            pre_selection = basic_cuts & (pt > pt_min)

        if "zmumu" in actual_reg_name:
            # Only inclusive category for zmumu CR — no Txbb pass/fail
            selection_dict = {"inclusive": pre_selection}
        elif "zgamma" in actual_reg_name:
            Txcc = data["FatJet0_ParTPXccVsQCD"]
            Txbb = data["FatJet0_ParTPXbbVsQCD"]
            Txbbxcc = data["FatJet0_ParTPXbbXcc"]
            selection_dict = {
                "pass_bb": pre_selection & (Txbbxcc > working_point) & (Txbb > Txcc),
                "pass_cc": pre_selection & (Txbbxcc > working_point) & (Txcc > Txbb),
                "fail": pre_selection & (Txbbxcc <= working_point),
                "pass": pre_selection & (Txbbxcc > working_point),
                "inclusive": pre_selection,
            }
        else:
            # Dark-shower signal region: anti-QCD tagger + nSV working point.
            TQCD = data["FatJet0_ParTPQCD"]
            nSV = data["FatJet0_nSV"]
            selection_dict = {
                "pass": pre_selection & (TQCD < 0.075) & (nSV > 6),
                "fail": pre_selection & (TQCD > 0.075) & (TQCD < 0.6) & (nSV > 6),
                "inclusive": pre_selection,
            }

        # --- 4. FILLING ---
        for category, selection in selection_dict.items():
            if category in h.axes["category"]:
                h.fill(
                    var_series[selection],
                    bin_data[selection],  # using dynamic bin data here
                    category=category,
                    genflavor=genflavordata[selection],
                    weight=weight_val[selection],
                )
    return h


def export_to_root(histograms, output_root_path, region_key, samples_qq, syst, data_key):
    suffix = "nominal" if syst == "nominal" else syst

    # Ensure directory exists
    output_root_path.parent.mkdir(parents=True, exist_ok=True)

    # Use uproot.update for writing
    # If the file doesn't exist, uproot.update will create it.
    with uproot.update(output_root_path) as fout:
        for process, h in histograms.items():
            # Standard Combine naming for data
            proc_name = "data_obs" if process == data_key else process

            # Check if process should be split (e.g., Wjets, Zjets)
            # and ensure we aren't trying to split the actual data stream
            should_split = any(s in process for s in samples_qq) and "data" not in process.lower()

            bin_axis = h.axes[1]
            bin_prefix = bin_axis.name  # e.g., "pt" or "mjj"

            for i_bin in range(len(bin_axis.edges) - 1):
                # Dynamically construct the bin string, e.g., pt1 or mjj1
                bin_str = f"{bin_prefix}{i_bin+1}"
                for category in h.axes["category"]:
                    base = f"{region_key}_{category}_{bin_str}_{proc_name}"

                    if should_split:
                        fout[f"{base}bb_{suffix}"] = h[:, i_bin, category, 3]
                        fout[f"{base}c_{suffix}"] = h[:, i_bin, category, 2]
                        fout[f"{base}light_{suffix}"] = h[:, i_bin, category, 1]    #1 = uds
                    else:
                        fout[f"{base}_{suffix}"] = h[:, i_bin, category, sum]


def main(args):
    with Path(args.setup).open() as f:
        setup = json.load(f)
    with Path("pmap_run3.json").open() as f:
        pmap = json.load(f)

    # We are working with private signal atm so it ends up in a different dir
    MAIN_DATA_DIR = "/eos/uscms/store/user/roguljic/lpchmdsrun3/"
    bkg_data_dir = (
        Path(args.data_dir) if args.data_dir else Path(MAIN_DATA_DIR) / args.tag / args.year
    )
    sig_data_dir = (
        Path(args.data_dir) if args.data_dir else Path(MAIN_DATA_DIR) / args.sig_tag / args.year
    )

    do_BDT_regions = setup.get("do_BDT_regions", False)

    for region_key, reg_cfg in setup["categories"].items():
        print("\n" + "=" * 50)
        print(f"STARTING REGION: {region_key}")
        print("=" * 50)

        pt_bins = np.array(reg_cfg["bins"])

        # Get dynamic branch and prefix (fallback to pt if not defined in json)
        bin_branch = reg_cfg.get("bin_branch", "FatJet0_pt")
        bin_prefix = reg_cfg.get("bin_prefix", "pt")

        obs = setup["observable"]
        region_to_load = REGION_MAP.get(region_key, region_key)
        if do_BDT_regions and not "control" in region_to_load:
            region_to_load = region_to_load + "-BDT"

        # Determine Data Stream (e.g., EGammadata for zgamma)
        if "zgamma" in region_to_load:
            data_map_key = "EGammadata"
        elif "zmumu" in region_to_load:
            data_map_key = "Muondata"
        elif "tt" in region_to_load:
            data_map_key = "Muondata"
        else:
            data_map_key = "Jetdata"

        folder_systs = ["JES", "JER", "UES", "MuonPTScale", "MuonPTRes"]
        systs_to_run = ["nominal"]
        if setup.get("do_systematics"):
            for s in setup.get("active_systematics", []):
                systs_to_run.extend([f"{s}Up", f"{s}Down"])

        obs_name = setup["observable"]["name"]  # e.g., "msd"

        output_root = Path(args.outdir) / f"fitting_{args.year}_{region_key}_{obs_name}.root"

        # Ensure outdir exists
        output_root.parent.mkdir(parents=True, exist_ok=True)

        # Delete existing file to start fresh for this region
        # (This avoids the errors by ensuring we never 'update' a corrupted or old file)
        if output_root.exists():
            print(f"Cleaning up existing file: {output_root}")
            output_root.unlink()

        # Initialize a fresh ROOT file
        uproot.recreate(output_root).close()

        # Update the axis_bin initialization for dynamic variable
        axis_var = hist.axis.Regular(obs["nbins"], obs["min"], obs["max"], name=obs["name"])
        axis_bin = hist.axis.Variable(pt_bins, name=bin_prefix)  # Replaced axis_pt
        axis_cat = hist.axis.StrCategory(
            ["pass_bb", "pass_cc", "fail", "pass", "inclusive"],
            name="category",
        )
        axis_flav = hist.axis.IntCategory([0, 1, 2, 3], name="genflavor")

        # Define columns to load
        cols = [
            "weight",
            "FatJet0_pt",
            "FatJet0_msd",
            "FatJet0_ParTPXbbVsQCD",
            "FatJet0_ParTPXccVsQCD",
            "FatJet0_ParTPXbbXcc",
            "FatJet0_ParTPQCD",
            "FatJet0_nSV",
            "GenFlavor",
        ]
        if data_map_key == "EGammadata":
            cols += [
                "Photon0_pt",
                "Photon0_phi",
                "FatJet0_phi",
                "MET",
                "Photon200",
                "Photon110EB_TightID_TightIso",
            ]

        # Ensure the dynamic bin branch is loaded
        if bin_branch not in cols:
            cols.append(bin_branch)

        obs_branch = setup["observable"]["branch_name"]
        if obs_branch not in cols:
            cols.append(obs_branch)

        # ------------------------------------------------------------------
        # Build loose PyArrow row filters from the setup config.
        # These are applied at read time (predicate pushdown) — rows that
        # fail are never loaded into RAM, which is critical for large MC
        # samples like GJets that have O(100M) events in the parquet.
        # Use slightly looser cuts than the analysis selection so we don't
        # accidentally lose events at bin edges.
        # ------------------------------------------------------------------
        pq_filters = [
            ("FatJet0_msd", ">=", float(obs["min"])),
            ("FatJet0_msd", "<=", float(obs["max"])),
            ("FatJet0_pt",  ">=", float(pt_bins[0])),
        ]
        if "zgamma" in region_to_load:
            # Photon0_pt > 120 is the analysis cut; pre-filter at 100 to
            # keep a small margin while cutting ~90% of low-pT GJets rows.
            pq_filters.append(("Photon0_pt", ">=", 100.0))

        for syst in systs_to_run:
            print(f"\n>>> Running Systematic Pass: {syst}")
            is_folder = any(fs in syst for fs in folder_systs)
            variation = syst if is_folder else "nominal"

            # Weight-based systematics are extra columns in the nominal parquet
            # (e.g. pileupUp, ISRPartonShowerDown). Load that column for this pass so
            # the fill can actually read it — otherwise it silently falls back to the
            # nominal weight and the systematic collapses to a null 1.0000 effect.
            weight_syst_col = syst if (not is_folder and syst != "nominal") else None
            load_cols = [*cols, weight_syst_col] if weight_syst_col else cols

            histograms = {}
            for process, datasets in pmap.items():
                if "data" in process.lower() and process != data_map_key:
                    continue
                if process == data_map_key and syst != "nominal":
                    continue

                h = hist.Hist(axis_var, axis_bin, axis_cat, axis_flav)
                for dataset in datasets:
                    events = utils.load_samples(
                        data_dir=sig_data_dir if process == "Signal" else bkg_data_dir,
                        samples={process: [dataset]},
                        columns=load_cols,
                        region=region_to_load,
                        variation=variation,
                        filters=pq_filters,
                        prescale=args.prescale,
                    )
                    if events:
                        # Pass the dynamic branch
                        h = fill_binned_histogram(
                            h,
                            events,
                            region_key,
                            setup,
                            bin_branch=bin_branch,
                            weight_syst=syst if not is_folder else "nominal",
                        )
                    # Memory management within dataset loop
                    del events
                    gc.collect()

                if h.sum() > 0:
                    histograms[process] = h

            if args.save_root:
                export_to_root(
                    histograms,
                    output_root,
                    region_key,
                    setup.get("samples_qq", []),
                    syst,
                    data_map_key,
                )
                print(
                    f"  [SUCCESS] Updated {output_root} with {len(histograms)} processes for systematic: {syst}"
                )

            # --- NEW: Save Pickles for plotting pipeline ---
            pickle_path = (
                Path(args.outdir) / f"hists_{args.year}_{region_key}_{obs_name}_{syst}.pkl"
            )
            with pickle_path.open("wb") as f:
                pickle.dump(histograms, f)
            print(f"  [SUCCESS] Pickles saved to: {pickle_path}")

            # Memory management after each systematic pass
            del histograms
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Histogram Maker for Signal and CR")
    parser.add_argument("--year", required=True, choices=["2022", "2022EE", "2023", "2023BPix", "2024"])
    parser.add_argument(
        "--tag", required=True,
        help="Tag for the skims directory",
    )
    parser.add_argument(
        "--sig-tag", default="260706_v15_private",
        help="Tag for the signals directory (while we are working with private signals)",
    )
    parser.add_argument("--setup", required=True, help="Path to setup.json file")
    parser.add_argument("--outdir", default="results", help="Directory to save ROOT files")
    parser.add_argument("--save-root", action="store_true", help="Actually write the ROOT file")
    parser.add_argument(
        "--data-dir", default=None,
        help="Override the full path to the parquet directory for this year (used for both "
             "background/data and Signal). Skips the --tag/--sig-tag-based path construction.",
    )
    parser.add_argument(
        "--prescale",
        type=int,
        default=10,
        help="If >1, blind data to a fixed 1/prescale subset via event %% prescale == 0. "
        "MC is never prescaled. Defaults to 10 (the current blinding policy); pass "
        "--prescale 1 for a full-lumi, unblinded run.",
    )

    args = parser.parse_args()

    # Ensure outdir exists before starting
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    main(args)
