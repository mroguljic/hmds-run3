"""
Datacard Maker - Fully Configuration-Driven
Supports: VBF Hbb Analysis, ZGamma Validation Region

Author(s): Gabi Hamilton, Lara Zygala, Cristina Mantilla
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import rhalphalib as rl
import ROOT
from card_utils import (
    add_systematics,
    badtemplate,
    get_merged_template,
    get_template,
    plot_mctf,
    one_bin
)

from hbb.common_vars import LUMI

ROOT.gROOT.SetBatch(True)
warnings.filterwarnings("ignore")
rl.util.install_roofit_helpers()

lumi_err = {"2022": 1.01, "2023": 1.02, "2024": 1.02}  # 2024: TODO get official CMS value
eps = 0.001 


def rhalphabet(args):
    # ---------------------------------------------------------
    # 1. SETUP & LOAD CONFIG
    # ---------------------------------------------------------
    year = args.year
    tag = args.tag
    analysis = args.analysis
    print(f"Running Card Maker for {year} (Analysis: {analysis})")

    # Define Paths
    working_dir = Path(f"{args.outdir}/{tag}/{year}")
    datacard_dir = working_dir / "datacards"
    initvals_dir = working_dir / "initial_vals"

    datacard_dir.mkdir(parents=True, exist_ok=True)
    initvals_dir.mkdir(parents=True, exist_ok=True)
    if not (initvals_dir / "initial_vals_data_vh_bb.json").exists():
        os.popen(f"cp -r initial_vals/* {initvals_dir}/ 2>/dev/null")

    # Load Configuration
    json_name = f"setup_{analysis}.json"
    with Path(json_name).open() as f:
        config = json.load(f)

    # ---------------------------------------------------------
    # 2. READ SETTINGS FROM JSON
    # ---------------------------------------------------------

    # Files & naming
    root_file_name = config.get("root_filename", "signalregion.root").replace("{year}", year)

    # [FIXED] Define infile_path here!
    infile_path = Path(args.indir) / root_file_name if args.indir else working_dir / root_file_name

    qcd_tf_proc = config.get("qcd_proc", "QCD")
    pt_min_scale = config.get("pt_min_scale", 450.0)
    regions_to_fit = config.get("regions_to_fit", ["bb"])
    rho_scaling_min = config.get("rho_scaling_min", -6.0)
    rho_scaling_max = config.get("rho_scaling_max", -2.1)

    # Process Definitions
    sample_dict = config.get("process_groups")
    for group in sample_dict.values():
        group["components"] = [tuple(c) for c in group["components"]]

    # ---------------------------------------------------------
    # 3. OBSERVABLES & SYSTEMATICS
    # ---------------------------------------------------------
    msd_cfg = config["observable"]
    msdbins = np.linspace(msd_cfg["min"], msd_cfg["max"], msd_cfg["nbins"] + 1)
    msd = rl.Observable(msd_cfg["name"], msdbins)

    cats_cfg = config["categories"]
    cats = list(cats_cfg.keys())
    if "mucr" in cats:
        cats.remove("mucr") #necessary since all categories get the same treatment

    # TT Independent Parameters
    tqqeffSF = rl.IndependentParameter(f'tqqeffSF_{year}', 1., -50, 50)
    tqqeffBCSF = rl.IndependentParameter(f'tqqeffBCSF_{year}', 1., -50, 50)
    tqqnormSF = rl.IndependentParameter(f'tqqnormSF_{year}', 1., -50, 50)

    do_muon_CR = config.get("do_muon_CR", False)

    # Standard Luminosity Uncertainty
    sys_lumi_uncor = rl.NuisanceParameter(f"CMS_lumi_13TeV_{year[:4]}", "lnN")

    do_systematics = config.get("do_systematics", False)
    syst_map = {}

    if do_systematics:
        # --- A. Experimental Systematics (from sys_dict) ---
        available_exp_systs = {
            "pileup": rl.NuisanceParameter(f"CMS_PU_{year}", "lnN"),
            "JES": rl.NuisanceParameter(f"CMS_scale_j_{year}", "lnN"),
            "JER": rl.NuisanceParameter(f"CMS_res_j_{year}", "lnN"),
            "UES": rl.NuisanceParameter(f"CMS_ues_j_{year}", "lnN"),
            "ISRPartonShower": rl.NuisanceParameter("ISRPartonShower", "lnN"),
            "FSRPartonShower": rl.NuisanceParameter("FSRPartonShower", "lnN"),
            "MuonPTScale": rl.NuisanceParameter(f"CMS_scale_m_{year}", "lnN"),
            "MuonPTRes": rl.NuisanceParameter(f"CMS_res_m_{year}", "lnN"),
            "btagSFb": rl.NuisanceParameter(f"CMS_btagSFb_{year}", "lnN"),
            "btagSFc": rl.NuisanceParameter(f"CMS_btagSFc_{year}", "lnN"),
            "btagSFlight": rl.NuisanceParameter(f"CMS_btagSFlight_{year}", "lnN"),
            "btagSFb_correlated": rl.NuisanceParameter(f"CMS_btagSFb_correlated_{year}", "lnN"),
            "btagSFc_correlated": rl.NuisanceParameter(f"CMS_btagSFc_correlated_{year}", "lnN"),
            "btagSFlight_correlated": rl.NuisanceParameter(
                f"CMS_btagSFlight_correlated_{year}", "lnN"
            ),
        }

        # --- B. Theory Systematics (PDF, Scale, ISR/FSR) ---
        theory_systs = {
            "pdf_ggF": rl.NuisanceParameter("pdf_Higgs_ggF", "lnN"),
            "pdf_VBF": rl.NuisanceParameter("pdf_Higgs_VBF", "lnN"),
            "pdf_VH": rl.NuisanceParameter("pdf_Higgs_VH", "lnN"),
            "pdf_ttH": rl.NuisanceParameter("pdf_Higgs_ttH", "lnN"),
            "scale_ggF": rl.NuisanceParameter("QCDscale_ggF", "lnN"),
            "scale_VBF": rl.NuisanceParameter("QCDscale_VBF", "lnN"),
            "scale_VH": rl.NuisanceParameter("QCDscale_VH", "lnN"),
            "scale_ttH": rl.NuisanceParameter("QCDscale_ttH", "lnN"),
            "isr_ggF": rl.NuisanceParameter("ISRPartonShower_ggF", "lnN"),
            "isr_VBF": rl.NuisanceParameter("ISRPartonShower_VBF", "lnN"),
            "isr_VH": rl.NuisanceParameter("ISRPartonShower_VH", "lnN"),
            "isr_ttH": rl.NuisanceParameter("ISRPartonShower_ttH", "lnN"),
            "fsr_ggF": rl.NuisanceParameter("FSRPartonShower_ggF", "lnN"),
            "fsr_VBF": rl.NuisanceParameter("FSRPartonShower_VBF", "lnN"),
            "fsr_VH": rl.NuisanceParameter("FSRPartonShower_VH", "lnN"),
            "fsr_ttH": rl.NuisanceParameter("FSRPartonShower_ttH", "lnN"),
        }

        # Combine all available systematics into one map
        all_available = {**available_exp_systs, **theory_systs}

        # Pull active list from JSON config
        active_list = config.get("active_systematics", [])
        for name in active_list:
            if name in all_available:
                syst_map[name] = all_available[name]
            else:
                print(f"Warning: Systematic {name} requested in JSON but not defined in script.")

    # ---------------------------------------------------------
    # 4. QCD ESTIMATION LOOP
    # ---------------------------------------------------------
    tf_params = {}
    validbins = {}

    for cat in cats:
        if "bins_pt" in cats_cfg[cat]:
            ptbins = np.array(cats_cfg[cat]["bins_pt"])
        else:
            ptbins = np.array(cats_cfg[cat]["bins"])
        npt = len(ptbins) - 1

        # Grid Setup
        ptpts, msdpts = np.meshgrid(
            ptbins[:-1] + 0.3 * np.diff(ptbins),
            msdbins[:-1] + 0.5 * np.diff(msdbins),
            indexing="ij",
        )
        rhopts = 2 * np.log(msdpts / ptpts)
        ptscaled = (ptpts - pt_min_scale) / (1200.0 - pt_min_scale)
        rhoscaled = (rhopts - rho_scaling_min) / (rho_scaling_max - rho_scaling_min)

        validbins[cat] = (rhoscaled >= 0.0) & (rhoscaled <= 1.0)
        rhoscaled[~validbins[cat]] = 1

        tf_params[cat] = {}
        fitfailed_qcd = {}

        for reg in regions_to_fit:
            fitfailed_qcd[reg] = 0
            while fitfailed_qcd[reg] < 5:
                qcdmodel = rl.Model(f"qcdmodel_{cat}_{reg}")
                qcdpass, qcdfail = 0.0, 0.0

                for ptbin in range(npt):
                    binindex = ptbin
                    if analysis == "vbf" and "hi" in cat:
                        binindex = 1

                    failCh = rl.Channel(f"ptbin{ptbin}{cat}fail{year}{reg}")
                    passCh = rl.Channel(f"ptbin{ptbin}{cat}pass{year}{reg}")
                    qcdmodel.addChannel(failCh)
                    qcdmodel.addChannel(passCh)

                    failTempl = get_template(
                        infile_path, qcd_tf_proc, "fail_", binindex + 1, cat, msd, "nominal"
                    )
                    passTempl = get_template(
                        infile_path, qcd_tf_proc, f"pass_{reg}_", binindex + 1, cat, msd, "nominal"
                    )

                    failCh.setObservation(failTempl, read_sumw2=True)
                    passCh.setObservation(passTempl, read_sumw2=True)
                    qcdfail += failCh.getObservation()[0].sum()
                    qcdpass += passCh.getObservation()[0].sum()

                qcdeff = qcdpass / qcdfail
                print(f"Inclusive P/F ({cat} {reg}) = {qcdeff:.4f}")

                # Initial Values Loading
                initF = initvals_dir / f"initial_vals_{cat}_{reg}.json"
                initial_vals = None
                if initF.exists():
                    with initF.open() as f:
                        loaded = np.array(json.load(f)["initial_vals"])
                    if (loaded.shape[0] - 1 == args.mc_pt_order) and (
                        loaded.shape[1] - 1 == args.mc_rho_order
                    ):
                        initial_vals = loaded
                    else:
                        print(f"Order Mismatch for {reg}. Resetting.")

                if initial_vals is None:
                    initial_vals = np.full((args.mc_pt_order + 1, args.mc_rho_order + 1), 1.0)

                tf_MCtempl = rl.BasisPoly(
                    f"tf_MCtempl_{cat}{reg}{year}",
                    (args.mc_pt_order, args.mc_rho_order),
                    ["pt", "rho"],
                    basis="Bernstein",
                    init_params=initial_vals,
                    limits=(0, 10),
                )

                tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)

                for ptbin in range(npt):
                    failCh = qcdmodel[f"ptbin{ptbin}{cat}fail{year}{reg}"]
                    passCh = qcdmodel[f"ptbin{ptbin}{cat}pass{year}{reg}"]

                    failObs = failCh.getObservation()[0]
                    qcdparams = np.array(
                        [
                            rl.IndependentParameter(f"qcdparam_ptbin{ptbin}{cat}{year}{reg}_{i}", 0)
                            for i in range(msd.nbins)
                        ]
                    )
                    scaledparams = (
                        failObs * (1 + 10.0 / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
                    )

                    fail_qcd = rl.ParametericSample(
                        f"ptbin{ptbin}{cat}fail{year}{reg}_qcd",
                        rl.Sample.BACKGROUND,
                        msd,
                        scaledparams,
                    )
                    failCh.addSample(fail_qcd)

                    pass_qcd = rl.TransferFactorSample(
                        f"ptbin{ptbin}{cat}pass{year}{reg}_qcd",
                        rl.Sample.BACKGROUND,
                        tf_MCtempl_params[ptbin, :],
                        fail_qcd,
                    )
                    passCh.addSample(pass_qcd)

                    failCh.mask = validbins[cat][ptbin]
                    passCh.mask = validbins[cat][ptbin]

                # Fit
                qcdfit_ws = ROOT.RooWorkspace("w")
                simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
                qcdfit = simpdf.fitTo(
                    obs,
                    ROOT.RooFit.Extended(True),
                    ROOT.RooFit.SumW2Error(True),
                    ROOT.RooFit.Strategy(2),
                    ROOT.RooFit.Save(),
                    ROOT.RooFit.PrintLevel(-1),
                )

                if qcdfit.status() != 0:
                    fitfailed_qcd[reg] += 1
                else:
                    allparams = dict(zip(qcdfit.nameArray(), qcdfit.valueArray()))
                    pvalues = [allparams[p.name] for p in tf_MCtempl.parameters.reshape(-1)]
                    new_values = np.array(pvalues).reshape(tf_MCtempl.parameters.shape)
                    with initF.open("w") as outfile:
                        json.dump({"initial_vals": new_values.tolist()}, outfile)
                    break

            if fitfailed_qcd[reg] >= 5:
                raise RuntimeError(f"Could not fit QCD for {cat} {reg} after 5 tries!")

            plot_mctf(
                tf_MCtempl,
                msdbins,
                f"{cat}_{reg}",
                year,
                tag,
                str(working_dir),
                pt_min=pt_min_scale,  # <--- Pass from config
                rho_min=rho_scaling_min,  # <--- Pass from config
                rho_max=rho_scaling_max,  # <--- Pass from config (-1.0 for ZG)
            )

            param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
            decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(
                tf_MCtempl.name + "_deco", qcdfit, param_names
            )
            tf_MCtempl.parameters = decoVector.correlated_params.reshape(
                tf_MCtempl.parameters.shape
            )

            # Residual
            resid_init = np.full((args.res_pt_order + 1, args.res_rho_order + 1), 1.0)
            tf_dataResidual = rl.BasisPoly(
                f"tf_dataResidual_{year}{cat}{reg}",
                (args.res_pt_order, args.res_rho_order),
                ["pt", "rho"],
                basis="Bernstein",
                init_params=resid_init,
                limits=(0, 20),
            )
            tf_params[cat][reg] = (
                qcdeff * tf_MCtempl(ptscaled, rhoscaled) * tf_dataResidual(ptscaled, rhoscaled)
            )

    # ---------------------------------------------------------
    # 5. MAIN MODEL BUILDING
    # ---------------------------------------------------------
    model = rl.Model(f"{analysis}Model_{year}")

    for cat in cats:
        if "bins_pt" in cats_cfg[cat]:
            ptbins = np.array(cats_cfg[cat]["bins_pt"])
        else:
            ptbins = np.array(cats_cfg[cat]["bins"])

        for ptbin in range(len(ptbins) - 1):
            binindex = ptbin
            # Handle the VBF hi/lo binning logic
            if analysis == "vbf" and "hi" in cat:
                binindex = 1

            regions = [f"pass_{r}_" for r in regions_to_fit] + ["fail_"]

            for region in regions:
                ch_name = f"ptbin{ptbin}{cat}{region.replace('_', '')}{year}"
                ch = rl.Channel(ch_name)
                model.addChannel(ch)

                for proc_name, info in sample_dict.items():
                    # proc_name is e.g., 'ggF', 'VBF', 'ttbar'
                    templ = get_merged_template(
                        infile_path, info["components"], region, binindex + 1, cat, msd
                    )
                    nominal = templ[0]

                    if badtemplate(nominal):
                        print(
                            f"Warning: Skipping template for {proc_name} in {ch_name} (failed badtemplate check)"
                        )
                        continue

                    stype = rl.Sample.SIGNAL if info["is_signal"] else rl.Sample.BACKGROUND
                    sample = rl.TemplateSample(ch.name + "_" + proc_name, stype, templ)

                    # Apply Luminosity
                    sample.setParamEffect(
                        sys_lumi_uncor, lumi_err[year[:4]] ** (LUMI[year[:4]] / LUMI["2022-2024"])
                    )

                    if do_systematics:
                        # 1. Automatic MC Statistical Uncertainties (Barlow-Beeston Lite)
                        # (Already handled inside add_systematics in card_utils.py)

                        # 2. Experimental Systematics (Shapes from ROOT file)
                        # Filter out theory systematics so they aren't double-applied
                        exp_syst_map = {
                            k: v
                            for k, v in syst_map.items()
                            if not k.startswith(("pdf_", "scale_", "isr_", "fsr_"))
                        }

                        add_systematics(
                            sample,
                            nominal,
                            exp_syst_map,
                            infile_path,
                            year,
                            info["components"],
                            region,
                            binindex + 1,
                            cat,
                            msd,
                        )

                        # 3. Theory Systematics (Process-Specific Logic)

                        # --- VBF / EWKZ Scale ---
                        if proc_name == "VBF" or proc_name == "EWKZ":
                            scale_up = get_merged_template(
                                infile_path,
                                info["components"],
                                region,
                                binindex + 1,
                                cat,
                                msd,
                                syst="scalevar_3ptUp",
                            )[0]
                            scale_do = get_merged_template(
                                infile_path,
                                info["components"],
                                region,
                                binindex + 1,
                                cat,
                                msd,
                                syst="scalevar_3ptDown",
                            )[0]
                            sample.setParamEffect(
                                syst_map["scale_VBF"],
                                np.sum(scale_up) / np.sum(nominal),
                                np.sum(scale_do) / np.sum(nominal),
                            )

                        # --- Higgs Signal Theory (PDF, ISR/FSR, Scale) ---
                        if proc_name in ["ggF", "VBF", "WH", "ZH", "ggZH", "ttH"]:
                            # Mapping logic: if proc is WH/ZH/ggZH, use "VH" for the nuisance name
                            proc_map_name = "VH" if proc_name in ["WH", "ZH", "ggZH"] else proc_name

                            for s_key, s_name in [
                                ("pdf", "PDF_weight"),
                                ("fsr", "FSRPartonShower"),
                                ("isr", "ISRPartonShower"),
                            ]:
                                s_up = get_merged_template(
                                    infile_path,
                                    info["components"],
                                    region,
                                    binindex + 1,
                                    cat,
                                    msd,
                                    syst=f"{s_name}Up",
                                )[0]
                                s_do = get_merged_template(
                                    infile_path,
                                    info["components"],
                                    region,
                                    binindex + 1,
                                    cat,
                                    msd,
                                    syst=f"{s_name}Down",
                                )[0]

                                # Look up using the mapped name (e.g., pdf_VH)
                                syst_obj = syst_map.get(f"{s_key}_{proc_map_name}")
                                if syst_obj:
                                    sample.setParamEffect(
                                        syst_obj,
                                        np.sum(s_up) / np.sum(nominal),
                                        np.sum(s_do) / np.sum(nominal),
                                    )

                            # ggF specific Scale (7pt)
                            if proc_name == "ggF":
                                sc_up = get_merged_template(
                                    infile_path,
                                    info["components"],
                                    region,
                                    binindex + 1,
                                    cat,
                                    msd,
                                    syst="scalevar_7ptUp",
                                )[0]
                                sc_do = get_merged_template(
                                    infile_path,
                                    info["components"],
                                    region,
                                    binindex + 1,
                                    cat,
                                    msd,
                                    syst="scalevar_7ptDown",
                                )[0]
                                sample.setParamEffect(
                                    syst_map["scale_ggF"],
                                    np.sum(sc_up) / np.sum(nominal),
                                    np.sum(sc_do) / np.sum(nominal),
                                )

                    ch.addSample(sample)

                # Data
                data_obs = get_template(
                    infile_path, "data_obs", region, binindex + 1, cat, msd, syst="nominal"
                )
                ch.setObservation(data_obs[0:3])

    # ---------------------------------------------------------
    # 6. ADD DATA-DRIVEN QCD
    # ---------------------------------------------------------
    print("Adding Data-Driven QCD Background...")
    for cat in cats:
        if "bins_pt" in cats_cfg[cat]:
            ptbins = np.array(cats_cfg[cat]["bins_pt"])
        else:
            ptbins = np.array(cats_cfg[cat]["bins"])

        for ptbin in range(len(ptbins) - 1):
            failCh = model[f"ptbin{ptbin}{cat}fail{year}"]

            initial_qcd = failCh.getObservation().astype(float)
            for sample in failCh:
                initial_qcd -= sample.getExpectation(nominal=True)
            initial_qcd[initial_qcd < 0] = 0

            qcdparams = np.array(
                [
                    rl.IndependentParameter(f"qcdparam_ptbin{ptbin}{cat}{year}_{i}", 0)
                    for i in range(msd.nbins)
                ]
            )
            scaledparams = (
                initial_qcd * (1 + 10.0 / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcdparams
            )
            fail_qcd = rl.ParametericSample(
                f"ptbin{ptbin}{cat}fail{year}_qcd", rl.Sample.BACKGROUND, msd, scaledparams
            )
            failCh.addSample(fail_qcd)

            # Add QCD to all pass regions defined in JSON
            for reg in regions_to_fit:
                passCh = model[f"ptbin{ptbin}{cat}pass{reg}{year}"]
                pass_qcd = rl.TransferFactorSample(
                    f"ptbin{ptbin}{cat}pass{reg}{year}_qcd",
                    rl.Sample.BACKGROUND,
                    tf_params[cat][reg][ptbin, :],
                    fail_qcd,
                )
                passCh.addSample(pass_qcd)

            if do_muon_CR:
                passChbb = model[f"ptbin{ptbin}{cat}passbb{year}"]
                passChcc = model[f"ptbin{ptbin}{cat}passcc{year}"]
                
                tqqpassbb = passChbb['ttbar']
                tqqpasscc = passChcc['ttbar']
                tqqfail = failCh['ttbar']

                sumPass = tqqpassbb.getExpectation(nominal=True).sum() + tqqpasscc.getExpectation(nominal=True).sum()
                sumFail = tqqfail.getExpectation(nominal=True).sum()

                sumPassbb = tqqpassbb.getExpectation(nominal=True).sum()
                sumPasscc = tqqpasscc.getExpectation(nominal=True).sum()

                if 'singlet' in passCh.samples:
                    stqqpassbb = passChbb['singlet']
                    stqqpasscc = passChcc['singlet']
                    stqqfail = failCh['singlet']
                    
                    sumPass += stqqpassbb.getExpectation(nominal=True).sum()
                    sumPass += stqqpasscc.getExpectation(nominal=True).sum()

                    sumPassbb += stqqpassbb.getExpectation(nominal=True).sum()
                    sumPasscc += stqqpasscc.getExpectation(nominal=True).sum()

                    sumFail += stqqfail.getExpectation(nominal=True).sum()
                    
                    tqqPF =  sumPass / sumFail
                    tqqBC = sumPassbb / sumPasscc
                    
                    stqqpassbb.setParamEffect(tqqeffSF, 1 * tqqeffSF)
                    stqqpasscc.setParamEffect(tqqeffSF, 1 * tqqeffSF)
                    stqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)

                    stqqpassbb.setParamEffect(tqqeffBCSF, 1 * tqqeffBCSF)
                    stqqpasscc.setParamEffect(tqqeffBCSF, (1 - tqqeffBCSF) * tqqBC + 1)

                    stqqpassbb.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                    stqqpasscc.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                    stqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)


                tqqPF =  sumPass / sumFail
                tqqBC = sumPassbb / sumPasscc

                tqqpassbb.setParamEffect(tqqeffSF, 1 * tqqeffSF)
                tqqpasscc.setParamEffect(tqqeffSF, 1 * tqqeffSF)
                tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)

                tqqpassbb.setParamEffect(tqqeffBCSF, 1 * tqqeffBCSF)
                tqqpasscc.setParamEffect(tqqeffBCSF, (1 - tqqeffBCSF) * tqqBC + 1)

                tqqpassbb.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                tqqpasscc.setParamEffect(tqqnormSF, 1 * tqqnormSF)
                tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

    muonCR_model = rl.Model('muonCR_'+year)
    if do_muon_CR:
        templates = {}
        samps = ['QCD','ttbar','singlet','Wjets','Zjetsc','Zjetslight','Zjetsbb']
        for region in ['pass_bb_', 'pass_cc_', 'fail_']:

            ch_name = 'muonCR%s%s' % (region.replace("_", ""), year)

            ch = rl.Channel(ch_name)
            muonCR_model.addChannel(ch)
            for sName in samps:
                templates[sName] = one_bin(infile_path, sName, region, 1, 'mucr_', syst='nominal')
                nominal = templates[sName][0]

                if nominal < eps:
                    print("Sample {} is too small, skipping".format(sName))
                    continue

                stype = rl.Sample.BACKGROUND
                sample = rl.TemplateSample(ch.name + '_' + sName, stype, templates[sName])

                sample.setParamEffect(sys_lumi_uncor, lumi_err[year[:4]] ** (LUMI[year[:4]] / LUMI["2022-2024"]))
                if do_systematics:

                    sample.autoMCStats(lnN=True) 

                ch.addSample(sample)
                
            data_obs = one_bin(infile_path, 'Muondata', region, 1, 'mucr_', syst='nominal')
            ch.setObservation(data_obs, read_sumw2=True)

        tqqpassbb = muonCR_model['muonCRpassbb'+year+'_ttbar']
        tqqpasscc = muonCR_model['muonCRpasscc'+year+'_ttbar']
        tqqfail = muonCR_model['muonCRfail'+year+'_ttbar']

        sumPass = tqqpassbb.getExpectation(nominal=True).sum() + tqqpasscc.getExpectation(nominal=True).sum()
        sumPassbb = tqqpassbb.getExpectation(nominal=True).sum()
        sumPasscc = tqqpasscc.getExpectation(nominal=True).sum()
        sumFail = tqqfail.getExpectation(nominal=True).sum()

        stqqpassbb = muonCR_model['muonCRpassbb'+year+'_singlet']
        stqqpasscc = muonCR_model['muonCRpasscc'+year+'_singlet']
        stqqfail = muonCR_model['muonCRfail'+year+'_singlet']

        sumPass += stqqpassbb.getExpectation(nominal=True).sum()
        sumPass += stqqpasscc.getExpectation(nominal=True).sum()

        sumPassbb += stqqpassbb.getExpectation(nominal=True).sum()
        sumPasscc += stqqpasscc.getExpectation(nominal=True).sum()
        sumFail += stqqfail.getExpectation(nominal=True).sum()

        tqqPF =  sumPass / sumFail
        tqqBC = sumPassbb / sumPasscc

        tqqpassbb.setParamEffect(tqqeffSF, 1 * tqqeffSF)
        tqqpasscc.setParamEffect(tqqeffSF, 1 * tqqeffSF)
        tqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)

        tqqpassbb.setParamEffect(tqqeffBCSF, 1 * tqqeffBCSF)
        tqqpasscc.setParamEffect(tqqeffBCSF, (1 - tqqeffBCSF) * tqqBC + 1)

        tqqpassbb.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        tqqpasscc.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        tqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

        stqqpassbb.setParamEffect(tqqeffSF, 1 * tqqeffSF)
        stqqpasscc.setParamEffect(tqqeffSF, 1 * tqqeffSF)
        stqqfail.setParamEffect(tqqeffSF, (1 - tqqeffSF) * tqqPF + 1)

        stqqpassbb.setParamEffect(tqqeffBCSF, 1 * tqqeffBCSF)
        stqqpasscc.setParamEffect(tqqeffBCSF, (1 - tqqeffBCSF) * tqqBC + 1)

        stqqpassbb.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        stqqpasscc.setParamEffect(tqqnormSF, 1 * tqqnormSF)
        stqqfail.setParamEffect(tqqnormSF, 1 * tqqnormSF)

    # ---------------------------------------------------------
    # 7. SAVE & RENDER
    # ---------------------------------------------------------
    with (datacard_dir / f"{analysis}Model_{year}.pkl").open("wb") as fout:
        pickle.dump(model, fout)
    modeldir = datacard_dir / f"{analysis}Model_{year}"
    if do_muon_CR:
        muonCR_model.renderCombine(modeldir)
    model.renderCombine(modeldir)
    print(f"Datacards saved to {modeldir}")

    out_cards = ""
    for ch in model:
        if "/" in ch.name:
            continue
        out_cards += f"{ch.name}={ch.name}.txt "
    if do_muon_CR:
        for ch in muonCR_model:
            if "/" in ch.name:
                continue
            out_cards += f"{ch.name}={ch.name}.txt "

    # 1. Get Physics Model Config from JSON
    # Default to simple signal strength 'r' if missing
    pm_config = config.get(
        "physics_model",
        {
            "model": "HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel",
            "maps": ["map=.*/.*:r[1,-20,20]"],
        },
    )

    model_cls = pm_config["model"]
    maps = " ".join([f"--PO '{m}'" for m in pm_config["maps"]])

    # Construct the text2workspace command dynamically
    t2w_cfg = f"-P {model_cls} --PO verbose {maps}"

    # Write the build script
    with (modeldir / "build.sh").open("w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"combineCards.py {out_cards} > model_combined.txt\n")
        f.write(f"text2workspace.py {t2w_cfg} model_combined.txt -o workspace.root\n")
        f.write("echo 'Workspace created: workspace.root'\n")

    (modeldir / "build.sh").chmod(0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--indir", default=None)
    parser.add_argument(
        "--outdir", default="results", help="Output directory for datacards and plots"
    )
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--mc-rho-order", type=int, default=1)
    parser.add_argument("--mc-pt-order", type=int, default=0)
    parser.add_argument("--res-rho-order", type=int, default=0)
    parser.add_argument("--res-pt-order", type=int, default=0)
    args = parser.parse_args()
    rhalphabet(args)
