"""
Datacard Utilities - common helper functions for reading templates,
merging histograms, and making transfer factor plots.

Author(s): Gabi Hamilton, Lara Zygala, Cristina Mantilla, Matej Roguljic
Date: Feb 2026
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rhalphalib as rl
import ROOT

# Constants
eps = 0.001


def badtemplate(hvalues, mask=None):
    """
    Checks if a template is statistically sufficient (non-empty and has >1 bins).
    """
    if mask is None:
        mask = np.ones(len(hvalues), dtype=bool)

    tot = np.sum(hvalues[mask])
    count_nonzeros = np.sum(hvalues[mask] > 0)
    return bool(tot < eps or count_nonzeros < 2)


def shape_to_num(var, nom, clip=1.5):
    """
    Converts a shape variation to a normalization variation (lnN).
    """
    nom_rate = np.sum(nom)
    var_rate = np.sum(var)

    if nom_rate == 0:
        return 1.0

    if abs(var_rate / nom_rate) > clip:
        var_rate = clip * nom_rate

    if var_rate < 0:
        var_rate = 0

    return var_rate / nom_rate


def get_template(filename, sName, region, ptbin, cat, obs, syst):
    """
    Read msd template from a specific ROOT file.
    Handles naming conventions for both VBF and ZGamma analyses.
    """
    f = ROOT.TFile.Open(str(filename))
    if not f:
        print(f"ERROR: Could not open {filename}")
        return (
            np.zeros(len(obs.binning) - 1),
            obs.binning,
            obs.name,
            np.zeros(len(obs.binning) - 1),
        )

    reg_clean = region.rstrip("_")

    # Construct the exact format: e.g. zgcr_fail_pt1_GJets_nominal
    name = f"{cat}_{reg_clean}"

    # All categories bin in pt except vbf, which bins in dijet mass (mjj).
    # HMDS does not use vbf
    if cat.startswith("vbf"):
        name += f"_mjj{ptbin}_"
    else:
        name += f"_pt{ptbin}_"

    name += f"{sName}_{syst}"

    h = f.Get(name)

    if not h:
        print(f"WARNING: Histogram {name} not found in {filename}")
        return (
            np.zeros(len(obs.binning) - 1),
            obs.binning,
            obs.name,
            np.zeros(len(obs.binning) - 1),
        )

    sumw = []
    sumw2 = []

    for i in range(1, h.GetNbinsX() + 1):
        content = h.GetBinContent(i)
        if content < 0:
            sumw.append(0)
            sumw2.append(0)
        else:
            sumw.append(content)
            sumw2.append(h.GetBinError(i) ** 2)

    return (np.array(sumw), obs.binning, obs.name, np.array(sumw2))


def get_merged_template(filename, process_groups, region, ptbin, cat, obs, syst="nominal"):
    """
    Helper function to sum histograms for a list of processes.
    Used for ZGamma analysis to group split processes (e.g. Zgamma + Zjets).
    """
    merged_sumw = None
    merged_sumw2 = None

    for proc_base, flavor_suffix in process_groups:
        proc_name = proc_base + flavor_suffix

        templ = get_template(filename, proc_name, region, ptbin, cat, obs, syst)
        sumw, binning, name, sumw2 = templ

        if merged_sumw is None:
            merged_sumw = np.zeros_like(sumw)
            merged_sumw2 = np.zeros_like(sumw2)

        merged_sumw += sumw
        merged_sumw2 += sumw2

    return (merged_sumw, obs.binning, obs.name, merged_sumw2)


def one_bin(filename, sName, region, ptbin, cat, syst):
    """
    Reads a single-bin template (e.g. for Muon Control Region).
    """
    f = ROOT.TFile.Open(str(filename))

    reg_clean = region.rstrip("_")
    name = f"{cat}{reg_clean}_pt{ptbin}_{sName}_{syst}"

    h = f.Get(name)
    if not h:
        return (np.array([0.0]), np.array([0.0, 1.0]), "onebin", np.array([0.0]))

    integral = h.Integral()
    # Approximate error as sumw2 of bin 1 (simplification)
    error2 = h.GetBinError(1) ** 2

    return (np.array([integral]), np.array([0.0, 1.0]), "onebin", np.array([error2]))


def fit_mc_tf(
    infile_path,
    qcd_tf_proc,
    cat,
    reg,
    year,
    msd,
    npt,
    ptscaled,
    rhoscaled,
    validbins,
    order,
    init_params=None,
    max_tries=5,
    bin_indices=None,
):
    """
    Fit a Bernstein tf_MCtempl(pt, rho) of the given (pt_order, rho_order) to the
    QCD MC pass/fail ratio via a standalone RooFit fit. Both the observation and
    the pdf here are built purely from MC templates (qcd_tf_proc)

    Returns (tf_MCtempl, qcdfit, qcdeff, fitted_values) where fitted_values is the
    fitted Bernstein-coefficient array (same shape as `order`-derived init_params).
    """
    mc_pt_order, mc_rho_order = order
    if init_params is None:
        init_params = np.full((mc_pt_order + 1, mc_rho_order + 1), 1.0)
    if bin_indices is None:
        bin_indices = [ptbin + 1 for ptbin in range(npt)]

    for _attempt in range(max_tries):
        qcdmodel = rl.Model(f"qcdmodel_{cat}_{reg}")
        qcdpass, qcdfail = 0.0, 0.0

        for ptbin in range(npt):
            failCh = rl.Channel(f"ptbin{ptbin}{cat}fail{year}{reg}")
            passCh = rl.Channel(f"ptbin{ptbin}{cat}pass{year}{reg}")
            qcdmodel.addChannel(failCh)
            qcdmodel.addChannel(passCh)

            failTempl = get_template(
                infile_path, qcd_tf_proc, "fail_", bin_indices[ptbin], cat, msd, "nominal"
            )
            passTempl = get_template(
                infile_path, qcd_tf_proc, f"pass_{reg}_", bin_indices[ptbin], cat, msd, "nominal"
            )

            failCh.setObservation(failTempl, read_sumw2=True)
            passCh.setObservation(passTempl, read_sumw2=True)
            qcdfail += failCh.getObservation()[0].sum()
            qcdpass += passCh.getObservation()[0].sum()

        qcdeff = qcdpass / qcdfail

        tf_MCtempl = rl.BasisPoly(
            f"tf_MCtempl_{cat}{reg}{year}",
            (mc_pt_order, mc_rho_order),
            ["pt", "rho"],
            basis="Bernstein",
            init_params=init_params,
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
                f"ptbin{ptbin}{cat}fail{year}{reg}_qcd", rl.Sample.BACKGROUND, msd, scaledparams
            )
            failCh.addSample(fail_qcd)

            pass_qcd = rl.TransferFactorSample(
                f"ptbin{ptbin}{cat}pass{year}{reg}_qcd",
                rl.Sample.BACKGROUND,
                tf_MCtempl_params[ptbin, :],
                fail_qcd,
            )
            passCh.addSample(pass_qcd)

            failCh.mask = validbins[ptbin]
            passCh.mask = validbins[ptbin]

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

        if qcdfit.status() == 0:
            param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
            allparams = dict(zip(qcdfit.nameArray(), qcdfit.valueArray()))
            fitted_values = np.array([allparams[p] for p in param_names]).reshape(
                tf_MCtempl.parameters.shape
            )
            # RooFit's fitTo() only updates its own internal RooRealVars; sync the
            # Python-side IndependentParameter.value too, otherwise tf_MCtempl(...,
            # nominal=True) (used for plotting) silently evaluates at init_params
            # instead of the fit result.
            tf_MCtempl.set_parvalues(fitted_values.reshape(-1))
            return tf_MCtempl, qcdfit, qcdeff, fitted_values

    raise RuntimeError(f"Could not fit MC TF for {cat} {reg} order {order} after {max_tries} tries!")


def raw_pass_fail_ratio(infile_path, qcd_tf_proc, reg, cat, msd, npt, bin_indices=None):
    """
    Raw QCD MC pass/fail ratio and its propagated statistical error, per pt bin
    and msd bin
    """
    if bin_indices is None:
        bin_indices = [ptbin + 1 for ptbin in range(npt)]

    ratio = np.full((npt, msd.nbins), np.nan)
    err = np.full((npt, msd.nbins), np.nan)
    for ptbin in range(npt):
        failT, _, _, failW2 = get_template(
            infile_path, qcd_tf_proc, "fail_", bin_indices[ptbin], cat, msd, "nominal"
        )
        passT, _, _, passW2 = get_template(
            infile_path, qcd_tf_proc, f"pass_{reg}_", bin_indices[ptbin], cat, msd, "nominal"
        )
        m = failT > 0
        ratio[ptbin][m] = passT[m] / failT[m]
        relvar = np.zeros_like(failT)
        relvar[m] = failW2[m] / failT[m] ** 2
        pm = m & (passT > 0)
        relvar[pm] += passW2[pm] / passT[pm] ** 2
        err[ptbin][m] = ratio[ptbin][m] * np.sqrt(relvar[m])
    return ratio, err


def mc_tf_chi2(tf_MCtempl, qcdeff, ptscaled, rhoscaled, validbins, ratio, err):
    """
    Chi2 of a fitted (post fit_mc_tf) tf_MCtempl against the raw MC ratio/err
    from raw_pass_fail_ratio, restricted to valid (unmasked, finite) bins.
    Returns (chi2, n_valid_bins).
    """
    mctf = qcdeff * tf_MCtempl(ptscaled, rhoscaled, nominal=True)
    resid = (ratio - mctf) / err
    m = np.isfinite(resid) & validbins
    return float(np.sum(resid[m] ** 2)), int(m.sum())


def plot_mctf(
    tf_MCtempl, msdbins, name, _year, _tag, out_dir_base,
    pt_min=450.0, rho_min=-6.0, rho_max=-2.1,
):
    import matplotlib.pyplot as plt
    import pandas as pd  # Ensure pandas is imported

    # Create directory
    outdir = Path(out_dir_base) / "plots" / "MCTF"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Create Grid
    pts = np.linspace(pt_min, 1200, 15)
    ptpts, msdpts = np.meshgrid(
        pts[:-1] + 0.5 * np.diff(pts), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing="ij"
    )

    # 2. Scale Coordinates
    ptpts_scaled = (ptpts - pt_min) / (1200.0 - pt_min)
    rhopts = 2 * np.log(msdpts / ptpts)
    rhopts_scaled = (rhopts - rho_min) / (rho_max - rho_min)

    # 3. SAFETY FILTER (Restoring the logic from old script)
    validbins = (
        (rhopts_scaled >= 0) & (rhopts_scaled <= 1) & (ptpts_scaled >= 0) & (ptpts_scaled <= 1)
    )

    # 4. Filter Arrays (Keep only valid points)
    ptpts_scaled = ptpts_scaled[validbins]
    rhopts_scaled = rhopts_scaled[validbins]
    msdpts = msdpts[validbins]
    ptpts = ptpts[validbins]

    # 5. Evaluate only on valid points
    tf_MCtempl_vals = tf_MCtempl(ptpts_scaled, rhopts_scaled, nominal=True)

    # 6. Reconstruct DataFrame
    df_plot = pd.DataFrame([])
    df_plot["msd"] = msdpts.reshape(-1)
    df_plot["pt"] = ptpts.reshape(-1)
    df_plot["MCTF"] = tf_MCtempl_vals.reshape(-1)

    # Plot
    fig, ax = plt.subplots()
    h = ax.hist2d(x=df_plot["msd"], y=df_plot["pt"], weights=df_plot["MCTF"], bins=(msdbins, pts))
    plt.xlabel("$m_{sd}$ [GeV]")
    plt.ylabel("$p_{T}$ [GeV]")
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label("Ratio (Pass/Fail)")
    fig.savefig(outdir / f"MCTF_msdpt_{name}.png", bbox_inches="tight")
    plt.close()

    print(f"Saved MCTF plots to {outdir}")


def add_systematics(
    sample, nominal, systs, infile_path, _year, components, region, ptbin, cat, obs
):
    """
    Applies lnN and Shape systematics to a Rhalphalib sample object.

    Args:
        sample (rl.TemplateSample): The sample to apply effects to.
        nominal (np.array): The nominal yield array.
        systs (dict): Dictionary mapping sys_name -> rl.NuisanceParameter.
        infile_path (Path): Path to ROOT file.
        year (str): Analysis year.
        components (list): List of (process, flavor) tuples for merging.
        region (str): Region string (e.g. 'pass_bb_').
        ptbin (int): Bin index.
        cat (str): Category name.
        obs (rl.Observable): Observable definition.
    """
    if not systs:
        return

    # 1. Statistical Uncertainty (Barlow-Beeston)
    sample.autoMCStats(lnN=True)

    # 2. Shape / Normalization Systematics
    for sys_name, nuisance_par in systs.items():
        # Get Up/Down Shapes (using merged template logic to handle groups)
        syst_up = get_merged_template(
            infile_path, components, region, ptbin, cat, obs, syst=sys_name + "Up"
        )[0]

        syst_down = get_merged_template(
            infile_path, components, region, ptbin, cat, obs, syst=sys_name + "Down"
        )[0]
        # Convert shape variation to single normalization number (lnN effect)
        eff_up = shape_to_num(syst_up, nominal)
        eff_do = shape_to_num(syst_down, nominal)

        sample.setParamEffect(nuisance_par, eff_up, eff_do)
