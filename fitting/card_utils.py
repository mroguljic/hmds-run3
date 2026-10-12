"""
Datacard Utilities - common helper functions for reading templates,
merging histograms, and making transfer factor plots.

Author(s): Gabi Hamilton, Lara Zygala, Cristina Mantilla, Matej Roguljic
Date: Feb 2026
"""

from __future__ import annotations

import warnings
from math import comb
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


def shape_to_ratio(var, nom, clip=2.0, min_yield_frac=1e-3):
    """
    Per-bin multiplicative effect of a shape variation, for setParamEffect.

    Bins carrying negligible yield are pinned to 1.0. A single MC event crossing a
    bin edge produces ratios of 10-50 in a near-empty bin, which is noise in the
    template rather than a real shape effect, and Combine would interpolate it as
    though it were physics. In the 2024 SR skim the eleven bins with JESUp/nominal
    above 3 have effective MC entries of 1-2.6 and carry 0.11 events between them.

    `clip` bounds what survives, so one poorly-populated-but-not-negligible bin
    cannot dominate the morphing.
    """
    nom = np.asarray(nom, dtype=float)
    var = np.asarray(var, dtype=float)
    effect = np.ones_like(nom)

    total = np.sum(nom)
    if total <= 0:
        return effect

    usable = (nom > min_yield_frac * total) & (var > 0)
    effect[usable] = var[usable] / nom[usable]
    return np.clip(effect, 1.0 / clip, clip)


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


class MCTFFitResult:
    """
    Result of the standalone least-squares MC transfer-factor fit.

    Stands in for the RooFitResult `fit_mc_tf` used to return, carrying the fitted
    Bernstein coefficients and their covariance. Unlike a RooFit covariance this
    one is closed form, (A^T W A)^-1, and positive definite by construction.
    """

    def __init__(self, param_names, values, cov, chi2, ndf):
        self.param_names = list(param_names)
        self.values = np.asarray(values, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.chi2 = float(chi2)
        self.ndf = int(ndf)

    def status(self):
        """0 if the covariance is usable. Mirrors RooFitResult.status() for callers."""
        return 0 if np.all(np.linalg.eigvalsh(self.cov) > 0) else 1

    @property
    def errors(self):
        return np.sqrt(np.diag(self.cov))

    def decorrelate(self, prefix):
        """
        Rotate to uncorrelated unit-Gaussian nuisances, as
        `DecorrelatedNuisanceVector.fromRooFitResult` did, but straight from numpy.
        """
        deco = rl.DecorrelatedNuisanceVector(prefix, self.values, self.cov)
        for p, name in zip(deco.correlated_params, self.param_names):
            p.name = name
        return deco


def _bernstein_basis(x, n):
    """Bernstein basis functions of degree n evaluated at x, stacked on a new last axis."""
    return np.stack([comb(n, k) * x**k * (1 - x) ** (n - k) for k in range(n + 1)], axis=-1)


def tf_design_matrix(ptscaled, rhoscaled, order):
    """
    Bernstein design matrix for the TF: element [..., i] is the i-th basis function
    of the flattened (pt, rho) coefficient grid, in the same C-order that
    `BasisPoly.parameters.reshape(-1)` uses.
    """
    basis = np.einsum(
        "...j,...k->...jk", _bernstein_basis(ptscaled, order[0]), _bernstein_basis(rhoscaled, order[1])
    )
    return basis.reshape(ptscaled.shape + (-1,))


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
    bin_indices=None,
):
    """
    Fit a Bernstein tf_MCtempl(pt, rho) of the given (pt_order, rho_order) to the raw
    QCD MC pass/fail ratio by weighted linear least squares.

    A Bernstein polynomial is linear in its coefficients, so this is an ordinary
    weighted least-squares problem: the solution and its covariance (A^T W A)^-1 are
    closed form, and the covariance is positive definite whenever the design matrix
    has full rank. The per-bin weights are the MC statistical errors on the ratio
    from `raw_pass_fail_ratio` (pass and fail are disjoint selections, so treating
    numerator and denominator as independent is exact to first order).

    This replaces the previous RooFit likelihood fit, whose covariance was unusable
    here. That fit asked RooFit for `SumW2Error(True)`, which corrects the covariance
    as V * C^-1 * V with C from a second Hesse on the *weight-squared* likelihood,
    evaluated at the minimum of the weighted one. Those two minima coincide only when
    the weights are near 1; the QCD MC weights are not (per-bin sum(w^2)/sum(w) runs
    0.017-0.245, and the fail channel is saturated, so the two minima sit a factor
    4-60 apart in every bin). The weight-squared Hessian there is indefinite, Minuit
    silently repairs it ("MnPosDef Matrix forced pos-def by adding to diagonal"), and
    inverting the repaired matrix gave TF coefficient errors of ~1e10 instead of
    ~0.03 - which in turn made every decorrelated nuisance move the pass-region QCD
    prediction by nine orders of magnitude per sigma, so no toy could be generated.
    Only `covQual` recorded that anything was wrong, and nothing checked it.

    Returns (tf_MCtempl, fitresult, qcdeff, fitted_values), where fitresult is an
    MCTFFitResult and fitted_values has the shape of tf_MCtempl.parameters.
    """
    if bin_indices is None:
        bin_indices = [ptbin + 1 for ptbin in range(npt)]

    # Inclusive pass/fail efficiency, over all bins (the TF carries the shape).
    qcdpass, qcdfail = 0.0, 0.0
    for ptbin in range(npt):
        failT, _, _, _ = get_template(
            infile_path, qcd_tf_proc, "fail_", bin_indices[ptbin], cat, msd, "nominal"
        )
        passT, _, _, _ = get_template(
            infile_path, qcd_tf_proc, f"pass_{reg}_", bin_indices[ptbin], cat, msd, "nominal"
        )
        qcdfail += np.sum(failT)
        qcdpass += np.sum(passT)
    qcdeff = qcdpass / qcdfail

    ratio, err = raw_pass_fail_ratio(infile_path, qcd_tf_proc, reg, cat, msd, npt, bin_indices)

    # Fit only bins inside the rho window that carry a usable MC error.
    fitmask = validbins & np.isfinite(ratio) & np.isfinite(err) & (err > 0)
    npar = (order[0] + 1) * (order[1] + 1)
    if fitmask.sum() < npar:
        raise RuntimeError(
            f"MC TF fit for {cat} {reg} order {order}: only {int(fitmask.sum())} usable "
            f"bins for {npar} parameters"
        )

    # ratio ~ qcdeff * sum_i c_i * B_i(pt, rho): linear in c, so solve directly.
    design = qcdeff * tf_design_matrix(ptscaled, rhoscaled, order)[fitmask]
    y = ratio[fitmask]
    w = 1.0 / err[fitmask] ** 2
    ata = design.T @ (design * w[:, None])
    aty = design.T @ (y * w)
    if np.linalg.matrix_rank(ata) < npar:
        raise RuntimeError(
            f"MC TF fit for {cat} {reg} order {order}: design matrix is rank deficient; "
            "the requested order is not constrained by the available bins"
        )
    coeffs = np.linalg.solve(ata, aty)
    cov = np.linalg.inv(ata)
    chi2 = float(np.sum(((y - design @ coeffs) / err[fitmask]) ** 2))

    fitted_values = coeffs.reshape(tuple(o + 1 for o in order))
    tf_MCtempl = rl.BasisPoly(
        f"tf_MCtempl_{cat}{reg}{year}",
        order,
        ["pt", "rho"],
        basis="Bernstein",
        init_params=fitted_values,
        limits=(0, 10),
    )
    # BasisPoly stores init_params separately from the parameter values; sync them so
    # tf_MCtempl(..., nominal=True) (used for plotting and the chi2) sees the fit.
    tf_MCtempl.set_parvalues(coeffs)

    if np.any(coeffs < 0):
        warnings.warn(
            f"MC TF fit for {cat} {reg} order {order} has negative Bernstein "
            f"coefficients {coeffs[coeffs < 0]}; the transfer factor is no longer "
            "guaranteed non-negative over the (pt, rho) plane.",
            stacklevel=2,
        )

    param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
    fitresult = MCTFFitResult(param_names, coeffs, cov, chi2, int(fitmask.sum()) - npar)
    return tf_MCtempl, fitresult, qcdeff, fitted_values


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

    # MC statistical uncertainties are per-bin and need the whole channel, so they
    # are added in make_datacards.py once all templates are in the channel.

    # Shape / Normalization Systematics
    for sys_name, nuisance_par in systs.items():
        # Get Up/Down Shapes (using merged template logic to handle groups)
        syst_up = get_merged_template(
            infile_path, components, region, ptbin, cat, obs, syst=sys_name + "Up"
        )[0]

        syst_down = get_merged_template(
            infile_path, components, region, ptbin, cat, obs, syst=sys_name + "Down"
        )[0]
        # A systematic whose variations are identical to nominal carries no
        # information (UES is exactly this in the dark-shower SR, since the
        # selection never touches MET); adding it would only clutter the impacts.
        if np.array_equal(syst_up, nominal) and np.array_equal(syst_down, nominal):
            continue

        # Honour how the nuisance was declared: a "shape" prior wants the per-bin
        # effect, "lnN" collapses it to a single yield ratio. Collapsing loses the
        # larger part of some systematics - JES shifts these templates by 7-40%
        # per bin while moving the yield by only 1-3%.
        if "shape" in nuisance_par.combinePrior:
            eff_up = shape_to_ratio(syst_up, nominal)
            eff_do = shape_to_ratio(syst_down, nominal)
        else:
            eff_up = shape_to_num(syst_up, nominal)
            eff_do = shape_to_num(syst_down, nominal)

        sample.setParamEffect(nuisance_par, eff_up, eff_do)
