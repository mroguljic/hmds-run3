"""
F-Test F-Statistic Plotter

Reads the Goodness-of-Fit ROOT files produced by run_ftest.py to calculate
the F-statistic for both the observed data and the generated toys.
Plots the resulting pseudo-experiment distribution against the theoretical
F-distribution and calculates the final p-value to justify model complexity.

Gabi Hamilton - Feb 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import scipy.stats as stats
import uproot

# Use CMS style for the plot
plt.style.use(hep.style.CMS)


def get_chi2_values(filename):
    """Reads the 'limit' branch (Chi2 value) from the ROOT file."""
    if not Path(filename).exists():
        print(f"Error: Could not find file {filename}")
        return None
    try:
        with uproot.open(filename) as f:
            return f["limit"]["limit"].array(library="np")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def calculate_f_statistic(chi2_null, chi2_alt, p1, p2, nbins):
    """
    Calculates the F-statistic based on Chi2 difference.
    Formula: F = [(Chi2_null - Chi2_alt) / (p2 - p1)] / [Chi2_alt / (nbins - p2)]
    """
    numerator = (chi2_null - chi2_alt) / (p2 - p1)
    denominator = chi2_alt / (nbins - p2)

    # Handle division by zero or negative denominator safely
    with np.errstate(divide="ignore", invalid="ignore"):
        f_val = numerator / denominator

    return f_val


def main(tag, nbins, p1, p2):
    suffix = f"_{tag}" if tag else ""
    print(f"--- F-Test Analysis for {tag} ---")

    # --- 1. DEFINE PARAMETERS ---
    # Order 0: (pt=0, rho=0) -> (0+1)*(0+1) = 1 parameter
    # Order 1: (pt=0, rho=1) -> (0+1)*(1+1) = 2 parameters
    # p1 = 1
    # p2 = 2

    print(f"Model Parameters: p_null={p1}, p_alt={p2}")
    print(f"Number of Bins: {nbins}")

    # --- 2. LOAD DATA ---
    file_obs_null = f"higgsCombine_Observed_Null{suffix}.GoodnessOfFit.mH120.root"
    file_obs_alt = f"higgsCombine_Observed_Alt{suffix}.GoodnessOfFit.mH120.root"

    chi2_obs_null = get_chi2_values(file_obs_null)
    chi2_obs_alt = get_chi2_values(file_obs_alt)

    if chi2_obs_null is None or chi2_obs_alt is None:
        return

    # Observed F-Stat
    obs_val_null = chi2_obs_null[0]
    obs_val_alt = chi2_obs_alt[0]
    f_obs = calculate_f_statistic(obs_val_null, obs_val_alt, p1, p2, nbins)

    print(f"Observed Chi2(Null): {obs_val_null:.2f}")
    print(f"Observed Chi2(Alt):  {obs_val_alt:.2f}")
    print(f"Observed F-Stat:     {f_obs:.4f}")

    # --- 3. LOAD TOYS ---
    seed = 123456
    file_toys_null = f"higgsCombine_Toys_Null{suffix}.GoodnessOfFit.mH120.{seed}.root"
    file_toys_alt = f"higgsCombine_Toys_Alt{suffix}.GoodnessOfFit.mH120.{seed}.root"

    vals_toys_null = get_chi2_values(file_toys_null)
    vals_toys_alt = get_chi2_values(file_toys_alt)

    if vals_toys_null is None or vals_toys_alt is None:
        return

    # Match array lengths
    n_toys = min(len(vals_toys_null), len(vals_toys_alt))
    vals_toys_null = vals_toys_null[:n_toys]
    vals_toys_alt = vals_toys_alt[:n_toys]

    # Calculate F-Stat for toys
    f_toys = calculate_f_statistic(vals_toys_null, vals_toys_alt, p1, p2, nbins)

    # Filter out failed fits (negative F or NaNs)
    # Failed fits usually produce negative DeltaChi2, resulting in negative F
    valid_mask = (f_toys > -10) & (~np.isnan(f_toys))
    f_toys_clean = f_toys[valid_mask]
    n_valid = len(f_toys_clean)

    # P-Value: Fraction of toys with F > F_observed
    n_above = np.sum(f_toys_clean > f_obs)
    p_value = n_above / n_valid if n_valid > 0 else 0.0

    print(f"Valid Toys: {n_valid}/{n_toys}")
    print(f"P-Value: {p_value:.4f}")

    # --- 4. PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define range for plot
    x_max = max(np.max(f_toys_clean), f_obs) + 2
    # if x_max > 20: x_max = 20 # Cap it if outliers exist
    # Use this instead:
    x_max = max(np.max(f_toys_clean), f_obs) * 1.1  # Add 10% padding
    x_bins = np.linspace(0, x_max, 25)

    # Plot Toy Histogram
    hist_vals, bins, _ = ax.hist(
        f_toys_clean,
        bins=x_bins,
        histtype="step",
        color="black",
        linewidth=1.5,
        label=f"Toys (Null)\n$N_{{toys}}={n_valid}$",
    )

    # Error bars on histogram (Poisson)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    y_err = np.sqrt(hist_vals)
    ax.errorbar(bin_centers, hist_vals, yerr=y_err, fmt="o", color="black", markersize=4)

    # Plot Observed Line
    ymax = ax.get_ylim()[1] * 1.2
    ax.vlines(f_obs, 0, ymax, color="red", linewidth=2, label=f"Observed\n$p={p_value:.3f}$")

    # Plot Theoretical F-Distribution PDF
    # We scale the PDF by (Number of Toys * Bin Width) to match the histogram area
    x_pdf = np.linspace(0, x_max, 500)
    bin_width = bins[1] - bins[0]
    # Degrees of Freedom: df1 = p2 - p1, df2 = nbins - p2
    df1 = p2 - p1
    df2 = nbins - p2
    pdf_curve = stats.f.pdf(x_pdf, df1, df2) * n_valid * bin_width

    ax.plot(x_pdf, pdf_curve, color="blue", linewidth=2, label=f"F-Dist ($d_1={df1}, d_2={df2}$)")

    # Labels and Style
    ax.set_ylim(0, ymax)
    ax.set_xlabel("F-Statistic", fontsize=18)
    ax.set_ylabel("Pseudo-experiments", fontsize=18)
    ax.legend(loc="best", fontsize=14, frameon=True)

    # CMS Label
    hep.cms.label(data=True, label="Preliminary", year=tag, ax=ax)

    outname = f"ftest_fstat_{tag}.png"
    plt.savefig(outname)
    print(f"Plot saved as {outname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="2022", help="Year tag")
    parser.add_argument("--nbins", default=128, type=int,
                        help="Bins entering the fit, i.e. rho-masked bins excluded "
                             "(128 for setup_sr.json; see fit_common.n_fit_bins)")
    # Add these two lines:
    parser.add_argument("--p1", required=True, type=int, help="Params in Null Model")
    parser.add_argument("--p2", required=True, type=int, help="Params in Alt Model")

    args = parser.parse_args()
    main(args.tag, args.nbins, args.p1, args.p2)
