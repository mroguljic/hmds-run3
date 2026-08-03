#!/usr/bin/env python3
"""
Shared naming, path and process helpers for the HMDS SR fit pipeline.

Imported by run_fit.py (the stage driver) and run_ftest_scan.py 
so both agree on the workspace/fitDiagnostics filename convention and on
where the datacard directory lives.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

FITTING_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------- naming
def order_tag(o):
    #mc pt rho; residual pt rho
    return f"mc{o[0]}{o[1]}_res{o[2]}{o[3]}"


def ws_name(o):
    return f"workspace_{order_tag(o)}.root"


def fitdiag_name(o):
    return f"fitDiagnostics_{order_tag(o)}.root"


def parse_order(s):
    parts = [int(x) for x in str(s).replace(" ", "").split(",")]
    if len(parts) != 4:
        raise SystemExit(f"order must be 4 ints 'mc_pt,mc_rho,res_pt,res_rho', got {s!r}")
    return tuple(parts)


def dc_dir(outdir, tag, year, model="srModel"):
    """Datacard/workspace dir written by make_datacards.py + build.sh."""
    return Path(outdir) / tag / year / "datacards" / f"{model}_{year}"


# ---------------------------------------------------------------- process
def run(cmd, cwd, check=True, prefix="run_fit"):
    """Echo then run `cmd` in `cwd`. check=True aborts on failure, else warns."""
    print(f"\n$ (cd {cwd}) {cmd}", flush=True)
    rc = subprocess.run(cmd, cwd=str(cwd), shell=True).returncode
    if rc != 0:
        if check:
            raise SystemExit(f"[{prefix}] command failed (exit {rc}): {cmd}")
        print(f"  [warn] command exit {rc}", flush=True)
    return rc


# ---------------------------------------------------------------- binning
def n_fit_bins(setup_path, apply_rho_mask=True):
    """
    Number of bins entering the likelihood, derived from a setup_*.json:
    (one pass region per entry in regions_to_fit, plus one shared fail) x pt bins
    x msd bins. For setup_sr.json the raw grid is 2 x 3 x 23 = 138.

    Bins outside the rho window are masked out of both the pass and fail channels
    (make_datacards.py: validbins -> Channel.mask) and do not enter the fit, so
    they are excluded here by default — that masked count is the correct
    denominator for an F-statistic. The mask is reproduced from the same grid
    make_datacards.py builds: pt at 30% into each pt bin, msd at bin centre,
    rho = 2 ln(msd/pt), kept when rho_scaling_min <= rho <= rho_scaling_max.

    Pass apply_rho_mask=False for the raw grid count.
    """
    cfg = json.loads(Path(setup_path).read_text())
    obs = cfg["observable"]
    nmsd = obs["nbins"]
    msdbins = np.linspace(obs["min"], obs["max"], nmsd + 1)
    rho_min = cfg.get("rho_scaling_min", -6.0)
    rho_max = cfg.get("rho_scaling_max", -2.1)
    nregions = len(cfg["regions_to_fit"]) + 1  # pass per region + shared fail

    nbins = 0
    for c in cfg["categories"].values():
        ptbins = np.array(c.get("bins_pt", c["bins"]))
        if not apply_rho_mask:
            nbins += (len(ptbins) - 1) * nmsd
            continue
        ptpts, msdpts = np.meshgrid(
            ptbins[:-1] + 0.3 * np.diff(ptbins),
            msdbins[:-1] + 0.5 * np.diff(msdbins),
            indexing="ij",
        )
        rhoscaled = (2 * np.log(msdpts / ptpts) - rho_min) / (rho_max - rho_min)
        nbins += int(((rhoscaled >= 0.0) & (rhoscaled <= 1.0)).sum())
    return nregions * nbins
