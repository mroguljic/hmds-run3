# HMDS SR Fitting

Runs from skimmed parquet files -> histograms -> datacards/workspace -> F-test -> fit -> plots/limits.
Everything below runs from `fitting/` in the `hbb` micromamba env. Examples use tag `20260721`, year `2024`, signal tag 20260721_private_v15_private (messy bookkeeping of a private signal). To be changed as necessary.

## TLDR

```bash

# 1) histograms (parquet skims -> ROOT templates)
python make_hists.py --year 2024 --tag 20260721_v15 --sig-tag 20260721_private_v15_private \
  --setup setup_sr.json --outdir results/20260721 --save-root

# 2) pick the background-model order (F-test)
python run_fit.py --tag 20260721 --year 2024 --stages ftest

# 3) datacards + workspace + fit + plots + GoF + limit
python run_fit.py --tag 20260721 --year 2024 --stages fitdiag,postfit,gof,limit
```

Note `make_hists.py --tag` is the *skim* tag (`20260721_v15`), while every
other `--tag` below is the *results* tag (`20260721`, i.e. `--outdir`'s
subdirectory), as shown in this example, they don't have to match.

Everything after step 1 is driven by `run_fit.py`, and results are stored in `results/20260721/2024/`.
Step 2 writes `chosen_order.txt`; step 3 reads it
automatically, so you don't need to pass `--order` yourself (pass it
explicitly, e.g. `--order 2,0,1,0`, to run a specific order without redoing the F-test).

## What "order" means

The background model has 4 integers: `(mc_pt, mc_rho, res_pt, res_rho)` — the
pt/rho Bernstein-polynomial degrees of the **MC transfer factor** and the
**data residual** on top of it. Two different fits, chosen two different ways:

- `mc_pt`/`mc_rho` — fit and chosen by F-test against the QCD MC templates alone (no fitting to observed data). Its results become Gaussian-constrained nuisances in the full model.
- `res_pt`/`res_rho` — a free (unconstrained) correction on top. Fit and chosen by F-test against observed data.

`--stages ftest` runs both in sequence (MC order first, then residual order with MC fixed) and saves the best order to `chosen_order.txt`.

## What each step does

1. **`make_hists.py`** — reads skims, writes `results/20260721/fitting_2024_sr_msd.root` (templates for the datacard maker) plus `.pkl` histograms (for `python/plot_histos.py`).
Builds nominal + JES/JER/UES/pileup/ISR/FSR up/down variations. Data is blinded to 1/10 by default (`--prescale`, default 10); pass `--prescale 1` for a full, unblinded run.
Model geometry (pt bins, msd binning, categories) comes from `setup_sr.json`.
The templates are prerequisite for `run_fit.py`.

2. **F-test (`--stages ftest`)** Orchestrates `run_ftest_scan.py`; run it directly instead of through `run_fit.py` for the step-by-step log:
   ```bash
   python run_ftest_scan.py --tag 20260721 --year 2024
   ```
   Stops in each direction once the next order gives no significant
   improvement (F-test p ≥ 0.05, via `scipy.stats.f`). `--nbins` overrides
   only the data-side bin count, normally derived from
   `setup_sr.json` by `fit_common.n_fit_bins()` (total pt x msd grid minus the bins rhalphalib masks outside the rho window). The MC-side bin    count comes from its own fit grid and isn't affected by `--nbins`.

   To compare two specific orders directly instead of scanning (from the
   datacard dir, with `workspace_<null>.root`/`workspace_<alt>.root` built):
   ```bash
   python run_ftest.py --null workspace_mc20_res00.root --alt workspace_mc20_res10.root \
     --ntoys 100 --seed 123456 --tag mc20res00_vs_mc20res10
   python plot_ftest.py --tag mc20res00_vs_mc20res10 --nbins 128 --p1 3 --p2 4
   # p1/p2 are parameter counts: (pt_order+1)*(rho_order+1)
   ```

3. **`--stages fitdiag,postfit,gof,limit`** — builds the datacard/workspace
   for the chosen order (if not already built; see `make_datacards.py` +
   `./build.sh` if you want to do this by hand), then:
   - `fitdiag` — `combine -M FitDiagnostics` (`--robustHesse 1`, needed for
     postfit bands on low-stat/blinded data) -> `fitDiagnostics_*.root`.
   - `postfit` — `plot_fit.py`: transfer-factor plots and per-channel
     prefit/postfit data/MC stacks.
   - `gof` — saturated-model goodness of fit, `--ntoys` toys (default 500) -> `plot_gof.py` prints the p-value (fraction of toys > observed).
   - `limit` — `combine -M AsymptoticLimits`, blinded/expected by default;
     add `--unblind` to use real data instead.

   Every output is named per order (`workspace_mc20_res10.root`,
   `fitDiagnostics_mc20_res10.root`, etc.).
## Useful extras

- **`plot_mctf_scan.py`** — fits the MC transfer factor at several candidate   orders and plots them against the raw MC pass/fail ratio:
  ```bash
  python plot_mctf_scan.py --tag 20260721 --year 2024 --orders "0,0;1,0;1,1;2,0;1,2"
  ```
