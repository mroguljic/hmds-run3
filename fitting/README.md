# Hbb Run 3 Fitting Framework
### Going from skims to datacards

## 1. Environment Setup
Activate your environment
```
micromamba activate hbb
```
Make sure you have the right packages needed for fitting. (You only need to do this once)
```
micromamba install root cms-combine -c conda-forge
```

## 2. Producing Histograms
The `make_hists.py` script reads skims from EOS and produces two outputs used for downstream analysis. It is completely configuration-driven via `setup.json` files, so remember to customize your setup file for your case.

### For zgamma control region:
```
python make_hists.py \
    --year 2022EE \
    --tag 26Feb03 \
    --setup setup_zgcr.json \
    --outdir results \
    --save-root
```
### For the signal region (VBF, ggF and VH):
In order to plot the BDT sorted signal regions, set do_BDT_regions = true in setup_sr.json
```
python make_hists.py \
    --year 2022EE \
    --tag 26Feb03 \
    --setup setup_sr.json \
    --outdir results \
    --save-root
```
Outputs:

- ROOT File: results/fitting_{year}_{region}.root (Used by Datacard Maker)

- Pickle File: results/hists_{year}_{region}_{variable}_nominal.pkl (Used by Plotter)

Note: Ensure your setup.json has the correct branch_name (e.g., FatJet0_msd) to avoid KeyErrors.

#### Important (Signal region only):
`make_hists.py` generates separate ROOT files for each signal category (ggf, vbf, vh). You must merge them before running the datacard maker:
```
hadd -f results/fitting_2022EE_signal_msd.root \
    results/fitting_2022EE_ggf_msd.root \
    results/fitting_2022EE_vbf_msd.root \
    results/fitting_2022EE_vh_msd.root
```

## 3. Validation Plotting
To verify the Data/MC agreement before running the fit, use the unified plotter. This script expects the .pkl files generated in Step 2.
```
python plot_control.py \
    --year 2022EE \
    --indir results \
    --outdir plots \
    --region control-zgamma \
    --setup setup_zgcr.json \
    --plot_type process
```
### Available `--plot_type` options:

- `process`: (Default) Plots stacked backgrounds grouped by physical process (e.g., Zgamma, Wjets, ttbar).

- `flavor`: Separates the V+Jets backgrounds by generator-level jet flavor (e.g., Wjets_c-jet, Zjets_b-jet) to validate heavy flavor modeling.

- `qcd_shape`: Compares the kinematic shapes of the data-driven proxy background (QCD or GJets) in the pass vs. fail regions. You can use --norm_type shape or --norm_type density.

- `inclusive`: Generates single, inclusive plots integrating across all pT bins.

## 4. Datacard Creation
The `make_datacards.py` script builds the statistical model using rhalphalib. It performs the QCD Transfer Factor (TF) fit and prepares the workspace for Combine.

### For the zgamma control region:
```
python make_datacards.py \
    --year 2022EE \
    --tag 26Feb03 \
    --indir results \
    --analysis zgcr
```

### For the signal region:
```
python make_datacards.py \
    --year 2022EE \
    --tag 26Feb03 \
    --indir results \
    --analysis vbf
```

*(Note: You can control the Bernstein polynomial degrees using --mc-pt-order and --mc-rho-order for the MC transfer factor, and --res-pt-order/--res-rho-order for the data residual).*

## 5. Compiling the workspace and running Combine:
Step 4 automatically generates the text datacards and a `build.sh` script. Navigate into the output directory to compile the Combine workspace (it will print out for you at the end of the run). Example:
```
cd results/26Feb03/2022EE/datacards/vbfModel_2022EE
./build.sh
```
Once `workspace.root` is created, you can run standard Combine tools. For example, to run a FitDiagnostics check:
```
combine -M FitDiagnostics -d workspace.root \
    -t -1 --setParameters rggF=1,rVBF=1,rVH=1 \
    --saveShapes --saveWithUncertainties \
    --cminDefaultMinimizerStrategy 0
```

## 6. Combining Multiple Years (Full Run 3)
To perform a full Run 3 fit, you need to generate the datacards for each year individually, compile them, and then stitch them together.

**Step 6.1: Prepare Individual Years**
For each year (2022, 2022EE, 2023), run Step 4 (`make_datacards.py`) and Step 5 (executing `./build.sh`).
*Running `./build.sh` is required because it generates the `model_combined.txt` file for that specific year.*

**Step 6.2: Combine the text datacards**
Once every year has its own `model_combined.txt`, stitch them together using `combineCards.py`. You assign a prefix (like `y22EE=`) to each text card so Combine knows how to correlate the systematics across the different eras.

```bash
combineCards.py \
    y22=results/26Feb03/2022/datacards/vbfModel_2022/model_combined.txt \
    y22EE=results/26Feb03/2022EE/datacards/vbfModel_2022EE/model_combined.txt \
    y23=results/26Feb03/2023/datacards/vbfModel_2023/model_combined.txt \
    > full_Run3_vbfModel.txt
```

**Step 6.3: Build the final workspace**
Run `text2workspace.py` on your new master datacard. This manual command does exactly what `build.sh` did for the single years, but for the entire Run 3 dataset.
```
text2workspace.py -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose full_Run3_vbfModel.txt -o workspace_Run3.root
```


## 7. F-Tests (Determining TF Polynomial Order)
To determine the optimal order for the QCD Bernstein polynomials, perform an F-test comparing a "Null" (simpler) model to an "Alt" (more complex) model.

**Step 7.1: Generate the two workspaces**
Run `make_datacards.py` (Step 4) and compile the workspace (Step 5) twice to create your two models. You can control the complexity of the fit using the optional flags you pass to the script:
* MC Transfer Factor: `--mc-pt-order` and `--mc-rho-order`
* Data Residual: `--res-pt-order` and `--res-rho-order`

*Example: Testing a Null model (MC pt=0, rho=1) vs. an Alt model (MC pt=1, rho=1):*

```bash
# 1. Build the Null Model
python make_datacards.py --year 2022EE --tag 26Feb03 --analysis vbf --mc-pt-order 0 --mc-rho-order 1
# (Navigate to the output directory, run ./build.sh, and rename the resulting workspace.root to workspace_01.root)

# 2. Build the Alt Model
python make_datacards.py --year 2022EE --tag 26Feb03 --analysis vbf --mc-pt-order 1 --mc-rho-order 1
# (Navigate to the output directory, run ./build.sh, and rename the resulting workspace.root to workspace_11.root)
```

### Step 7.2: Run the F-Test toys
This script generates snapshots, calculates the observed Goodness-of-Fit, and generates/fits pseudo-experiments (toys).

```bash
python run_ftest.py \
    --null workspace_01.root \
    --alt workspace_11.root \
    --ntoys 100 \
    --seed 123456 \
    --tag vbf_01_vs_11
```

### Step 7.3: Plot the F-Statistic Distribution
Visualize the results and calculate the p-value to see if the complex model is justified. You must provide the total number of bins and the number of parameters for each model (p=(pt_order+1)×(rho_order+1)).

```bash
python plot_ftest.py \
    --tag vbf_01_vs_11 \
    --nbins 128 \
    --p1 2 \
    --p2 4
```

## 8. Post-Fit and Pre-Fit Visualization
Use the [`combine_postfits`](https://github.com/andrzejnovak/combine_postfits) repository to generate publication-quality plots. This tool allows for automatic category merging and handles the complex multi-signal strengths used in the signal region.

### 8.1 Setup the Style File

Before plotting, ensure a `sty.yml` exists to define process colors and labels. If it doesn't exist, running the script once will generate a default one that you can then edit.

### 8.2 Generate Signal Region Plots

To plot the signal region (VBF, ggF, VH) using the Asimov dataset, use the `--MC` flag. Note that the plotter natively supports up to two explicit signals; additional signals (like WH and ZH) will be stacked as backgrounds.

```bash
python3 make_plots.py \
    -i fitDiagnostics.root \
    -o plots_signal_vbf \
    --MC \
    --bkgs qcd,ttbar,Wjets,Zjetsbb,singlet,VV,ttH \
    --sigs VBF,ggF \
    --rmap 'VBF:rVBF,ggF:rggF' \
    --onto qcd \
    --xlabel '$m_{SD}$ [GeV]' \
    --year 2022EE \
    --lumi 26.7 \
    --cmslabel "Preliminary (Asimov)" \
    --cats "VBF_PassBB:*vbfpassbb*;VBF_Fail:*vbffail*;ggF_PassBB:*ggfpassbb*" \
    -p 10
```

### 8.3 Key Argument Descriptions

- `--rmap`: Maps your signal processes to the specific POIs (rVBF, rggF, etc.) found in the fitDiagnostics.root file to ensure correct post-fit scaling.

- `--cats`: Uses wildcards (e.g., *vbfpassbb*) to automatically find and merge all pT​ bins belonging to a specific category into a single plot.

- `--onto`: Specifies a background (usually QCD) to plot as an unfilled line, stacking other processes on top of it.

- `-p`: Enables multiprocessing to speed up the rendering of multiple categories simultaneously.
