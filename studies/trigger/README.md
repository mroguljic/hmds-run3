# Trigger efficiency and scale factors

`trigger_eff.py` measures the trigger efficiency (`src/hbb/triggers.json`:
`AK8PFJet250_SoftDropMass40_PNetBB0p06 || AK8PFJet425_SoftDropMass30`) in the mu+jets ttbar control
region (`control-tt` skims), with `Mu50` as the reference trigger, and derives data/MC scale
factors binned in AK8 jet pT x PNet Xbb (`FatJet0_pnetTXbb`).

## Usage (from the repo root)

```bash
python studies/trigger/trigger_eff.py all --outdir studies/trigger/output/<YYMMDD>
```

- `load`: skims -> `<outdir>/cache/` (~2 min). By default reads the `261005_v15` (control-tt) and
 `261005_v15_private` (signal) skims from `root://cmseos.fnal.gov//store/group/lpchbbrun3/roguljic/hmds/`
 over xrootd; for other skims, pass `--skim-dir` / `--signal-dir` (a local path or `root://` URL).
- `analyze`: cache -> `<outdir>/plots/`, `trigger_sf_<year>.json` (correctionlib) and `summary.md` (~1 min).
  `summary.md` has the per-bin efficiencies, SFs and uncertainties, and the signal-weighted SF per SR pT bin.
- `all`: both.

The correctionlib `trigger_sf` takes `(systematic, pt, xbb)` with systematic in `nominal`, `up`/`down`
(nominal ± total), `stat_up/down` and `syst_up/down`; pt/xbb are clamped outside the measured range (pT 350-1200).

## Method

**Events (denominator).** `control-tt` skim (Mu50, exactly one muon with pT > 55 and |eta| < 2.1, no
electrons, >= 1 b-tagged AK4 jet on the leptonic side, MET > 50; probe = highest-nSV AK8 jet with
dphi(muon) > 2pi/3), plus:
- Mu50 fired (reference trigger);
- probe 40 <= msd < 201 (fit mass window);
- no good AK8 jet (pT > 200) on the leptonic side, so only the probe can fire the jet trigger;
- probe P(QCD) < 0.075 (SR pass region; ~98% top in MC).

Data: `Muon` PD, 2024C-I. MC: all control-tt backgrounds (ttbar, single top, W(lnu)+jets, QCD,
V(qq), VV), each event weighted by w = xsec x lumi x genWeight x pileup x (b-tag SF of the leptonic-side
AK4 jets) x (muon ID SF) x (muon ISO SF) / sum(genWeight) of its sample.

**Binning.** pT edges 350, 400, 450, 500, 1200 GeV (one wide bin above 500 GeV, where the efficiency is
flat); Xbb edges 0, 0.3, 0.6, 0.8, 1. The Xbb edges were chosen by hand to split the SR signal into
reasonably populated bins (~6%, 26%, 40%, 29%) while keeping >= ~100 data events per cell.

**Efficiency in a (pT, Xbb) bin.** Numerator = denominator events that also fire the jet-trigger OR.
- Data: eps_data = k / n. Uncertainty: 68.3% Clopper-Pearson interval.
- MC: eps_MC = sum_pass(w) / sum_all(w). Uncertainty: Clopper-Pearson with effective entries
  N_eff = (sum w)^2 / sum(w^2) and k_eff = eps_MC x N_eff.

Asymmetric intervals are symmetrized (mean of up and down) when propagated to the SF.

**SF and its uncertainty in a bin.**
- SF = eps_data / eps_MC.
- stat = SF x sqrt((d eps_data / eps_data)^2 + (d eps_MC / eps_MC)^2).
- syst = |SF(P(QCD) < 0.075) - SF(no P(QCD) cut)|, i.e. the difference between the measurements with and
  without the P(QCD) < 0.075 cut.
- total = sqrt(stat^2 + syst^2).

**Signal-weighted SF.** Look up SF(pT, Xbb) for each signal event in the SR pass region
(40 < msd < 201, P(QCD) < 0.075, nSV > NSV_MIN) and average it with the signal weights per SR pT bin.
The quoted uncertainty is the weighted average of the per-bin total uncertainty, i.e. fully correlated
across bins.

## Plots

| file | content |
|---|---|
| `region_pt`, `region_xbb`, `region_msd` | measurement region, data vs MC stack (MC scaled to the data yield) |
| `turnon_pt_paths` | OR and the two individual paths |
| `turnon_pt_xbb` | OR turn-on in the four Xbb bins of the SF |
| `eff_maps` | eps_data and eps_MC per (pT, Xbb) bin |
| `sf_map` | SF ± total uncertainty per (pT, Xbb) bin |
| `signal_impact` | where the SR signal sits in (pT, Xbb) and the signal-weighted SF per SR pT bin |
