#!/usr/bin/env python3
"""
HMDS SR fit driver — run any subset of the fit stages for a chosen model order.
A model order is four ints: (mc_pt, mc_rho, res_pt, res_rho).

Stages:
  datacard  make_datacards + build.sh, copy workspace.root -> workspace_<order>.root
  fitdiag   FitDiagnostics (postfit shapes + residual params), robustFit+robustHesse
  postfit   plot_fit.py (transfer-factor panels + prefit/postfit stacks)
  gof       saturated Goodness-of-Fit: snapshot -> observed -> N toys -> plot
  limit     AsymptoticLimits (blinded/expected by default) + parsed printout

F-test is separately used to provide the best order:
  --stages ftest   runs the sequential scan, prints + saves the winning order to
                   <results>/<tag>/<year>/chosen_order.txt, then stops.

Intended workflow:
  # pick an order via F-test:
  python run_fit.py --tag 20260721 --year 2024 --stages ftest
  # run the full chain on a specific order:
  python run_fit.py --tag 20260721 --year 2024 --order 2,0,1,0 \
      --stages fitdiag,postfit,gof,limit
  
Example of just running one step, e.g. GoF (auto-builds datacard/workspace if needed):
  python run_fit.py --tag 20260721 --year 2024 --order 2,0,1,0 --stages gof

Specified stages always run and overwrite their outputs

Non-specified stages are not rerun if they are not needed or if \
their output already exists
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from fit_common import FITTING_DIR, dc_dir, fitdiag_name, order_tag, parse_order, run, ws_name

STEP_ORDER = ["datacard", "fitdiag", "postfit", "gof", "limit"] # intended order of execution after F-test


class FitDriver:
    def __init__(self, args):
        self.tag = args.tag
        self.year = args.year
        self.order = args.order
        self.model = args.model
        self.outdir = Path(args.outdir)
        self.indir = Path(args.indir) if args.indir else Path(f"results/{args.tag}")
        self.ntoys = args.ntoys
        self.seed = args.seed
        self.setup = args.setup
        self.blind = not args.unblind
        self.dc = dc_dir(self.outdir, self.tag, self.year, self.model)
        self.templates = self.indir / f"fitting_{self.year}_sr_msd.root"
        self._done = set()  # stages completed in this run

    def require_templates(self):
        if not self.templates.exists():
            raise SystemExit(
                f"[run_fit] templates missing: {self.templates}\n"
                f"  Build them first:\n"
                f"  python make_hists.py --year {self.year} --tag {self.tag}_v15 "
                f"--sig-tag {self.tag}_private_v15_private --setup setup_sr.json "
                f"--outdir {self.indir} --save-root"
            )

    # -- ensure a stage (and its prerequisites) has run once this invocation
    def ensure(self, stage):
        if stage in self._done:
            return
        getattr(self, f"stage_{stage}")()
        self._done.add(stage)

    # ---------------------------------------------------------- stages
    def stage_datacard(self):
        self.require_templates()
        o = self.order
        run(
            f"python make_datacards.py --year {self.year} --tag {self.tag} "
            f"--indir {self.indir} --outdir {self.outdir} --analysis sr "
            f"--mc-pt-order {o[0]} --mc-rho-order {o[1]} "
            f"--res-pt-order {o[2]} --res-rho-order {o[3]}",
            cwd=FITTING_DIR,
        )
        run("./build.sh", cwd=self.dc)
        src, dst = self.dc / "workspace.root", self.dc / ws_name(o)
        shutil.copy(src, dst)
        print(f"[run_fit] workspace -> {dst}")

    def stage_fitdiag(self):
        if not (self.dc / ws_name(self.order)).exists():
            self.ensure("datacard")
        run(
            f"combine -M FitDiagnostics {ws_name(self.order)} "
            f"--saveShapes --saveWithUncertainties --saveNormalizations "
            f"-n _{order_tag(self.order)} ",
            #f"--cminDefaultMinimizerStrategy 0 --robustFit 1 --robustHesse 1",
            cwd=self.dc,
        )
        print(f"[run_fit] fit diagnostics -> {self.dc / fitdiag_name(self.order)}")

    def stage_postfit(self): #postfit plots
        if not (self.dc / fitdiag_name(self.order)).exists():
            self.ensure("fitdiag")
        o = self.order
        run(
            f"python plot_fit.py --tag {self.tag} --year {self.year} "
            f"--outdir {self.outdir} --model {self.model} "
            f"--fitdiag {fitdiag_name(o)} "
            f"--mc-pt-order {o[0]} --mc-rho-order {o[1]} "
            f"--res-pt-order {o[2]} --res-rho-order {o[3]}",
            cwd=FITTING_DIR,
        )
        print(f"[run_fit] plots -> {self.dc / 'fit_plots'}")

    def stage_gof(self):
        if not (self.dc / ws_name(self.order)).exists():
            self.ensure("datacard")
        o = self.order
        gtag = f"gof_{order_tag(o)}"
        snap = f"higgsCombine_{gtag}_Snapshot.MultiDimFit.mH120.root"
        # Run fit, and save best fit parameters
        run(
            f"combine -M MultiDimFit -d {ws_name(o)} -n _{gtag}_Snapshot "
            f"--saveWorkspace --cminDefaultMinimizerStrategy 0",
            cwd=self.dc,
        )
        # Load snapshot and get the observed GoF value
        run(
            f"combine -M GoodnessOfFit -d {snap} --snapshotName MultiDimFit "
            f"-n _{gtag}_Observed --algo saturated",
            cwd=self.dc,
        )
        # Throw toys based on snapshot, needed because of data-driven bkg
        # Report the usage of these options in the AN for stat review
        run(
            f"combine -M GoodnessOfFit -d {snap} --snapshotName MultiDimFit "
            f"--bypassFrequentistFit --toysFrequentist -n _{gtag}_Toys --algo saturated "
            f"-t {self.ntoys} --seed {self.seed}",
            cwd=self.dc,
        )
        # Plot GoF results
        run(
            f"python plot_gof.py --dir {self.dc} --tag {gtag} --seed {self.seed} "
            f'--label "HMDS SR  {order_tag(o)}"',
            cwd=FITTING_DIR,
        )

    def stage_limit(self):
        if not (self.dc / ws_name(self.order)).exists():
            self.ensure("datacard")
        o = self.order
        ltag = order_tag(o)
        run_opt = "--run blind" if self.blind else ""
        run(
            f"combine -M AsymptoticLimits {ws_name(o)} {run_opt} "
            f"-n _{ltag}",
            cwd=self.dc,
        )
        self._report_limit(self.dc / f"higgsCombine_{ltag}.AsymptoticLimits.mH120.root")

    def _report_limit(self, path):
        try:
            import uproot

            with uproot.open(path) as f:
                q = f["limit"]["quantileExpected"].array(library="np")
                r = f["limit"]["limit"].array(library="np")
        except Exception as e:  # noqa: BLE001
            print(f"[run_fit] could not parse limit file {path}: {e}")
            return
        labels = {0.025: "-2sigma", 0.16: "-1sigma", 0.5: "median",
                  0.84: "+1sigma", 0.975: "+2sigma", -1.0: "observed"}
        print(f"\n[run_fit] Asymptotic {'expected (blind)' if self.blind else ''} limits on r:")
        for qi, ri in zip(q, r):
            key = min(labels, key=lambda k: abs(k - qi))
            print(f"    {labels[key]:>8s} (q={qi:+.3f}) :  r < {ri:.3f}")

    # ---------------------------------------------------------- ftest mode
    def run_ftest(self):
        # run_ftest_scan relies on toys instead of assuming the PDF
        # Takes several minutes per tested order
        from run_ftest_scan import scan

        order = scan(
            tag=self.tag, year=self.year, model=self.model,
            outdir=str(self.outdir), indir=str(self.indir),
            setup=self.setup, seed=self.seed,
        )
        outdir = self.outdir / self.tag / self.year
        outdir.mkdir(parents=True, exist_ok=True)
        dest = outdir / "chosen_order.txt"
        txt = ",".join(str(x) for x in order)
        dest.write_text(txt + "\n")
        print(f"\n[run_fit] F-test winner: order {txt}")
        print(f"[run_fit] saved -> {dest}")
        print(f"[run_fit] now run: python run_fit.py --tag {self.tag} "
              f"--year {self.year} --order {txt} "
              f"--stages fitdiag,postfit,gof,limit")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--year", required=True)
    p.add_argument("--order", default=None,
                   help="mc_pt,mc_rho,res_pt,res_rho (e.g. 2,0,1,0). "
                        "If omitted, falls back to chosen_order.txt from an F-test run.")
    p.add_argument("--stages", required=True,
                   help="comma list: any of ftest,datacard,fitdiag,postfit,gof,limit")
    p.add_argument("--model", default="srModel")
    p.add_argument("--outdir", default="results")
    p.add_argument("--indir", default=None, help="dir with fitting_<year>_sr_msd.root (default results/<tag>)")
    p.add_argument("--setup", default="setup_sr.json",
                   help="setup json; ftest derives its bin count from it")
    p.add_argument("--ntoys", type=int, default=500, help="GoF toys (default 500)")
    p.add_argument("--seed", type=int, default=123456)
    p.add_argument("--unblind", action="store_true", help="limits: use observed data (default: blinded/expected)")
    args = p.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STEP_ORDER) - {"ftest"}
    if unknown:
        raise SystemExit(f"unknown stage(s): {sorted(unknown)}; valid: ftest,{','.join(STEP_ORDER)}")

    # ftest is exclusive (order-provider mode)
    if "ftest" in stages:
        if len(stages) > 1:
            print("[run_fit] NOTE: 'ftest' is order-provider only; ignoring other stages. "
                  "Re-run with the chosen --order to continue.")
        args.order = args.order  # unused by ftest
        FitDriver(_with_order(args, (0, 0, 0, 0))).run_ftest()
        return

    # resolve order for the consuming stages
    if args.order:
        order = parse_order(args.order)
    else:
        chosen = Path(args.outdir) / args.tag / args.year / "chosen_order.txt"
        if chosen.exists():
            order = parse_order(chosen.read_text().strip())
            print(f"[run_fit] using order {order} from {chosen}")
        else:
            raise SystemExit("no --order given and no chosen_order.txt found; "
                             "run --stages ftest first or pass --order.")

    drv = FitDriver(_with_order(args, order))
    for st in STEP_ORDER:
        if st in stages:
            drv.ensure(st)
    print("\n[run_fit] done:", ", ".join(s for s in STEP_ORDER if s in stages))


def _with_order(args, order):
    args.order = order
    return args


if __name__ == "__main__":
    sys.exit(main())
