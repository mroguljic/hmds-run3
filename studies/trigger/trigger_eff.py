#!/usr/bin/env python3
"""
AK8 jet-trigger efficiency and data/MC scale factors, measured in the mu+jets ttbar
control region (``control-tt`` skims) with the single-muon trigger as reference.

Efficiency of the SR trigger OR (``triggers.json``) for the CR probe jet (the nSV-sorted
AK8 jet on the hadronic side), in data (Muon PD) and MC (all control-tt backgrounds):

    eps = N(reference & jet-trigger OR) / N(reference)

Denominator: control-tt skim selection (Mu50, one muon, lepton-side b-tag, MET > 50),
probe 40 <= msd < 201 (SR window) and no good AK8 jet on the leptonic side
(``LepSideFatJet_pt < 200``), so the probe is the only AK8 jet that can fire the
trigger -- otherwise a boosted leptonic top fires it and fakes efficiency in the turn-on.
The nominal SF uses SR-pass-like probes, P(QCD) < 0.075 (~98% top): at fixed pT and Xbb
the efficiency still depends on P(QCD), and the SF is applied in the SR pass region.

The SF is binned in probe pT x offline PNet Xbb (``FatJet0_pnetTXbb``): the
``AK8PFJet250_SoftDropMass40_PNetBB0p06`` path dominates the signal in the 350-450 GeV
turn-on and its efficiency depends strongly on the (HLT) PNet bb score.

Subcommands:

``load``
    control-tt + signal skims -> reduced parquet cache in ``<outdir>/cache/`` (skims are
    read through fsspec, so ``--skim-dir`` can be a local/FUSE path or a ``root://`` URL).

``analyze``
    cache -> ``<outdir>/plots/`` (.png/.pdf), ``trigger_sf_<year>.json`` (correctionlib)
    and ``summary.md``.

``all``
    load, then analyze.

SF uncertainty per bin: data (Clopper-Pearson) and MC (effective-entries) statistics, plus a
systematic |SF(P(QCD) < 0.075) - SF(no P(QCD) cut)|, added in quadrature.

Example (from the repo root):
    python studies/trigger/trigger_eff.py all --outdir studies/trigger/output/261012
"""
from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import correctionlib.schemav2 as cs
import fsspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import beta

import hbb
from hbb.common_vars import LUMI, NSV_MIN

YEAR = "2024"
HBB_DIR = Path(hbb.__file__).parent
TRIGGERS = json.loads((HBB_DIR / "triggers.json").read_text())[YEAR]
REF_TRIGGERS = json.loads((HBB_DIR / "muon_triggers.json").read_text())[YEAR]

EOS = "root://cmseos.fnal.gov//store/group/lpchbbrun3/roguljic/hmds"
DEFAULT_SKIMS = f"{EOS}/261005_v15/{YEAR}"
DEFAULT_SIGNAL = f"{EOS}/261005_v15_private/{YEAR}/Signal"

DATA_PREFIX = "Muon_Run"
# MC groups for the region-composition plots (order = stacking order, bottom first)
MC_GROUPS = {
    "tt1l": (r"$t\bar{t}$ (1$\ell$)", ["TTtoLNu2Q"]),
    "tt2l": (r"$t\bar{t}$ (2$\ell$)", ["TTto2L2Nu"]),
    "singletop": ("Single top", ["TWminus", "TbarWplus", "TBbarQ", "TbarBQ"]),
    "wjets": (r"W($\ell\nu$)+jets", ["WtoLNu"]),
    "qcd": ("QCD", ["QCD_HT"]),
    "other": (r"Other ($t\bar{t}$ 0$\ell$, V(qq), VV)", ["TTto4Q", "Wto2Q", "Zto2Q", "WW", "WZ", "ZZ"]),
}

PATH_PNET, PATH_AK8 = "AK8PFJet250_SoftDropMass40_PNetBB0p06", "AK8PFJet425_SoftDropMass30"
CR_COLUMNS = [
    "FatJet0_pt", "FatJet0_msd", "FatJet0_pnetTXbb", "FatJet0_ParTPQCD", "FatJet0_nSV",
    "LepSideFatJet_pt", "nFatJet", "event", "weight", *TRIGGERS, *REF_TRIGGERS,
]
SIG_COLUMNS = [
    "FatJet0_pt", "FatJet0_msd", "FatJet0_pnetTXbb", "FatJet0_ParTPQCD", "FatJet0_nSV",
    "weight", *TRIGGERS,
]

# measurement / SR definitions
MSD_WINDOW = (40.0, 201.0)
LEPSIDE_MAX = 200.0  # good AK8 jets have pT > 200, so this vetoes any leptonic-side AK8 jet
PQCD_PASS = 0.075  # fitting/make_hists.py
SR_PT_BINS = [350, 450, 550, 700, 1200]  # fitting/setup_sr.json
SF_PT_BINS = [350, 400, 450, 500, 1200]  # flat plateau above 500 GeV
SF_XBB_BINS = [0.0, 0.3, 0.6, 0.8, 1.0]
TURNON_PT_BINS = [*range(200, 600, 25), 600, 650, 700, 800, 1000, 1200]

# colors: CMS-recommended (Petroff) categorical set; data in black
C_MC = "#5790fc"
STACK_COLORS = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#7a21dd", "#9c9ca1"]


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def _read_parquets(fs, files, columns):
    def one(f):
        names = pq.read_schema(f, filesystem=fs).names
        df = pq.read_table(f, columns=[c for c in columns if c in names], filesystem=fs).to_pandas()
        for c in df.columns:
            # parts where e.g. a column is all-None are written with a null type
            if df[c].dtype == object:
                df[c] = df[c].astype(np.float32)
        return df

    with ThreadPoolExecutor(16) as ex:
        dfs = [d for d in ex.map(one, files) if len(d)]
    return pd.concat(dfs, ignore_index=True) if dfs else None


def _sumw(fs, sample_dir):
    total = 0.0
    for f in fs.glob(f"{sample_dir}/pickles/*.pkl"):
        with fs.open(f, "rb") as fh:
            out = pickle.load(fh)
        for key in out:
            total += next(iter(out[key]["nominal"]["sumw"].values()))
    return total


def mc_group(sample):
    for g, (_, prefixes) in MC_GROUPS.items():
        if any(sample.startswith(p) for p in prefixes):
            return g
    return None


def load(args):
    cache = args.outdir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    fs, root = fsspec.core.url_to_fs(args.skim_dir)
    samples = sorted(Path(p).name for p in fs.ls(root, detail=False))
    dfs = []
    for s in samples:
        is_data = s.startswith(DATA_PREFIX)
        group = "data" if is_data else mc_group(s)
        if group is None:
            continue
        files = fs.glob(f"{root}/{s}/parquet/nominal/control-tt/*.parquet")
        df = _read_parquets(fs, files, CR_COLUMNS)
        if df is None:
            print(f"  {s}: no control-tt events, skipped")
            continue
        if is_data:
            df["w"] = 1.0
        else:
            df["w"] = df["weight"] / _sumw(fs, f"{root}/{s}")
        df = df.drop(columns="weight")
        df["group"] = group
        df["sample"] = s
        print(f"  {s:45s} {group:9s} {len(df):9d} rows, yield {df['w'].sum():11.1f}", flush=True)
        dfs.append(df)
    cr = pd.concat(dfs, ignore_index=True)
    for c in cr.columns:
        if cr[c].dtype == np.float64:
            cr[c] = cr[c].astype(np.float32)
    cr[["group", "sample"]] = cr[["group", "sample"]].astype("category")
    cr.to_parquet(cache / "control_tt.parquet")
    print(f"wrote {cache / 'control_tt.parquet'} ({len(cr)} rows)")

    fs, root = fsspec.core.url_to_fs(args.signal_dir)
    sig = _read_parquets(fs, fs.glob(f"{root}/parquet/nominal/signal-all/*.parquet"), SIG_COLUMNS)
    sig.to_parquet(cache / "signal.parquet")
    print(f"wrote {cache / 'signal.parquet'} ({len(sig)} rows)")


# ---------------------------------------------------------------------------
# efficiencies
# ---------------------------------------------------------------------------
def clopper_pearson(k, n, cl=0.682689):
    a = (1 - cl) / 2
    lo = np.where(k > 0, beta.ppf(a, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - a, k + 1, n - k), 1.0)
    return lo, hi


def efficiency(passed, w, idx, nbins):
    """Binned efficiency with (eff, err_lo, err_hi, n_eff). idx: bin index per entry, -1 = skip."""
    ok = idx >= 0
    sw = np.bincount(idx[ok], weights=w[ok], minlength=nbins)
    sw2 = np.bincount(idx[ok], weights=w[ok] ** 2, minlength=nbins)
    sp = np.bincount(idx[ok], weights=(w * passed)[ok], minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = sp / sw
        neff = np.where(sw2 > 0, sw**2 / sw2, 0.0)
    # Clopper-Pearson on effective entries (exact for unit weights, i.e. data)
    lo, hi = clopper_pearson(np.clip(eff * neff, 0, neff), neff)
    lo, hi = np.where(neff > 0, lo, np.nan), np.where(neff > 0, hi, np.nan)
    return eff, eff - lo, hi - eff, neff


def bin_index(x, edges):
    i = np.digitize(x, edges) - 1
    return np.where((i >= 0) & (i < len(edges) - 1) & np.isfinite(x), i, -1)


def bin_index_2d(x, xe, y, ye):
    ix, iy = bin_index(x, xe), bin_index(y, ye)
    return np.where((ix >= 0) & (iy >= 0), ix * (len(ye) - 1) + iy, -1)


class CR:
    """control-tt events in the measurement denominator."""

    def __init__(self, df):
        msd = df.FatJet0_msd.to_numpy()
        ref = np.zeros(len(df), bool)
        for t in REF_TRIGGERS:
            ref |= df[t].to_numpy().astype(bool)
        sel = (
            ref
            & (msd >= MSD_WINDOW[0])
            & (msd < MSD_WINDOW[1])
            & (df.LepSideFatJet_pt.to_numpy() < LEPSIDE_MAX)
            & np.isfinite(df.FatJet0_pnetTXbb.to_numpy())
        )
        self.df = df[sel].reset_index(drop=True)
        d = self.df
        self.is_data = (d.group == "data").to_numpy()
        self.w = d.w.to_numpy().astype(np.float64)
        self.pt = d.FatJet0_pt.to_numpy()
        self.xbb = d.FatJet0_pnetTXbb.to_numpy()
        self.paths = {t: d[t].to_numpy().astype(bool) for t in TRIGGERS}
        self.OR = np.any(np.stack(list(self.paths.values())), axis=0)
        self.pqcd = np.nan_to_num(d.FatJet0_ParTPQCD.to_numpy(), nan=1.0)
        self.pass_like = self.pqcd < PQCD_PASS

    def eff(self, idx, nbins, passed=None, extra=None, probes="pass"):
        """probes: "pass" = SR-pass-like probes (P(QCD) < 0.075, nominal), "all" = any P(QCD)."""
        passed = self.OR if passed is None else passed
        d, m = self.is_data, ~self.is_data
        if probes == "pass":
            d, m = d & self.pass_like, m & self.pass_like
        if extra is not None:
            d, m = d & extra, m & extra
        ed = efficiency(passed, np.ones_like(self.w), np.where(d, idx, -1), nbins)
        em = efficiency(passed, self.w, np.where(m, idx, -1), nbins)
        return ed, em


def sym(e):
    return 0.5 * (e[1] + e[2])


def ratio_err(ed, em):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = ed[0] / em[0]
        err = r * np.sqrt((sym(ed) / ed[0]) ** 2 + (sym(em) / em[0]) ** 2)
    return r, err


def measure_sf(cr):
    nx, ny = len(SF_PT_BINS) - 1, len(SF_XBB_BINS) - 1
    idx = bin_index_2d(np.clip(cr.pt, None, SF_PT_BINS[-1] - 1e-3), SF_PT_BINS, cr.xbb, SF_XBB_BINS)
    ed, em = cr.eff(idx, nx * ny)
    ed_all, em_all = cr.eff(idx, nx * ny, probes="all")
    sf, stat = ratio_err(ed, em)
    sf_allpq = ed_all[0] / em_all[0]
    syst = np.abs(sf - sf_allpq)
    total = np.hypot(stat, syst)
    shape = (nx, ny)
    return {
        "eff_data": ed[0].reshape(shape),
        "eff_data_err": sym(ed).reshape(shape),
        "eff_mc": em[0].reshape(shape),
        "eff_mc_err": sym(em).reshape(shape),
        "n_data": ed[3].reshape(shape),
        "sf": sf.reshape(shape),
        "stat": stat.reshape(shape),
        "syst": syst.reshape(shape),
        "sf_allpq": sf_allpq.reshape(shape),
        "total": total.reshape(shape),
    }


def sf_lookup(res, pt, xbb, key="sf"):
    ix = np.clip(np.digitize(pt, SF_PT_BINS) - 1, 0, len(SF_PT_BINS) - 2)
    iy = np.clip(np.digitize(xbb, SF_XBB_BINS) - 1, 0, len(SF_XBB_BINS) - 2)
    return res[key][ix, iy]


def write_correctionlib(res, path):
    def content(values):
        return cs.MultiBinning(
            nodetype="multibinning",
            inputs=["pt", "xbb"],
            edges=[SF_PT_BINS, SF_XBB_BINS],
            content=[float(v) for v in values.ravel()],
            flow="clamp",
        )

    variations = {
        "nominal": res["sf"],
        "up": res["sf"] + res["total"],
        "down": res["sf"] - res["total"],
        "stat_up": res["sf"] + res["stat"],
        "stat_down": res["sf"] - res["stat"],
        "syst_up": res["sf"] + res["syst"],
        "syst_down": res["sf"] - res["syst"],
    }
    corr = cs.Correction(
        name="trigger_sf",
        version=1,
        description=(
            f"{YEAR} AK8 jet-trigger OR ({' | '.join(TRIGGERS)}) data/MC SF, measured in the "
            f"mu+jets ttbar CR w.r.t. {' | '.join(REF_TRIGGERS)}. Inputs: candidate AK8 jet pT and "
            "PNet XbbVsQCD (FatJet0_pnetTXbb); clamped outside the measured range."
        ),
        inputs=[
            cs.Variable(name="systematic", type="string", description=", ".join(variations)),
            cs.Variable(name="pt", type="real", description="AK8 jet pT [GeV]"),
            cs.Variable(name="xbb", type="real", description="AK8 jet PNet XbbVsQCD"),
        ],
        output=cs.Variable(name="weight", type="real"),
        data=cs.Category(
            nodetype="category",
            input="systematic",
            content=[cs.CategoryItem(key=k, value=content(v)) for k, v in variations.items()],
        ),
    )
    cset = cs.CorrectionSet(schema_version=2, corrections=[corr])
    path.write_text(cset.model_dump_json(exclude_unset=True, indent=1))


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def setup_mpl():
    mpl.use("Agg")
    plt.style.use(hep.style.CMS)
    return plt, hep


def cms_label(hep, ax, data=True, fontsize=20):
    kw = {"lumi": round(LUMI[YEAR] / 1000)} if data else {}
    hep.cms.label("WiP", data=data, year=YEAR, com=13.6, ax=ax, fontsize=fontsize, **kw)


def savefig(fig, plotdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(plotdir / f"{name}.{ext}", bbox_inches="tight", dpi=150 if ext == "png" else None)
    print(f"  {name}")


PROBE_CAPTION = (rf"$\mu$+jets $t\bar{{t}}$ CR, {MSD_WINDOW[0]:g} $\leq m_{{SD}}$ < {MSD_WINDOW[1]:g}, "
                 rf"P(QCD) < {PQCD_PASS:g}")


def caption(fig, text):
    fig.text(0.5, 0.925, text, ha="center", va="bottom", fontsize=18)


def pt_label(lo, hi):
    return rf"{lo:g} $\leq p_T$ < {hi:g} GeV"


def xbb_label(lo, hi):
    return rf"{lo:g} $\leq$ Xbb < {hi:g}"


def draw_eff_panel(ax, rax, edges, ed, em, label_mc="MC"):
    """Efficiency (data points, MC band) on ax and data/MC on rax."""
    edges = np.asarray(edges, float)
    c = 0.5 * (edges[1:] + edges[:-1])
    xerr = 0.5 * np.diff(edges)
    lo_m, hi_m = em[0] - em[1], em[0] + em[2]
    ax.stairs(em[0], edges, color=C_MC, lw=2, label=label_mc)
    ax.stairs(hi_m, edges, baseline=lo_m, fill=True, color=C_MC, alpha=0.3, lw=0)
    ok = ed[3] > 0
    ax.errorbar(c[ok], ed[0][ok], yerr=[ed[1][ok], ed[2][ok]], xerr=xerr[ok], fmt="o", color="black",
                ms=7, label="Data", zorder=5)
    ax.set_ylim(0, 1.25)
    ax.axhline(1, color="0.6", lw=1, ls=":")

    with np.errstate(invalid="ignore", divide="ignore"):
        r = ed[0] / em[0]
        r_lo, r_hi = ed[1] / em[0], ed[2] / em[0]
        band = np.where(em[0] > 0, sym(em) / em[0], np.nan)
    rax.stairs(1 + band, edges, baseline=1 - band, fill=True, color=C_MC, alpha=0.3, lw=0, label="MC stat.")
    rax.errorbar(c[ok], r[ok], yerr=[r_lo[ok], r_hi[ok]], xerr=xerr[ok], fmt="o", color="black", ms=6)
    rax.axhline(1, color="0.4", lw=1, ls="--")
    rax.set_ylim(0.7, 1.3)
    rax.set_ylabel("Data / MC", fontsize=18)
    for a in (ax, rax):
        a.axvspan(edges[0], SR_PT_BINS[0], color="0.92", lw=0, zorder=0)
    ax.text(SR_PT_BINS[0] - 8, 1.03, "below SR", ha="right", va="bottom", fontsize=14, color="0.45")


def ratio_axes(plt, ncols=1, width=10, height=10, sharey=True):
    fig, axs = plt.subplots(
        2, ncols, figsize=(width * ncols, height), sharex="col", sharey="row" if sharey else False,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05, "wspace": 0.06},
        squeeze=False,
    )
    return fig, axs


def plot_region(cr, plt, hep, plotdir):
    """Probe-jet pT / Xbb in the denominator: data vs MC stack (shape)."""
    d, m = cr.is_data & cr.pass_like, ~cr.is_data & cr.pass_like
    norm = d.sum() / cr.w[m].sum()
    groups = cr.df.group.to_numpy()
    for var, values, edges, xlabel, logy in [
        ("pt", cr.pt, TURNON_PT_BINS[:-1], r"Probe AK8 jet $p_T$ [GeV]", True),
        ("xbb", cr.xbb, np.linspace(0, 1, 21), "Probe AK8 jet PNet Xbb", False),
        ("msd", cr.df.FatJet0_msd.to_numpy(), np.linspace(*MSD_WINDOW, 24), r"Probe AK8 jet $m_{SD}$ [GeV]", False),
    ]:
        fig, axs = ratio_axes(plt)
        ax, rax = axs[0, 0], axs[1, 0]
        width = np.diff(edges)
        stack = [np.histogram(values[m & (groups == g)], edges, weights=cr.w[m & (groups == g)] * norm)[0] / width
                 for g in MC_GROUPS]
        hep.histplot(stack, edges, stack=True, histtype="fill", ax=ax, color=STACK_COLORS,
                     label=[lab for lab, _ in MC_GROUPS.values()], edgecolor="white", lw=1)
        hd = np.histogram(values[d], edges)[0]
        hep.histplot(hd / width, edges, yerr=np.sqrt(hd) / width, histtype="errorbar", color="black",
                     ax=ax, label="Data", markersize=7)
        tot = np.sum(stack, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rax.errorbar(0.5 * (np.asarray(edges[1:]) + edges[:-1]), hd / width / tot,
                         yerr=np.sqrt(hd) / width / tot, fmt="o", color="black", ms=6)
        rax.axhline(1, color="0.4", lw=1, ls="--")
        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel("Data / MC", fontsize=18)
        rax.set_xlabel(xlabel)
        ax.set_ylabel("Events / bin width" if var == "xbb" else "Events / GeV")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(top=ax.get_ylim()[1] * 300)
        h, lab = ax.get_legend_handles_labels()
        ax.legend(h[::-1], lab[::-1], fontsize=15, ncol=2, loc="upper right")
        caption(fig, PROBE_CAPTION.replace(", P(QCD)", "\nP(QCD)") + f", MC scaled to data ($\\times${norm:.2f})")
        cms_label(hep, ax)
        savefig(fig, plotdir, f"region_{var}")
        plt.close(fig)


def plot_turnons(cr, plt, hep, plotdir):
    nb = len(TURNON_PT_BINS) - 1
    idx = bin_index(cr.pt, TURNON_PT_BINS)

    # trigger OR + individual paths
    passes = {"OR of AK8 jet triggers": cr.OR, **{t: cr.paths[t] for t in TRIGGERS}}
    fig, axs = ratio_axes(plt, ncols=len(passes), width=9)
    for j, (name, passed) in enumerate(passes.items()):
        ed, em = cr.eff(idx, nb, passed=passed)
        draw_eff_panel(axs[0, j], axs[1, j], TURNON_PT_BINS, ed, em, label_mc="MC")
        axs[0, j].text(0.96, 0.30, name.replace("_", " ").replace("SoftDrop", "\nSoftDrop"), ha="right",
                       transform=axs[0, j].transAxes, fontsize=16, va="top")
        axs[1, j].set_xlabel(r"Probe AK8 jet $p_T$ [GeV]")
        axs[1, j].set_xlim(TURNON_PT_BINS[0], 1000)
        axs[1, j].set_xticks([200, 400, 600, 800])
    axs[0, 0].set_ylabel("Trigger efficiency")
    axs[0, 0].legend(loc="center right", fontsize=16)
    caption(fig, PROBE_CAPTION)
    cms_label(hep, axs[0, 0])
    savefig(fig, plotdir, "turnon_pt_paths")
    plt.close(fig)

    # OR vs pT in Xbb bins (small multiples)
    nx = len(SF_XBB_BINS) - 1
    fig, axs = ratio_axes(plt, ncols=nx, width=8)
    for j, (lo, hi) in enumerate(zip(SF_XBB_BINS[:-1], SF_XBB_BINS[1:])):
        sel = (cr.xbb >= lo) & (cr.xbb < hi + (1e-6 if hi == 1 else 0))
        ed, em = cr.eff(idx, nb, extra=sel)
        draw_eff_panel(axs[0, j], axs[1, j], TURNON_PT_BINS, ed, em, label_mc="MC")
        axs[1, j].set_xlabel(r"Probe $p_T$ [GeV]")
        axs[1, j].set_xlim(TURNON_PT_BINS[0], 1000)
        axs[1, j].set_xticks([200, 400, 600, 800])
        axs[0, j].text(0.96, 0.06, xbb_label(lo, hi), ha="right", transform=axs[0, j].transAxes, fontsize=18)
    axs[0, 0].set_ylabel("Trigger efficiency (OR)")
    axs[0, 0].legend(loc="center right", fontsize=16)
    caption(fig, PROBE_CAPTION)
    cms_label(hep, axs[0, 0], fontsize=18)
    savefig(fig, plotdir, "turnon_pt_xbb")
    plt.close(fig)


def _diverging_cmap():
    # two hues, neutral gray midpoint at SF = 1
    return LinearSegmentedColormap.from_list("sf", ["#1f4fb4", "#5790fc", "#ececec", "#f89c20", "#c4321e"])


def _annotated_map(ax, values, text, cmap, vmin, vmax, xedges, yedges):
    X, Y = np.meshgrid(xedges, yedges, indexing="ij")
    mesh = ax.pcolormesh(X, Y, values, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=2)
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            ax.text(0.5 * (xedges[i] + xedges[i + 1]), 0.5 * (yedges[j] + yedges[j + 1]), text[i][j],
                    ha="center", va="center", fontsize=14, color="black")
    return mesh


def plot_maps(res, plt, hep, plotdir):
    # x axis drawn in "bin index" space for the wide last pt bin, labeled with edges
    xe = np.arange(len(SF_PT_BINS))
    ye = np.asarray(SF_XBB_BINS)

    def style(ax):
        ax.set_xticks(xe, [f"{e:g}" for e in SF_PT_BINS])
        ax.set_xlabel(r"AK8 jet $p_T$ [GeV]")
        ax.set_ylabel("AK8 jet PNet Xbb")
        ax.set_yticks(ye)

    # SF map
    fig, ax = plt.subplots(figsize=(13, 10))
    txt = [[f"{res['sf'][i, j]:.3f}\n± {res['total'][i, j]:.3f}"
            for j in range(len(ye) - 1)] for i in range(len(xe) - 1)]
    mesh = _annotated_map(ax, res["sf"], txt, _diverging_cmap(), 0.8, 1.2, xe, ye)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("Data / MC scale factor")
    style(ax)
    caption(fig, PROBE_CAPTION)
    cms_label(hep, ax, fontsize=18)
    savefig(fig, plotdir, "sf_map")
    plt.close(fig)

    # efficiency maps, data and MC
    fig, axs = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={"wspace": 0.25})
    for ax, key, title in [(axs[0], "eff_data", "Data"), (axs[1], "eff_mc", "MC")]:
        err = res[f"{key}_err"]
        txt = [[f"{res[key][i, j]:.3f}\n±{err[i, j]:.3f}" for j in range(len(ye) - 1)] for i in range(len(xe) - 1)]
        mesh = _annotated_map(ax, res[key], txt, "Blues", 0, 1.4, xe, ye)
        style(ax)
        ax.text(0.03, 0.97, title, transform=ax.transAxes, fontsize=22, va="top", weight="bold")
        cms_label(hep, ax, fontsize=18)
    cb = fig.colorbar(mesh, ax=axs, pad=0.02)
    cb.set_label("Trigger efficiency (OR)")
    caption(fig, PROBE_CAPTION)
    savefig(fig, plotdir, "eff_maps")
    plt.close(fig)


def signal_impact(sig, res):
    pt, msd = sig.FatJet0_pt.to_numpy(), sig.FatJet0_msd.to_numpy()
    sr = (
        (msd > MSD_WINDOW[0]) & (msd < MSD_WINDOW[1]) & (pt > SR_PT_BINS[0]) & (pt < SR_PT_BINS[-1])
        & (np.nan_to_num(sig.FatJet0_ParTPQCD.to_numpy(), nan=1) < PQCD_PASS)
        & (sig.FatJet0_nSV.to_numpy() > NSV_MIN)
        & np.isfinite(sig.FatJet0_pnetTXbb.to_numpy())
    )
    s = sig[sr]
    w = s.weight.to_numpy()
    pt, xbb = s.FatJet0_pt.to_numpy(), s.FatJet0_pnetTXbb.to_numpy()
    sf = sf_lookup(res, pt, xbb)
    unc = sf_lookup(res, pt, xbb, "total")
    # fraction of signal per SF cell
    idx = bin_index_2d(np.clip(pt, None, SF_PT_BINS[-1] - 1e-3), SF_PT_BINS, xbb, SF_XBB_BINS)
    frac = np.bincount(idx[idx >= 0], weights=w[idx >= 0], minlength=res["sf"].size) / w.sum()
    rows = []
    for lo, hi in [*zip(SR_PT_BINS[:-1], SR_PT_BINS[1:]), (SR_PT_BINS[0], SR_PT_BINS[-1])]:
        b = (pt >= lo) & (pt < hi)
        # SF uncertainties treated as fully correlated across cells (conservative)
        rows.append({
            "pt_lo": lo, "pt_hi": hi, "frac": w[b].sum() / w.sum(),
            "sf": np.average(sf[b], weights=w[b]), "unc": np.average(unc[b], weights=w[b]),
        })
    return frac.reshape(res["sf"].shape), rows


def plot_signal(frac, rows, plt, hep, plotdir):
    fig, axs = plt.subplots(1, 2, figsize=(24, 10), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.25})
    xe = np.arange(len(SF_PT_BINS))
    ye = np.asarray(SF_XBB_BINS)
    txt = [[f"{100 * frac[i, j]:.1f}%" for j in range(len(ye) - 1)] for i in range(len(xe) - 1)]
    mesh = _annotated_map(axs[0], 100 * frac, txt, "Purples", 0, 100 * frac.max() * 1.6, xe, ye)
    fig.colorbar(mesh, ax=axs[0], pad=0.02).set_label("Share of SR signal yield [%]")
    axs[0].set_xticks(xe, [f"{e:g}" for e in SF_PT_BINS])
    axs[0].set_xlabel(r"Candidate AK8 jet $p_T$ [GeV]")
    axs[0].set_ylabel("Candidate AK8 jet PNet Xbb")
    axs[0].set_yticks(ye)
    caption(fig, rf"Signal, SR pass region: P(QCD) < {PQCD_PASS:g}, nSV > {NSV_MIN}, "
                 rf"{MSD_WINDOW[0]:g} < $m_{{SD}}$ < {MSD_WINDOW[1]:g}")
    cms_label(hep, axs[0], data=False, fontsize=18)

    ax = axs[1]
    per_bin = rows[:-1]
    edges = np.array(SR_PT_BINS, float)
    x = np.arange(len(per_bin))
    ax.errorbar(x, [r["sf"] for r in per_bin], yerr=[r["unc"] for r in per_bin], fmt="o", color="black", ms=9,
                capsize=6, label="Signal-weighted SF ± total unc.")
    for xi, r in zip(x, per_bin):
        ax.text(xi, r["sf"] - r["unc"] - 0.006, f"{r['sf']:.3f}\n± {r['unc']:.3f}", ha="center", va="top", fontsize=14)
    incl = rows[-1]
    ax.axhspan(incl["sf"] - incl["unc"], incl["sf"] + incl["unc"], color=C_MC, alpha=0.2, lw=0)
    ax.axhline(incl["sf"], color=C_MC, lw=2, ls="--", label=f"All SR: {incl['sf']:.3f} ± {incl['unc']:.3f}")
    ax.axhline(1, color="0.5", lw=1, ls=":")
    ax.set_xticks(x, [f"{edges[i]:g}-{edges[i + 1]:g}\n({100 * r['frac']:.0f}%)" for i, r in enumerate(per_bin)])
    ax.set_xlim(-0.5, len(per_bin) - 0.5)
    ax.set_ylim(0.85, 1.2)
    ax.set_xlabel(r"SR $p_T$ bin [GeV] (share of signal)")
    ax.set_ylabel("Trigger SF applied to signal")
    ax.legend(fontsize=16, loc="upper right")
    cms_label(hep, ax, data=False, fontsize=18)
    savefig(fig, plotdir, "signal_impact")
    plt.close(fig)


def write_summary(path, res, rows, cr):
    nd = int(cr.is_data.sum())
    lines = [
        f"# Trigger efficiency / SF summary ({YEAR})",
        "",
        f"- Trigger OR: {', '.join(f'`{t}`' for t in TRIGGERS)}; reference: {', '.join(f'`{t}`' for t in REF_TRIGGERS)}",
        (
            f"- Denominator: control-tt, {MSD_WINDOW[0]:g} <= probe msd < {MSD_WINDOW[1]:g}, "
            f"no leptonic-side AK8 jet (LepSideFatJet_pt < {LEPSIDE_MAX:g}); {nd} data events (all probe pT)."
        ),
        f"- Nominal probes: P(QCD) < {PQCD_PASS:g} (SR pass region).",
        (
            "- SF uncertainty: stat (data Clopper-Pearson, MC effective entries) (+) syst, "
            "syst = |SF - SF(no P(QCD) cut)|."
        ),
        "",
        "## SF per (pT, Xbb) bin",
        "",
        (
            "| pT [GeV] | Xbb | N data | eff data | eff MC | SF | total | stat | syst "
            "| SF (no P(QCD) cut) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, (plo, phi) in enumerate(zip(SF_PT_BINS[:-1], SF_PT_BINS[1:])):
        for j, (xlo, xhi) in enumerate(zip(SF_XBB_BINS[:-1], SF_XBB_BINS[1:])):
            lines.append(
                f"| {plo}-{phi} | {xlo:g}-{xhi:g} | {res['n_data'][i, j]:.0f} | {res['eff_data'][i, j]:.3f} "
                f"| {res['eff_mc'][i, j]:.3f} | {res['sf'][i, j]:.3f} | {res['total'][i, j]:.3f} | {res['stat'][i, j]:.3f} "
                f"| {res['syst'][i, j]:.3f} "
                f"| {res['sf_allpq'][i, j]:.3f} |"
            )
    lines += ["", "## Signal-weighted SF (SR pass region)", "", "| SR pT bin | share of signal | SF | unc. |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['pt_lo']}-{r['pt_hi']} | {100 * r['frac']:.1f}% | {r['sf']:.3f} | {r['unc']:.3f} |")
    path.write_text("\n".join(lines) + "\n")


def analyze(args):
    plotdir = args.outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    cr = CR(pd.read_parquet(args.outdir / "cache" / "control_tt.parquet"))
    sig = pd.read_parquet(args.outdir / "cache" / "signal.parquet")
    print(f"denominator: {cr.is_data.sum()} data ({(cr.is_data & cr.pass_like).sum()} with P(QCD) < {PQCD_PASS}), "
          f"{(~cr.is_data).sum()} MC events")

    res = measure_sf(cr)
    write_correctionlib(res, args.outdir / f"trigger_sf_{YEAR}.json")
    frac, rows = signal_impact(sig, res)

    plt, hep = setup_mpl()
    plot_region(cr, plt, hep, plotdir)
    plot_turnons(cr, plt, hep, plotdir)
    plot_maps(res, plt, hep, plotdir)
    plot_signal(frac, rows, plt, hep, plotdir)
    write_summary(args.outdir / "summary.md", res, rows, cr)
    print((args.outdir / "summary.md").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["load", "analyze", "all"])
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--skim-dir", default=DEFAULT_SKIMS, help="<tag>/<year> skim dir (local path or root:// URL)")
    p.add_argument("--signal-dir", default=DEFAULT_SIGNAL, help="signal sample skim dir (local path or root:// URL)")
    args = p.parse_args()
    if args.step in ("load", "all"):
        load(args)
    if args.step in ("analyze", "all"):
        analyze(args)


if __name__ == "__main__":
    main()
