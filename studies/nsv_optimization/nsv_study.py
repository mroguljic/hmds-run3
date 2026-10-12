#!/usr/bin/env python3
"""
nSV definition and cut optimization, run directly on NanoAOD.

The skims only store the final count ``FatJet0_nSV`` (all NanoAOD SVs within dR < 0.8
of the jet), so per-SV quantities (ntracks, dlenSig) need NanoAOD.  Subcommands:

``dump``
    NanoAOD -> reduced parquet in ``<outdir>/dump/``: event-level ``signal-all`` cuts
    (trigger OR, MET filters, jet veto map, lepton veto, >= 1 good AK8 jet passing
    minjetkin with P(QCD) < 0.6); every good AK8 jet with its SVs (dR < 0.8) and the
    leading 4 good AK4 jets; per-file genEventSumw.  Already-dumped files are skipped.

``analyze``
    For each SV definition on the (min. ntracks x min. dlenSig) grid: recount nSV, re-pick
    the candidate jet like categorizer.py (highest nSV, tiebreak pT), apply minjetkin and the
    opposite-hemisphere anti-b-tag, fill (nSV, P(QCD), pT bin, msd bin) histograms.  Writes
    to ``<outdir>/plots/``: sv_cuts, sv_definition, joint_map, nsv_threshold, pqcd_cut,
    passfail (.png/.pdf) and summary.txt.

``all``
    dump, then analyze.

Approximations w.r.t. the processor: no JEC/JER re-application, uncorrected muon pT
(closure for the all-SVs definition: nSV > 6 fraction 0.330 here vs 0.327 in the skim).
Background NanoAOD is read over xrootd -> needs a valid grid proxy.

Example (from the repo root):
    python studies/nsv_optimization/nsv_study.py all --outdir studies/nsv_optimization/output/261001
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import awkward as ak
import hbb
import numpy as np
import uproot

from hbb.common_vars import LUMI
from hbb.corrections import correct_jetid, get_jetveto_event
from hbb.taggers import b_taggers
from hbb.xsecs import xsecs

warnings.filterwarnings("ignore", message="Missing cross-reference")

YEAR = "2024"
REPO = Path(__file__).resolve().parents[2]
HBB_DIR = Path(hbb.__file__).parent
SIGNAL_FILE = "/eos/uscms/store/user/roguljic/HMDS/signal/Merged.root"
TRIGGERS = json.loads((HBB_DIR / "triggers.json").read_text())[YEAR]
MET_FILTERS = json.loads((HBB_DIR / "metfilters.json").read_text())[YEAR]["mc"]
BTAG_CUT = b_taggers[YEAR]["AK4"]["btagUParTAK4B"]["M"]

# QCD HT bins below 400 contribute negligibly at AK8 pT > 300
DEFAULT_SUBSAMPLES = {
    "QCD": [
        "QCD_HT-400to600",
        "QCD_HT-600to800",
        "QCD_HT-800to1000",
        "QCD_HT-1000to1200",
        "QCD_HT-1200to1500",
        "QCD_HT-1500to2000",
        "QCD_HT-2000",
    ],
    "TT": ["TTto4Q", "TTtoLNu2Q"],
}

SV_FIELDS = [
    "pt", "eta", "phi", "mass", "ntracks", "chi2", "ndof", "charge",
    "dlen", "dlenSig", "dxy", "dxySig", "pAngle",
]
JETID_FIELDS = ["chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "chMultiplicity", "neMultiplicity"]

# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------


def _collection(arrays, name, fields):
    return ak.zip({f: arrays[f"{name}_{f}"] for f in fields}, depth_limit=None)


def _branches(t, is_mc):
    br = [f"HLT_{h}" for h in TRIGGERS if f"HLT_{h}" in t.keys()]
    br += [f"Flag_{f}" for f in MET_FILTERS if f"Flag_{f}" in t.keys()]
    br += ["event"] + (["genWeight"] if is_mc else [])
    br += [f"FatJet_{f}" for f in ["pt", "eta", "phi", "msoftdrop", "globalParT3_QCD", *JETID_FIELDS]]
    br += [f"Jet_{f}" for f in ["pt", "eta", "phi", "btagUParTAK4B", *JETID_FIELDS]]
    br += ["Muon_pt", "Muon_eta", "Muon_mediumId", "Muon_pfIsoId"]
    br += ["Electron_pt", "Electron_eta", "Electron_mvaIso_WP90"]
    br += [f"SV_{f}" for f in SV_FIELDS]
    return br


def select_chunk(a, is_mc):
    """Event-level preselection + per-jet SV association for one chunk."""
    trig = np.zeros(len(a), dtype=bool)
    for h in TRIGGERS:
        if f"HLT_{h}" in a.fields:
            trig = trig | ak.to_numpy(a[f"HLT_{h}"])
    for flag in MET_FILTERS:
        if f"Flag_{flag}" in a.fields:
            trig = trig & ak.to_numpy(a[f"Flag_{flag}"])

    nmu = ak.sum((a.Muon_pt > 10) & (abs(a.Muon_eta) < 2.4) & a.Muon_mediumId & (a.Muon_pfIsoId >= 3), axis=1)
    nel = ak.sum((a.Electron_pt > 10) & (abs(a.Electron_eta) < 2.5) & a.Electron_mvaIso_WP90, axis=1)

    fj = _collection(a, "FatJet", ["pt", "eta", "phi", "msoftdrop", "globalParT3_QCD", *JETID_FIELDS])
    fj = correct_jetid(fj, "AK8", YEAR)
    fj = fj[fj.jetidtight & (fj.pt > 200) & (abs(fj.eta) < 2.5)]

    jets = _collection(a, "Jet", ["pt", "eta", "phi", "btagUParTAK4B", *JETID_FIELDS])
    jets = correct_jetid(jets, "AK4", YEAR)
    jetveto = ak.to_numpy(get_jetveto_event(jets, YEAR))
    jets = jets[jets.jetidtight & jets.jetidtightlepveto & (jets.pt > 30) & (abs(jets.eta) < 2.5)][:, :4]

    kin = (fj.pt >= 300) & (fj.pt < 1200) & (fj.msoftdrop >= 40) & (fj.msoftdrop < 201)
    any_cand = ak.any(kin & (fj.globalParT3_QCD < 0.6), axis=1)
    sel = trig & jetveto & ak.to_numpy((nmu == 0) & (nel == 0) & any_cand)
    if not np.any(sel):
        return None

    a, fj, jets = a[sel], fj[sel], jets[sel]
    sv = _collection(a, "SV", SV_FIELDS)
    pairs = ak.cartesian({"fj": fj, "sv": sv}, axis=1, nested=True)
    deta = pairs.fj.eta - pairs.sv.eta
    dphi = (pairs.fj.phi - pairs.sv.phi + np.pi) % (2 * np.pi) - np.pi
    dr = np.sqrt(deta**2 + dphi**2)
    in_jet = dr < 0.8
    fj_svs = ak.zip({f: pairs.sv[f][in_jet] for f in SV_FIELDS} | {"dR": dr[in_jet]}, depth_limit=None)

    out = {
        "event": a.event,
        "weight": a.genWeight if is_mc else ak.ones_like(a.event, dtype=np.float32),
        "fj": ak.zip(
            {
                "pt": fj.pt,
                "eta": fj.eta,
                "phi": fj.phi,
                "msd": fj.msoftdrop,
                "pqcd": fj.globalParT3_QCD,
                "sv": fj_svs,
            },
            depth_limit=2,
        ),
        "jet": ak.zip({"pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "btag": jets.btagUParTAK4B}),
    }
    return ak.zip(out, depth_limit=1)


def dump_file(fname, outfile, is_mc, step_size):
    with uproot.open(fname, timeout=180) as f:
        sumw = float(np.sum(f["Runs"]["genEventSumw"].array(library="np"))) if is_mc else 0.0
        t = f["Events"]
        nevents = t.num_entries
        chunks = []
        for a in t.iterate(_branches(t, is_mc), step_size=step_size):
            c = select_chunk(a, is_mc)
            if c is not None:
                chunks.append(c)
    npass = 0
    if chunks:
        arr = ak.concatenate(chunks)
        npass = len(arr)
        ak.to_parquet(arr, outfile)
    Path(outfile).with_suffix(".json").write_text(
        json.dumps({"file": fname, "sumw": sumw, "nevents": nevents, "npass": npass})
    )
    return fname, nevents, npass


def run_dump(args):
    index = json.loads((REPO / "data" / "nanoindex_v15.json").read_text())[YEAR]
    jobs = []
    for sample in args.samples:
        if sample == "Signal":
            jobs.append(("Signal", 0, SIGNAL_FILE))
            continue
        for sub in args.subsamples or DEFAULT_SUBSAMPLES[sample]:
            files = list(index[sample][sub])
            random.Random(args.seed).shuffle(files)
            jobs += [(sub, i, f) for i, f in enumerate(files[: args.nfiles])]

    todo = []
    for sub, i, fname in jobs:
        outfile = Path(args.outdir) / "dump" / sub / f"{i}.parquet"
        if outfile.with_suffix(".json").exists() and not args.overwrite:
            continue
        outfile.parent.mkdir(parents=True, exist_ok=True)
        todo.append((fname, str(outfile), sub != "Signal"))
    print(f"{len(jobs)} files requested, {len(todo)} to process")

    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(dump_file, f, o, mc, args.step_size): f for f, o, mc in todo}
        for fut in as_completed(futures):
            try:
                fname, n, npass = fut.result()
                print(f"  done {fname.rsplit('/', 3)[-3]}/.../{fname.rsplit('/', 1)[-1]}: {npass}/{n}")
            except Exception as e:  # noqa: BLE001 -- keep going on xrootd hiccups, rerun picks them up
                failed += 1
                print(f"  FAILED {futures[fut]}: {str(e)[:200]}")
    if failed:
        print(f"{failed} files failed; rerun the same command to retry them (normalization only uses read files)")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

NSV_EDGES = np.arange(-0.5, 16.5)  # last bin collects nSV >= 15
SV_PLOT_BINS = {"ntracks": np.arange(1.5, 12.5, 1), "dlenSig": np.logspace(0, 3, 31)}
NO_CUT = "all SVs"


def def_name(ntrk, dlensig):
    """Name of an SV definition (ntrk = min. ntracks, written as ntrk > ntrk-1); NanoAOD SVs have
    >= 2 tracks and dlenSig > 3, so ntrk 2 / dlenSig 0 = no cut."""
    parts = []
    if ntrk > 2:
        parts.append(f"ntrk>{ntrk - 1}")
    if dlensig > 0:
        parts.append(f"dlenSig>{dlensig:g}")
    return "&".join(parts) or NO_CUT


def load_sample(outdir, sub):
    d = Path(outdir) / "dump" / sub
    meta = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None, meta
    return ak.concatenate([ak.from_parquet(f) for f in files]), meta


def candidate(events, ntrk, dlensig, pt_range, msd_range):
    """Recount nSV with the SV definition, pick the candidate like categorizer.py (highest
    nSV, tiebreak pT), apply minjetkin + opposite-hemisphere anti-btag."""
    fj = events.fj
    nsv = ak.sum((fj.sv.ntracks >= ntrk) & (fj.sv.dlenSig > dlensig), axis=2)
    order = ak.argsort(ak.values_astype(nsv, np.float64) * 1e5 + fj.pt, axis=1, ascending=False)
    cand = ak.firsts(fj[order])
    cand_nsv = ak.firsts(nsv[order])

    dphi = abs((events.jet.phi - cand.phi + np.pi) % (2 * np.pi) - np.pi)
    antib = ak.fill_none(ak.max(events.jet.btag[dphi > np.pi / 2], axis=1) < BTAG_CUT, True)
    ok = (
        (cand.pt >= pt_range[0]) & (cand.pt < pt_range[1])
        & (cand.msd >= msd_range[0]) & (cand.msd < msd_range[1])
        & (abs(cand.eta) < 2.5) & antib
    )

    def np_(x, fill):
        return ak.to_numpy(ak.fill_none(x, fill))

    return np_(cand_nsv, -1), np_(cand.pqcd, 2.0), np_(cand.pt, -1), np_(cand.msd, -1), np_(ok, False), cand


class Hists:
    """(nSV, P(QCD), pT, msd) histograms per (definition, group): sumw, sumw2, raw counts."""

    def __init__(self, pqcd_edges, pt_edges, msd_edges):
        self.edges = (NSV_EDGES, np.asarray(pqcd_edges), np.asarray(pt_edges), np.asarray(msd_edges))
        self.h = {}

    def fill(self, dname, group, nsv, pqcd, pt, msd, w):
        sample = np.column_stack([np.minimum(nsv, 15), pqcd, pt, msd])
        out = self.h.setdefault(
            (dname, group), {k: np.zeros([len(e) - 1 for e in self.edges]) for k in ("w", "w2", "n")}
        )
        for key, ww in (("w", w), ("w2", w**2), ("n", None)):
            out[key] += np.histogramdd(sample, bins=self.edges, weights=ww)[0]

    def pqcd_index(self, cut):
        idx = np.flatnonzero(np.isclose(self.edges[1], cut))
        if len(idx) != 1:
            raise ValueError(f"P(QCD) cut {cut} is not a histogram edge")
        return int(idx[0])

    def get(self, dname, group, key="w", k=None, pqcd_lo=None, pqcd_hi=None, per_pt=False, per_msd=False):
        """Sum over nSV > k and pqcd_lo <= P(QCD) < pqcd_hi; pT / msd collapsed unless requested."""
        if (dname, group) not in self.h:
            return 0.0
        a = self.h[(dname, group)][key][(k + 1 if k is not None else 0):]
        lo = self.pqcd_index(pqcd_lo) if pqcd_lo is not None else 0
        hi = self.pqcd_index(pqcd_hi) if pqcd_hi is not None else a.shape[1]
        a = a[:, lo:hi].sum(axis=(0, 1))
        if not per_msd:
            a = a.sum(axis=-1)
        if not per_pt:
            a = a.sum(axis=0)
        return a


def run_analyze(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    plt.style.use(hep.style.CMS)
    plotdir = Path(args.outdir) / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)

    def cms_label(ax):
        hep.cms.label("WiP", data=False, year=YEAR, lumi=round(LUMI[YEAR] / 1000), ax=ax, fontsize=18)

    def save(fig, name):
        fig.savefig(plotdir / f"{name}.png", bbox_inches="tight")
        fig.savefig(plotdir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    def textbox(ax, text, loc="upper right"):
        at = matplotlib.offsetbox.AnchoredText(text, loc=loc, prop={"fontsize": 15}, frameon=True)
        at.patch.set_alpha(0.8)
        ax.add_artist(at)

    def mark(ax, x):
        ax.axvline(x, color="k", ls=":", lw=2, label="chosen")

    # ---- configuration ----------------------------------------------------
    setup = json.loads(Path(args.fit_setup).read_text())
    pt_edges = setup["categories"]["sr"]["bins"]
    obs = setup["observable"]
    msd_edges = np.linspace(obs["min"], obs["max"], obs["nbins"] + 1)

    cd, ck, cc = args.chosen.rsplit(":", 2)
    ck, cc = int(ck), float(cc)
    params = {def_name(nt, ds): (nt, ds) for nt in args.ntrk_grid for ds in args.dlensig_grid}
    if cd not in params:
        raise SystemExit(f"chosen definition '{cd}' not on the grid: {list(params)}")
    c_nt, c_ds = params[cd]

    pqcd_grid = sorted(set(args.pqcd_grid) | {cc})
    pqcd_fine = np.round(np.arange(0.0125, 0.3001, 0.0125), 4)  # for the 1D P(QCD) scan
    pqcd_coarse = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0 + 1e-6]  # for its distribution
    pqcd_edges = np.array(sorted({*pqcd_grid, *pqcd_fine, *pqcd_coarse, args.pqcd_fail_max}))

    # ---- fill histograms --------------------------------------------------
    groups = {"Signal": ["Signal"], "QCD": DEFAULT_SUBSAMPLES["QCD"], "TT": DEFAULT_SUBSAMPLES["TT"]}
    hists = Hists(pqcd_edges, pt_edges, msd_edges)
    sv_vals = {}  # var -> group -> [(values, weights)], SVs in the candidate jet before SV cuts
    for group, subs in groups.items():
        for sub in subs:
            ev, meta = load_sample(args.outdir, sub)
            if ev is None:
                continue
            if group == "Signal":
                w = np.ones(len(ev))  # efficiencies only
            else:
                w = ak.to_numpy(ev.weight) * xsecs[sub] * LUMI[YEAR] / sum(m["sumw"] for m in meta)
            print(f"{sub}: {len(ev)} events from {len(meta)} files")
            for dname, (nt, ds) in params.items():
                nsv, pqcd, pt, msd, ok, cand = candidate(ev, nt, ds, (pt_edges[0], pt_edges[-1]),
                                                         (msd_edges[0], msd_edges[-1]))
                hists.fill(dname, group, nsv[ok], pqcd[ok], pt[ok], msd[ok], w[ok])
                if dname == NO_CUT:
                    sel = ok & (pqcd < args.pqcd_fail_max)
                    sv = cand.sv[sel]
                    wsv = ak.to_numpy(ak.flatten(ak.broadcast_arrays(w[sel], sv.pt)[0]))
                    for var in SV_PLOT_BINS:
                        sv_vals.setdefault(var, {}).setdefault(group, []).append((ak.to_numpy(ak.flatten(sv[var])), wsv))
            del ev
    bkg = [g for g in ("QCD", "TT") if any(key[1] == g for key in hists.h)]
    if not any(key[1] == "Signal" for key in hists.h):
        raise SystemExit("no signal dump found")

    # ---- helpers on the histograms ----------------------------------------
    def eff(d, g, k=None, cut=None):
        """Efficiency of nSV > k and P(QCD) < cut w.r.t. the preselection; g = group or 'B' (QCD + tt)."""
        gs = bkg if g == "B" else [g]
        den = sum(hists.get(d, x) for x in gs)
        return sum(hists.get(d, x, k=k, pqcd_hi=cut) for x in gs) / den if den > 0 else 0.0

    def bkg_yield(d, k, cut):
        return sum(hists.get(d, g, k=k, pqcd_hi=cut) for g in bkg)

    def raw_qcd(d, k, cut):
        return hists.get(d, "QCD", "n", k=k, pqcd_hi=cut)

    def min_bkg_pt(d, k, cut):
        """Pass-region background yield of the least populated fit pT bin."""
        return float(np.min(sum(hists.get(d, g, k=k, pqcd_hi=cut, per_pt=True) for g in bkg)))

    def fom(d, k, cut):
        b = bkg_yield(d, k, cut)
        return eff(d, "Signal", k, cut) / np.sqrt(b) if b > 0 else np.nan

    nmin = args.min_bkg_per_ptbin
    excl_label = f"< {nmin:g} bkg. events in a $p_T$ bin"
    lines = [
        f"Chosen working point: SV definition '{cd}', nSV > {ck}, P(QCD) < {cc:g}",
        "Efficiencies w.r.t. the preselection (trigger, lepton veto, candidate minjetkin + anti-b-tag);",
        f"B = QCD + tt MC yield for {round(LUMI[YEAR] / 1000)} fb^-1 ({YEAR}); requirement: >= {nmin:g} background",
        "events in the pass region of every fit pT bin.",
    ]

    # ---- sv_cuts: per-SV ntracks and dlenSig with the chosen cuts ------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    lines.append(f"\n=== Fraction of candidate-jet SVs removed by the SV requirements (P(QCD) < {args.pqcd_fail_max:g}) ===")
    for ax, var, cutval, cutlabel in ((axes[0], "ntracks", c_nt - 0.5, f"ntracks > {c_nt - 1}"),
                                      (axes[1], "dlenSig", c_ds, f"dlenSig > {c_ds:g}")):
        bins = SV_PLOT_BINS[var]
        for group, parts in sv_vals[var].items():
            x = np.concatenate([p[0] for p in parts])
            w = np.concatenate([p[1] for p in parts])
            removed = w[x < cutval].sum() / w.sum()
            h, _ = np.histogram(np.clip(x, bins[0], bins[-1]), bins=bins, weights=w)
            label = "$t\\bar{t}$" if group == "TT" else group
            hep.histplot(h / h.sum(), bins, ax=ax, label=label)
            lines.append(f"{cutlabel:14s} {group:7s} {100 * removed:.0f}%")
        if var == "dlenSig":
            ax.set_xscale("log")
        ax.axvline(cutval, color="k", ls=":", lw=2, label=f"cut: {cutlabel}")
        ax.set_xlabel("SV track multiplicity" if var == "ntracks" else "SV 3D decay-length significance")
        ax.set_ylabel("Fraction of SVs")
        ax.legend(fontsize=13)
        cms_label(ax)
    save(fig, "sv_cuts")

    # ---- sv_definition: ROC curves for SV definitions ------------------------
    kk = np.arange(0, 15)
    fig, ax = plt.subplots(figsize=(11, 9))
    for i, d in enumerate(args.roc_defs):
        if d not in params:
            continue
        es = np.array([eff(d, "Signal", k, cc) for k in kk])
        eb = np.array([eff(d, "QCD", k, cc) for k in kk])
        good = (eb > 0) & (np.array([raw_qcd(d, k, cc) for k in kk]) >= args.min_mc)
        ax.plot(es[good], eb[good], "o-" if d == cd else "o--", ms=4, lw=3 if d == cd else 1.5, color=plt.cm.tab10(i % 10),
                label=d.replace(">", " > ").replace("&", ", "))
        if d == cd:
            for k in kk[good]:
                if 2 <= k <= 7:
                    ax.annotate(f"nSV > {k}", (es[k], eb[k]), textcoords="offset points", xytext=(8, -14), fontsize=12)
    ax.plot([eff(cd, "Signal", ck, cc)], [eff(cd, "QCD", ck, cc)], "r*", ms=22, label=f"chosen: nSV > {ck}")
    ax.set_yscale("log")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("QCD multijet efficiency")
    ax.legend(fontsize=12, loc="upper left", title="SV definition", title_fontsize=13)
    textbox(ax, f"P(QCD) < {cc:g}\npoints: nSV > k", loc="lower right")
    cms_label(ax)
    save(fig, "sv_definition")

    target = eff(cd, "Signal", ck, cc)
    lines.append(f"\n=== SV definitions at signal efficiency {target:.3f}, P(QCD) < {cc:g}: QCD only, as in sv_definition "
                 "(nSV threshold interpolated) ===")
    for d in params:
        es = np.array([eff(d, "Signal", k, cc) for k in kk])
        eb = np.array([eff(d, "QCD", k, cc) for k in kk])
        by = np.array([hists.get(d, "QCD", k=k, pqcd_hi=cc) for k in kk])
        m = eb > 0
        lines.append(
            f"{d:24s} eps_QCD {np.exp(np.interp(target, es[m][::-1], np.log(eb[m][::-1]))):.3e}  "
            f"QCD {np.exp(np.interp(target, es[m][::-1], np.log(by[m][::-1]))):9.0f}  "
            f"(nSV > {np.interp(target, es[m][::-1], kk[m][::-1]):.1f})"
        )

    # ---- joint_map: S/sqrt(B) over (nSV threshold, P(QCD) cut) ---------------
    kgrid = np.arange(args.kmin, args.kmax + 1)
    f = np.full((len(pqcd_grid), len(kgrid)), np.nan)
    cells = {}
    for i, c in enumerate(pqcd_grid):
        for j, k in enumerate(kgrid):
            cells[(i, j)] = (eff(cd, "Signal", k, c), bkg_yield(cd, k, c), min_bkg_pt(cd, k, c))
            if raw_qcd(cd, k, c) >= args.min_mc:
                f[i, j] = fom(cd, k, c)
    f /= np.nanmax(f)
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(f, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    best = (None, -1.0)
    for (i, j), (es_, by, mpt) in cells.items():
        valid = not np.isnan(f[i, j])
        def n(x):
            return f"{x:.0f}" if x < 1e5 else f"{x:.1e}"

        txt = (f"{f[i, j]:.2f}\n" if valid else "") + f"$\\epsilon_S$ {es_:.2f}\nB {n(by)}\n$B_{{p_T}}^{{min}}$ {n(mpt)}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="w" if valid else "gray")
        if mpt < nmin:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//", ec="0.75", lw=0))
        elif valid and f[i, j] > best[1]:
            best = ((kgrid[j], pqcd_grid[i]), f[i, j])
    ax.add_patch(plt.Rectangle((list(kgrid).index(ck) - 0.5, pqcd_grid.index(cc) - 0.5), 1, 1, fill=False, ec="r", lw=4))
    ax.set_xticks(range(len(kgrid)), [f">{k}" for k in kgrid])
    ax.set_yticks(range(len(pqcd_grid)), [f"<{c:g}" for c in pqcd_grid])
    ax.set_xlabel("nSV threshold")
    ax.set_ylabel("P(QCD) cut")
    fig.colorbar(im, ax=ax, label=r"S/$\sqrt{B}$ / max")
    cms_label(ax)
    save(fig, "joint_map")
    lines.append(f"\nBest allowed (nSV threshold, P(QCD) cut) on the grid: nSV > {best[0][0]}, P(QCD) < {best[0][1]:g}")

    # ---- nsv_threshold / pqcd_cut: 1D scans ----------------------------------
    def scan_figure(xs, values, xlabel, chosen_x, dist, dist_edges, dist_xlabel, cut_line, fixed, outname):
        """Distribution / efficiencies / S/sqrt(B) and least-populated pT bin vs one cut.  Returns best allowed x."""
        es, eb, by, raw, minpt = (np.array(v) for v in values)
        allowed = minpt >= nmin
        step = np.diff(xs).astype(float)
        half_lo = np.concatenate([[step[0]], step]) / 2
        half_hi = np.concatenate([step, [step[-1]]]) / 2

        def shade(ax):
            for x, a, lo, hi in zip(xs, allowed, half_lo, half_hi):
                if not a:
                    ax.axvspan(x - lo, x + hi, color="0.85", zorder=0, lw=0)
            return matplotlib.patches.Patch(color="0.85", label=f"excluded: {excl_label}")

        fig, axes = plt.subplots(1, 3, figsize=(30, 9))
        fig.subplots_adjust(wspace=0.35)
        ax = axes[0]
        widths = np.diff(dist_edges)
        for g, h in dist.items():
            hep.histplot(h / h.sum() / widths, dist_edges, ax=ax, label="$t\\bar{t}$" if g == "TT" else g)
        ax.axvline(cut_line, color="k", ls=":", lw=2, label="cut")
        ax.set_yscale("log")
        ax.set_xlabel(dist_xlabel)
        ax.set_ylabel("Normalized density")
        ax.legend(fontsize=13)
        textbox(ax, fixed, loc="lower left")
        cms_label(ax)

        ax = axes[1]
        patch = shade(ax)
        ax.plot(xs, es, "o-", label="Signal")
        ax.plot(xs, eb, "s-", label="Background (QCD + $t\\bar{t}$)")
        mark(ax, chosen_x)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Efficiency")
        h, l_ = ax.get_legend_handles_labels()
        ax.legend(h + [patch], l_ + [patch.get_label()], fontsize=12)
        cms_label(ax)

        ax = axes[2]
        patch = shade(ax)
        ok = (by > 0) & (raw >= args.min_mc)
        fv = np.where(ok, es / np.sqrt(np.where(by > 0, by, 1)), np.nan)
        fv = fv / np.nanmax(fv)
        ax.plot(xs[ok], fv[ok], "o-", color="k", label=r"S/$\sqrt{B}$ / max")
        mark(ax, chosen_x)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"S/$\sqrt{B}$ / max")
        ax2 = ax.twinx()
        m = minpt > 0
        ax2.plot(xs[m], minpt[m], "s--", color="tab:blue", label="bkg. in least populated $p_T$ bin")
        ax2.axhline(nmin, color="tab:blue", ls=":", lw=1.5)
        ax2.set_yscale("log")
        ax2.set_ylabel("Background in least populated $p_T$ bin", color="tab:blue")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2 + [patch], l1 + l2 + [patch.get_label()], fontsize=12, loc="lower center")
        cms_label(ax)
        save(fig, outname)
        fa = np.where(ok & allowed, fv, -1)
        return xs[int(np.argmax(fa))]

    def scan_values(pairs):
        return ([eff(cd, "Signal", k, c) for k, c in pairs], [eff(cd, "B", k, c) for k, c in pairs],
                [bkg_yield(cd, k, c) for k, c in pairs], [raw_qcd(cd, k, c) for k, c in pairs],
                [min_bkg_pt(cd, k, c) for k, c in pairs])

    kk = np.arange(0, 11)
    dist = {}
    for g in ["Signal", *bkg]:
        cum = np.array([hists.get(cd, g, k=k - 1, pqcd_hi=cc) for k in range(16)] + [0.0])
        dist[g] = cum[:-1] - cum[1:]
    best_k = scan_figure(kk, scan_values([(k, cc) for k in kk]), "nSV threshold k (nSV > k)", ck, dist, NSV_EDGES,
                         "Candidate-jet nSV", ck + 0.5, f"P(QCD) < {cc:g}", "nsv_threshold")

    cuts = pqcd_edges[(pqcd_edges > 0) & (pqcd_edges <= args.pqcd_fail_max + 1e-9)]
    idx = [hists.pqcd_index(c) for c in pqcd_coarse]
    dist = {g: np.add.reduceat(hists.h[(cd, g)]["w"][ck + 1:].sum(axis=(0, 2, 3)), idx[:-1]) for g in ["Signal", *bkg]}
    best_c = scan_figure(cuts, scan_values([(ck, c) for c in cuts]), "P(QCD) cut (P(QCD) < x)", cc, dist,
                         np.array(pqcd_coarse[:-1] + [1.0]), "Candidate-jet P(QCD)", cc, f"nSV > {ck}", "pqcd_cut")
    lines.append(f"Best allowed nSV threshold at P(QCD) < {cc:g}: nSV > {best_k}; "
                 f"best allowed P(QCD) cut at nSV > {ck}: {best_c:g}")

    # ---- passfail: msd in pass / fail per pT bin at the chosen point ---------
    centers = 0.5 * (msd_edges[1:] + msd_edges[:-1])
    npt = len(pt_edges) - 1
    fig, axes = plt.subplots(npt, 3, figsize=(22, 5 * npt), sharex=True)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    lines.append(f"\n=== Chosen working point: signal eff {eff(cd, 'Signal', ck, cc):.3f}, "
                 f"background eff {eff(cd, 'B', ck, cc):.2e}, B {bkg_yield(cd, ck, cc):.0f} ===")
    lines.append(f"fail: {cc:g} < P(QCD) < {args.pqcd_fail_max:g}, same nSV requirement")
    lines.append(f"{'pT bin':>10s} {'QCD pass':>10s} {'QCD fail':>10s} {'tt pass':>8s} {'pass/fail':>9s} "
                 f"{'rawQCD P':>8s} {'rawQCD F':>8s}")
    for ib in range(npt):
        per = {}
        for g in ["Signal", *bkg]:
            for cat, lo, hi in (("pass", None, cc), ("fail", cc, args.pqcd_fail_max)):
                for key in ("w", "w2", "n"):
                    per[(g, cat, key)] = hists.get(cd, g, key, k=ck, pqcd_lo=lo, pqcd_hi=hi, per_pt=True, per_msd=True)[ib]
        for col, cat in enumerate(("pass", "fail")):
            ax = axes[ib, col]
            stack = [per[(g, cat, "w")] for g in bkg]
            hep.histplot(stack, msd_edges, ax=ax, stack=True, histtype="fill", alpha=0.7,
                         label=["$t\\bar{t}$" if g == "TT" else g for g in bkg])
            tot = sum(stack)
            ax.errorbar(centers, tot, yerr=np.sqrt(sum(per[(g, cat, "w2")] for g in bkg)), fmt="none", color="k", lw=1)
            sig = per[("Signal", cat, "w")]
            if sig.max() > 0 and tot.max() > 0:
                hep.histplot(sig / sig.max() * tot.max(), msd_edges, ax=ax, color="r", label="Signal (shape)")
            ax.set_title(f"{cat}, {pt_edges[ib]}-{pt_edges[ib + 1]} GeV: B = {tot.sum():.0f}", fontsize=14)
            ax.set_ylabel("Events / bin")
            ax.legend(fontsize=11)
        p, fl = per[("QCD", "pass", "w")], per[("QCD", "fail", "w")]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(fl > 0, p / fl, np.nan)
            rerr = r * np.sqrt(np.where(p > 0, per[("QCD", "pass", "w2")] / p**2, 0)
                               + np.where(fl > 0, per[("QCD", "fail", "w2")] / fl**2, 0))
        axes[ib, 2].errorbar(centers, r, yerr=rerr, fmt="o", color="k")
        axes[ib, 2].set_title("QCD pass / fail", fontsize=14)
        axes[ib, 2].set_ylabel("pass / fail")
        tt = per[("TT", "pass", "w")].sum() if "TT" in bkg else 0.0
        lines.append(f"{pt_edges[ib]:>4}-{pt_edges[ib + 1]:<5} {p.sum():10.0f} {fl.sum():10.0f} {tt:8.0f} "
                     f"{p.sum() / fl.sum() if fl.sum() > 0 else np.nan:9.3f} "
                     f"{int(per[('QCD', 'pass', 'n')].sum()):8d} {int(per[('QCD', 'fail', 'n')].sum()):8d}")
    for ax in axes[-1]:
        ax.set_xlabel("Candidate-jet $m_{SD}$ [GeV]")
    fig.suptitle(r"$\bf{CMS}$ $\it{Simulation\ WiP}$" + f"    {YEAR}, {round(LUMI[YEAR] / 1000)} fb$^{{-1}}$",
                 fontsize=24, y=0.93)
    save(fig, "passfail")

    report = "\n".join(lines)
    print(report)
    (plotdir / "summary.txt").write_text(report + "\n")
    print(f"plots and summary in {plotdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_dump_args(p):
        p.add_argument("--samples", nargs="+", default=["Signal", "QCD", "TT"], choices=["Signal", "QCD", "TT"])
        p.add_argument("--subsamples", nargs="+", default=None, help="override subsamples (non-signal)")
        p.add_argument("--nfiles", type=int, default=20, help="files per subsample (random, seeded)")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--workers", type=int, default=6)
        p.add_argument("--step-size", default="200 MB")
        p.add_argument("--overwrite", action="store_true")

    def add_analyze_args(p):
        p.add_argument("--chosen", default="ntrk>2&dlenSig>4:4:0.075",
                       help="chosen working point 'definition:k:pqcd' (nSV > k, P(QCD) < pqcd)")
        p.add_argument("--min-bkg-per-ptbin", type=float, default=500,
                       help="min. expected background (QCD + tt) in the pass region of every fit pT bin")
        p.add_argument("--fit-setup", default=str(REPO / "fitting" / "setup_sr.json"),
                       help="pT bins and msd binning are taken from here")
        p.add_argument("--ntrk-grid", type=int, nargs="+", default=[2, 3, 4], help="min. SV ntracks (2 = no cut)")
        p.add_argument("--dlensig-grid", type=float, nargs="+", default=[0, 4, 5, 6, 10],
                       help="min. SV dlenSig (0 = no cut; NanoAOD already requires > 3)")
        p.add_argument("--roc-defs", nargs="+",
                       default=["all SVs", "ntrk>2", "dlenSig>4", "ntrk>2&dlenSig>4", "ntrk>2&dlenSig>6", "ntrk>2&dlenSig>10", "ntrk>3"],
                       help="SV definitions drawn in sv_definition")
        p.add_argument("--pqcd-grid", type=float, nargs="+", default=[0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                       help="P(QCD) cuts in joint_map")
        p.add_argument("--kmin", type=int, default=2, help="lowest nSV threshold in joint_map")
        p.add_argument("--kmax", type=int, default=8, help="highest nSV threshold in joint_map")
        p.add_argument("--pqcd-fail-max", type=float, default=0.6, help="fail: cut < P(QCD) < this")
        p.add_argument("--min-mc", type=int, default=20, help="min. raw QCD MC events to show S/sqrt(B)")

    for name, adders in (("dump", [add_dump_args]), ("analyze", [add_analyze_args]),
                         ("all", [add_dump_args, add_analyze_args])):
        p = sub.add_parser(name)
        p.add_argument("--outdir", required=True, help="holds dump/ and plots/")
        for add in adders:
            add(p)

    args = parser.parse_args()
    if args.cmd in ("dump", "all"):
        run_dump(args)
    if args.cmd in ("analyze", "all"):
        run_analyze(args)


if __name__ == "__main__":
    main()
