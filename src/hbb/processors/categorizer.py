from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import awkward as ak
import dask_awkward as dak
import numpy as np
import xgboost as xgb
from coffea.analysis_tools import PackedSelection, Weights
from coffea.ml_tools import xgboost_wrapper
from hist.dask import Hist

from hbb.corrections import (
    add_btag_weights,
    add_muon_weights,
    add_pdf_weight,
    add_photon_weights,
    add_pileup_weight,
    add_ps_weight,
    add_scalevar_3pt,
    add_scalevar_7pt,
    apply_jerc,
    correct_met,
    correct_muons,
    get_jetveto_event,
    lumiMasks,
    mupt_variations,
)
from hbb.jerc_eras import jerc_variations
from hbb.processors.SkimmerABC import SkimmerABC
from hbb.taggers import b_taggers

from .GenSelection import (
    bosonFlavor,
    gen_selection_Hbb,
    gen_selection_V,
    gen_selection_Vg,
    getBosons,
    sv_gen_matched,
    top_merging_category,
    weak_hf_hadrons,
)
from .objects import (
    SV_VARIATIONS,
    delta_r_eta_phi,
    good_ak4jets,
    good_ak8jets,
    good_electrons,
    loose_muons,
    highpt_muons,
    good_photons,
    set_ak4jets,
    set_ak8jets,
    sv_quality_mask,
    tight_photons,
)

logger = logging.getLogger(__name__)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "Hto2B": gen_selection_Hbb,
    "Wto2Q-": gen_selection_V,
    "Zto2Q-": gen_selection_V,
    "ZGto2QG-": gen_selection_Vg,
}


def get_BDT_model(BDT_file: str):
    bdt_features = [
        "nFatJet",
        "nJet",
        "FatJet0_phi",
        "FatJet0_eta",
        "FatJet0_n2b1",
        "FatJet0_n3b1",
        "FatJet1_pt",
        "FatJet1_phi",
        "FatJet1_eta",
        "FatJet1_msd",
        "FatJet1_pnetMass",
        "FatJet1_pnetTXbb",
        "FatJet1_pnetTXcc",
        "FatJet1_pnetTXqq",
        "FatJet1_pnetTXgg",
        "VBFPair_mjj",
        "VBFPair_deta",
        "Photon0_pt",
        "Jet0_pt",
        "Jet0_eta",
        "Jet0_phi",
        "Jet0_mass",
        "Jet0_btagPNetB",
        "Jet0_btagPNetCvB",
        "Jet0_btagPNetCvL",
        "Jet0_btagPNetQvG",
        "Jet1_pt",
        "Jet1_eta",
        "Jet1_phi",
        "Jet1_mass",
        "Jet1_btagPNetB",
        "Jet1_btagPNetCvB",
        "Jet1_btagPNetCvL",
        "Jet1_btagPNetQvG",
        "Jet2_pt",
        "Jet2_eta",
        "Jet2_phi",
        "Jet2_mass",
        "Jet2_btagPNetB",
        "Jet2_btagPNetCvB",
        "Jet2_btagPNetCvL",
        "Jet2_btagPNetQvG",
        "Jet3_pt",
        "Jet3_eta",
        "Jet3_phi",
        "Jet3_mass",
        "Jet3_btagPNetB",
        "Jet4_btagPNetCvB",
        "Jet4_btagPNetCvL",
        "Jet4_btagPNetQvG",
        "JetClosestFatJet0_pt",
        "JetClosestFatJet0_eta",
        "JetClosestFatJet0_phi",
        "JetClosestFatJet0_mass",
    ]

    class xgboost_model(xgboost_wrapper):
        # Define how to prepare awkward arrays for BDT evaluation
        def prepare_awkward(self, events):
            features = []
            for name in bdt_features:
                feat = events[name]
                feat = ak.fill_none(feat, -999.0)
                features.append(feat[:, np.newaxis])
            ret = ak.concatenate(features, axis=1)
            return [], dict(data=ret)

    booster = xgb.Booster()
    booster.load_model(Path.cwd() / BDT_file)
    booster.feature_names = None  # Disable feature name checking
    model = xgboost_model(booster)
    return model


class categorizer(SkimmerABC):
    # control-tt: number of probe-jet SVs (all, highest dlenSig first) saved with per-SV properties
    N_PROBE_SVS = 8
    # control-tt columns kept for the JES/JER/UES/muon-pt shifted skims (plus the trigger bits)
    TT_SHIFT_COLUMNS = (
        "GenFlavor",
        "FatJet0_pt",
        "FatJet0_eta",
        "FatJet0_msd",
        "FatJet0_msdmatched",
        "FatJet0_nSV",
        "FatJet0_nSVall",
        "FatJet0_pnetTXbb",
        "FatJet0_ParTPQCD",
        "FatJet0_ParTPXbbVsQCD",
        "FatJet0_nGenB",
        "FatJet0_nGenC",
        "FatJet0_nSVmatched",
        "FatJet0_GenTopCat",
        "MET",
        "LepW_pt",
        "LepSideFatJet_pt",
        "nLepSideBJet",
        "weight",
        "genWeight",
    )

    def __init__(
        self,
        year="2022",
        nano_version="v12",
        xsecs: dict = None,
        systematics=False,
        save_skim=False,
        skim_outpath="",
        evaluate_BDT=True,
        btag_eff=False,
        save_skim_nosysts=False,
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb

        self._year = year
        self._nano_version = nano_version
        self._systematics = systematics
        self._skip_syst = save_skim_nosysts
        self._save_skim = save_skim
        if self._skip_syst:
            self._save_skim = True
        self._skim_outpath = skim_outpath
        self._evaluate_BDT = evaluate_BDT
        self._btag_eff = btag_eff
        self._btagger, self._btag_wp = "btagPNetB", "M"
        if year == "2024":
            self._btagger = "btagUParTAK4B"
        self._btag_cut = b_taggers[self._year]["AK4"][self._btagger][self._btag_wp]
        self._mupt_type = "ptcorr"
        if self._evaluate_BDT:
            self.bdt_model = get_BDT_model("src/hbb/data/MultiClassBDT_23Oct25.ubj")

        with Path("src/hbb/muon_triggers.json").open() as f:
            self._muontriggers = json.load(f)

        with Path("src/hbb/egamma_triggers.json").open() as f:
            self._egammatriggers = json.load(f)

        with Path("src/hbb/triggers.json").open() as f:
            self._triggers = json.load(f)

        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with Path("src/hbb/metfilters.json").open() as f:
            self._met_filters = json.load(f)

        self.make_output = lambda: {
            "sumw": {},
            "cutflow": Hist.new.StrCat([], growth=True, name="region", label="Region")
            .StrCat([], growth=True, name="dataset", label="Dataset")
            .Reg(15, 0, 15, name="cut", label="Cut index")
            .Variable([0, 1, 2, 3, 4], name="genflavor", label="Gen. jet flavor")
            .Weight(),
            "btagWeight": Hist.new.Reg(50, 0, 3, name="val", label="BTag correction").Weight(),
            "skim": {},
        }

        # btag efficiency plots - binning according to:
        # https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#b-tagging-efficiencies-in-simulation
        self.make_btag_output = lambda: (
            Hist.new.StrCat([], growth=True, name="tagger", label="Tagger")
            .Reg(2, 0, 2, name="passWP", label="passWP")
            .Variable([0, 4, 5], name="flavor", label="Jet hadronFlavour")
            .Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt", label="Jet pt")
            .Reg(4, 0, 2.5, name="abseta", label="Jet abseta")
            .Weight()
        )

    def process(self, events):

        # process only nominal case
        if self._skip_syst or not self._save_skim or not hasattr(events, "genWeight"):
            return {"nominal": self.process_shift(events, "nominal")}

        """
        Add `Up and Down` to total list of energy variations
        Muon Energy: `MuonPTScale and MuonPTResolution` defined in mupt_variations
        Jet Energy: `JES, JER, UES` defined in jerc_variations
        """
        total_variations = (
            ["nominal"]
            + [f"{var}_{dir}" for var in jerc_variations for dir in ["Up", "Down"]]
            + [f"{var}_{dir}" for var in mupt_variations for dir in ["Up", "Down"]]
        )

        """
        run processor for each shift defined in total_variations
        return output as dict {variation: output}
        """
        return {var: self.process_shift(events, var) for var in total_variations}

    def add_common_weights(self, weights, events, dataset):
        """
        Add weights that are not region specific
        """

        weights.add("genweight", events.genWeight)

        if not self._skip_syst:
            add_pileup_weight(weights, self._year, events.Pileup.nTrueInt)
            add_ps_weight(weights, events.PSWeight)

            # Easier to save nominal weights for rest of MC with all of the syst names for grabbing columns in post-processing
            # Need to fix
            # flag_syst = ("Hto2B" in dataset) or ("Hto2C" in dataset) or ("VBFZto" in dataset)
            # add_pdf_weight(weights, getattr(events, "LHEPdfWeight", None) if flag_syst else None)
            # add_scalevar_7pt(
            #     weights, getattr(events, "LHEScaleWeight", None) if flag_syst else None
            # )
            # add_scalevar_3pt(
            #     weights, getattr(events, "LHEScaleWeight", None) if flag_syst else None
            # )

        return

    def add_region_weights(
        self, region, weights, events, btag_jets=None, muons=None, muon_type="", photons=None
        ):
        """
        Add weights that are region specific, depending on objects queried.
        Weights will be differentiated by "REGION{region}_" , which will be used for sorting in the partial_weight call
        """

        weight_str = f"REGION{region}_"

        btag_SF = ak.ones_like(events.run)
        if not self._skip_syst:

            if not self._btag_eff and btag_jets is not None:
                btag_SF = add_btag_weights(
                    weights, btag_jets, self._btagger, self._btag_wp, self._year, alt_str=weight_str
                )

            if muons is not None:
                add_muon_weights(weights, self._year, muons, self._mupt_type, muon_type, alt_str=weight_str)

            if photons is not None:
                add_photon_weights(weights, self._year, photons, alt_str=weight_str)

        return btag_SF

    def get_weight_dict(self, region, weights, dataset) -> tuple[dict, dict]:
        """
        Calculate the partial weights and the systematic variations for specified region.
        Saved to dictionary to be output in skim files.
        """

        #Sort the region specific weights
        include_weights = []
        for weight_key in weights._weights.keys():
            if "REGION" in weight_key:
                if region in weight_key:
                    include_weights.append(weight_key)
            else:
                include_weights.append(weight_key)

        logger.debug("weights", extra=weights._weights.keys())
        # dictionary of all weights and variations
        weights_dict = {}
        # dictionary of total # events for norm preserving variations for normalization in postprocessing
        totals_dict = {}

        # nominal
        weights_dict["weight"] = weights.partial_weight(include=include_weights)

        # systematics
        for systematic in weights.variations:
            if "REGION" in systematic:
                if region in systematic:
                    syst_dict = systematic.replace(f"REGION{region}_", "")
                    weights_dict[syst_dict] = weights.partial_weight(include=include_weights, modifier=systematic)
            else:
                weights_dict[systematic] = weights.partial_weight(include=include_weights, modifier=systematic)

        ###################### Normalization (Step 1) ######################
        # strip the year from the dataset name
        dataset_no_year = dataset.replace(f"{self._year}_", "")
        weight_norm = self.get_dataset_norm(self._year, dataset_no_year)
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.partial_weight(include=include_weights)

        return weights_dict, totals_dict

    def process_shift(self, events, shift_name):

        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        output = self.make_output() if not self._btag_eff else self.make_btag_output()
        weights = Weights(None, storeIndividual=True)
        if shift_name == "nominal" and not isRealData and not self._btag_eff:
            output["sumw"][dataset] = ak.sum(events.genWeight)

        trigger = ak.values_astype(ak.zeros_like(events.run), bool)
        for t in self._triggers[self._year]:
            if t in events.HLT.fields:
                trigger = trigger | events.HLT[t]
        selection.add("trigger", trigger)
        del trigger

        if isRealData:
            selection.add("lumimask", lumiMasks[self._year[:4]](events.run, events.luminosityBlock))
        else:
            selection.add("lumimask", ak.values_astype(ak.ones_like(events.run), bool))

        trigger = ak.values_astype(ak.zeros_like(events.run), bool)
        for t in self._muontriggers[self._year]:
            if t in events.HLT.fields:
                trigger = trigger | events.HLT[t]
        selection.add("muontrigger", trigger)
        del trigger

        trigger = ak.values_astype(ak.zeros_like(events.run), bool)
        for t in self._egammatriggers[self._year]:
            if t in events.HLT.fields:
                trigger = trigger | events.HLT[t]
        selection.add("egammatrigger", trigger)
        del trigger

        metfilter = ak.values_astype(ak.ones_like(events.run), bool)
        for flag in self._met_filters[self._year]["data" if isRealData else "mc"]:
            if flag in events.Flag.fields:
                metfilter = metfilter & events.Flag[flag]
        selection.add("metfilter", metfilter)
        del metfilter

        fatjets = set_ak8jets(
            events.FatJet, isRealData, self._year, self._nano_version, events.Rho.fixedGridRhoFastjetAll
        )
        jets = set_ak4jets(
            events.Jet, isRealData, self._year, self._nano_version, events.Rho.fixedGridRhoFastjetAll
        )

        if self._nano_version == "v14_private" or self._nano_version == "v15":
            # subjets in PFNano reprocessing break the fatjet jercs for whatever reason
            keep_fields = [
                f
                for f in fatjets.fields
                if (
                    ("nConstituents" not in f)
                    and ("IdxG" not in f)
                    and ("Idx1G" not in f)
                    and ("Idx2G" not in f)
                )
            ]

            fatjets = ak.zip(
                {f: fatjets[f] for f in keep_fields}, with_name="FatJet", behavior=fatjets.behavior
            )
        met = events.PuppiMET
        # Apply jerc corrections to jets, fatjets, and met collections
        if not self._skip_syst:
            ak8_mass_fields = ("msd", "pnetmass", "ParTmassGeneric", "ParTmassX2p")
            jets = apply_jerc(jets, "AK4", self._year, not isRealData, events.run, events.event)
            fatjets = apply_jerc(
                fatjets,
                "AK8",
                self._year,
                not isRealData,
                events.run,
                events.event,
                mass_fields=ak8_mass_fields,
            )
            met = correct_met(met, jets)  # PuppiMET Recommended for Run3

        # Select jets, fatjets, and met collections according to jerc variation shift
        if shift_name != "nominal" and "Muon" not in shift_name:
            var, direction = shift_name.split("_")
            attr = jerc_variations[var]
            if var in ("JES", "JER"):
                jets = getattr(getattr(jets, attr), direction.lower())
                fatjets = getattr(getattr(fatjets, attr), direction.lower())
                met = getattr(getattr(met, attr), direction.lower())
            elif var == "UES":
                met = getattr(getattr(met, attr), direction.lower())

        goodfatjets = good_ak8jets(fatjets)
        goodjets = good_ak4jets(jets)

        cut_jetveto = get_jetveto_event(jets, self._year)
        selection.add("ak4jetveto", cut_jetveto)

        selection.add("2FJ", ak.num(goodfatjets, axis=1) == 2)
        selection.add("not2FJ", ak.num(goodfatjets, axis=1) != 2)

        # Count SVs passing the SV definition (objects.py::sv_quality_mask) within dR < 0.8 of each
        # good FatJet, then sort by nSV desc (tiebreak: pt desc).
        # Use ak.cartesian to avoid metric_table, which calls delta_r on SecondaryVertexArray
        # and fails during dask typetracer meta-inference (SV is not a vector.Vector)
        fj_sv_pairs = ak.cartesian({"fj": goodfatjets, "sv": events.SV}, axis=1, nested=True)
        sv_in_fj = (
            delta_r_eta_phi(
                fj_sv_pairs["fj"].eta, fj_sv_pairs["fj"].phi, fj_sv_pairs["sv"].eta, fj_sv_pairs["sv"].phi
            )
            < 0.8
        )
        nSV_per_jet = ak.sum(sv_in_fj & sv_quality_mask(fj_sv_pairs["sv"]), axis=2)
        goodfatjets["nSV"] = nSV_per_jet
        for sv_var, (min_ntracks, min_dlensig) in SV_VARIATIONS.items():
            goodfatjets[sv_var] = ak.sum(
                sv_in_fj & sv_quality_mask(fj_sv_pairs["sv"], min_ntracks, min_dlensig), axis=2
            )
        sv_sort_key = ak.values_astype(nSV_per_jet, np.float32) * 1e5 + goodfatjets.pt
        svfatjets = goodfatjets[ak.argsort(sv_sort_key, axis=1, ascending=False)]

        candidatejet = ak.firsts(svfatjets[:, 0:1])
        subleadingjet = ak.firsts(svfatjets[:, 1:2])


        selection.add(
            "minjetkin",
            (candidatejet.pt >= 300) & (candidatejet.pt < 1200) & (candidatejet.msd >= 40.0)
            # & (candidatejet.msd < 201.0)
            & (abs(candidatejet.eta) < 2.5),
        )

        selection.add(
            "minjetkin_zgamma",
            (candidatejet.pt >= 200)  # Loosened pt cut
            & (candidatejet.pt < 1200)
            & (candidatejet.msd >= 0.0)  # Loosened msd cut
            & (candidatejet.msd < 201.0)
            & (abs(candidatejet.eta) < 2.5),
        )

        selection.add("particleNetXbbpass", (candidatejet.particleNet_XbbVsQCD >= 0.5))

        # only consider 4 AK4 jets leading in pT to be consistent with old framework
        jets = goodjets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        dR = jets.delta_r(candidatejet)
        ak4_opphem_ak8 = jets[dphi > np.pi / 2]
        ak4_outside_ak8 = jets[dR > 0.8]

        # ak4 closest to ak8
        ak4_closest_ak8 = ak.firsts(
            ak4_outside_ak8[ak.argmin(ak4_outside_ak8.delta_r(candidatejet), axis=1, keepdims=True)]
        )

        selection.add(
            "antiak4btagMediumOppHem",
            ak.max(getattr(ak4_opphem_ak8, self._btagger), axis=1, mask_identity=False)
            < self._btag_cut,
        )
        selection.add(
            "antiak4btagMedium",
            ak.max(getattr(ak4_outside_ak8, self._btagger), axis=1, mask_identity=False)
            < self._btag_cut,
        )
        selection.add(
            "ak4btagMedium08",
            ak.max(getattr(ak4_outside_ak8, self._btagger), axis=1, mask_identity=False)
            > self._btag_cut,
        )

        selection.add("lowmet", met.pt < 140.0)

        # VBF specific variables
        jet1_away = ak.firsts(ak4_outside_ak8[:, 0:1])
        jet2_away = ak.firsts(ak4_outside_ak8[:, 1:2])
        jet3_away = ak.firsts(ak4_outside_ak8[:, 2:3])
        jet4_away = ak.firsts(ak4_outside_ak8[:, 3:4])

        vbf_deta = abs(jet1_away.eta - jet2_away.eta)
        vbf_mjj = (jet1_away + jet2_away).mass
        vbf_deta = ak.fill_none(vbf_deta, -1)
        vbf_mjj = ak.fill_none(vbf_mjj, -1)

        isvbf = (vbf_deta > 3.5) & (vbf_mjj > 1000)
        isvbf = ak.fill_none(isvbf, False)
        isnotvbf = ak.fill_none(~isvbf, True)

        selection.add("isvbf", isvbf)
        selection.add("notvbf", isnotvbf)

        muons = correct_muons(events.Muon, events, self._year, isRealData)
        if shift_name != "nominal" and "Muon" in shift_name:
            var, direction = shift_name.split("_")
            self._mupt_type = f"{mupt_variations[var]}_{direction.lower()}"

        loosemuon = loose_muons(muons, self._mupt_type)
        highptmuon = highpt_muons(muons, self._mupt_type)
        nmuons = ak.num(loosemuon, axis=1)
        ttbarmuon = ak.firsts(loosemuon)
        leadingmuon = loosemuon[:, :1]
        ttbarmuon_sf = leadingmuon[getattr(leadingmuon, self._mupt_type) > 55.0]
        # low pt muons break sf (lower bound 15GeV)
        # ak.firsts records array breaks the evaluator also, the inverse breaks selection.add
            # TODO figure this out later

        zmm_muons = highptmuon[:, :2]   #Collection to pass for sfs
        zmm_lead = ak.firsts(zmm_muons[:, 0:1])
        zmm_sublead = ak.firsts(zmm_muons[:, 1:2])
        nmuons_zmm = ak.num(highptmuon, axis=1)

        zmm_mll = (zmm_lead + zmm_sublead).mass
        zmm_charge = zmm_lead.charge * zmm_sublead.charge   # >0 same sign, <0 opp sign
        zmm_pt = getattr(zmm_lead, self._mupt_type) + getattr(zmm_sublead, self._mupt_type)

        dR_leadm = goodfatjets.delta_r(zmm_lead)
        dR_subleadm = goodfatjets.delta_r(zmm_sublead)
        ak8_outside_dimuon = goodfatjets[(dR_leadm > 0.8) & (dR_subleadm > 0.8)]
        nak8_zmm = ak.num(ak8_outside_dimuon, axis=1)

        goodelectron = good_electrons(events.Electron)
        nelectrons = ak.num(goodelectron, axis=1)

        selection.add("noleptons", (nmuons == 0) & (nelectrons == 0))
        selection.add("onemuon", (nmuons == 1) & (nelectrons == 0))
        selection.add("twoloosemuon", (nmuons == 2) & (nelectrons == 0))
        selection.add("twomuon_zmm", (nmuons_zmm == 2) & (nelectrons == 0))
        selection.add(
            "muonkin_leadzmm", (getattr(zmm_lead, self._mupt_type) > 60.0)
        )
        selection.add(
            "muonpairkin_zmm", (zmm_mll >= 80) & (zmm_mll <= 100) & (zmm_charge < 0) & (zmm_pt > 300)
        )

        selection.add(
            "muonkin", (getattr(ttbarmuon, self._mupt_type) > 55.0) & (abs(ttbarmuon.eta) < 2.1)
        )
        selection.add("muonDphiAK8", abs(ttbarmuon.delta_phi(candidatejet)) > 2 * np.pi / 3)

        # ttbar CR (control-tt) probe: max-nSV AK8 jet on the hadronic side (dphi(muon) > 2pi/3).
        # Restricting the candidates *before* picking keeps events where a leptonic-side AK8 jet
        # (boosted b from t -> b mu nu) has more SVs; picking first and then cutting on dphi
        # (muonDphiAK8) would veto them and bias the probe nSV spectrum.
        muon_phi = ak.fill_none(ttbarmuon.phi, np.nan)  # nan: no hadronic/leptonic side
        dphi_fj_muon = abs((svfatjets.phi - muon_phi + np.pi) % (2 * np.pi) - np.pi)
        probejet = ak.firsts(svfatjets[dphi_fj_muon > 2 * np.pi / 3][:, 0:1])
        # leading leptonic-side AK8 pt, to check whether that jet could fire the AK8 jet triggers
        lepside_fatjet_pt = ak.fill_none(
            ak.max(svfatjets[dphi_fj_muon <= 2 * np.pi / 3].pt, axis=1), 0.0
        )

        probe_phi = ak.fill_none(probejet.phi, np.nan)
        ak4_lepside = goodjets[
            abs((goodjets.phi - probe_phi + np.pi) % (2 * np.pi) - np.pi) > np.pi / 2
        ]
        lepbjet = ak.firsts(
            ak4_lepside[ak.argmax(getattr(ak4_lepside, self._btagger), axis=1, keepdims=True)]
        )
        n_ak4_lepside_btag = ak.sum(getattr(ak4_lepside, self._btagger) > self._btag_cut, axis=1)

        mu_pt = ak.fill_none(getattr(ttbarmuon, self._mupt_type), 0.0)
        lepw_px = mu_pt * np.cos(ak.fill_none(ttbarmuon.phi, 0.0)) + met.pt * np.cos(met.phi)
        lepw_py = mu_pt * np.sin(ak.fill_none(ttbarmuon.phi, 0.0)) + met.pt * np.sin(met.phi)
        lepw_pt = np.sqrt(lepw_px**2 + lepw_py**2)

        selection.add("probejetkin", ak.fill_none(probejet.pt > 0, False))  # pt/eta from good_ak8jets
        selection.add("ttlepbtag", n_ak4_lepside_btag >= 1)
        selection.add("ttmet", met.pt > 50.0)

        # all SVs within dR < 0.8 of the SR candidate / CR probe, highest dlenSig first
        def svs_in_jet(jet):
            jet_eta, jet_phi = ak.fill_none(jet.eta, np.nan), ak.fill_none(jet.phi, np.nan)
            svs = events.SV[delta_r_eta_phi(events.SV.eta, events.SV.phi, jet_eta, jet_phi) < 0.8]
            return svs[ak.argsort(svs.dlenSig, axis=1, ascending=False)]

        candidate_svs = svs_in_jet(candidatejet)
        probe_svs = svs_in_jet(probejet)

        goodphotons = good_photons(events.Photon)
        nphotons = ak.num(goodphotons, axis=1)

        ntightphotons = ak.num(tight_photons(events.Photon), axis=1)
        vgammaphoton = ak.firsts(tight_photons(events.Photon))

        selection.add("onephoton", (nphotons == 1))
        selection.add("atleastonephoton", (ntightphotons >= 1))
        selection.add("passphotonveto", (nphotons == 0))

        if self._evaluate_BDT:
            # Construct BDT input
            bdt_ak_array = {
                "nFatJet": ak.num(goodfatjets, axis=1),
                "nJet": ak.num(goodjets, axis=1),
                "FatJet0_phi": candidatejet.phi,
                "FatJet0_eta": candidatejet.eta,
                "FatJet0_n2b1": candidatejet.n2b1,
                "FatJet0_n3b1": candidatejet.n3b1,
                "FatJet1_pt": subleadingjet.pt,
                "FatJet1_phi": subleadingjet.phi,
                "FatJet1_eta": subleadingjet.eta,
                "FatJet1_msd": subleadingjet.msd,
                "FatJet1_pnetMass": subleadingjet.pnetmass,
                "FatJet1_pnetTXbb": subleadingjet.particleNet_XbbVsQCD,
                "FatJet1_pnetTXcc": subleadingjet.particleNet_XccVsQCD,
                "FatJet1_pnetTXqq": subleadingjet.particleNet_XqqVsQCD,
                "FatJet1_pnetTXgg": subleadingjet.particleNet_XggVsQCD,
                "VBFPair_mjj": vbf_mjj,
                "VBFPair_deta": vbf_deta,
                "Photon0_pt": vgammaphoton.pt,
                "Jet0_pt": jet1_away.pt,
                "Jet0_eta": jet1_away.eta,
                "Jet0_phi": jet1_away.phi,
                "Jet0_mass": jet1_away.mass,
                "Jet0_btagPNetB": jet1_away.btagPNetB,
                "Jet0_btagPNetCvB": jet1_away.btagPNetCvB,
                "Jet0_btagPNetCvL": jet1_away.btagPNetCvL,
                "Jet0_btagPNetQvG": jet1_away.btagPNetQvG,
                "Jet1_pt": jet2_away.pt,
                "Jet1_eta": jet2_away.eta,
                "Jet1_phi": jet2_away.phi,
                "Jet1_mass": jet2_away.mass,
                "Jet1_btagPNetB": jet2_away.btagPNetB,
                "Jet1_btagPNetCvB": jet2_away.btagPNetCvB,
                "Jet1_btagPNetCvL": jet2_away.btagPNetCvL,
                "Jet1_btagPNetQvG": jet2_away.btagPNetQvG,
                "Jet2_pt": jet3_away.pt,
                "Jet2_eta": jet3_away.eta,
                "Jet2_phi": jet3_away.phi,
                "Jet2_mass": jet3_away.mass,
                "Jet2_btagPNetB": jet3_away.btagPNetB,
                "Jet2_btagPNetCvB": jet3_away.btagPNetCvB,
                "Jet2_btagPNetCvL": jet3_away.btagPNetCvL,
                "Jet2_btagPNetQvG": jet3_away.btagPNetQvG,
                "Jet3_pt": jet4_away.pt,
                "Jet3_eta": jet4_away.eta,
                "Jet3_phi": jet4_away.phi,
                "Jet3_mass": jet4_away.mass,
                "Jet3_btagPNetB": jet4_away.btagPNetB,
                "Jet4_btagPNetCvB": jet4_away.btagPNetCvB,
                "Jet4_btagPNetCvL": jet4_away.btagPNetCvL,
                "Jet4_btagPNetQvG": jet4_away.btagPNetQvG,
                "JetClosestFatJet0_pt": ak4_closest_ak8.pt,
                "JetClosestFatJet0_eta": ak4_closest_ak8.eta,
                "JetClosestFatJet0_phi": ak4_closest_ak8.phi,
                "JetClosestFatJet0_mass": ak4_closest_ak8.mass,
            }
            bdt_input = ak.zip(bdt_ak_array, depth_limit=1)

            # Evaluate BDT
            bdt_model = self.bdt_model
            bdt_scores = bdt_model(bdt_input)

            # assign scores to selections
            selection.add("BDTisVBF", (bdt_scores == 0))
            selection.add("BDTisVH", (bdt_scores == 1))
            selection.add("BDTisggF", (bdt_scores == 2))

        gen_variables = {}
        gen_jet_sr = {}
        gen_jet_tt = {}
        btag_SF = ak.ones_like(events.run)
        btag_SF_mu = ak.ones_like(events.run)
        if isRealData:
            genflavor = ak.zeros_like(candidatejet.pt)
            genflavor_probe = ak.zeros_like(probejet.pt)
            genBosonPt = ak.zeros_like(candidatejet.pt)
        else:

            self.add_common_weights(weights, events, dataset)
            # signal regions
            btag_SF = self.add_region_weights(
                "signal", weights, events, btag_jets=ak4_opphem_ak8
            )
            # muon region
            btag_SF_mu = self.add_region_weights(
                "control-tt", weights, events, btag_jets=ak4_lepside, muons=ttbarmuon_sf, muon_type="medium"
            )
            # gamma region
            btag_SF_gamma = self.add_region_weights(
                "control-zgamma", weights, events, btag_jets=ak4_outside_ak8, photons=vgammaphoton
            )
            # zmumu muon region 
            btag_SF_zmm = self.add_region_weights(
                "control-zmumu", weights, events, muons=zmm_muons, muon_type="highpt"
            )

            weights_dict, totals_temp = self.get_weight_dict("signal", weights, dataset)
            weights_dict_mu, totals_temp_mu = self.get_weight_dict("control-tt", weights, dataset)
            weights_dict_gamma, totals_temp_gamma = self.get_weight_dict("control-zgamma", weights, dataset)
            weights_dict_zmm, totals_temp_zmm = self.get_weight_dict("control-zmumu", weights, dataset)

            for d, gen_func in gen_selection_dict.items():
                if d in dataset:
                    # match goodfatjets
                    gen_variables = gen_func(events, goodfatjets)

            bosons = getBosons(events.GenPart)

            def boson_flavor(jet):
                matchedBoson = jet.nearest(bosons, axis=None, threshold=0.8)
                match_mask = (abs(jet.pt - matchedBoson.pt) / matchedBoson.pt < 0.5) & (
                    abs(jet.msd - matchedBoson.mass) / matchedBoson.mass < 0.3
                )
                return bosonFlavor(ak.mask(matchedBoson, match_mask))

            genflavor = boson_flavor(candidatejet)
            genflavor_probe = boson_flavor(probejet)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)

            # heavy-flavour content and SV gen-matching for the per-SV nSV scale factor model
            hadrons_b, hadrons_c_prompt, hadrons_c = weak_hf_hadrons(events.GenPart)

            def gen_jet_content(jet, jet_svs):
                jet_eta, jet_phi = ak.fill_none(jet.eta, np.nan), ak.fill_none(jet.phi, np.nan)
                sv_matched = sv_gen_matched(jet_svs, (hadrons_b, hadrons_c))

                def n_in_jet(hadrons):
                    return ak.sum(delta_r_eta_phi(hadrons.eta, hadrons.phi, jet_eta, jet_phi) < 0.8, axis=1)

                content = {
                    "FatJet0_nGenB": n_in_jet(hadrons_b),
                    "FatJet0_nGenC": n_in_jet(hadrons_c_prompt),
                    "FatJet0_nSVmatched": ak.sum(sv_matched & sv_quality_mask(jet_svs), axis=1),
                    "FatJet0_GenTopCat": top_merging_category(jet_eta, jet_phi, events.GenPart),
                }
                return content, sv_matched

            gen_jet_sr, _ = gen_jet_content(candidatejet, candidate_svs)
            gen_jet_tt, probe_sv_matched = gen_jet_content(probejet, probe_svs)

        # softdrop mass, 0 for genflavor == 0
        msd_matched = candidatejet.msd * (genflavor > 0) + candidatejet.msd * (genflavor == 0)
        msd_matched_probe = probejet.msd * (genflavor_probe > 0) + probejet.msd * (genflavor_probe == 0)

        regions = {
            "signal-all": [
                "trigger",
                "lumimask",
                "metfilter",
                "ak4jetveto",
                "minjetkin",
                "antiak4btagMediumOppHem",
                "noleptons",
            ],
            # "signal-ggf": [
            #     "trigger",
            #     "lumimask",
            #     "metfilter",
            #     "ak4jetveto",
            #     "minjetkin",
            #     "antiak4btagMediumOppHem",
            #     "lowmet",
            #     "noleptons",
            #     "notvbf",
            #     "not2FJ",
            # ],
            # "signal-vh": [
            #     "trigger",
            #     "lumimask",
            #     "metfilter",
            #     "ak4jetveto",
            #     "minjetkin",
            #     "antiak4btagMediumOppHem",
            #     "lowmet",
            #     "noleptons",
            #     "notvbf",
            #     "2FJ",
            # ],
            # "signal-vbf": [
            #     "trigger",
            #     "lumimask",
            #     "metfilter",
            #     "ak4jetveto",
            #     "minjetkin",
            #     "antiak4btagMediumOppHem",
            #     "lowmet",
            #     "noleptons",
            #     "isvbf",
            # ],
            # ttbar (mu+jets) CR for the trigger / nSV / anti-QCD scale factors.
            # Jet triggers are not required (per-path bits are saved); loose probe kinematics
            # (good_ak8jets: pt > 200) so the trigger turn-on is visible.
            "control-tt": [
                "muontrigger",
                "lumimask",
                "metfilter",
                "ak4jetveto",
                "onemuon",
                "muonkin",
                "probejetkin",
                "ttlepbtag",
                "ttmet",
            ],
            # H->bb version, probe picked over all AK8 jets before the dphi cut:
            # "control-tt": [
            #     "muontrigger",
            #     "lumimask",
            #     "metfilter",
            #     "ak4jetveto",
            #     "minjetkin",
            #     "ak4btagMedium08",
            #     "onemuon",
            #     "muonkin",
            #     "muonDphiAK8",
            # ],
            # "control-zgamma": [
            #     "egammatrigger",
            #     "lumimask",
            #     "metfilter",
            #     "minjetkin_zgamma",
            #     "atleastonephoton",
            #     "antiak4btagMedium",
            # ],
            # "control-zmumu": [
            #     "muontrigger",
            #     "lumimask",
            #     "metfilter",
            #     "twoloosemuon",
            #     "twomuon_zmm",
            #     "muonkin_leadzmm",
            #     "muonpairkin_zmm",
            # ],
        }
        if self._evaluate_BDT:
            #replace orthogonality cuts with BDT score
            regions.update(
                {
                    "signal-ggf-BDT": [x for x in regions["signal-ggf"] if x not in ["notvbf", "not2FJ"]] + ["BDTisggF"],
                    "signal-vh-BDT": [x for x in regions["signal-vh"] if x not in ["notvbf", "2FJ"]] + ["BDTisVH"],
                    "signal-vbf-BDT": [x for x in regions["signal-vbf"] if x not in ["isvbf"]] + ["BDTisVBF"],
                }
            )

        btag_eff_cuts = [
            "trigger",
            "lumimask",
            "metfilter",
            "ak4jetveto",
            "minjetkin",
            "lowmet",
            "noleptons",
        ]

        tic = time.time()

        nominal_weight = ak.ones_like(events.run) if isRealData else weights_dict["weight"]
        nominal_weight_tt = ak.ones_like(events.run) if isRealData else weights_dict_mu["weight"]
        gen_weight = ak.ones_like(events.run) if isRealData else events.genWeight

        egamma_trigger_booleans = {}
        for t in self._egammatriggers[self._year]:
            if t in events.HLT.fields:
                egamma_trigger_booleans[t] = events.HLT[t]
            else:
                egamma_trigger_booleans[t] = ak.values_astype(ak.zeros_like(events.run), bool)

        # per-path jet (and reference muon) trigger bits, for the trigger efficiency in control-tt
        trigger_booleans = {}
        for t in self._triggers[self._year] + self._muontriggers[self._year]:
            if t in events.HLT.fields:
                trigger_booleans[t] = events.HLT[t]
            else:
                trigger_booleans[t] = ak.values_astype(ak.zeros_like(events.run), bool)

        if self._btag_eff:
            cut = selection.all(*btag_eff_cuts)
            flat_gj = ak.flatten(goodjets)

            output.fill(
                tagger=self._btagger,
                abseta=self.normalize(abs(flat_gj.eta), cut),
                pt=self.normalize(flat_gj.pt, cut),
                flavor=self.normalize(flat_gj.hadronFlavour, cut),
                passWP=self.normalize(getattr(flat_gj, self._btagger) > self._btag_cut, cut),
            )
            return output

        output_array = None
        if self._save_skim:

            output_array = {
                "event": events.event,
                "GenBoson_pt": genBosonPt,
                "GenFlavor": genflavor,
                "nFatJet": ak.num(goodfatjets, axis=1),
                "nJet": ak.num(goodjets, axis=1),
                "FatJet0_pt": candidatejet.pt,
                "FatJet0_phi": candidatejet.phi,
                "FatJet0_eta": candidatejet.eta,
                "FatJet0_nSV": candidatejet.nSV,
                **{f"FatJet0_{sv_var}": candidatejet[sv_var] for sv_var in SV_VARIATIONS},
                "FatJet0_msd": candidatejet.msd,
                "FatJet0_mass": candidatejet.mass,
                "FatJet0_msdmatched": msd_matched,
                "FatJet0_n2b1": candidatejet.n2b1,
                "FatJet0_n3b1": candidatejet.n3b1,
                "FatJet0_pnetMass": candidatejet.pnetmass,
                "FatJet0_pnetTXbb": candidatejet.particleNet_XbbVsQCD,
                "FatJet0_pnetTXcc": candidatejet.particleNet_XccVsQCD,
                "FatJet0_pnetTXqq": candidatejet.particleNet_XqqVsQCD,
                "FatJet0_pnetTXgg": candidatejet.particleNet_XggVsQCD,
                "FatJet0_pnetTQCD": candidatejet.particleNet_QCD,
                "FatJet0_pnetXbbXcc": candidatejet.pnetXbbXcc,
                "FatJet1_pt": subleadingjet.pt,
                "FatJet1_phi": subleadingjet.phi,
                "FatJet1_eta": subleadingjet.eta,
                "FatJet1_nSV": subleadingjet.nSV,
                "FatJet1_nSVall": subleadingjet.nSVall,
                "FatJet1_msd": subleadingjet.msd,
                "FatJet1_pnetMass": subleadingjet.pnetmass,
                "FatJet1_pnetTXbb": subleadingjet.particleNet_XbbVsQCD,
                "FatJet1_pnetTXcc": subleadingjet.particleNet_XccVsQCD,
                "FatJet1_pnetTXqq": subleadingjet.particleNet_XqqVsQCD,
                "FatJet1_pnetTXgg": subleadingjet.particleNet_XggVsQCD,
                "VBFPair_mjj": vbf_mjj,
                "VBFPair_deta": vbf_deta,
                "Photon0_pt": vgammaphoton.pt,
                "Photon0_phi": vgammaphoton.phi,
                "Photon0_eta": vgammaphoton.eta,
                "MET": met.pt,
                "weight": nominal_weight,
                "genWeight": gen_weight,
                **gen_variables,
                **gen_jet_sr,
                **egamma_trigger_booleans,
                **trigger_booleans,
            }
            if self._evaluate_BDT:
                output_array.update({"BDT_score": bdt_scores})

            output_array_zmm = {
                "Zmm_MuonLead_pt": (getattr(zmm_lead, self._mupt_type)),
                "Zmm_MuonLead_eta": zmm_lead.eta,
                "Zmm_MuonLead_phi": zmm_lead.phi,
                "Zmm_MuonLead_charge": zmm_lead.charge,

                "Zmm_MuonSublead_pt": (getattr(zmm_sublead, self._mupt_type)),
                "Zmm_MuonSubLead_eta": zmm_sublead.eta,
                "Zmm_MuonSubLead_phi": zmm_sublead.phi,
                "Zmm_MuonSubLead_charge": zmm_sublead.charge,

                "Zmm_MuonPair_mll": zmm_mll,
                "Zmm_MuonPair_pt": zmm_pt,

                "Zmm_nak8": nak8_zmm,
                "Zmm_ntightPhotons": ntightphotons,
            }

            # reduced output array for energy variation shift
            energy_var_array = {
                "GenBoson_pt": genBosonPt,
                "GenFlavor": genflavor,
                "FatJet0_pt": candidatejet.pt,
                "FatJet0_msd": candidatejet.msd,
                "FatJet0_msdmatched": msd_matched,
                "FatJet0_nSV": candidatejet.nSV,
                "FatJet0_nSVall": candidatejet.nSVall,
                "FatJet0_pnetTXbb": candidatejet.particleNet_XbbVsQCD,
                "FatJet0_pnetTXcc": candidatejet.particleNet_XccVsQCD,
                "FatJet0_pnetXbbXcc": candidatejet.pnetXbbXcc,
                "VBFPair_mjj": vbf_mjj,
                "weight": nominal_weight,
                "genWeight": gen_weight,
            }

            if "v12" not in self._nano_version:
                parT_array = {
                    "FatJet0_ParTPQCD": candidatejet.ParTPQCD,
                    "FatJet0_ParTPXbb": candidatejet.ParTPXbb,
                    "FatJet0_ParTPXcc": candidatejet.ParTPXcc,
                    "FatJet0_ParTPXqq": candidatejet.ParTPXqq,
                    "FatJet0_ParTPXcs": candidatejet.ParTPXcs,
                    "FatJet0_ParTPXbbVsQCD": candidatejet.ParTPXbbVsQCD,
                    "FatJet0_ParTPXccVsQCD": candidatejet.ParTPXccVsQCD,
                    "FatJet0_ParTPXbbXcc": candidatejet.ParTPXbbXcc,
                    "FatJet0_ParTmassGeneric": candidatejet.ParTmassGeneric,
                    "FatJet0_ParTmassX2p": candidatejet.ParTmassX2p,
                    "FatJet1_ParTPQCD": subleadingjet.ParTPQCD,
                    "FatJet1_ParTPXbb": subleadingjet.ParTPXbb,
                    "FatJet1_ParTPXcc": subleadingjet.ParTPXcc,
                    "FatJet1_ParTPXqq": subleadingjet.ParTPXqq,
                    "FatJet1_ParTPXcs": subleadingjet.ParTPXcs,
                    "FatJet1_ParTPXbbVsQCD": subleadingjet.ParTPXbbVsQCD,
                    "FatJet1_ParTPXccVsQCD": subleadingjet.ParTPXccVsQCD,
                    "FatJet1_ParTPXbbXcc": subleadingjet.ParTPXbbXcc,
                    "FatJet1_ParTmassGeneric": subleadingjet.ParTmassGeneric,
                    "FatJet1_ParTmassX2p": subleadingjet.ParTmassX2p,
                }
                output_array = {**output_array, **parT_array}

                energy_var_array_parT = {
                    "FatJet0_ParTPXbbVsQCD": candidatejet.ParTPXbbVsQCD,
                    "FatJet0_ParTPXccVsQCD": candidatejet.ParTPXccVsQCD,
                    "FatJet0_ParTPXbbXcc": candidatejet.ParTPXbbXcc,
                    "FatJet0_ParTPQCD": candidatejet.ParTPQCD,
                }
                energy_var_array = {**energy_var_array, **energy_var_array_parT}

            # extra variables for big array
            output_array_extra = {
                # AK4 Jets away from FatJet0
                "Jet0_pt": jet1_away.pt,
                "Jet0_eta": jet1_away.eta,
                "Jet0_phi": jet1_away.phi,
                "Jet0_mass": jet1_away.mass,
                "Jet0_btagPNetB": jet1_away.btagPNetB,
                "Jet0_btagPNetCvB": jet1_away.btagPNetCvB,
                "Jet0_btagPNetCvL": jet1_away.btagPNetCvL,
                "Jet0_btagPNetQvG": jet1_away.btagPNetQvG,
                "Jet1_pt": jet2_away.pt,
                "Jet1_eta": jet2_away.eta,
                "Jet1_phi": jet2_away.phi,
                "Jet1_mass": jet2_away.mass,
                "Jet1_btagPNetB": jet2_away.btagPNetB,
                "Jet1_btagPNetCvB": jet2_away.btagPNetCvB,
                "Jet1_btagPNetCvL": jet2_away.btagPNetCvL,
                "Jet1_btagPNetQvG": jet2_away.btagPNetQvG,
                "Jet2_pt": jet3_away.pt,
                "Jet2_eta": jet3_away.eta,
                "Jet2_phi": jet3_away.phi,
                "Jet2_mass": jet3_away.mass,
                "Jet2_btagPNetB": jet3_away.btagPNetB,
                "Jet2_btagPNetCvB": jet3_away.btagPNetCvB,
                "Jet2_btagPNetCvL": jet3_away.btagPNetCvL,
                "Jet2_btagPNetQvG": jet3_away.btagPNetQvG,
                "Jet3_pt": jet4_away.pt,
                "Jet3_eta": jet4_away.eta,
                "Jet3_phi": jet4_away.phi,
                "Jet3_mass": jet4_away.mass,
                "Jet3_btagPNetB": jet4_away.btagPNetB,
                "Jet4_btagPNetCvB": jet4_away.btagPNetCvB,
                "Jet4_btagPNetCvL": jet4_away.btagPNetCvL,
                "Jet4_btagPNetQvG": jet4_away.btagPNetQvG,
                # AK4 Jet away but closest to FatJet0
                "JetClosestFatJet0_pt": ak4_closest_ak8.pt,
                "JetClosestFatJet0_eta": ak4_closest_ak8.eta,
                "JetClosestFatJet0_phi": ak4_closest_ak8.phi,
                "JetClosestFatJet0_mass": ak4_closest_ak8.mass,
            }

            # control-tt: the probe jet is saved under the FatJet0_ names used for the SR candidate
            probe_svs_padded = ak.pad_none(probe_svs, self.N_PROBE_SVS, axis=1, clip=True)
            probe_sv_array = {
                f"FatJet0_SV{i}_{var}": probe_svs_padded[var][:, i]
                for i in range(self.N_PROBE_SVS)
                for var in ("ntracks", "dlenSig", "mass", "pt", "dxy")
            }
            if not isRealData:
                probe_sv_matched_padded = ak.pad_none(probe_sv_matched, self.N_PROBE_SVS, axis=1, clip=True)
                probe_sv_array.update(
                    {
                        f"FatJet0_SV{i}_genMatched": probe_sv_matched_padded[:, i]
                        for i in range(self.N_PROBE_SVS)
                    }
                )

            output_array_tt = {
                "event": events.event,
                "GenFlavor": genflavor_probe,
                "nFatJet": ak.num(goodfatjets, axis=1),
                "nJet": ak.num(goodjets, axis=1),
                "FatJet0_pt": probejet.pt,
                "FatJet0_phi": probejet.phi,
                "FatJet0_eta": probejet.eta,
                "FatJet0_nSV": probejet.nSV,
                **{f"FatJet0_{sv_var}": probejet[sv_var] for sv_var in SV_VARIATIONS},
                "FatJet0_msd": probejet.msd,
                "FatJet0_mass": probejet.mass,
                "FatJet0_msdmatched": msd_matched_probe,
                "FatJet0_n2b1": probejet.n2b1,
                "FatJet0_n3b1": probejet.n3b1,
                "FatJet0_pnetMass": probejet.pnetmass,
                "FatJet0_pnetTXbb": probejet.particleNet_XbbVsQCD,
                "FatJet0_pnetTXcc": probejet.particleNet_XccVsQCD,
                "FatJet0_pnetTXqq": probejet.particleNet_XqqVsQCD,
                "FatJet0_pnetTQCD": probejet.particleNet_QCD,
                **probe_sv_array,
                **gen_jet_tt,
                "LepSideFatJet_pt": lepside_fatjet_pt,
                "Muon0_pt": getattr(ttbarmuon, self._mupt_type),
                "Muon0_eta": ttbarmuon.eta,
                "Muon0_phi": ttbarmuon.phi,
                "MET": met.pt,
                "MET_phi": met.phi,
                "LepW_pt": lepw_pt,
                "LepBJet_pt": lepbjet.pt,
                "LepBJet_eta": lepbjet.eta,
                "LepBJet_phi": lepbjet.phi,
                "LepBJet_btag": getattr(lepbjet, self._btagger),
                "nLepSideBJet": n_ak4_lepside_btag,
                **trigger_booleans,
                "weight": nominal_weight_tt,
                "genWeight": gen_weight,
            }
            if "v12" not in self._nano_version:
                output_array_tt.update(
                    {
                        "FatJet0_ParTPQCD": probejet.ParTPQCD,
                        "FatJet0_ParTPXbb": probejet.ParTPXbb,
                        "FatJet0_ParTPXcc": probejet.ParTPXcc,
                        "FatJet0_ParTPXqq": probejet.ParTPXqq,
                        "FatJet0_ParTPXbbVsQCD": probejet.ParTPXbbVsQCD,
                        "FatJet0_ParTmassGeneric": probejet.ParTmassGeneric,
                        "FatJet0_ParTmassX2p": probejet.ParTmassX2p,
                    }
                )
            # reduced control-tt array for energy variation shifts
            energy_var_array_tt = {
                k: v
                for k, v in output_array_tt.items()
                if k in self.TT_SHIFT_COLUMNS or k in trigger_booleans
            }

        def skim(region, output_array, region_weight, region_genflavor, region_btag_SF):
            selections = regions[region]
            cut = selection.all(*selections)

            # to debug...
            # print(output_array.compute())
            # print(output_array[cut].compute())

            if "root:" in self._skim_outpath:
                skim_path = f"{self._skim_outpath}/{shift_name.replace('_', '')}/{self._year}/{dataset}/{region}"
            else:
                skim_path = (
                    Path(self._skim_outpath)
                    / shift_name.replace("_", "")
                    / self._year
                    / dataset
                    / region
                )
                skim_path.mkdir(parents=True, exist_ok=True)
            print("Saving skim to: ", skim_path)

            # possible TODO: add systematic weights?
            output["skim"][region] = dak.to_parquet(
                output_array[cut],
                str(skim_path),
                compute=False,
            )

            if shift_name == "nominal":

                # Fill cutflow hist
                allcuts = set()
                cut = selection.all(*allcuts)
                output["cutflow"].fill(
                    dataset=dataset,
                    region=region,
                    genflavor=self.normalize(region_genflavor, None),
                    cut=0,
                    weight=region_weight,
                )
                for i, cut in enumerate(selections):
                    allcuts.add(cut)
                    cumulative_cut = selection.all(*allcuts)
                    output["cutflow"].fill(
                        dataset=dataset,
                        region=region,
                        genflavor=self.normalize(region_genflavor, cumulative_cut),
                        cut=i + 1,
                        weight=region_weight[cumulative_cut],
                    )

                # Fill btag SF hist
                cut = selection.all(*selections)
                output["btagWeight"].fill(val=self.normalize(region_btag_SF, cut))

        if self._save_skim:
            # per region: (nominal columns, energy-shift columns, weights, nominal weight, genflavor, btag SF)
            sr_arrays = (
                {**output_array, **output_array_extra},
                energy_var_array,
                {} if isRealData else weights_dict,
                nominal_weight,
                genflavor,
                btag_SF,
            )
            region_arrays = {
                "control-tt": (
                    output_array_tt,
                    energy_var_array_tt,
                    {} if isRealData else weights_dict_mu,
                    nominal_weight_tt,
                    genflavor_probe,
                    btag_SF_mu,
                ),
            }
            for region in regions:
                nominal_arr, shift_arr, region_weights, region_weight, region_genflavor, region_btag_SF = (
                    region_arrays.get(region, sr_arrays)
                )
                arr = nominal_arr if shift_name == "nominal" else shift_arr
                skim(
                    region,
                    ak.zip({**arr, **region_weights}, depth_limit=1),
                    region_weight,
                    region_genflavor,
                    region_btag_SF,
                )

        toc = time.time()
        output["filltime"] = toc - tic
        print(f"Time to fill histograms: {toc - tic:.2f} seconds")
        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        pass
