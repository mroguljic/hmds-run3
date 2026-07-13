from __future__ import annotations

year_map = {
    "2022": ["2022"],
    "2022EE": ["2022EE"],
    "2023": ["2023"],
    "2023BPix": ["2023BPix"],
    "2022-2023": ["2022", "2022EE", "2023", "2023BPix"],
    "2024": ["2024"],
}

# HMDS signal + backgrounds actually skimmed for the dark-shower "signal-all" region.
# Subsample names must match what's under EOS, e.g. <tag>/2024/<subsample>/parquet/nominal/signal-all/.
common_mc = {
    "Signal": {"Signal"},
    "qcd": {
        "QCD_HT-100to200",
        "QCD_HT-200to400",
        "QCD_HT-400to600",
        "QCD_HT-600to800",
        "QCD_HT-800to1000",
        "QCD_HT-1000to1200",
        "QCD_HT-1200to1500",
        "QCD_HT-1500to2000",
        "QCD_HT-2000",
    },
    "tt": {"TTto2L2Nu", "TTto4Q", "TTtoLNu2Q"},
    "diboson": {"WW", "WZ", "ZZ"},
    "wjets": {
        "Wto2Q-3Jets_Bin-HT-100to400",
        "Wto2Q-3Jets_Bin-HT-400to800",
        "Wto2Q-3Jets_Bin-HT-800to1500",
        "Wto2Q-3Jets_Bin-HT-1500to2500",
        "Wto2Q-3Jets_Bin-HT-2500",
    },
    "zjets": {
        "Zto2Q-4Jets_Bin-HT-100to400",
        "Zto2Q-4Jets_Bin-HT-400to800",
        "Zto2Q-4Jets_Bin-HT-800to1500",
        "Zto2Q-4Jets_Bin-HT-1500to2500",
        "Zto2Q-4Jets_Bin-HT-2500",
    },
}

data_by_year = {
    "2022": {
        "JetMET_Run2022C_single",
        "JetMET_Run2022C",
        "JetMET_Run2022D",
    },
    "2022EE": {
        "JetMET_Run2022E",
        "JetMET_Run2022F",
        "JetMET_Run2022G",
    },
    "2023": {
        "JetMET_Run2023Cv1",
        "JetMET_Run2023Cv2",
        "JetMET_Run2023Cv3",
        "JetMET_Run2023Cv4",
    },
    "2023BPix": {
        "JetMET_Run2023D",
    },
    "2024": {
        "JetMET_Run2024C",
        "JetMET_Run2024D",
        "JetMET_Run2024E",
        "JetMET_Run2024F",
        "JetMET_Run2024G",
        "JetMET_Run2024H",
        "JetMET_Run2024I",
    },
}

# --- ADDED for control-tt region ---
data_by_year_muon = {
    "2022": {"Muon_Run2022C", "Muon_Run2022D"},
    "2022EE": {"Muon_Run2022E", "Muon_Run2022F", "Muon_Run2022G"},
    "2023": {"Muon_Run2023Cv1", "Muon_Run2023Cv2", "Muon_Run2023Cv3", "Muon_Run2023Cv4"},
    "2023BPix": {"Muon_Run2023D"},
}

# --- ADDED for control-zgamma region ---
data_by_year_zgamma = {
    "2022": {"EGamma_Run2022C", "EGamma_Run2022D"},
    "2022EE": {"EGamma_Run2022E", "EGamma_Run2022F", "EGamma_Run2022G"},
    "2023": {"EGamma_Run2023Cv1", "EGamma_Run2023Cv2", "EGamma_Run2023Cv3", "EGamma_Run2023Cv4"},
    "2023BPix": {"EGamma_Run2023D"},
}
