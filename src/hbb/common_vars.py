from __future__ import annotations

DATA_SAMPLES = ["JetMET", "Muon", "ParkingHH", "ParkingSingleMuon", "ParkingVBF"]

LUMI = {
    "2022": 7980.5,
    "2022EE": 26671.6,
    "2023": 18084.4,
    "2023BPix": 9692.1,
    "2024": 109080.0,
    "2022-2023": 62428.6,
    "2022-2024": 171508.6
}

# SR nSV working point (nSV > NSV_MIN) for the SV definition ntracks >= 3 & dlenSig > 4
# (processors/objects.py::sv_quality_mask). Skims made before that definition have FatJet0_nSV = all SVs
NSV_MIN = 4

norm_preserving_weights = ["genweight", "pileup", "ISRPartonShower", "FSRPartonShower"]

data_key = "data"
