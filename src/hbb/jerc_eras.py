"""JERC payload configuration.

Corrections stored in `hbb/data/jerc/`, copied from:

    /cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2026-07-16/
            jet_jerc.json.gz      (AK4PFPuppi)
            fatJet_jerc.json.gz   (AK8PFPuppi)
    /cvmfs/cms-griddata.cern.ch/cat/metadata/JME/JER-Smearing/2025-11-03/
            jer_smear.json.gz
"""

from __future__ import annotations

jerc_files = {
    "AK4": "jet_jerc.json.gz",
    "AK8": "fatJet_jerc.json.gz",
}
jer_smear_file = "jer_smear.json.gz"

jec_tags = {
    "2024": "Summer24Prompt24_V5",
}

jer_tags = {
    "2024": "Summer24Prompt24_JRV2",
}

jet_algos = {
    "AK4": "AK4PFPuppi",
    "AK8": "AK8PFPuppi",
}

# jet radius, for the JER gen-matching criterion (dR < R/2)
cone_sizes = {
    "AK4": 0.4,
    "AK8": 0.8,
}

jerc_variations = {
    "JES": "JES_jes",
    "JER": "JER",
    "UES": "MET_UnclusteredEnergy",
}
