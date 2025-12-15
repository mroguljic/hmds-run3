from __future__ import annotations

qcd_ht_bins = [
    "100to200",
    "200to400",
    "400to600",
    "600to800",
    "800to1000",
    "1000to1200",
    "1200to1500",
    "1500to2000",
    "2000",
]

def get_datasets():  ## NanoAOD for now, will need to switch to PRIVATE PFNano production for jet foundation model
    return {
        "2024": {
            "JetMET": {
                "JetMET_Run2024C": [
                    "/JetMET0/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                    "/JetMET1/Run2024C-MINIv6NANOv15-v1/NANOAOD",
                ],
                "JetMET_Run2024D": [
                    "/JetMET0/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                    "/JetMET1/Run2024D-MINIv6NANOv15-v1/NANOAOD",
                ],
                "JetMET_Run2024E": [
                    "/JetMET0/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                    "/JetMET1/Run2024E-MINIv6NANOv15-v1/NANOAOD",
                ],
                "JetMET_Run2024F": [
                    "/JetMET0/Run2024F-MINIv6NANOv15-v2/NANOAOD",
                    "/JetMET1/Run2024F-MINIv6NANOv15-v2/NANOAOD",
                ],
                "JetMET_Run2024G": [
                    "/JetMET0/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                    "/JetMET1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
                ],
                "JetMET_Run2024H": [
                    "/JetMET0/Run2024H-MINIv6NANOv15-v2/NANOAOD",
                    "/JetMET1/Run2024H-MINIv6NANOv15-v2/NANOAOD",
                ],
                "JetMET_Run2024I": [
                    "/JetMET0/Run2024I-MINIv6NANOv15-v2/NANOAOD",
                    "/JetMET1/Run2024I-MINIv6NANOv15-v1/NANOAOD",
                ]
            },
            "QCD": { #TODO check 100-200
                **{
                    f"QCD_HT-{qbin}": f"/QCD-4Jets_Bin-HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
            },
            "TT": {
                "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
        }
    }
