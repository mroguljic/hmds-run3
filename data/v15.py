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

def get_datasets():
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
                    # runs 386694-386951 (end of 2024I) are only in the _v2 processing
                    "/JetMET0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD",
                    "/JetMET1/Run2024I-MINIv6NANOv15_v2-v2/NANOAOD",
                ]
            },
            "Muon": {
                **{
                    f"Muon_Run2024{era}": [
                        f"/Muon0/Run2024{era}-MINIv6NANOv15-{v0}/NANOAOD",
                        f"/Muon1/Run2024{era}-MINIv6NANOv15-{v1}/NANOAOD",
                    ]
                    for era, v0, v1 in [
                        ("C", "v1", "v1"),
                        ("D", "v1", "v1"),
                        ("E", "v1", "v1"),
                        ("F", "v1", "v1"),
                        ("G", "v1", "v2"),
                        ("H", "v1", "v2"),
                    ]
                },
                "Muon_Run2024I": [
                    "/Muon0/Run2024I-MINIv6NANOv15-v1/NANOAOD",
                    "/Muon1/Run2024I-MINIv6NANOv15-v1/NANOAOD",
                    # runs 386694-386951 (end of 2024I) are only in the _v2 processing
                    "/Muon0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD",
                    "/Muon1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD",
                ],
            },
            "QCD": {
                **{
                    f"QCD_HT-{qbin}": f"/QCD-4Jets_Bin-HT-{qbin}_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
                    for qbin in qcd_ht_bins
                },
            },
            "TT": {
                "TTto2L2Nu": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "TTto4Q": "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "TTtoLNu2Q": "/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
            "Diboson": {
                "WW": "/WW_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WZ": "/WZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "ZZ": "/ZZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
            "VJets_had_LO": {
                # Wto2Q - New Binning
                "Wto2Q-3Jets_Bin-HT-100to400": "/Wto2Q-3Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_Bin-HT-400to800": "/Wto2Q-3Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_Bin-HT-800to1500": "/Wto2Q-3Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Wto2Q-3Jets_Bin-HT-1500to2500": "/Wto2Q-3Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "Wto2Q-3Jets_Bin-HT-2500": "/Wto2Q-3Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                # Zto2Q - New Binning
                "Zto2Q-4Jets_Bin-HT-100to400": "/Zto2Q-4Jets_Bin-HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_Bin-HT-400to800": "/Zto2Q-4Jets_Bin-HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_Bin-HT-800to1500": "/Zto2Q-4Jets_Bin-HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_Bin-HT-1500to2500": "/Zto2Q-4Jets_Bin-HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "Zto2Q-4Jets_Bin-HT-2500": "/Zto2Q-4Jets_Bin-HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
            },
            # backgrounds for the mu+jets ttbar CR (control-tt)
            "SingleTop": {
                **{
                    f"{tw}to{decay}": f"/{tw}to{decay}_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
                    for tw in ["TWminus", "TbarWplus"]
                    for decay in ["4Q", "LNu2Q", "2L2Nu"]
                },
                **{
                    f"{t}toLNu-t-channel-4FS": f"/{t}toLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM"
                    for t in ["TBbarQ", "TbarBQ"]
                },
            },
            "VJets_lep_NLO": {
                "WtoLNu-2Jets_Bin-1J-PTLNu-40to100": "/WtoLNu-2Jets_Bin-1J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "WtoLNu-2Jets_Bin-1J-PTLNu-100to200": "/WtoLNu-2Jets_Bin-1J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "WtoLNu-2Jets_Bin-1J-PTLNu-200to400": "/WtoLNu-2Jets_Bin-1J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-1J-PTLNu-400to600": "/WtoLNu-2Jets_Bin-1J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-1J-PTLNu-600": "/WtoLNu-2Jets_Bin-1J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-40to100": "/WtoLNu-2Jets_Bin-2J-PTLNu-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-100to200": "/WtoLNu-2Jets_Bin-2J-PTLNu-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-200to400": "/WtoLNu-2Jets_Bin-2J-PTLNu-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-400to600": "/WtoLNu-2Jets_Bin-2J-PTLNu-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
                "WtoLNu-2Jets_Bin-2J-PTLNu-600": "/WtoLNu-2Jets_Bin-2J-PTLNu-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM",
            },
        }
    }