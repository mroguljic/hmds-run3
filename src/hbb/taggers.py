b_taggers = {
    "2022": {
        "AK4": {
            "btagPNetB" : {
                "L" : 0.047,
                "M" : 0.245,
                "T" : 0.6734,
                "XT" : 0.7862,
                "XXT" : 0.961
                }
            }

        },
    "2022EE": {
        "AK4": {
            "btagPNetB" : {
                "L" : 0.0499,
                "M" : 0.2605,
                "T" : 0.6915,
                "XT" : 0.8033,
                "XXT" : 0.9664
                }
            }

        },
    "2023": {
        "AK4": {
            "btagPNetB" : {
                "L" : 0.0358,
                "M" : 0.1917,
                "T" : 0.6172,
                "XT" : 0.7515,
                "XXT" : 0.9659
                }
            }

        },
    "2023BPix": {
        "AK4": {
            "btagPNetB" : {
                "L" : 0.0359,
                "M" : 0.1919,
                "T" : 0.6133,
                "XT" : 0.7544,
                "XXT" : 0.9688
                }
            }

        },
    #Will need to be changed: https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer24/
    #From the 2024 campaign onwards, only the UParTAK4 tagger is supported, and only json files are provided.
    #UParT taggers do not exist in PFNano_v2 so we use btagPNetB for now, with values copied from 2023BPix
    "2024": {
        "AK4": {
            "btagPNetB" : {
                "L" : 0.0359,
                "M" : 0.1919,
                "T" : 0.6133,
                "XT" : 0.7544,
                "XXT" : 0.9688
                }
            }

        },
}
