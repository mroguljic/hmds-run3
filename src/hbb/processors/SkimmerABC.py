"""
Skimmer Base Class - common functions for all skimmers.
Author(s): Raghav Kansal
"""

from __future__ import annotations

import logging
from abc import abstractmethod

from coffea import processor
import awkward as ak
import numpy as np

from hbb.common_vars import LUMI

logging.basicConfig(level=logging.INFO)


class SkimmerABC(processor.ProcessorABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).

    Args:
        xsecs (dict, optional): sample cross sections,
          if sample not included no lumi and xsec will not be applied to weights
    """

    XSECS = None

    def get_dataset_norm(self, year, dataset):
        """
        Cross section * luminosity normalization for a given dataset and year.
        This still needs to be normalized with the acceptance of the pre-selection in post-processing.
        (Done in postprocessing/utils.py:load_samples())
        """
        if dataset not in self.XSECS:
            # A missing xsec silently normalizing by 1 is never what we want
            raise KeyError(
                f"Dataset '{dataset}' not found in XSECS (hbb/xsecs.py); Add its cross section before skimming."
            )

        xsec = self.XSECS[dataset]
        weight_norm = xsec * LUMI[year]
        logging.info(f"XSEC: {xsec}, LUMI: {LUMI[year]}")
        print("weight_norm", weight_norm)

        return weight_norm

    def normalize(self, val, cut):
        """
        Fills dak.array nones with nan and applies selection cut
        Used for filling hist.Hist, 
        where the events with nones in the object collections throw errors on fill during compute
        """
        if cut is None:
            ar = ak.fill_none(val, np.nan)
            return ar
        else:
            ar = ak.fill_none(val[cut], np.nan)
            return ar
