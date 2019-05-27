"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

BinProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import (
    CompositeProcessor, LeafProcessor, SharedProperty,
    StopCompositionProcessing
)
from ..exceptions import ProcessingError
from ...metadata import Metadata as mt
from ...config import AnalysisType, BinMode
from ...helpers import profiler


class BinProcessor(CompositeProcessor):
    """BinProcessor class.

    Bing data.

    Attributes:
        n_bins (int): number of bins.
        bin_range (tuple): bin range.
        analysis_type (AnalysisType): binning analysis type.
        mode (BinMode): binning mode.

    """
    n_bins = SharedProperty()
    bin_range = SharedProperty()
    analysis_type = SharedProperty()
    mode = SharedProperty()

    class Data:
        def __init__(self):
            self.bins = None
            self.centers = None
            self.counts = None
            self.values = None

    def __init__(self):
        super().__init__()

        self._data = self.Data()

        self.add(BinGeneralProcessor())

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.BIN_PROC)

        reset = False
        if 'reset' in cfg:
            self._meta.delete(mt.BIN_PROC, 'reset')
            reset = True

        analysis_type = AnalysisType(int(cfg['analysis_type']))
        self._update_analysis(analysis_type)

        self.mode = AnalysisType(int(cfg['mode']))

        n_bins = int(cfg['n_bins'])
        bin_range = self.str2tuple(cfg['bin_range'])

        if reset or n_bins != self.n_bins or bin_range != self.bin_range:
            bins = np.linspace(bin_range[0], bin_range[1], n_bins+1)

            self._data.bins = bins
            self._data.centers = (bins[1:] + bins[:-1])/2.0
            self._data.counts = np.zeros(n_bins)
            self._data.values = None

            self.n_bins = n_bins
            self.bin_range = bin_range


class BinGeneralProcessor(LeafProcessor):
    @profiler("Bin image processor")
    def process(self, processed, raw=None):
        bins = self._data.bins
        centers = self._data.centers
        counts = self._data.counts
        values = self._data.values

        # get the bin value
        bin_v = bins[0] + (bins[-1] - bins[0]) * np.random.rand()
        index = np.digitize(bin_v, bins)
        if len(bins) >= index > 0:
            counts[index-1] += 1

            if self._has_analysis(AnalysisType.AZIMUTHAL_INTEG):
                new_value = processed.ai.intensity_mean
                x = processed.ai.momentum
            else:
                return

            if new_value is None:
                return

            if values is None:
                # initialization
                values = np.zeros((len(counts), len(new_value)))
                values[index-1][:] = new_value
            else:
                values[index-1][:] += new_value

            processed.bin.centers = centers
            processed.bin.counts = counts
            processed.bin.x = x
            processed.bin.values = values
