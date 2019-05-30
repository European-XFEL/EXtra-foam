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
    CompositeProcessor, _get_slow_data, LeafProcessor, SharedProperty,
    StopCompositionProcessing
)
from ..exceptions import ProcessingError
from ...metadata import Metadata as mt
from ...config import AnalysisType, BinMode
from ...utils import profiler


class BinProcessor(CompositeProcessor):
    """BinProcessor class.

    Bing data.

    Attributes:
        analysis_type (AnalysisType): binning analysis type.
        mode (BinMode): binning mode.
        device_id_x (str): device ID x.
        property_x (str): property of the device x.
        n_bins_x (int): number of x bins.
        bin_range_x (tuple): bin x range.
        device_id_y (str): device ID y.
        property_y (str): property of the device y.
        n_bins_y (int): number of y bins.
        bin_range_y (tuple): bin y range.
    """
    analysis_type = SharedProperty()
    mode = SharedProperty()
    device_id_x = SharedProperty()
    device_id_y = SharedProperty()
    property_x = SharedProperty()
    property_y = SharedProperty()
    n_bins_x = SharedProperty()
    n_bins_y = SharedProperty()
    bin_range_x = SharedProperty()
    bin_range_y = SharedProperty()

    class Data:
        def __init__(self):
            self.bins_x = None
            self.bins_y = None
            self.centers_x = None
            self.centers_y = None
            self.counts_x = None
            self.counts_y = None
            self.values = None

    def __init__(self):
        super().__init__()

        self._data = self.Data()

        self.add(Bin1DProcessor())
        self.add(Bin2DProcessor())

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.BIN_PROC)

        reset = False
        if 'reset' in cfg:
            self._meta.delete(mt.BIN_PROC, 'reset')
            reset = True

        self._update_analysis(AnalysisType(int(cfg['analysis_type'])))

        self.mode = AnalysisType(int(cfg['mode']))

        self.device_id_x = cfg['device_id_x']
        self.property_x = cfg['property_x']
        if not self.device_id_x or not self.property_x:
            # x is required for both 1D and 2D binning
            raise StopCompositionProcessing
        n_bins_x = int(cfg['n_bins_x'])
        bin_range_x = self.str2tuple(cfg['bin_range_x'])

        if reset or n_bins_x != self.n_bins_x \
                or bin_range_x != self.bin_range_x:

            bins_x = np.linspace(bin_range_x[0], bin_range_x[1], n_bins_x+1)

            self._data.bins_x = bins_x
            self._data.centers_x = (bins_x[1:] + bins_x[:-1])/2.0
            self._data.counts_x = np.zeros(n_bins_x)
            self._data.values = None

            self.n_bins_x = n_bins_x
            self.bin_range_x = bin_range_x


class Bin1DProcessor(LeafProcessor):
    """Bin data based to the x parameter."""
    @profiler("Bin 1D processor")
    def process(self, processed, raw=None):
        bins = self._data.bins_x
        centers = self._data.centers_x
        counts = self._data.counts_x
        values = self._data.values

        # get the group value
        device_id = self.device_id_x
        ppt = self.property_x

        group_v = _get_slow_data(processed.tid, raw, device_id, ppt)

        index = np.digitize(group_v, bins)
        if len(centers) >= index > 0:
            counts[index-1] += 1

            if self.analysis_type == AnalysisType.AZIMUTHAL_INTEG:
                new_value = processed.ai.intensity_mean
                x = processed.ai.momentum
            else:
                return

            if new_value is None:
                return

            if values is None:
                # initialization
                values = np.zeros((len(counts), len(new_value)))
                self._data.values = values
                values[index-1][:] = new_value
            else:
                values[index-1][:] += new_value

            processed.bin.centers_x = centers
            processed.bin.counts_x = counts
            processed.bin.x = x
            processed.bin.values = values


class Bin2DProcessor(LeafProcessor):
    pass
