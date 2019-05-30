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
            self.edge_x = None
            self.edge_y = None
            self.center_x = None
            self.center_y = None
            self.count_x = None
            self.count_y = None
            self.data_x = None
            self.data_y = None

    def __init__(self):
        super().__init__()

        self._data = self.Data()

        self.add(Bin1DProcessorX())
        self.add(Bin1DProcessorY())
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
        self.device_id_y = cfg['device_id_y']
        self.property_y = cfg['property_y']

        n_bins_x = int(cfg['n_bins_x'])
        bin_range_x = self.str2tuple(cfg['bin_range_x'])
        n_bins_y = int(cfg['n_bins_y'])
        bin_range_y = self.str2tuple(cfg['bin_range_y'])

        if reset or n_bins_x != self.n_bins_x \
                or bin_range_x != self.bin_range_x:

            edge_x = np.linspace(bin_range_x[0], bin_range_x[1], n_bins_x+1)

            self._data.edge_x = edge_x
            self._data.center_x = (edge_x[1:] + edge_x[:-1])/2.0
            self._data.count_x = np.zeros(n_bins_x)
            self._data.data_x = None

            self.n_bins_x = n_bins_x
            self.bin_range_x = bin_range_x

        if reset or n_bins_y != self.n_bins_y \
                or bin_range_y != self.bin_range_y:

            edge_y = np.linspace(bin_range_y[0], bin_range_y[1], n_bins_y+1)

            self._data.edge_y = edge_y
            self._data.center_y = (edge_y[1:] + edge_y[:-1])/2.0
            self._data.count_y = np.zeros(n_bins_y)
            self._data.data_y = None

            self.n_bins_y = n_bins_y
            self.bin_range_y = bin_range_y


class Bin1DProcessorX(LeafProcessor):
    """Bin data based to the x parameter."""
    @profiler("Bin 1D processor")
    def process(self, processed, raw=None):
        if not self.device_id_x or not self.property_x:
            return

        edge = self._data.edge_x
        center = self._data.center_x
        count = self._data.count_x
        data_x = self._data.data_x

        # get the group value
        device_id = self.device_id_x
        ppt = self.property_x

        group_v = _get_slow_data(processed.tid, raw, device_id, ppt)

        index = np.digitize(group_v, edge)
        if len(center) >= index > 0:
            count[index-1] += 1

            if self.analysis_type == AnalysisType.AZIMUTHAL_INTEG:
                new_value = processed.ai.intensity_mean
                x = processed.ai.momentum
            else:
                return

            if new_value is None:
                return

            if data_x is None:
                # initialization
                data_x = np.zeros((len(count), len(new_value)))
                self._data.values = data_x
                data_x[index-1][:] = new_value
            else:
                data_x[index-1][:] += new_value

            processed.bin.center_x = center
            processed.bin.count_x = count
            processed.bin.x = x
            processed.bin.data_x = data_x


class Bin1DProcessorY(LeafProcessor):
    """Bin data based to the y parameter."""
    @profiler("Bin 1D processor")
    def process(self, processed, raw=None):
        if not self.device_id_y or not self.property_y:
            return

        edge = self._data.edge_y
        center = self._data.center_y
        count = self._data.count_y
        data_y = self._data.data_y

        # get the group value
        device_id = self.device_id_y
        ppt = self.property_y

        group_v = _get_slow_data(processed.tid, raw, device_id, ppt)

        index = np.digitize(group_v, edge)
        if len(center) >= index > 0:
            count[index-1] += 1

            if self.analysis_type == AnalysisType.AZIMUTHAL_INTEG:
                new_value = processed.ai.intensity_mean
                y = processed.ai.momentum
            else:
                return

            if new_value is None:
                return

            if data_y is None:
                # initialization
                data_y = np.zeros((len(count), len(new_value)))
                self._data.values = data_y
                data_y[index-1][:] = new_value
            else:
                data_y[index-1][:] += new_value

            processed.bin.center_y = center
            processed.bin.count_y = count
            processed.bin.y = y
            processed.bin.data_y = data_y


class Bin2DProcessor(LeafProcessor):
    pass
