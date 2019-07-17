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
    CompositeProcessor, _get_slow_data
)
from ..exceptions import ProcessingError
from ...metadata import Metadata as mt
from ...config import AnalysisType, BinMode
from ...utils import profiler


def compute_bin_edge(n_bins, bin_range):
    edge = np.linspace(bin_range[0], bin_range[1], n_bins+1)
    center = (edge[1:] + edge[:-1]) / 2.0
    return edge, center


class BinProcessor(CompositeProcessor):
    """BinProcessor class.

    Bin data based on 1 and/or 2 scalar data.

    Attributes:
        analysis_type (AnalysisType): binning analysis type.
        _mode (BinMode): binning mode.
        _device_id1 (str): device ID 1.
        _property1 (str): property of the device 1.
        _device_id2 (str): device ID 2.
        _property2 (str): property of the device 2.
        _range1 (tuple): bin 1 range.
        _range2 (tuple): bin 2 range.
        _n_bins1 (int): number of 1 bins.
        _n_bins2 (int): number of 2 bins.
        _edge1 (numpy.array): bin edges for parameter 1.
        _edge2 (numpy.array): bin edges for parameter 2.
        _center1 (numpy.array): bin centers for parameter 1.
        _center2 (numpy.array): bin centers for parameter 2.
        _reset1 (bool): True for resetting history data for param1.
        _reset2 (bool): True for resetting history data for param2.
        _vec1_hist (numpy.array): vector heatmap for 1D binning with param1.
            shape = (vector length, _n_bins1)
        _fom1_hist (numpy.array): 1D FOM array for 1D binning with param1.
            shape = (_n_bins1,)
        _count1_hist (numpy.array): 1D count array for 1D binning with param1.
            shape = (_n_bins1,)
        _vec2_hist (numpy.array): vector heatmap for 1D binning with param2.
            shape = (vector length, _n_bins2)
        _fom2_hist (numpy.array): 1D FOM array for 1D binning with param2.
            shape = (_n_bins2,)
        _count2_hist (numpy.array): 1D count array for 1D binning with param2.
            shape = (_n_bins2,)
        _fom12_hist (numpy.array): FOM heatmap for 2D binning.
            shape = (_n_bins2, _n_bins1)
        _count12_hist (numpy.array): count heatmap for 2D binning.
            shape = (_n_bins2, _n_bins1)
    """

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

        self._mode = None

        self._device_id1 = ''
        self._device_id2 = ''
        self._property1 = ''
        self._property2 = ''

        self._range1 = None
        self._range2 = None
        self._n_bins1 = None
        self._n_bins2 = None
        self._edge1 = None
        self._edge2 = None
        self._center1 = None
        self._center2 = None

        self._reset1 = False
        self._reset2 = False

        # 1D binning
        self._vec1_hist = None
        self._fom1_hist = None
        self._count1_hist = None
        self._vec2_hist = None
        self._fom2_hist = None
        self._count2_hist = None

        # 2D binning
        self._fom12_hist = None
        self._count12_hist = None

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.BIN_PROC)

        if self._update_analysis(AnalysisType(int(cfg['analysis_type']))):
            # reset when analysis type changes
            self._reset1 = True
            self._reset2 = True

        mode = BinMode(int(cfg['mode']))
        if mode != self._mode:
            # reset when bin mode changes
            self._reset1 = True
            self._reset2 = True
            self._mode = mode

        device_id1 = cfg['device_id1']
        property1 = cfg['property1']
        if device_id1 != self._device_id1 or property1 != self._property1:
            self._device_id1 = device_id1
            self._property1 = property1
            # reset when slow data source changes
            self._reset1 = True

        device_id2 = cfg['device_id2']
        property2 = cfg['property2']
        if device_id2 != self._device_id2 or property2 != self._property2:
            self._device_id2 = device_id2
            self._property2 = property2
            # reset when slow data source changes
            self._reset2 = True

        n_bins1 = int(cfg['n_bins1'])
        bin_range1 = self.str2tuple(cfg['bin_range1'])
        if n_bins1 != self._n_bins1 or bin_range1 != self._range1:
            self._edge1, self._center1 = compute_bin_edge(n_bins1, bin_range1)
            self._n_bins1 = n_bins1
            self._range1 = bin_range1
            # reset when number of bins and bin range change
            self._reset1 = True

        n_bins2 = int(cfg['n_bins2'])
        bin_range2 = self.str2tuple(cfg['bin_range2'])
        if n_bins2 != self._n_bins2 or bin_range2 != self._range2:
            self._edge2, self._center2 = compute_bin_edge(n_bins2, bin_range2)
            self._n_bins2 = n_bins2
            self._range2 = bin_range2
            # reset when number of bins and bin range change
            self._reset2 = True

        if 'reset' in cfg:
            self._meta.delete(mt.BIN_PROC, 'reset')
            # reset when commanded by the GUI
            self._reset1 = True
            self._reset2 = True

        # reset history data

        if self._reset1:
            self._fom1_hist = np.zeros(self._n_bins1, dtype=np.float32)
            self._count1_hist = np.zeros(self._n_bins1, dtype=np.uint32)
            # Real initialization could take place later then valid vec
            # is received.
            self._vec1_hist = None

        if self._reset2:
            self._fom2_hist = np.zeros(self._n_bins2, dtype=np.float32)
            self._count2_hist = np.zeros(self._n_bins2, dtype=np.uint32)
            # Real initialization could take place later then valid vec
            # is received.
            self._vec2_hist = None

        if self._reset1 or self._reset2:
            self._fom12_hist = np.zeros((self._n_bins2, self._n_bins1),
                                        dtype=np.float32)
            self._count12_hist = np.zeros((self._n_bins2, self._n_bins1),
                                          dtype=np.uint32)

    @profiler("Binning Processor")
    def process(self, data):
        processed = data['processed']

        # Guarantee initialization and reset arrays in the main process.

        processed.mode = self._mode

        processed.bin.n_bins1 = self._n_bins1
        processed.bin.center1 = self._center1
        processed.bin.edge1 = self._edge1
        processed.bin.label1 = self._device_id1 + ":" + self._property1
        processed.bin.reset1 = self._reset1
        self._reset1 = False

        processed.bin.n_bins2 = self._n_bins2
        processed.bin.center2 = self._center2
        processed.bin.edge2 = self._edge2
        processed.bin.label2 = self._device_id2 + ":" + self._property2
        processed.bin.reset2 = self._reset2
        self._reset2 = False

        # Try to get FOM first.

        if self.analysis_type == AnalysisType.PUMP_PROBE:
            ret = processed.pp
            # Don't raise an Exception here if fom is None since it does not
            # work well if on- and off- pulses are in different trains.
        elif self.analysis_type == AnalysisType.ROI1:
            ret = processed.roi.roi1
            if ret.fom is None:
                raise ProcessingError("ROI1 sum result is not available")
        elif self.analysis_type == AnalysisType.ROI2:
            ret = processed.roi.roi2
            if ret.fom is None:
                raise ProcessingError("ROI2 sum result is not available")
        elif self.analysis_type == AnalysisType.ROI1_SUB_ROI2:
            ret = processed.roi.roi1_sub_roi2
            if ret.fom is None:
                raise ProcessingError(
                    "ROI1 - ROI2 sum result is not available")
        elif self.analysis_type == AnalysisType.ROI1_ADD_ROI2:
            ret = processed.roi.roi1_add_roi2
            if ret.fom is None:
                raise ProcessingError(
                    "ROI1 + ROI2 sum result is not available")
        elif self.analysis_type == AnalysisType.PROJ_ROI1:
            ret = processed.roi.proj1
            if ret.fom is None:
                raise ProcessingError(
                    "ROI1 projection result is not available")
        elif self.analysis_type == AnalysisType.PROJ_ROI2:
            ret = processed.roi.proj2
            if ret.fom is None:
                raise ProcessingError(
                    "ROI2 projection result is not available")
        elif self.analysis_type == AnalysisType.PROJ_ROI1_SUB_ROI2:
            ret = processed.roi.proj1_sub_proj2
            if ret.fom is None:
                raise ProcessingError(
                    "ROI1 - ROI2 projection result is not available")
        elif self.analysis_type == AnalysisType.PROJ_ROI1_ADD_ROI2:
            ret = processed.roi.proj1_add_proj2
            if ret.fom is None:
                raise ProcessingError(
                    "ROI1 + ROI2 projection result is not available")
        elif self.analysis_type == AnalysisType.TRAIN_AZIMUTHAL_INTEG:
            ret = processed.ai
            if ret.fom is None:
                raise ProcessingError(
                    "Azimuthal integration result is not available")
        else:
            return

        vec_x = ret.x
        vec = ret.vfom
        fom = ret.fom
        vec_label = ret.x_label

        if fom is None:
            # If it is not available, we stop getting slow data.
            return

        # try to get slow data.

        iloc1 = -1
        iloc2 = -1
        error_messages = []
        if self._device_id1 and self._property1:
            try:
                slow1, _ = _get_slow_data(processed.tid,
                                          data['raw'],
                                          self._device_id1,
                                          self._property1)

                iloc1 = np.searchsorted(self._edge1, slow1) - 1
            except ProcessingError as e:
                error_messages.append(repr(e))

        if self._device_id2 and self._property2:
            try:
                slow2, _ = _get_slow_data(processed.tid,
                                          data['raw'],
                                          self._device_id2,
                                          self._property2)

                iloc2 = np.searchsorted(self._edge2, slow2) - 1
            except ProcessingError as e:
                error_messages.append(repr(e))

        # 1d binning

        if 0 <= iloc1 < self._n_bins1:
            self._count1_hist[iloc1] += 1
            if self._mode == BinMode.ACCUMULATE:
                self._fom1_hist[iloc1] += fom
            else:  # self._mode == BinMode.AVERAGE
                self._fom1_hist[iloc1] += \
                    (fom - self._fom1_hist[iloc1]) / self._count1_hist[iloc1]

            processed.bin.iloc1 = iloc1
            processed.bin.fom1 = self._fom1_hist[iloc1]

            if vec is not None:
                if self._vec1_hist is None or len(vec_x) != self._vec1_hist.shape[0]:
                    # initialization
                    self._vec1_hist = np.zeros(
                        (len(vec_x), self._n_bins1), dtype=np.float32)

                if self._mode == BinMode.ACCUMULATE:
                    self._vec1_hist[:, iloc1] += vec
                else:  # self._mode == BinMode.AVERAGE
                    self._vec1_hist[:, iloc1] += \
                        (vec - self._vec1_hist[:, iloc1]) / self._count1_hist[iloc1]

                processed.bin.vec1 = self._vec1_hist[:, iloc1]

        if 0 <= iloc2 < self._n_bins2:
            self._count2_hist[iloc2] += 1

            if self._mode == BinMode.ACCUMULATE:
                self._fom2_hist[iloc2] += fom
            else:  # self._mode == BinMode.AVERAGE
                self._fom2_hist[iloc2] += \
                    (fom - self._fom2_hist[iloc2]) / self._count2_hist[iloc2]

            processed.bin.iloc2 = iloc2
            processed.bin.fom2 = self._fom2_hist[iloc2]

            if vec is not None:
                if self._vec2_hist is None or len(vec_x) != self._vec2_hist.shape[0]:
                    # initialization
                    self._vec2_hist = np.zeros(
                        (len(vec_x), self._n_bins2), dtype=np.float32)

                if self._mode == BinMode.ACCUMULATE:
                    self._vec2_hist[:, iloc2] += vec
                else:  # self._mode == BinMode.AVERAGE
                    self._vec2_hist[:, iloc2] += \
                        (vec - self._vec2_hist[:, iloc2]) / self._count2_hist[iloc2]

                processed.bin.vec2 = self._vec2_hist[:, iloc2]

        # 2D binning
        if 0 <= iloc1 < self._n_bins1 and 0 <= iloc2 < self._n_bins2:
            self._count12_hist[iloc2, iloc1] += 1
            if self._mode == BinMode.ACCUMULATE:
                self._fom12_hist[iloc2, iloc1] += fom
            else:  # self._mode == BinMode.AVERAGE
                self._fom12_hist[iloc2, iloc1] += \
                    (fom - self._fom12_hist[iloc2, iloc1]) / self._count12_hist[iloc2, iloc1]

            processed.bin.fom12 = self._fom12_hist[iloc2, iloc1]

        processed.bin.vec_x = vec_x
        processed.bin.vec_label = vec_label

        for msg in error_messages:
            raise ProcessingError(msg)
