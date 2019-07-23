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
        vfom1_heat (numpy.array): VFOM heatmap for 1D binning with param1.
            shape = (VFOM length, _n_bins1)
        _fom1_hist (numpy.array): 1D FOM array for 1D binning with param1.
            shape = (_n_bins1,)
        _count1_hist (numpy.array): 1D count array for 1D binning with param1.
            shape = (_n_bins1,)
        _vfom2_heat (numpy.array): VFOM heatmap for 1D binning with param2.
            shape = (VFOM length, _n_bins2)
        _fom2_hist (numpy.array): 1D FOM array for 1D binning with param2.
            shape = (_n_bins2,)
        _count2_hist (numpy.array): 1D count array for 1D binning with param2.
            shape = (_n_bins2,)
        _fom12_heat (numpy.array): FOM heatmap for 2D binning.
            shape = (_n_bins2, _n_bins1)
        _count12_heat (numpy.array): count heatmap for 2D binning.
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
        self._vfom1_heat = None
        self._fom1_hist = None
        self._count1_hist = None
        self._vfom2_heat = None
        self._fom2_hist = None
        self._count2_hist = None

        # 2D binning
        self._fom12_heat = None
        self._count12_heat = None

        # used to check whether pump-probe FOM is available
        self._pp_fail_flag = 0

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

    @profiler("Binning Processor")
    def process(self, data):
        processed = data['processed']
        tid = processed.tid
        raw = data['raw']

        # for speed
        bin1 = processed.bin.bin1
        bin2 = processed.bin.bin2
        bin12 = processed.bin.bin12

        # try to get the FOM
        # raise ProcessingError if FOM is not available
        fom, has_vfom, x, vfom, x_label, vfom_label = \
            self._get_analysis_ret(processed)

        if fom is None:
            # needed for pump-probe
            return

        if has_vfom:
            # reset all when VFOM length changes
            if self._vfom1_heat is not None and len(x) != self._vfom1_heat.shape[0]:
                self._reset1 = True
                self._reset2 = True
            elif self._vfom2_heat is not None and len(x) != self._vfom2_heat.shape[0]:
                self._reset1 = True
                self._reset2 = True

        # reset data

        if self._reset1:
            self._fom1_hist = np.zeros(self._n_bins1, dtype=np.float32)
            self._count1_hist = np.zeros(self._n_bins1, dtype=np.uint32)
            if has_vfom:
                self._vfom1_heat = np.zeros(
                    (len(x), self._n_bins1), dtype=np.float32)
            else:
                self._vfom1_heat = None

            bin1.updated = True

        if self._reset2:
            self._fom2_hist = np.zeros(self._n_bins2, dtype=np.float32)
            self._count2_hist = np.zeros(self._n_bins2, dtype=np.uint32)
            if has_vfom:
                self._vfom2_heat = np.zeros(
                    (len(x), self._n_bins2), dtype=np.float32)
            else:
                self._vfom2_heat = None

            bin2.updated = True

        if self._reset1 or self._reset2:
            self._fom12_heat = np.zeros(
                (self._n_bins2, self._n_bins1), dtype=np.float32)
            self._count12_heat = np.zeros(
                (self._n_bins2, self._n_bins1), dtype=np.float32)

            bin12.updated = True

        self._reset1 = False
        self._reset2 = False

        # try to get slow data.

        err_msgs = []

        iloc1, err = self._get_iloc(
            tid, raw, self._device_id1, self._property1, self._edge1)
        if err:
            err_msgs.append(err)

        iloc2, err = self._get_iloc(
            tid, raw, self._device_id2, self._property2, self._edge2)
        if err:
            err_msgs.append(err)

        # 1D binning

        if 0 <= iloc1 < self._n_bins1:
            self._update_fom_hist_1d(
                iloc1, fom, self._fom1_hist, self._count1_hist)

            self._update_vfom_heat_1d(
                iloc1, vfom, self._vfom1_heat, self._count1_hist)

            bin1.updated = True

        if 0 <= iloc2 < self._n_bins2:
            self._update_fom_hist_1d(
                iloc2, fom, self._fom2_hist, self._count2_hist)

            self._update_vfom_heat_1d(
                iloc2, vfom, self._vfom2_heat, self._count2_hist)

            bin2.updated = True

        # 2D binning

        if 0 <= iloc1 < self._n_bins1 and 0 <= iloc2 < self._n_bins2:
            self._update_fom_heat_2d(
                iloc1, iloc2, fom, self._fom12_heat, self._count12_heat)

            bin12.updated = True

        bin1.center = self._center1
        bin1.label = f"{self._device_id1} | {self._property1}"
        bin1.has_vfom = has_vfom
        bin1.vfom_heat = self._vfom1_heat
        bin1.vfom_label = vfom_label
        bin1.vfom_x = x
        bin1.vfom_x_label = x_label
        bin1.fom_hist = self._fom1_hist
        bin1.count_hist = self._count1_hist

        bin2.center = self._center2
        bin2.label = f"{self._device_id2} | {self._property2}"
        bin2.has_vfom = has_vfom
        bin2.vfom_heat = self._vfom2_heat
        bin2.vfom_label = vfom_label
        bin2.vfom_x = x
        bin2.vfom_x_label = x_label
        bin2.fom_hist = self._fom2_hist
        bin2.count_hist = self._count2_hist

        bin12.center_x = self._center1
        bin12.center_y = self._center2
        bin12.x_label = bin1.label
        bin12.y_label = bin2.label
        bin12.fom_heat = self._fom12_heat
        bin12.count_heat = self._count12_heat

        if err_msgs:
            raise ProcessingError(f"[Binning] {err_msgs[0]}")

    def _get_analysis_ret(self, processed):
        err = ''

        analysis_type = self.analysis_type
        if analysis_type == AnalysisType.PUMP_PROBE:
            ret = processed.pp
            if ret.fom is None:
                self._pp_fail_flag += 1
                # if on/off pulses are in different trains, pump-probe FOM is
                # only calculated every other train.
                if self._pp_fail_flag == 2:
                    self._pp_fail_flag = 0
                    err = "Pump-probe result is not available"
            else:
                self._pp_fail_flag = 0
        elif analysis_type == AnalysisType.ROI1:
            ret = processed.roi.roi1
            if ret.fom is None:
                err = "ROI1 sum result is not available"
        elif analysis_type == AnalysisType.ROI2:
            ret = processed.roi.roi2
            if ret.fom is None:
                err = "ROI2 sum result is not available"
        elif analysis_type == AnalysisType.ROI1_SUB_ROI2:
            ret = processed.roi.roi1_sub_roi2
            if ret.fom is None:
                err = "ROI1 - ROI2 sum result is not available"
        elif analysis_type == AnalysisType.ROI1_ADD_ROI2:
            ret = processed.roi.roi1_add_roi2
            if ret.fom is None:
                err = "ROI1 + ROI2 sum result is not available"
        elif analysis_type == AnalysisType.PROJ_ROI1:
            ret = processed.roi.proj1
            if ret.fom is None:
                err = "ROI1 projection result is not available"
        elif analysis_type == AnalysisType.PROJ_ROI2:
            ret = processed.roi.proj2
            if ret.fom is None:
                err = "ROI2 projection result is not available"
        elif analysis_type == AnalysisType.PROJ_ROI1_SUB_ROI2:
            ret = processed.roi.proj1_sub_proj2
            if ret.fom is None:
                err = "ROI1 - ROI2 projection result is not available"
        elif analysis_type == AnalysisType.PROJ_ROI1_ADD_ROI2:
            ret = processed.roi.proj1_add_proj2
            if ret.fom is None:
                err = "ROI1 + ROI2 projection result is not available"
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            ret = processed.ai
            if ret.fom is None:
                err = "Azimuthal integration result is not available"
        else:  # self.analysis_type == AnalysisType.UNDEFINED
            ret = None

        if err:
            raise ProcessingError(f"[Binning] {err}")

        if ret is None:
            return None, None, None, None, None, None

        return ret.fom, ret.has_vfom, ret.x, ret.vfom, ret.x_label, ret.vfom_label

    def _get_iloc(self, tid, raw, device_id, property, bin_edge):
        slow, err = _get_slow_data(tid, raw, device_id, property)

        if slow is None:
            iloc = -1
        else:
            iloc = np.searchsorted(bin_edge, slow) - 1

        return iloc, err

    def _update_fom_hist_1d(self, iloc, fom, fom_hist, count_hist):
        if fom is None:
            return

        count_hist[iloc] += 1
        if self._mode == BinMode.ACCUMULATE:
            fom_hist[iloc] += fom
        else:  # self._mode == BinMode.AVERAGE
            fom_hist[iloc] += (fom - fom_hist[iloc]) / count_hist[iloc]

    def _update_vfom_heat_1d(self, iloc, vfom, vfom_heat, count_hist):
        if vfom is None:
            return

        if self._mode == BinMode.ACCUMULATE:
            vfom_heat[:, iloc] += vfom
        else:  # self._mode == BinMode.AVERAGE
            vfom_heat[:, iloc] += \
                (vfom - vfom_heat[:, iloc]) / count_hist[iloc]

    def _update_fom_heat_2d(self, iloc1, iloc2, fom, fom_heat, count_heat):
        if fom is None:
            return

        count_heat[iloc2, iloc1] += 1
        if self._mode == BinMode.ACCUMULATE:
            fom_heat[iloc2, iloc1] += fom
        else:  # self._mode == BinMode.AVERAGE
            fom_heat[iloc2, iloc1] += \
                fom - fom_heat[iloc2, iloc1] / count_heat[iloc2, iloc1]
