"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from scipy import stats

from .base_processor import _BaseProcessor, SimpleSequence, SimpleVectorSequence
from ..exceptions import ProcessingError, UnknownParameterError
from ...database import Metadata as mt
from ...config import AnalysisType, BinMode
from ...utils import profiler
from ...ipc import process_logger as logger


class _BinMixin:
    @staticmethod
    def edges2centers(edges):
        if edges is None:
            return None
        return (edges[1:] + edges[:-1]) / 2.0

    @staticmethod
    def searchsorted(edges, v):
        """A wrapper for np.searchsorted.

        This is to match the behavior of scipy.stats.binned_statistic:

        All but the last (righthand-most) bin is half-open. In other words,
        if bins is [1, 2, 3, 4], then the first bin is [1, 2) (including 1,
        but excluding 2) and the second [2, 3). The last bin, however,
        is [3, 4], which includes 4.
        """
        s = len(edges)
        if s <= 1:
            return -1
        if v == edges[-1]:
            return s - 2
        # use side = 'right' to match the result from scipy
        return np.searchsorted(edges, v, side='right') - 1


class BinProcessor(_BaseProcessor, _BinMixin):
    """BinProcessor class.

    Bin data based on 1 or 2 slow (control) data. For 1D binning, only the
    first slow data will be used. For the heat map of 2D binning, x and y
    are the first and second slow data, respectively.

    Attributes:
        analysis_type (AnalysisType): binning analysis type.
        _pp_analysis_type (AnalysisType): pump-probe analysis type.
        _mode (BinMode): binning mode.
        _device_id1 (str): device ID 1.
        _property1 (str): property of the device 1.
        _device_id2 (str): device ID 2.
        _property2 (str): property of the device 2.
        _range1 (tuple): bin 1 range.
        _range2 (tuple): bin 2 range.
        _n_bins1 (int): number of 1 bins.
        _n_bins2 (int): number of 2 bins.
        _edges1 (numpy.array): bin edges for parameter 1.
        _edges2 (numpy.array): bin edges for parameter 2.
        _stats1 (numpy.array): 1D FOM array for 1D binning with param1.
            shape = (_n_bins1,)
        _counts1 (numpy.array): 1D count array for 1D binning with param1.
            shape = (_n_bins1,)
        _vfom_heat1 (numpy.array): VFOM heatmap for 1D binning with param1.
            shape = (VFOM length, _n_bins1)
        _vfom_x1 (numpy.array): y coordinates of VFOM heatmap for 1D binning
            with param1.
        _stats2 (numpy.array): 1D FOM array for 1D binning with param2.
            shape = (_n_bins2,)
        _counts2 (numpy.array): 1D count array for 1D binning with param2.
            shape = (_n_bins2,)
        _heat (numpy.array): FOM heatmap for 2D binning.
            shape = (_n_bins2, _n_bins1)
        _heat_count (numpy.array): count heatmap for 2D binning.
            shape = (_n_bins2, _n_bins1)
        _has_param1 (bool): True if both 'device ID' and 'property' are
            specified for param1.
        _has_param2 (bool): True if both 'device ID' and 'property' are
            specified for param2.
        _bin1d (bool): a flag indicates whether data need to be re-binned
            with respect to param1.
        _bin2d (bool): a flag indicates whether data need to be re-binned
            with respect to param1 and param2.
        _reset (bool): True for clearing all the existing data.
    """

    # 10 pulses/train * 60 seconds * 30 minutes = 18,000
    # In case of VFOM, assuming the vector has a length of 1000, the array
    # size will be 18,000,000, which is still affordable.
    _MAX_POINTS = 18000

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED
        self._pp_analysis_type = AnalysisType.UNDEFINED

        self._mode = None

        self._device_id1 = ''
        self._property1 = ''
        self._device_id2 = ''
        self._property2 = ''

        # For performance reason, I prefer defining 'slow1', 'slow2' instead
        # of 'slows', which can be a tuple or list, although it leads to a
        # little bit of code bloating.
        self._slow1 = SimpleSequence(max_len=self._MAX_POINTS)
        self._slow2 = SimpleSequence(max_len=self._MAX_POINTS)
        self._fom = SimpleSequence(max_len=self._MAX_POINTS)
        self._vfom = None

        self._range1 = None
        self._n_bins1 = None
        self._range2 = None
        self._n_bins2 = None

        self._edges1 = None
        self._counts1 = None
        self._stats1 = None
        self._vfom_heat1 = None
        self._vfom_x1 = None
        self._edges2 = None
        self._counts2 = None
        self._stats2 = None
        self._heat = None
        self._heat_count = None

        # used to check whether pump-probe FOM is available
        self._pp_fail_flag = 0

        self._has_param1 = False
        self._has_param2 = False

        self._bin1d = True
        self._bin2d = True

        self._reset = True

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.BIN_PROC)

        if self._update_analysis(AnalysisType(int(cfg['analysis_type']))):
            # reset when analysis type changes
            self._reset = True

        mode = BinMode(int(cfg['mode']))
        if mode != self._mode:
            # reset when bin mode changes
            self._bin1d = True
            self._bin2d = True
            self._mode = mode

        device_id1 = cfg['device_id1']
        property1 = cfg['property1']
        if device_id1 != self._device_id1 or property1 != self._property1:
            self._device_id1 = device_id1
            self._property1 = property1
            if device_id1 and property1:
                self._has_param1 = True
            else:
                self._has_param1 = False
            self._reset = True

        device_id2 = cfg['device_id2']
        property2 = cfg['property2']
        if device_id2 != self._device_id2 or property2 != self._property2:
            self._device_id2 = device_id2
            self._property2 = property2
            if device_id2 and property2:
                self._has_param2 = True
            else:
                self._has_param2 = False
            # When the 2nd parameter (2D binning) is activated, the current
            # 1D binning result will be cleared.
            self._reset = True

        n_bins1 = int(cfg['n_bins1'])
        bin_range1 = self.str2tuple(cfg['bin_range1'])
        if n_bins1 != self._n_bins1 or bin_range1 != self._range1:
            self._n_bins1 = n_bins1
            self._range1 = bin_range1
            self._bin1d = True
            self._bin2d = True

        n_bins2 = int(cfg['n_bins2'])
        bin_range2 = self.str2tuple(cfg['bin_range2'])
        if n_bins2 != self._n_bins2 or bin_range2 != self._range2:
            self._n_bins2 = n_bins2
            self._range2 = bin_range2
            self._bin2d = True

        if 'reset' in cfg:
            self._meta.hdel(mt.BIN_PROC, 'reset')
            # reset when commanded by the GUI
            self._reset = True

    @profiler("Binning Processor")
    def process(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return

        if not self._has_param1:
            return

        processed = data['processed']

        if self.analysis_type == AnalysisType.PUMP_PROBE:
            pp_analysis_type = processed.pp.analysis_type
            if self._pp_analysis_type != pp_analysis_type:
                self._reset = True
                self._pp_analysis_type = pp_analysis_type

        if self._reset:
            self._clear_history()
            self._reset = False

        if self._bin1d:
            self._new_1d_binning()

        if self._has_param2 and self._bin2d:
            self._new_2d_binning()

        try:
            fom, vfom_x, vfom, s1, s2 = self._get_data_point(
                processed, data['raw'])

            # if the above line of code raises, no data point will be added

            if vfom is not None:
                if self._vfom is None:
                    self._init_vfom_binning(vfom, vfom_x)
                else:
                    try:
                        self._vfom.append(vfom)
                    except ValueError:
                        # vfom cannot be scalar data. Therefore, we assume the
                        # ValueError is caused by length mismatch.
                        self._init_vfom_binning(vfom, vfom_x)

            if fom is not None:
                self._slow1.append(s1)
                if self._has_param2:
                    self._slow2.append(s2)
                self._fom.append(fom)

                self._update_1d_binning(fom, vfom, s1)
                if self._has_param2:
                    self._update_2d_binning(fom, s1, s2)

        except ProcessingError as e:
            logger.error(f"[Bin] {str(e)}!")

        bin = processed.bin
        bin1 = bin[0]
        bin1.device_id = self._device_id1
        bin1.property = self._property1
        bin1.centers = self.edges2centers(self._edges1)
        bin1.stats = self._stats1
        bin1.counts = self._counts1
        bin1.heat = self._vfom_heat1
        bin1.x = self._vfom_x1

        bin2 = bin[1]
        bin2.device_id = self._device_id2
        bin2.property = self._property2
        bin2.centers = self.edges2centers(self._edges2)

        bin.heat = self._heat
        bin.heat_count = self._heat_count

    def _get_data_point(self, processed, raw):
        analysis_type = self.analysis_type
        if analysis_type == AnalysisType.PUMP_PROBE:
            ret = processed.pp
            if ret.fom is None:
                self._pp_fail_flag += 1
                # if on/off pulses are in different trains, pump-probe FOM is
                # only calculated every other train.
                if self._pp_fail_flag == 2:
                    self._pp_fail_flag = 0
                    raise ProcessingError("Pump-probe FOM is not available")
            else:
                self._pp_fail_flag = 0
        elif analysis_type == AnalysisType.ROI_FOM:
            ret = processed.roi
            if ret.fom is None:
                raise ProcessingError("ROI FOM is not available")
        elif analysis_type == AnalysisType.ROI_PROJ:
            ret = processed.roi.proj
            if ret.fom is None:
                raise ProcessingError("ROI projection FOM is not available")
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            ret = processed.ai
            if ret.fom is None:
                raise ProcessingError(
                    "Azimuthal integration FOM is not available")
        else:
            raise UnknownParameterError(
                f"[Bin] Unknown analysis type: {self.analysis_type}")

        tid = processed.tid

        slow1, err = self._fetch_property_data(
            tid, raw, self._device_id1, self._property1)
        if err:
            raise ProcessingError(err)

        slow2, err = self._fetch_property_data(
            tid, raw, self._device_id2, self._property2)
        if err:
            raise ProcessingError(err)

        return ret.fom, ret.x, ret.y, slow1, slow2

    def _new_1d_binning(self):
        self._stats1, self._edges1, _ = \
            stats.binned_statistic(self._slow1.data(),
                                   self._fom.data(),
                                   self._statistics(),
                                   self._n_bins1,
                                   self._range1)
        np.nan_to_num(self._stats1, copy=False)

        if self._vfom is not None:
            self._vfom_heat1, _, _ = \
                stats.binned_statistic(self._slow1.data(),
                                       self._vfom.data().T,
                                       self._statistics(),
                                       self._n_bins1,
                                       self._range1)
            np.nan_to_num(self._vfom_heat1, copy=False)

        self._counts1, _, _ = \
            stats.binned_statistic(self._slow1.data(),
                                   self._fom.data(),
                                   'count',
                                   self._n_bins1,
                                   self._range1)
        np.nan_to_num(self._counts1, copy=False)

        self._bin1d = False

    def _init_vfom_binning(self, vfom, vfom_x):
        self._clear_history()
        self._vfom = SimpleVectorSequence(
            len(vfom), max_len=self._MAX_POINTS, order='F')
        self._vfom_x1 = vfom_x
        # caveat: sequence of the following two lines
        self._new_1d_binning()
        if self._has_param2:
            self._new_2d_binning()
        self._vfom.append(vfom)

    def _update_1d_binning(self, fom, vfom, s1):
        iloc = self.searchsorted(self._edges1, s1)
        if 0 <= iloc < self._n_bins1:
            self._counts1[iloc] += 1
            count = self._counts1[iloc]
            if self._mode == BinMode.ACCUMULATE:
                self._stats1[iloc] += fom
                if vfom is not None:
                    self._vfom_heat1[:, iloc] += vfom
            else:  # self._mode == BinMode.AVERAGE
                self._stats1[iloc] += (fom - self._stats1[iloc]) / count
                if vfom is not None:
                    self._vfom_heat1[:, iloc] += \
                        (vfom - self._vfom_heat1[:, iloc]) / count

    def _new_2d_binning(self):
        # Note: the return array from 'stats.binned_statistic_2d' has a swap x and y
        # axis compared to conventional image data
        self._heat, self._edges2, _, _ = \
            stats.binned_statistic_2d(self._slow2.data(),
                                      self._slow1.data(),
                                      self._fom.data(),
                                      self._statistics(),
                                      [self._n_bins2, self._n_bins1],
                                      [self._range2, self._range1])
        np.nan_to_num(self._heat, copy=False)

        self._heat_count, _, _, _ = \
            stats.binned_statistic_2d(self._slow2.data(),
                                      self._slow1.data(),
                                      self._fom.data(),
                                      'count',
                                      [self._n_bins2, self._n_bins1],
                                      [self._range2, self._range1])
        np.nan_to_num(self._heat_count, copy=False)

        self._bin2d = False

    def _update_2d_binning(self, fom, s1, s2):
        iloc_x = self.searchsorted(self._edges1, s1)
        iloc_y = self.searchsorted(self._edges2, s2)
        if 0 <= iloc_x < self._n_bins1 and 0 <= iloc_y < self._n_bins2:
            self._heat_count[iloc_y, iloc_x] += 1
            if self._mode == BinMode.ACCUMULATE:
                self._heat[iloc_y, iloc_x] += fom
            else:   # self._mode == BinMode.AVERAGE
                self._heat[iloc_y, iloc_x] += \
                    (fom - self._heat[iloc_y, iloc_x]) / \
                    self._heat_count[iloc_y, iloc_x]

    def _statistics(self):
        """Return the statistic mode string used in Scipy."""
        return 'sum' if self._mode == BinMode.ACCUMULATE else 'mean'

    def _clear_history(self):
        self._slow1.reset()
        self._slow2.reset()
        self._fom.reset()
        self._vfom = None

        self._edges1 = None
        self._counts1 = None
        self._stats1 = None
        self._vfom_heat1 = None
        self._vfom_x1 = None
        self._edges2 = None
        self._counts2 = None
        self._stats2 = None
        self._heat = None
        self._heat_count = None

        self._bin1d = True
        self._bin2d = True
