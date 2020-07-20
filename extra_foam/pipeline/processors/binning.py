"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import math

import numpy as np
from scipy import stats

from .base_processor import _BaseProcessor
from ..exceptions import ProcessingError, UnknownParameterError
from ...algorithms import SimpleSequence, SimpleVectorSequence
from ...database import Metadata as mt
from ...config import AnalysisType, BinMode
from ...utils import profiler
from ...ipc import process_logger as logger


class _BinMixin:
    @staticmethod
    def edges2centers(edges):
        if edges is None:
            return None, None
        return (edges[1:] + edges[:-1]) / 2.0, edges[1] - edges[0]

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

    @staticmethod
    def get_actual_range(data, bin_range, auto_range):
        # It is guaranteed that bin_range[0] < bin_range[1]
        if not auto_range[0] and not auto_range[1]:
            return bin_range

        if auto_range[0]:
            v_min = None if data.size == 0 else data.min()
        else:
            v_min = bin_range[0]

        if auto_range[1]:
            v_max = None if data.size == 0 else data.max()
        else:
            v_max = bin_range[1]

        # The following three cases caused by zero-sized array.
        if v_min is None and v_max is None:
            return 0., 1.
        if v_min is None:
            return v_max - 1., v_max
        if v_max is None:
            return v_min, v_min + 1.

        if auto_range[0] and auto_range[1]:
            if v_min == v_max:
                # all elements have the same value
                return v_min - 0.5, v_max + 0.5
        elif v_min >= v_max:
            # two tricky corner cases
            if auto_range[0]:
                v_min = v_max - 1.0
            elif auto_range[1]:
                v_max = v_min + 1.0
            # else cannot happen
        return v_min, v_max


class BinningProcessor(_BaseProcessor, _BinMixin):
    """BinningProcessor class.

    Bin data based on 1 or 2 slow (control) data. For 1D binning, only the
    first slow data will be used. For the heat map of 2D binning, x and y
    are the first and second slow data, respectively.

    Attributes:
        analysis_type (AnalysisType): binning analysis type.
        _pp_analysis_type (AnalysisType): pump-probe analysis type.
        _mode (BinMode): binning mode.
        _source1 (str): source name 1.
        _source2 (str): source name 2.
        _slow1 (SimpleSequence): store train-resolved data of source 1.
        _slow2 (SimpleSequence): store train-resolved data of source 2.
        _fom (SimpleSequence): store train-resolved FOM data.
        _vfom (SimpleVectorSequence): store train-resolved VFOM data.
        _bin_range1 (tuple): requested range of bin 1.
        _actual_range1 (tuple): actual bin range used in bin 1.
        _n_bins1 (int): number of bins of bin 1.
        _bin_range2 (tuple): requested range of bin 2.
        _actual_range2 (tuple): actual bin range used in bin 2.
        _n_bins2 (int): number of bins of bin 2.
        _edges1 (numpy.array): edges of bin 1. shape = (_n_bins1 + 1,)
        _counts1 (numpy.array): counts of bin 1. shape = (_n_bins1,)
        _stats1 (numpy.array): 1D binning of FOM with respect to source 1.
            shape = (_n_bins1,)
        _vfom_heat1 (numpy.array): 1D binning of VFOM with respect to
            source 1. shape = (VFOM length, _n_bins1)
        _vfom_x1 (numpy.array): x coordinates of _vfom_heat1.
        _edges2 (numpy.array): edges of bin 2. shape = (_n_bins2 + 1,)
        _stats2 (numpy.array): 1D binning of FOM with respect to source 2.
            shape = (_n_bins2,)
        _heat (numpy.array): 2D binning of FOM. shape = (_n_bins2, _n_bins1)
        _heat_count (numpy.array): counts of 2D binning of FOM.
            shape = (_n_bins2, _n_bins1)
        _has_param1 (bool): True if source 1 is not empty.
        _has_param2 (bool): True if source 2 is not empty.
        _bin1d (bool): a flag indicates whether data need to be re-binned
            with respect to source 1.
        _bin2d (bool): a flag indicates whether data need to be re-binned
            with respect to source 1 and source 2.
        _reset (bool): True for clearing all the existing data.
        _reset_bin2d (bool): True for clearing all the existing data which
            only relate to 2D binning.
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

        self._source1 = ''
        self._source2 = ''

        # For performance reason, I prefer defining 'slow1', 'slow2' instead
        # of 'slows', which can be a tuple or list, although it leads to a
        # little bit of code bloating.
        self._slow1 = SimpleSequence(max_len=self._MAX_POINTS)
        self._slow2 = SimpleSequence(max_len=self._MAX_POINTS)
        self._fom = SimpleSequence(max_len=self._MAX_POINTS)
        self._vfom = None

        self._bin_range1 = None
        self._auto_range1 = [False, False]
        self._actual_range1 = None
        self._n_bins1 = None
        self._bin_range2 = None
        self._auto_range2 = [False, False]
        self._actual_range2 = None
        self._n_bins2 = None

        self._edges1 = None
        self._counts1 = None
        self._stats1 = None
        self._vfom_heat1 = None
        self._vfom_x1 = None
        self._edges2 = None
        self._stats2 = None
        self._heat = None
        self._heat_count = None

        # used to check whether pump-probe FOM is available
        self._pp_fail_flag = 0

        self._has_param1 = False
        self._has_param2 = False

        self._bin1d = False
        self._bin2d = False

        self._reset = False
        self._reset_bin2d = False

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.BINNING_PROC)
        if 'analysis_type' not in cfg:
            # BinningWindow not initialized
            return

        if self._update_analysis(AnalysisType(int(cfg['analysis_type']))):
            # reset when analysis type changes
            self._reset = True

        mode = BinMode(int(cfg['mode']))
        if mode != self._mode:
            # reset when bin mode changes
            self._bin1d = True
            self._bin2d = True
            self._mode = mode

        source1 = cfg['source1']
        if source1 != self._source1:
            self._source1 = source1
            self._has_param1 = bool(source1)
            self._reset = True

        source2 = cfg['source2']
        if source2 != self._source2:
            self._source2 = source2
            if source2:
                self._has_param2 = True
                # When the 2nd parameter (2D binning) is changed, the current
                # 1D binning result will also be cleared.
                self._reset = True
            else:
                self._has_param2 = False
                self._reset_bin2d = True

        n_bins1 = int(cfg['n_bins1'])
        if n_bins1 != self._n_bins1:
            self._n_bins1 = n_bins1
            self._bin1d = True
            self._bin2d = True

        n_bins2 = int(cfg['n_bins2'])
        if n_bins2 != self._n_bins2:
            self._n_bins2 = n_bins2
            self._bin2d = True

        bin_range1 = self.str2tuple(cfg['bin_range1'])
        if bin_range1 != self._bin_range1:
            self._bin_range1 = bin_range1
            self._auto_range1[:] = [math.isinf(v) for v in bin_range1]
            # whether the data should be re-binned is determined by the
            # change of self._actual_range

        bin_range2 = self.str2tuple(cfg['bin_range2'])
        if bin_range2 != self._bin_range2:
            self._bin_range2 = bin_range2
            self._auto_range2[:] = [math.isinf(v) for v in bin_range2]

        if 'reset' in cfg:
            self._meta.hdel(mt.BINNING_PROC, 'reset')
            # reset when commanded by the GUI
            self._reset = True

    @profiler("Binning Processor")
    def process(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return

        if not self._has_param1:
            return

        processed, raw = data['processed'], data['raw']

        if self.analysis_type == AnalysisType.PUMP_PROBE:
            pp_analysis_type = processed.pp.analysis_type
            if self._pp_analysis_type != pp_analysis_type:
                self._reset = True
                self._pp_analysis_type = pp_analysis_type

        if self._reset:
            self._clear_history()
        elif self._reset_bin2d:
            self._clear_bin2d_history()

        fom, vfom, vfom_x, s1, s2 = None, None, None, None, None
        try:
            fom, vfom, vfom_x, s1, s2 = self._update_data_point(
                processed, raw)
        except ProcessingError as e:
            logger.error(f"[Binning] {str(e)}!")

        actual_range1 = self.get_actual_range(
            self._slow1.data(), self._bin_range1, self._auto_range1)
        if actual_range1 != self._actual_range1:
            self._actual_range1 = actual_range1
            self._bin1d = True
            self._bin2d = True

        if self._bin1d:
            self._new_1d_binning()
            self._bin1d = False
        else:
            if fom is not None:
                self._update_1d_binning(fom, vfom, s1)

        if self._has_param2:
            actual_range2 = self.get_actual_range(
                self._slow2.data(), self._bin_range2, self._auto_range2)
            if actual_range2 != self._actual_range2:
                self._actual_range2 = actual_range2
                self._bin2d = True

            if self._bin2d:
                self._new_2d_binning()
                self._bin2d = False
            else:
                if fom is not None:
                    self._update_2d_binning(fom, s1, s2)

        bin = processed.bin

        bin1 = bin[0]
        bin1.source = self._source1
        bin1.centers, bin1.size = self.edges2centers(self._edges1)
        bin1.stats = self._stats1
        bin1.counts = self._counts1
        bin1.heat = self._vfom_heat1
        bin1.x = self._vfom_x1

        bin2 = bin[1]
        bin2.source = self._source2
        bin2.centers, bin2.size = self.edges2centers(self._edges2)

        bin.heat = self._heat
        bin.heat_count = self._heat_count

    def _update_data_point(self, processed, raw):
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
                return None, None, None, None, None
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
                f"[Binning] Unknown analysis type: {self.analysis_type}")

        tid = processed.tid

        slow1, err = self._fetch_property_data(tid, raw, self._source1)
        if err:
            # The number of data points in slow1, slow2 (if requested),
            # fom and vfom (if applicable) must be kept the same. So we
            # drop the train if any of them is not available.
            raise ProcessingError(err)

        slow2, err = self._fetch_property_data(tid, raw, self._source2)
        if err:
            raise ProcessingError(err)

        fom, vfom, vfom_x = ret.fom, ret.y, ret.x

        if vfom is not None:
            if self._vfom is None:
                # after analysis type changed
                self._init_vfom_binning(vfom, vfom_x)
            else:
                try:
                    self._vfom.append(vfom)
                except ValueError:
                    # vfom cannot be scalar data. Therefore, we assume the
                    # ValueError is caused by length mismatch.

                    # The number of VFOM and FOM data are always the same.
                    self._clear_history()
                    self._bin1d = True
                    self._bin2d = True
                    self._init_vfom_binning(vfom, vfom_x)

        self._slow1.append(slow1)
        if self._has_param2:
            self._slow2.append(slow2)
        self._fom.append(fom)

        return fom, vfom, vfom_x, slow1, slow2

    def _new_1d_binning(self):
        if self._actual_range1[0] is None:
            # deal with the case when actual_range1 == (None, None) and
            # there is no (FOM, etc.) data
            self._actual_range1 = (0, 0)

        self._stats1, self._edges1, _ = \
            stats.binned_statistic(self._slow1.data(),
                                   self._fom.data(),
                                   self._statistics(),
                                   self._n_bins1,
                                   self._actual_range1)
        np.nan_to_num(self._stats1, copy=False)

        if self._vfom is not None:
            self._vfom_heat1, _, _ = \
                stats.binned_statistic(self._slow1.data(),
                                       self._vfom.data().T,
                                       self._statistics(),
                                       self._n_bins1,
                                       self._actual_range1)
            np.nan_to_num(self._vfom_heat1, copy=False)

        self._counts1, _, _ = \
            stats.binned_statistic(self._slow1.data(),
                                   self._fom.data(),
                                   'count',
                                   self._n_bins1,
                                   self._actual_range1)
        np.nan_to_num(self._counts1, copy=False)

    def _init_vfom_binning(self, vfom, vfom_x):
        self._vfom = SimpleVectorSequence(
            len(vfom), max_len=self._MAX_POINTS, order='F')
        self._vfom_x1 = vfom_x
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
        if self._actual_range2[0] is None:
            # deal with the case when actual_range2 == (None, None) and
            # there is no (FOM, etc.) data
            self._actual_range2 = (0, 0)

        # Note: the return array from 'stats.binned_statistic_2d' has a swap x and y
        # axis compared to conventional image data
        self._heat, self._edges2, _, _ = \
            stats.binned_statistic_2d(self._slow2.data(),
                                      self._slow1.data(),
                                      self._fom.data(),
                                      self._statistics(),
                                      [self._n_bins2, self._n_bins1],
                                      [self._actual_range2, self._actual_range1])
        np.nan_to_num(self._heat, copy=False)

        self._heat_count, _, _, _ = \
            stats.binned_statistic_2d(self._slow2.data(),
                                      self._slow1.data(),
                                      self._fom.data(),
                                      'count',
                                      [self._n_bins2, self._n_bins1],
                                      [self._actual_range2, self._actual_range1])
        np.nan_to_num(self._heat_count, copy=False)

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
        self._fom.reset()
        self._vfom = None
        self._reset = False
        self._bin1d = True

        self._edges1 = None
        self._counts1 = None
        self._stats1 = None
        self._vfom_heat1 = None
        self._vfom_x1 = None

        self._pp_fail_flag = 0

        self._clear_bin2d_history()

    def _clear_bin2d_history(self):
        self._slow2.reset()
        self._reset_bin2d = False
        self._bin2d = True

        self._edges2 = None
        self._stats2 = None

        self._heat = None
        self._heat_count = None
