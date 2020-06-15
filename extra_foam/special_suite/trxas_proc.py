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

from extra_foam.algorithms import compute_spectrum_1d, nansum
from extra_foam.algorithms import SimpleSequence
from extra_foam.pipeline.processors.binning import _BinMixin
from extra_foam.pipeline.exceptions import ProcessingError

from .special_analysis_base import profiler, QThreadWorker

_DEFAULT_N_BINS = 20
_DEFAULT_BIN_RANGE = "-inf, inf"


class TrXasProcessor(QThreadWorker, _BinMixin):
    """Time-resolved XAS processor.

    The implementation of tr-XAS processor is easier than bin processor
    since it cannot have empty device ID or property. Moreover, it does
    not include VFOM heatmap.

    Absorption ROI-i/ROI-j is defined as -log(sum(ROI-i)/sum(ROI-j)).

    Attributes:
        _device_id1 (str): device ID 1.
        _ppt1 (str): property of device 1.
        _device_id2 (str): device ID 2.
        _ppt2 (str): property of device 2.
        _slow1 (SimpleSequence): store train-resolved data of source 1.
        _slow2 (SimpleSequence): store train-resolved data of source 2.
        _a13 (SimpleSequence): store train-resolved absorption ROI1/ROI3.
        _a23 (SimpleSequence): store train-resolved absorption ROI2/ROI3.
        _a21 (SimpleSequence): store train-resolved absorption ROI2/ROI1.
        _edges1 (numpy.array): edges of bin 1. shape = (_n_bins1 + 1,)
        _counts1 (numpy.array): counts of bin 1. shape = (_n_bins1,)
        _a13_stats (numpy.array): 1D binning of absorption ROI1/ROI3 with
            respect to source 1.
        _a23_stats (numpy.array): 1D binning of absorption ROI2/ROI3 with
            respect to source 1.
        _a21_stats (numpy.array): 1D binning of absorption ROI2/ROI1 with
            respect to source 1.
        _edges2 (numpy.array): edges of bin 2. shape = (_n_bins2 + 1,)
        _a21_heat (numpy.array): 2D binning of absorption ROI2/ROI1.
            shape = (_n_bins2, _n_bins1)
        _a21_heat_count (numpy.array): counts of 2D binning of absorption
            ROI2/ROI1. shape = (_n_bins2, _n_bins1)
        _bin_range1 (tuple): bin 1 range requested.
        _actual_range1 (tuple): actual bin range used in bin 1.
        _n_bins1 (int): number of bins of bin 1.
        _bin_range2 (tuple): bin 2 range requested.
        _actual_range2 (tuple): actual bin range used in bin 2.
        _n_bins2 (int): number of bins of bin 2.
        _bin1d (bool): a flag indicates whether data need to be re-binned
            with respect to source 1.
        _bin2d (bool): a flag indicates whether data need to be re-binned
            with respect to both source 1 and source 2.
        _reset (bool): True for clearing all the existing data.
    """

    # 10 pulses/train * 60 seconds * 30 minutes = 18,000
    _MAX_POINTS = 100 * 60 * 60

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._device_id1 = ""
        self._ppt1 = ""
        self._device_id2 = ""
        self._ppt2 = ""

        self._slow1 = SimpleSequence(max_len=self._MAX_POINTS)
        self._slow2 = SimpleSequence(max_len=self._MAX_POINTS)
        self._a13 = SimpleSequence(max_len=self._MAX_POINTS)
        self._a23 = SimpleSequence(max_len=self._MAX_POINTS)
        self._a21 = SimpleSequence(max_len=self._MAX_POINTS)

        self._edges1 = None
        self._counts1 = None
        self._a13_stats = None
        self._a23_stats = None
        self._a21_stats = None

        self._edges2 = None
        self._a21_heat = None
        self._a21_heat_count = None

        self._bin_range1 = self.str2range(_DEFAULT_BIN_RANGE)
        self._actual_range1 = None
        self._auto_range1 = [True, True]
        self._n_bins1 = _DEFAULT_N_BINS
        self._bin_range2 = self.str2range(_DEFAULT_BIN_RANGE)
        self._actual_range2 = None
        self._auto_range2 = [True, True]
        self._n_bins2 = _DEFAULT_N_BINS

        self._bin1d = True
        self._bin2d = True

    def onDeviceId1Changed(self, value: str):
        self._device_id1 = value

    def onProperty1Changed(self, value: str):
        self._ppt1 = value

    def onDeviceId2Changed(self, value: str):
        self._device_id2 = value

    def onProperty2Changed(self, value: str):
        self._ppt2 = value

    def onNBins1Changed(self, value: str):
        n_bins = int(value)
        if n_bins != self._n_bins1:
            self._n_bins1 = n_bins
            self._bin1d = True
            self._bin2d = True

    def onBinRange1Changed(self, value: tuple):
        if value != self._bin_range1:
            self._bin_range1 = value
            self._auto_range1[:] = [math.isinf(v) for v in value]
            self._bin1d = True
            self._bin2d = True

    def onNBins2Changed(self, value: str):
        n_bins = int(value)
        if n_bins != self._n_bins2:
            self._n_bins2 = n_bins
            self._bin2d = True

    def onBinRange2Changed(self, value: tuple):
        if value != self._bin_range2:
            self._bin_range2 = value
            self._auto_range2[:] = [math.isinf(v) for v in value]

    def sources(self):
        """Override."""
        return [
            (self._device_id1, self._ppt1, 0),
            (self._device_id2, self._ppt2, 0),
        ]

    @profiler("tr-XAS Processor")
    def process(self, data):
        """Override."""
        processed = data["processed"]

        roi1, roi2, roi3 = None, None, None
        a13, a23, a21, s1, s2 = None, None, None, None, None
        try:
            roi1, roi2, roi3, a13, a23, a21, s1, s2 = \
                self._update_data_point(processed, data['raw'])
        except ProcessingError as e:
            self.log.error(repr(e))

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
            if a21 is not None:
                self._update_1d_binning(a13, a23, a21, s1)

        actual_range2 = self.get_actual_range(
            self._slow2.data(), self._bin_range2, self._auto_range2)
        if actual_range2 != self._actual_range2:
            self._actual_range2 = actual_range2
            self._bin2d = True

        if self._bin2d:
            self._new_2d_binning()
            self._bin2d = False
        else:
            if a21 is not None:
                self._update_2d_binning(a21, s1, s2)

        self.log.info(f"Train {processed.tid} processed")

        return {
            "roi1": roi1,
            "roi2": roi2,
            "roi3": roi3,
            "centers1": self.edges2centers(self._edges1)[0],
            "counts1": self._counts1,
            "centers2": self.edges2centers(self._edges2)[0],
            "a13_stats": self._a13_stats,
            "a23_stats": self._a23_stats,
            "a21_stats": self._a21_stats,
            "a21_heat": self._a21_heat,
            "a21_heat_count": self._a21_heat_count
        }

    def _update_data_point(self, processed, raw):
        roi = processed.roi
        masked = processed.image.masked_mean

        # get three ROIs
        roi1 = roi.geom1.rect(masked)
        if roi1 is None:
            raise ProcessingError("ROI1 is not available!")
        roi2 = roi.geom2.rect(masked)
        if roi2 is None:
            raise ProcessingError("ROI2 is not available!")
        roi3 = roi.geom3.rect(masked)
        if roi3 is None:
            raise ProcessingError("ROI3 is not available!")

        # get sums of the three ROIs
        sum1 = nansum(roi1)
        if sum1 <= 0:
            raise ProcessingError("ROI1 sum <= 0!")
        sum2 = nansum(roi2)
        if sum2 <= 0:
            raise ProcessingError("ROI2 sum <= 0!")
        sum3 = nansum(roi3)
        if sum3 <= 0:
            raise ProcessingError("ROI3 sum <= 0!")

        # calculate absorptions
        a13 = -np.log(sum1 / sum3)
        a23 = -np.log(sum2 / sum3)
        a21 = -np.log(sum2 / sum1)

        # update historic data
        self._a13.append(a13)
        self._a23.append(a23)
        self._a21.append(a21)

        # fetch slow data
        s1 = self.getPropertyData(raw, self._device_id1, self._ppt1)
        self._slow1.append(s1)
        s2 = self.getPropertyData(raw, self._device_id2, self._ppt2)
        self._slow2.append(s2)

        return roi1, roi2, roi3, a13, a23, a21, s1, s2

    def _new_1d_binning(self):
        self._a13_stats, _, _ = compute_spectrum_1d(
            self._slow1.data(),
            self._a13.data(),
            n_bins=self._n_bins1,
            bin_range=self._actual_range1,
            edge2center=False,
            nan_to_num=True
        )

        self._a23_stats, _, _ = compute_spectrum_1d(
            self._slow1.data(),
            self._a23.data(),
            n_bins=self._n_bins1,
            bin_range=self._actual_range1,
            edge2center=False,
            nan_to_num=True
        )

        self._a21_stats, edges, counts = compute_spectrum_1d(
            self._slow1.data(),
            self._a21.data(),
            n_bins=self._n_bins1,
            bin_range=self._actual_range1,
            edge2center=False,
            nan_to_num=True
        )
        self._edges1 = edges
        self._counts1 = counts

    def _update_1d_binning(self, a13, a23, a21, delay):
        iloc_x = self.searchsorted(self._edges1, delay)
        if 0 <= iloc_x < self._n_bins1:
            self._counts1[iloc_x] += 1
            count = self._counts1[iloc_x]
            self._a13_stats[iloc_x] += (a13 - self._a13_stats[iloc_x]) / count
            self._a23_stats[iloc_x] += (a23 - self._a23_stats[iloc_x]) / count
            self._a21_stats[iloc_x] += (a21 - self._a21_stats[iloc_x]) / count

    def _new_2d_binning(self):
        # to have energy on x axis and delay on y axis
        # Note: the return array from 'stats.binned_statistic_2d' has a swap x and y
        # axis compared to conventional image data
        self._a21_heat, _, self._edges2, _ = \
            stats.binned_statistic_2d(self._slow1.data(),
                                      self._slow2.data(),
                                      self._a21.data(),
                                      'mean',
                                      [self._n_bins1, self._n_bins2],
                                      [self._actual_range1, self._actual_range2])
        np.nan_to_num(self._a21_heat, copy=False)

        self._a21_heat_count, _, _, _ = \
            stats.binned_statistic_2d(self._slow1.data(),
                                      self._slow2.data(),
                                      self._a21.data(),
                                      'count',
                                      [self._n_bins1, self._n_bins2],
                                      [self._actual_range1, self._actual_range2])
        np.nan_to_num(self._a21_heat_count, copy=False)

    def _update_2d_binning(self, a21, energy, delay):
        iloc_x = self.searchsorted(self._edges2, energy)
        iloc_y = self.searchsorted(self._edges1, delay)
        if 0 <= iloc_x < self._n_bins2 \
                and 0 <= iloc_y < self._n_bins1:
            self._a21_heat_count[iloc_y, iloc_x] += 1
            self._a21_heat[iloc_y, iloc_x] += \
                (a21 - self._a21_heat[iloc_y, iloc_x]) / \
                self._a21_heat_count[iloc_y, iloc_x]

    def reset(self):
        """Override."""
        self._slow1.reset()
        self._slow2.reset()
        self._a13.reset()
        self._a23.reset()
        self._a21.reset()

        self._edges1 = None
        self._counts1 = None
        self._a13_stats = None
        self._a23_stats = None
        self._a21_stats = None
        self._edges2 = None
        self._a21_heat = None
        self._a21_heat_count = None

        self._bin1d = True
        self._bin2d = True
