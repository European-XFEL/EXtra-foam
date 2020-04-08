"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from scipy import stats

from .special_analysis_base import QThreadWorker
from ..algorithms import nansum
from ..pipeline.processors.base_processor import SimpleSequence
from ..pipeline.processors.binning import _BinMixin
from ..pipeline.exceptions import ProcessingError
from ..utils import profiler


class TrxasProcessor(QThreadWorker, _BinMixin):
    """Time-resolved XAS processor.

    The implementation of tr-XAS processor is easier than bin processor
    since it cannot have empty device ID or property. Moreover, it does
    not include VFOM heatmap.

    Attributes:
        _delays (SimpleSequence): train-resolved time delays.
        _energies (SimpleSequence): train-resolved photon energies.
        _a13 (SimpleSequence): train-resolved -log(sum(ROI1)/sum(ROI3)).
        _a23 (SimpleSequence): train-resolved -log(sum(ROI2)/sum(ROI3)).
        _a21 (SimpleSequence): train-resolved -log(sum(ROI2)/sum(ROI1)).
        _delay_bin_edges (numpy.array): edges of delay bins.
        _delay_bin_counts (numpy.array): count of data points in each
            delay bin.
        _a13_stats (numpy.array): mean of -log(sum(ROI1)/sum(ROI3)) in
            each delay bin.
        _a23_stats (numpy.array): mean of -log(sum(ROI2)/sum(ROI3)) in
            each delay bin.
        _a21_stats (numpy.array): mean of -log(sum(ROI2)/sum(ROI1)) in
            each delay bin.
        _energy_bin_edges (numpy.array): edges of energy bins.
        _a21_heat (numpy.array): mean of -log(sum(ROI2)/sum(ROI1)) in each
            delay/energy bin.
        _a21_heatcount (numpy.array): count of data points in each
            delay/energy bin.
        _delay_device (str): delay data device ID.
        _delay_ppt (str): delay data property name.
        _energy_device (str): energy data device ID.
        _energy_ppt (str): energy data property name.
        _n_delay_bins (int): number of delay bins.
        _delay_range (tuple): (lower boundary, upper boundary) of energy bins.
        _n_energy_bins (int): number of energy bins.
        _energy_range (tuple): (lower boundary, upper boundary) of delay bins.
        _bin1d (bool): a flag indicates whether data need to be re-binned
            with respect to delay.
        _bin2d (bool): a flag indicates whether data need to be re-binned
            with respect to delay and energy.
        _reset (bool): True for clearing all the existing data.
    """

    # 10 pulses/train * 60 seconds * 30 minutes = 18,000
    _MAX_POINTS = 100 * 60 * 60

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._delays = SimpleSequence(max_len=self._MAX_POINTS)
        self._energies = SimpleSequence(max_len=self._MAX_POINTS)
        self._a13 = SimpleSequence(max_len=self._MAX_POINTS)
        self._a23 = SimpleSequence(max_len=self._MAX_POINTS)
        self._a21 = SimpleSequence(max_len=self._MAX_POINTS)

        self._delay_bin_edges = None
        self._delay_bin_counts = None
        self._a13_stats = None
        self._a23_stats = None
        self._a21_stats = None
        self._energy_bin_edges = None
        self._a21_heat = None
        self._a21_heatcount = None

        self._delay_device = ""
        self._delay_ppt = ""
        self._energy_device = ""
        self._energy_ppt = ""

        self._n_delay_bins = None
        self._delay_range = None
        self._n_energy_bins = None
        self._energy_range = None

        self._bin1d = True
        self._bin2d = True

        self._reset = False

    def onDelayDeviceChanged(self, value: str):
        self._delay_device = value

    def onDelayPropertyChanged(self, value: str):
        self._delay_ppt = value

    def onEnergyDeviceChanged(self, value: str):
        self._energy_device = value

    def onEnergyPropertyChanged(self, value: str):
        self._energy_ppt = value

    def onNoDelayBinsChanged(self, value: str):
        n_bins = int(value)
        if n_bins != self._n_delay_bins:
            self._n_delay_bins = n_bins
            self._bin1d = True
            self._bin2d = True

    def onDelayRangeChanged(self, value: tuple):
        if value != self._delay_range:
            self._delay_range = value
            self._bin1d = True
            self._bin2d = True

    def onNoEnergyBinsChanged(self, value: str):
        n_bins = int(value)
        if n_bins != self._n_energy_bins:
            self._n_energy_bins = n_bins
            self._bin2d = True

    def onEnergyRangeChanged(self, value: tuple):
        if value != self._energy_range:
            self._energy_range = value
            self._bin2d = True

    def onReset(self):
        self._reset = True

    @profiler("tr-XAS Processor")
    def process(self, data):
        """Override."""
        processed = data["processed"]

        if self._reset:
            self._clear_history()
            self._reset = False

        if self._bin1d:
            self._new_1d_binning()

        if self._bin2d:
            self._new_2d_binning()

        roi1, roi2, roi3 = None, None, None
        try:
            roi1, roi2, roi3, sum1, sum2, sum3, delay, energy = \
                self._get_data_point(processed, data['raw'])

            a13 = -np.log(sum1 / sum3)
            a23 = -np.log(sum2 / sum3)
            a21 = -np.log(sum2 / sum1)
            # update historic data
            self._a13.append(a13)
            self._a23.append(a23)
            self._a21.append(a21)
            self._delays.append(delay)
            self._energies.append(energy)

            self._update_1d_binning(a13, a23, a21, delay)
            self._update_2d_binning(a21, energy, delay)

        except ProcessingError as e:
            self.log.error(repr(e))

        self.log.info(f"Train {processed.tid} processed")

        return {
            "roi1": roi1,
            "roi2": roi2,
            "roi3": roi3,
            "delay_bin_centers": self.edges2centers(self._delay_bin_edges)[0],
            "delay_bin_counts": self._delay_bin_counts,
            "energy_bin_centers": self.edges2centers(self._energy_bin_edges)[0],
            "a13_stats": self._a13_stats,
            "a23_stats": self._a23_stats,
            "a21_stats": self._a21_stats,
            "a21_heat": self._a21_heat,
            "a21_heatcount": self._a21_heatcount
        }

    def _get_data_point(self, processed, raw):
        tid = processed.tid
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

        # fetch energy and delay

        delay_src = f"{self._delay_device} {self._delay_ppt}"
        delay, err = self._fetch_property_data(tid, raw, delay_src)
        if err:
            raise ProcessingError(err)

        energy_src = f"{self._energy_device} {self._energy_ppt}"
        energy, err = self._fetch_property_data(tid, raw, energy_src)
        if err:
            raise ProcessingError(err)

        return roi1, roi2, roi3, sum1, sum2, sum3, delay, energy

    def _new_1d_binning(self):
        self._a13_stats, self._delay_bin_edges, _ = \
            stats.binned_statistic(self._delays.data(),
                                   self._a13.data(),
                                   'mean',
                                   self._n_delay_bins,
                                   self._delay_range)
        np.nan_to_num(self._a13_stats, copy=False)

        self._delay_bin_counts, _, _ = \
            stats.binned_statistic(self._delays.data(),
                                   self._a13.data(),
                                   'count',
                                   self._n_delay_bins,
                                   self._delay_range)
        np.nan_to_num(self._delay_bin_counts, copy=False)

        self._a23_stats, _, _ = \
            stats.binned_statistic(self._delays.data(),
                                   self._a23.data(),
                                   'mean',
                                   self._n_delay_bins,
                                   self._delay_range)
        np.nan_to_num(self._a23_stats, copy=False)

        self._a21_stats, _, _ = \
            stats.binned_statistic(self._delays.data(),
                                   self._a21.data(),
                                   'mean',
                                   self._n_delay_bins,
                                   self._delay_range)
        np.nan_to_num(self._a21_stats, copy=False)

        self._bin1d = False

    def _update_1d_binning(self, a13, a23, a21, delay):
        iloc_x = self.searchsorted(self._delay_bin_edges, delay)
        if 0 <= iloc_x < self._n_delay_bins:
            self._delay_bin_counts[iloc_x] += 1
            count = self._delay_bin_counts[iloc_x]
            self._a13_stats[iloc_x] += (a13 - self._a13_stats[iloc_x]) / count
            self._a23_stats[iloc_x] += (a23 - self._a23_stats[iloc_x]) / count
            self._a21_stats[iloc_x] += (a21 - self._a21_stats[iloc_x]) / count

    def _new_2d_binning(self):
        # to have energy on x axis and delay on y axis
        # Note: the return array from 'stats.binned_statistic_2d' has a swap x and y
        # axis compared to conventional image data
        self._a21_heat, _, self._energy_bin_edges, _ = \
            stats.binned_statistic_2d(self._delays.data(),
                                      self._energies.data(),
                                      self._a21.data(),
                                      'mean',
                                      [self._n_delay_bins, self._n_energy_bins],
                                      [self._delay_range, self._energy_range])
        np.nan_to_num(self._a21_heat, copy=False)

        self._a21_heatcount, _, _, _ = \
            stats.binned_statistic_2d(self._delays.data(),
                                      self._energies.data(),
                                      self._a21.data(),
                                      'count',
                                      [self._n_delay_bins, self._n_energy_bins],
                                      [self._delay_range, self._energy_range])
        np.nan_to_num(self._a21_heatcount, copy=False)

        self._bin2d = False

    def _update_2d_binning(self, a21, energy, delay):
        iloc_x = self.searchsorted(self._energy_bin_edges, energy)
        iloc_y = self.searchsorted(self._delay_bin_edges, delay)
        if 0 <= iloc_x < self._n_energy_bins \
                and 0 <= iloc_y < self._n_delay_bins:
            self._a21_heatcount[iloc_y, iloc_x] += 1
            self._a21_heat[iloc_y, iloc_x] += \
                (a21 - self._a21_heat[iloc_y, iloc_x]) / \
                self._a21_heatcount[iloc_y, iloc_x]

    def _clear_history(self):
        self._delays.reset()
        self._energies.reset()
        self._a13.reset()
        self._a23.reset()
        self._a21.reset()

        self._delay_bin_edges = None
        self._delay_bin_counts = None
        self._a13_stats = None
        self._a23_stats = None
        self._a21_stats = None
        self._energy_bin_edges = None
        self._a21_heat = None
        self._a21_heatcount = None

        self._bin1d = True
        self._bin2d = True
