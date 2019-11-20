"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from scipy import stats

from .base_processor import _BaseProcessor
from ..exceptions import ProcessingError
from ...config import AnalysisType
from ...utils import profiler
from ...database import Metadata as mt


class TrXasProcessor(_BaseProcessor):
    """Tr-XAS processor.

    Attributes:
        analysis_type (AnalysisType): TrXAS analysis type.
        _delays (list): train-resolved time delays.
        _energies (list): train-resolved photon energies.
        _a13 (list): train-resolved -log(sum(ROI1)/sum(ROI3)).
        _a23 (list): train-resolved -log(sum(ROI2)/sum(ROI3)).
        _a21 (list): train-resolved -log(sum(ROI2)/sum(ROI1)).
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
        _delay_device (str): device ID of the device which provides time
            delay data.
        _delay_ppt (str): property of the delay device which provides time
            delay data.
        _energy_device (str): device ID of the device which provides photon
            energy data.
        _energy_ppt (str): property of the energy device which provides
            photon energy data.
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

    _MAX_POINTS = 100 * 60 * 60

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

        self._delays = []
        self._energies = []
        self._a13 = []
        self._a23 = []
        self._a21 = []

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

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.TR_XAS_PROC)

        try:
            self._update_analysis(AnalysisType(int(cfg['analysis_type'])))
        except KeyError:
            self._update_analysis(AnalysisType.UNDEFINED)

        self._delay_device = cfg["delay_device"]
        self._delay_ppt = cfg["delay_property"]

        self._energy_device = cfg["energy_device"]
        self._energy_ppt = cfg["energy_property"]

        n_delay_bins = int(cfg["n_delay_bins"])
        if n_delay_bins != self._n_delay_bins:
            self._n_delay_bins = n_delay_bins
            self._bin1d = True
            self._bin2d = True

        delay_range = self.str2tuple(cfg["delay_range"])
        if delay_range != self._delay_range:
            self._delay_range = delay_range
            self._bin1d = True
            self._bin2d = True

        n_energy_bins = int(cfg["n_energy_bins"])
        if n_energy_bins != self._n_energy_bins:
            self._n_energy_bins = n_energy_bins
            self._bin2d = True

        energy_range = self.str2tuple(cfg["energy_range"])
        if energy_range != self._energy_range:
            self._energy_range = energy_range
            self._bin2d = True

        if 'reset' in cfg:
            self._meta.hdel(mt.TR_XAS_PROC, 'reset')
            self._reset = True

    @profiler("tr-XAS Processor")
    def process(self, data):
        """Override."""
        if not self._meta.has_analysis(AnalysisType.TR_XAS):
            return

        processed = data['processed']

        if self._reset:
            self._clear_history()
            self._reset = False

        if len(self._delays) > self._MAX_POINTS:
            raise ProcessingError(f"[tr-XAS] Maximum number of points "
                                  f"({self._MAX_POINTS}) reached!")

        err_msg = ""
        try:
            sum1, sum2, sum3, delay, energy = self._get_data_point(
                processed, data['raw'])

            a13 = -np.log(sum1 / sum3)
            a23 = -np.log(sum2 / sum3)
            a21 = -np.log(sum2 / sum1)
            # update historic data
            self._a13.append(a13)
            self._a23.append(a23)
            self._a21.append(a21)
            self._delays.append(delay)
            self._energies.append(energy)
        except ProcessingError as e:
            err_msg = str(e)

        # update 1d binning

        if self._bin1d:
            self._new_1d_binning()
        else:
            if not err_msg:
                self._update_1d_binning(a13, a23, a21, delay)

        # update 2d binning

        if self._bin2d:
            self._new_2d_binning()
        else:
            if not err_msg:
                self._update_2d_binning(a21, energy, delay)

        # update processed
        xas = processed.trxas
        xas.delay_bin_centers = self.edges2centers(self._delay_bin_edges)
        xas.delay_bin_counts = self._delay_bin_counts
        xas.a13_stats = self._a13_stats
        xas.a23_stats = self._a23_stats
        xas.a21_stats = self._a21_stats
        xas.energy_bin_centers = self.edges2centers(self._energy_bin_edges)
        xas.a21_heat = self._a21_heat
        xas.a21_heatcount = self._a21_heatcount

        if err_msg:
            raise ProcessingError(err_msg)

    def _get_data_point(self, processed, raw):
        tid = processed.tid
        roi = processed.roi
        masked = processed.image.masked_mean

        # get three ROIs
        roi1 = roi[0].rect(masked)
        if roi1 is None:
            raise ProcessingError("ROI1 is not available!")
        roi2 = roi[1].rect(masked)
        if roi2 is None:
            raise ProcessingError("ROI2 is not available!")
        roi3 = roi[2].rect(masked)
        if roi3 is None:
            raise ProcessingError("ROI3 is not available!")

        # get sums of the three ROIs
        sum1 = np.sum(roi1)
        if sum1 <= 0:
            raise ProcessingError("ROI1 sum <= 0!")
        sum2 = np.sum(roi2)
        if sum2 <= 0:
            raise ProcessingError("ROI2 sum <= 0!")
        sum3 = np.sum(roi3)
        if sum3 <= 0:
            raise ProcessingError("ROI3 sum <= 0!")

        delay, err = self._fetch_property_data(
            tid, raw, self._delay_device, self._delay_ppt)
        if err:
            raise ProcessingError(err)

        # fetch energy and delay
        energy, err = self._fetch_property_data(
            tid, raw, self._energy_device, self._energy_ppt)
        if err:
            raise ProcessingError(err)

        return sum1, sum2, sum3, delay, energy

    def _new_1d_binning(self):
        self._a13_stats, self._delay_bin_edges, _ = \
            stats.binned_statistic(self._delays, self._a13, 'mean',
                                   self._n_delay_bins, self._delay_range)
        np.nan_to_num(self._a13_stats, copy=False)

        self._delay_bin_counts, _, _ = \
            stats.binned_statistic(self._delays, self._a13, 'count',
                                   self._n_delay_bins, self._delay_range)
        np.nan_to_num(self._delay_bin_counts, copy=False)

        self._a23_stats, _, _ = \
            stats.binned_statistic(self._delays, self._a23, 'mean',
                                   self._n_delay_bins, self._delay_range)
        np.nan_to_num(self._a23_stats, copy=False)

        self._a21_stats, _, _ = \
            stats.binned_statistic(self._delays, self._a21, 'mean',
                                   self._n_delay_bins, self._delay_range)
        np.nan_to_num(self._a21_stats, copy=False)

        self._bin1d = False

    def _update_1d_binning(self, a13, a23, a21, delay):
        # use side = 'right' to match the result from scipy
        iloc_x = np.searchsorted(self._delay_bin_edges, delay, side='right') - 1
        if 0 <= iloc_x < self._n_delay_bins:
            self._delay_bin_counts[iloc_x] += 1
            count = self._delay_bin_counts[iloc_x]
            self._a13_stats[iloc_x] += (a13 - self._a13_stats[iloc_x]) / count
            self._a23_stats[iloc_x] += (a23 - self._a23_stats[iloc_x]) / count
            self._a21_stats[iloc_x] += (a21 - self._a21_stats[iloc_x]) / count

        return iloc_x

    def _new_2d_binning(self):
        # to have energy on x axis and delay on y axis
        # Note: the return array from 'stats.binned_statistic_2d' has a swap x and y
        # axis compared to conventional image data
        self._a21_heat, _, self._energy_bin_edges, _ = \
            stats.binned_statistic_2d(self._delays,
                                      self._energies,
                                      self._a21,
                                      'mean',
                                      [self._n_delay_bins, self._n_energy_bins],
                                      [self._delay_range, self._energy_range])
        np.nan_to_num(self._a21_heat, copy=False)

        self._a21_heatcount, _, _, _ = \
            stats.binned_statistic_2d(self._delays,
                                      self._energies,
                                      self._a21,
                                      'count',
                                      [self._n_delay_bins, self._n_energy_bins],
                                      [self._delay_range, self._energy_range])
        np.nan_to_num(self._a21_heatcount, copy=False)

        self._bin2d = False

    def _update_2d_binning(self, a21, energy, delay):
        iloc_x = np.searchsorted(self._energy_bin_edges, energy, side='right') - 1
        iloc_y = np.searchsorted(self._delay_bin_edges, delay, side='right') - 1
        if 0 <= iloc_x < self._n_energy_bins \
                and 0 <= iloc_y < self._n_delay_bins:
            self._a21_heatcount[iloc_y, iloc_x] += 1
            self._a21_heat[iloc_y, iloc_x] += \
                (a21 - self._a21_heat[iloc_y, iloc_x]) / \
                self._a21_heatcount[iloc_y, iloc_x]

    @staticmethod
    def edges2centers(edges):
        if edges is None:
            return None
        return (edges[1:] + edges[:-1]) / 2.0

    def _clear_history(self):
        self._delays.clear()
        self._energies.clear()
        self._a13.clear()
        self._a23.clear()
        self._a21.clear()

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
