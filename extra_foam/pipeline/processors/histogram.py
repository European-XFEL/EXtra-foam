"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import math

import numpy as np

from .base_processor import _BaseProcessor, SimpleSequence
from ..exceptions import UnknownParameterError
from ...algorithms import compute_statistics, find_actual_range
from ...ipc import process_logger as logger
from ...database import Metadata as mt
from ...config import AnalysisType
from ...utils import profiler


class HistogramProcessor(_BaseProcessor):
    """HistogramProcessor class.

    - Calculate the FOM of each pulse in a train.
    - Calculate histogram of accumulative pulse-/train-resolved FOMs.

    Attributes:
        analysis_type (AnalysisType): binning analysis type.
        _fom (SimpleSequence): accumulative pulse-/train-resolved FOMs.
        _n_bins (int): number of bins for calculating histogram.
        _bin_range (tuple): range of bins for calculating histogram.
        _pulse_resolved (bool): True for calculating pulse-resolved FOMs,
            otherwise train-resolved.
    """
    # 128 pulses/s * 10 trains/s * 60 seconds * 30 minutes = 2304000
    _MAX_POINTS = 128 * 10 * 60 * 30

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED
        self._fom = SimpleSequence(max_len=self._MAX_POINTS)
        self._n_bins = None
        self._bin_range = (-math.inf, math.inf)
        self._pulse_resolved = False
        self._reset = False

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.HISTOGRAM_PROC)
        self._pulse_resolved = cfg['pulse_resolved'] == 'True'

        self._bin_range = self.str2tuple(cfg["bin_range"])
        self._n_bins = int(cfg['n_bins'])

        analysis_type = AnalysisType(int(cfg['analysis_type']))
        if self._pulse_resolved and analysis_type != AnalysisType.UNDEFINED:
            analysis_type = AnalysisType(analysis_type + 2700)

        if self._update_analysis(analysis_type):
            self._reset = True

        if 'reset' in cfg:
            self._meta.hdel(mt.HISTOGRAM_PROC, 'reset')
            self._reset = True

    @profiler("Histogram processor")
    def process(self, data):
        """Override."""
        if self.analysis_type == AnalysisType.UNDEFINED:
            return
        processed = data['processed']

        if self._reset:
            self._fom.reset()
            self._reset = False

        if self._pulse_resolved:
            if self.analysis_type == AnalysisType.ROI_FOM_PULSE:
                fom = processed.pulse.roi.fom
                if fom is None:
                    logger.error(
                        "[Histogram] Pulse resolved ROI FOM is not available")
                else:
                    processed.pulse.hist.pulse_foms = \
                        fom if self._pulse_resolved else None
                    self._fom.extend(fom)
            else:
                raise UnknownParameterError(
                    f"[Histogram] Unknown analysis type: {self.analysis_type}")

            self._process_poi(processed)

        else:
            if self.analysis_type == AnalysisType.ROI_FOM:
                fom = processed.roi.fom
                if fom is None:
                    logger.error("[Histogram] ROI FOM is not available")
                else:
                    self._fom.append(fom)
            else:
                raise UnknownParameterError(
                    f"[Histogram] Unknown analysis type: {self.analysis_type}")

        # ought to calculate the histogram even if there is no new FOM coming
        data = self._fom.data()
        if data.size != 0:
            v_min, v_max = find_actual_range(data, self._bin_range)
            filtered = data[(data >= v_min) & (data <= v_max)]
            hist, bin_edges = np.histogram(
                filtered, bins=self._n_bins, range=(v_min, v_max))
            train_hist = processed.hist
            train_hist.hist = hist
            train_hist.bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
            train_hist.mean, train_hist.median, train_hist.std = \
                compute_statistics(filtered)

    def _process_poi(self, processed):
        """Calculate histograms of FOMs of POI pulses."""
        n_pulses = processed.n_pulses
        pulse_hist = processed.pulse.hist
        image_data = processed.image
        for i in image_data.poi_indices:
            if i >= n_pulses:
                return

            poi_fom = self._fom.data()[i::n_pulses]
            if poi_fom.size == 0:
                continue

            v_min, v_max = find_actual_range(poi_fom, self._bin_range)
            filtered = poi_fom[(poi_fom >= v_min) & (poi_fom <= v_max)]
            hist, bin_edges = np.histogram(
                filtered, bins=self._n_bins, range=(v_min, v_max))
            mean, median, std = compute_statistics(filtered)
            pulse_hist[i] = (hist, (bin_edges[1:] + bin_edges[:-1]) / 2.0,
                             mean, median, std)

            if image_data.poi_indices[1] == image_data.poi_indices[0]:
                # skip the second one if two POIs have the same index
                break
