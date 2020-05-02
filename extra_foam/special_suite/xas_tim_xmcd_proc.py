"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from extra_foam.pipeline.processors.base_processor import (
    SimpleSequence, SimplePairSequence
)
from extra_foam.algorithms import compute_spectrum_1d

from .special_analysis_base import profiler
from .xas_tim_proc import XasTimProcessor, _MAX_WINDOW

_DEFAULT_CURRENT_THRESHOLD = 1e-6


class XasTimXmcdProcessor(XasTimProcessor):
    """XAS-TIM XMCD processor.

    Attributes:
        _magnet_device_id (str): Magnet device ID.
        _magnet_ppt (str): Magnet property name for current.
        _current_threshold (float): Lower boundary of the magnet current.
            Pulses will be dropped if the absolute current is below the
            threshold.
        _current (SimpleSequence): Store pulse magnet currents.
        _current_scan (SimplePairSequence): A sequence of (train ID, curent).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._magnet_device_id = ""
        self._magnet_ppt = "value"  # amazing property name

        self._current_threshold = _DEFAULT_CURRENT_THRESHOLD

        self._current = SimpleSequence(max_len=_MAX_WINDOW)

        self._current_scan = SimplePairSequence(max_len=_MAX_WINDOW)

    def onMagnetDeviceChanged(self, device: str):
        self._magnet_device_id = device

    def onMagnetThresholdChanged(self, value: str):
        self._current_threshold = float(value)

    def sources(self):
        """Override."""
        srcs = super().sources()
        srcs.append((self._magnet_device_id, self._magnet_ppt))
        return srcs

    @profiler("XAS-TIM-XMCD Processor")
    def process(self, data):
        """Override."""
        tid, xgm_intensity, digitizer_apds, energy = \
            self._update_data_history(data)

        current = self.getPropertyData(
            data['raw'], self._magnet_device_id, self._magnet_ppt)
        self._current.extend([current] * len(xgm_intensity))
        self._current_scan.append((tid, current))

        # apply filters
        flt = np.logical_and(
            self._i0.data() > self._i0_threshold,
            np.abs(self._current.data()) > self._current_threshold
        )
        i0 = self._i0.data()[flt][-self._window:]
        i1 = [None] * 4
        for i, _item in enumerate(self._i1):
            if self._digitizer_channels[i]:
                i1[i] = _item.data()[flt][-self._window:]
        energy = self._energy.data()[flt][-self._window:]
        current = self._current.data()[flt][-self._window:]

        # compute spectra
        p_flt = current > 0
        n_flt = current < 0
        e_p, e_n = energy[p_flt], energy[n_flt]
        stats = []
        for i, _item in enumerate(i1):
            if self._digitizer_channels[i]:
                mcp_stats_p, _, _ = compute_spectrum_1d(
                    e_p, _item[p_flt], n_bins=self._n_bins)
                mcp_stats_n, _, _ = compute_spectrum_1d(
                    e_n, _item[n_flt], n_bins=self._n_bins)
                stats.append([mcp_stats_p, mcp_stats_n])
            else:
                # Do not calculate spectrum which is not requested to display
                stats.append((None, None))

        i0_stats_p, _, _ = compute_spectrum_1d(
            e_p, i0[p_flt], n_bins=self._n_bins)
        i0_stats_n, _, _ = compute_spectrum_1d(
            e_n, i0[n_flt], n_bins=self._n_bins)
        i0_stats, centers, counts = compute_spectrum_1d(
            energy, i0, n_bins=self._n_bins)

        for i, (p, n) in enumerate(stats):
            if p is not None:
                if i < 3:
                    stats[i][0] = -np.log(-p / i0_stats_p)
                    stats[i][1] = -np.log(-n / i0_stats_n)
                else:
                    # MCP4 has a different spectrum
                    stats[i][0] = -p / i0_stats_p
                    stats[i][1] = -n / i0_stats_n

        stats.append(i0_stats)

        self.log.info(f"Train {tid} processed")

        return {
            "xgm_intensity": xgm_intensity,
            "digitizer_apds": digitizer_apds,
            "energy_scan": self._energy_scan.data(),
            "current_scan": self._current_scan.data(),
            "correlation_length": self._correlation_window,
            "i0": i0,
            "i1": i1,
            "spectra": (stats, centers, counts),
        }

    def reset(self):
        """Override."""
        super().reset()
        self._current.reset()
        self._current_scan.reset()
