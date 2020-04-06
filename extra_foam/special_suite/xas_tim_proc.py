"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import itertools

import numpy as np

from .special_analysis_base import ProcessingError, QThreadWorker
from ..utils import profiler
from ..pipeline.processors.base_processor import (
    SimpleSequence, SimplePairSequence
)
from ..algorithms import compute_spectrum_1d


_DEFAULT_N_PULSES_PER_TRAIN = 1
_DEFAULT_XGM_THRESHOLD = 0.0
_MAX_WINDOW = 180000  # 60 s * 10 train/s * 300 pulses/train
_MAX_CORRELATION_WINDOW = 3000
_MAX_N_BINS = 999
_DEFAULT_N_BINS = 30
# MCP 1 - 4
_DIGITIZER_CHANNEL_NAMES = ['D', 'C', 'B', 'A']


class XasTimProcessor(QThreadWorker):
    """XAS-TIM processor.

    Attributes:
        _xgm_output_channel (str): XGM output channel name.
        _xgm_ppt (str): XGM property name for pulse-resolved intensity.
        _digitizer_output_channel (str): Digitizer output channel name.
        _digitizer_ppts (list): A list of property names for different
            digitizer channels.
        _mono_device_id (str): Soft mono device ID.
        _mono_ppt (str): Soft mono property name for energy.
        _digitizer_channel_mask (int):
        _n_pulses_per_train (int): Number of pulses per train.
        _apd_stride (int): Pulse index stride of the digitizer APD data.
        _xgm_threshold (float): Lower boundary of the XGM data. The pulse
            will be dropped if its intensity if below the threshold.
        _window (int): Maximum number of pulses kept.
        _correlation_window (int): Number of points in correlation plots.
            It includes the points which are dropped by the filter.
        _n_bins (int): Number of bins in spectra calculation.
        _i0 (SimpleSequence): Pulse intensities of XGM.
        _i1 (list): A list of SimpleSequence, which stores pulse apd data
            for each channel of the digitizer.
        _energy (SimpleSequence): Pulse energies.
        _energy_scan (SimplePairSequence): A sequence of (train ID, energy).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._xgm_output_channel = ""
        self._xgm_ppt = "data.intensitySa3TD"
        self._digitizer_output_channel = ""
        self._digitizer_ppts = [
            f"digitizers.channel_1_{ch}.apd.pulseIntegral"
            for ch in _DIGITIZER_CHANNEL_NAMES
        ]
        self._mono_device_id = ""
        self._mono_ppt = "actualEnergy"

        self._digitizer_channel_mask = 0x00

        self._n_pulses_per_train = _DEFAULT_N_PULSES_PER_TRAIN
        self._apd_stride = 1
        self._xgm_threshold = _DEFAULT_XGM_THRESHOLD
        self._window = _MAX_WINDOW
        self._correlation_window = _MAX_CORRELATION_WINDOW
        self._n_bins = _DEFAULT_N_BINS

        self._i0 = SimpleSequence(max_len=_MAX_WINDOW)
        self._i1 = [SimpleSequence(max_len=_MAX_WINDOW)
                    for _ in _DIGITIZER_CHANNEL_NAMES]
        self._energy = SimpleSequence(max_len=_MAX_WINDOW)

        self._energy_scan = SimplePairSequence(max_len=_MAX_WINDOW)

    def onXgmOutputChannelChanged(self, ch: str):
        self._xgm_output_channel = ch

    def onDigitizerOutputChannelChanged(self, ch: str):
        self._digitizer_output_channel = ch

    def onMonoDeviceChanged(self, device: str):
        self._mono_device_id = device

    def onNPulsesPerTrainChanged(self, value: str):
        self._n_pulses_per_train = int(value)

    def onApdStrideChanged(self, value: str):
        self._apd_stride = int(value)

    def onSpectraDisplayedChanged(self, index: int, value: bool):
        # the index starts from 0 here
        if value:
            self._digitizer_channel_mask |= (1 << index)
        else:
            self._digitizer_channel_mask &= ~(1 << index)

    def onXgmThresholdChanged(self, value: str):
        self._xgm_threshold = float(value)

    def onPulseWindowChanged(self, value: str):
        self._window = int(value)

    def onCorrelationWindowChanged(self, value: str):
        self._correlation_window = int(value)

    def onNoBinsChanged(self, value: str):
        self._n_bins = int(value)

    @profiler("XAS-TIM Processor")
    def process(self, data):
        """Override."""
        data, meta = data

        tid = self._get_tid(meta)

        xgm_intensity = self._fetch_property_data(
            tid, data, self._xgm_output_channel, self._xgm_ppt)
        if xgm_intensity is None:
            return

        digitizer_apds = []
        for i, ppt in enumerate(self._digitizer_ppts):
            apd = self._fetch_property_data(
                tid, data, self._digitizer_output_channel, ppt)
            digitizer_apds.append(apd)
            if apd is None:
                # It must contain all apd data. Otherwise, it will result
                # in length mismatch between apd data and other data.
                return

        energy = self._fetch_property_data(
            tid, data, self._mono_device_id, self._mono_ppt)
        if energy is None:
            return

        # check and slice XGM intensity
        pulse_slicer = slice(0, self._n_pulses_per_train)
        if len(xgm_intensity) < self._n_pulses_per_train:
            raise ProcessingError(f"Length of {self._xgm_ppt} is less "
                                  f"than {self._n_pulses_per_train}: "
                                  f"actual {len(xgm_intensity)}")
        xgm_intensity = xgm_intensity[pulse_slicer]

        # check and slice digitizer APD data
        for i, (apd, ppt) in enumerate(zip(digitizer_apds,
                                           self._digitizer_ppts)):
            if apd is not None:
                v = apd[::self._apd_stride]
                if len(v) < self._n_pulses_per_train:
                    raise ProcessingError(
                        f"Length of {ppt} (sliced) is less than "
                        f"{self._n_pulses_per_train}: actual {len(v)}")
                digitizer_apds[i] = v[:self._n_pulses_per_train]

        # apply filter and add data points
        flt = xgm_intensity > self._xgm_threshold

        self._i0.extend(itertools.compress(xgm_intensity, flt))

        for i, apd in enumerate(digitizer_apds):
            if apd is not None:
                self._i1[i].extend(itertools.compress(apd, flt))
        self._energy.extend([energy] * sum(flt))

        self._energy_scan.append((tid, energy))

        # compute spectra
        i0 = self._i0.data()[-self._window:]
        i1 = [i.data()[-self._window:] for i in self._i1]
        energy = self._energy.data()[-self._window:]

        stats = []
        for i, item in enumerate(i1):
            if (1 << i) & self._digitizer_channel_mask:
                mcp_stats, _, _ = compute_spectrum_1d(
                    energy, item, n_bins=self._n_bins)
                stats.append(mcp_stats)
            else:
                # Do not calculate spectrum which is not requested to display
                stats.append(None)

        xgm_stats, centers, counts = compute_spectrum_1d(
            energy, i0, n_bins=self._n_bins)
        for i, item in enumerate(stats):
            if item is not None:
                if i < 3:
                    stats[i] = -np.log(-item / xgm_stats)
                else:
                    # MCP4 has a different spectrum
                    stats[i] = -item / xgm_stats
        stats.append(xgm_stats)

        self.log.info(f"Train {tid} processed")

        return {
            "xgm": xgm_intensity,
            "digitizer": digitizer_apds,
            "mono": self._energy_scan.data(),
            "correlation_length": self._correlation_window,
            "i0": i0,
            "i1": i1,
            "spectra": (stats, centers, counts),
        }

    def onReset(self):
        """Override."""
        self._i0.reset()
        for q in self._i1:
            q.reset()
        self._energy.reset()
        self._energy_scan.reset()
