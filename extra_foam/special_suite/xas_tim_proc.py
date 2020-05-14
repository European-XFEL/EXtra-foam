"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from extra_foam.algorithms import compute_spectrum_1d
from extra_foam.pipeline.processors.base_processor import (
    SimpleSequence, SimplePairSequence
)

from .special_analysis_base import ProcessingError, profiler, QThreadWorker

_DEFAULT_N_PULSES_PER_TRAIN = 1
_DEFAULT_I0_THRESHOLD = 0.0
_MAX_WINDOW = 180000  # 60 s * 10 train/s * 300 pulses/train
_MAX_CORRELATION_WINDOW = 3000
_MAX_N_BINS = 999
_DEFAULT_N_BINS = 80
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
        _digitizer_channels (list): A list of boolean to indicates the
            required digitizer channel.
        _n_pulses_per_train (int): Number of pulses per train.
        _apd_stride (int): Pulse index stride of the digitizer APD data.
        _i0_threshold (float): Lower boundary of the XGM intensity. Pulses
            will be dropped if the intensity is below the threshold.
        _window (int): Maximum number of pulses used to calculating spectra.
        _correlation_window (int): Maximum number of pulses in correlation
            plots. It includes the pulses which are dropped by the filter.
        _n_bins (int): Number of bins in spectra calculation.
        _i0 (SimpleSequence): Store XGM pulse intensities.
        _i1 (list): A list of SimpleSequence, which stores pulsed apd data
            for each digitizer channel.
        _energy (SimpleSequence): Store pulse energies.
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

        self._digitizer_channels = [False] * 4

        self._n_pulses_per_train = _DEFAULT_N_PULSES_PER_TRAIN
        self._apd_stride = 1
        self._i0_threshold = _DEFAULT_I0_THRESHOLD
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

    def onDigitizerChannelsChanged(self, index: int, value: bool):
        self._digitizer_channels[index] = value
        if value:
            # reset the data history when a new channel is added in order to
            # ensure the same length of data history
            self.reset()

    def onMonoDeviceChanged(self, device: str):
        self._mono_device_id = device

    def onNPulsesPerTrainChanged(self, value: str):
        self._n_pulses_per_train = int(value)

    def onApdStrideChanged(self, value: str):
        self._apd_stride = int(value)

    def onI0ThresholdChanged(self, value: str):
        self._i0_threshold = float(value)

    def onPulseWindowChanged(self, value: str):
        self._window = int(value)

    def onCorrelationWindowChanged(self, value: str):
        self._correlation_window = int(value)

    def onNoBinsChanged(self, value: str):
        self._n_bins = int(value)

    def sources(self):
        """Override."""
        return [
            (self._xgm_output_channel, self._xgm_ppt, 1),
            *[(self._digitizer_output_channel, ppt, 1)
              for ppt in self._digitizer_ppts],
            (self._mono_device_id, self._mono_ppt, 0)
        ]

    def _update_data_history(self, data):
        data, meta = data["raw"], data["meta"]

        tid = self.getTrainId(meta)

        xgm_intensity = self.getPropertyData(
            data, self._xgm_output_channel, self._xgm_ppt)

        digitizer_apds = []
        if sum(self._digitizer_channels) == 0:
            raise ProcessingError(
                "At least one digitizer channel is required!")
        for i, ppt in enumerate(self._digitizer_ppts):
            if self._digitizer_channels[i]:
                apd = self.getPropertyData(
                    data, self._digitizer_output_channel, ppt)
                if apd is None:
                    raise ProcessingError(
                        f"Digitizer channel {ppt} not found!")
                digitizer_apds.append(apd)
            else:
                digitizer_apds.append(None)

        energy = self.getPropertyData(
            data, self._mono_device_id, self._mono_ppt)

        # Check and slice XGM intensity.
        pulse_slicer = slice(0, self._n_pulses_per_train)
        if len(xgm_intensity) < self._n_pulses_per_train:
            raise ProcessingError(f"Length of {self._xgm_ppt} is less "
                                  f"than {self._n_pulses_per_train}: "
                                  f"actual {len(xgm_intensity)}")
        xgm_intensity = xgm_intensity[pulse_slicer]

        # Check and slice digitizer APD data.
        for i, (apd, ppt) in enumerate(zip(digitizer_apds,
                                           self._digitizer_ppts)):
            if self._digitizer_channels[i]:
                v = apd[::self._apd_stride]
                if len(v) < self._n_pulses_per_train:
                    raise ProcessingError(
                        f"Length of {ppt} (sliced) is less than "
                        f"{self._n_pulses_per_train}: actual {len(v)}")
                digitizer_apds[i] = v[:self._n_pulses_per_train]

        # update data history
        self._i0.extend(xgm_intensity)
        for i, apd in enumerate(digitizer_apds):
            if self._digitizer_channels[i]:
                self._i1[i].extend(apd)
        self._energy.extend([energy] * len(xgm_intensity))

        self._energy_scan.append((tid, energy))

        return tid, xgm_intensity, digitizer_apds, energy

    @profiler("XAS-TIM Processor")
    def process(self, data):
        """Override."""
        tid, xgm_intensity, digitizer_apds, energy = \
            self._update_data_history(data)

        # apply filter
        flt = self._i0.data() > self._i0_threshold
        i0 = self._i0.data()[flt][-self._window:]
        i1 = [None] * 4
        for i, _item in enumerate(self._i1):
            if self._digitizer_channels[i]:
                i1[i] = _item.data()[flt][-self._window:]
        energy = self._energy.data()[flt][-self._window:]

        # compute spectra
        stats = []
        for i, item in enumerate(i1):
            if self._digitizer_channels[i]:
                mcp_stats, _, _ = compute_spectrum_1d(
                    energy, item, n_bins=self._n_bins)
                stats.append(mcp_stats)
            else:
                # Do not calculate spectrum which is not requested to display
                stats.append(None)

        i0_stats, centers, counts = compute_spectrum_1d(
            energy, i0, n_bins=self._n_bins)
        for i, _item in enumerate(stats):
            if _item is not None:
                if i < 3:
                    stats[i] = -np.log(-_item / i0_stats)
                else:
                    # MCP4 has a different spectrum
                    stats[i] = -_item / i0_stats
        stats.append(i0_stats)

        self.log.info(f"Train {tid} processed")

        return {
            "xgm_intensity": xgm_intensity,
            "digitizer_apds": digitizer_apds,
            "energy_scan": self._energy_scan.data(),
            "correlation_length": self._correlation_window,
            "i0": i0,
            "i1": i1,
            "spectra": (stats, centers, counts),
        }

    def reset(self):
        """Override."""
        self._i0.reset()
        for item in self._i1:
            item.reset()
        self._energy.reset()
        self._energy_scan.reset()
