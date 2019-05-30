"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeProcessors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import (
    LeafProcessor, CompositeProcessor, SharedProperty,
)
from ..exceptions import ProcessingError
from ...algorithms import Stack
from ...config import PumpProbeMode, AnalysisType
from ...metadata import Metadata as mt
from ...utils import profiler


class PumpProbeProcessor(CompositeProcessor):
    """PumpProbeProcessor.

    Attributes:
        mode (PumpProbeMode): pump-probe mode.
        analysis_type (AnalysisType): pump-probe analysis type.
        on_pulse_indices (list): a list of laser-on pulse indices.
        off_pulse_indices (list): a list of laser-off pulse indices.
        abs_difference (bool): True for calculating absolute different
            between on/off pulses.
        ma_window (int): moving average window size.
    """
    mode = SharedProperty()
    analysis_type = SharedProperty()
    on_pulse_indices = SharedProperty()
    off_pulse_indices = SharedProperty()

    abs_difference = SharedProperty()
    ma_window = SharedProperty()

    def __init__(self):
        super().__init__()

        self.add(PumpProbeImageProcessor())

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.PUMP_PROBE_PROC)
        if cfg is None:
            return

        self.mode = PumpProbeMode(int(cfg['mode']))
        self.on_pulse_indices = self.str2list(cfg['on_pulse_indices'], handler=int)
        self.off_pulse_indices = self.str2list(cfg['off_pulse_indices'], handler=int)
        self._update_analysis(AnalysisType(int(cfg['analysis_type'])))
        self.ma_window = int(cfg['ma_window'])
        self.abs_difference = cfg['abs_difference'] == 'True'


class PumpProbeImageProcessor(LeafProcessor):
    def __init__(self):
        super().__init__()

        self._buffer = Stack()

    def _buffer_image(self, item):
        if len(self._buffer) == 0:
            if item[0] == 'on':
                self._buffer.push(item)
            else:
                return  # 'off' will not be acknowledged along
        else:
            # 'on' is already buffered
            if item[0] == 'on':
                self._buffer.pop()  # remove the old on

            self._buffer.push(item)

    @profiler("Pump-probe image processor")
    def process(self, processed, raw=None):
        if self.mode == PumpProbeMode.UNDEFINED or \
                self.analysis_type == AnalysisType.UNDEFINED:
            return

        self._check_pulse_indices(processed.n_pulses)

        if self.mode == PumpProbeMode.PRE_DEFINED_OFF:
            on_image = processed.image.masked_mean
            off_image = processed.image.masked_ref
            if off_image is None:
                off_image = np.zeros_like(on_image)
            self._buffer_image(('on', on_image))
            self._buffer_image(('off', off_image))
        else:
            handler = processed.image.sliced_masked_mean

            if self.mode == PumpProbeMode.SAME_TRAIN:
                self._buffer_image(('on', handler(self.on_pulse_indices)))
                self._buffer_image(('off', handler(self.off_pulse_indices)))
            else:
                if self.mode == PumpProbeMode.EVEN_TRAIN_ON:
                    flag = 0
                elif self.mode == PumpProbeMode.ODD_TRAIN_ON:
                    flag = 1
                else:
                    raise ProcessingError(
                        f"Unknown pump-probe mode: {self.mode}")

                if processed.tid % 2 == 1 ^ flag:
                    self._buffer_image(('on', handler(self.on_pulse_indices)))
                else:
                    self._buffer_image(('off', handler(self.off_pulse_indices)))

        if len(self._buffer) == 2:
            processed.pp.analysis_type = self.analysis_type
            # setting processing parameters should come before setting data
            processed.pp.ma_window = self.ma_window
            processed.pp.abs_difference = self.abs_difference

            processed.pp.off_image_mean = self._buffer.pop()[1]
            processed.pp.on_image_mean = self._buffer.pop()[1]

    def _check_pulse_indices(self, n_pulses):
        max_on_pulse_index = max(self.on_pulse_indices)
        if max_on_pulse_index >= n_pulses:
            raise ProcessingError(
                f"Out of range: on-pulse index = {max_on_pulse_index}, "
                f"total number of pulses = {n_pulses}")

        if self.mode != PumpProbeMode.PRE_DEFINED_OFF:
            max_off_pulse_index = max(self.off_pulse_indices)
            if max_off_pulse_index >= n_pulses:
                raise ProcessingError(
                    f"Out of range: off-pulse index = {max_off_pulse_index}, "
                    f"total number of pulses = {n_pulses}")
