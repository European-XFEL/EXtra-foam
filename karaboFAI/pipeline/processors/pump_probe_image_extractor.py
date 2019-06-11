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


class PumpProbeImageExtractor(CompositeProcessor):
    """PumpProbeImageExtractor.

    Extract the pump and probe images in a train.

    Attributes:
        mode (PumpProbeMode): pump-probe mode.
        analysis_type (AnalysisType): pump-probe analysis type.
        on_indices (list): a list of laser-on pulse indices.
        off_indices (list): a list of laser-off pulse indices.
        abs_difference (bool): True for calculating absolute different
            between on/off pulses.
        ma_window (int): moving average window size.
    """
    mode = SharedProperty()
    analysis_type = SharedProperty()
    on_indices = SharedProperty()
    off_indices = SharedProperty()

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
        self.on_indices = self.str2list(cfg['on_pulse_indices'],
                                        handler=int)
        self.off_indices = self.str2list(cfg['off_pulse_indices'],
                                         handler=int)
        self._update_analysis(AnalysisType(int(cfg['analysis_type'])))
        self.ma_window = int(cfg['ma_window'])
        self.abs_difference = cfg['abs_difference'] == 'True'


class PumpProbeImageProcessor(LeafProcessor):
    """Calculate the masked average on/off images ."""
    def __init__(self):
        super().__init__()

        self._buffer = Stack()

    @profiler("Pump-probe Image Processor")
    def process(self, data):
        if self.analysis_type == AnalysisType.UNDEFINED:
            return

        processed = data['processed']

        processed.pp.analysis_type = self.analysis_type

        # setting processing parameters should come before setting data
        processed.pp.ma_window = self.ma_window
        processed.pp.abs_difference = self.abs_difference

        mode = self.mode
        handler = processed.image.sliced_masked_mean

        # on and off are not from different trains
        if mode in (PumpProbeMode.PRE_DEFINED_OFF, PumpProbeMode.SAME_TRAIN):
            on_image = self._slice_pulses(handler, self.on_indices)

            if mode == PumpProbeMode.PRE_DEFINED_OFF:
                off_image = processed.image.masked_ref
                if off_image is None:
                    off_image = np.zeros_like(on_image)
            else:
                off_image = self._slice_pulses(handler, self.off_indices)

            processed.pp.on_image_mean = on_image
            processed.pp.off_image_mean = off_image

            return

        # on and off are from different trains
        if self.mode == PumpProbeMode.EVEN_TRAIN_ON:
            flag = 0
        elif self.mode == PumpProbeMode.ODD_TRAIN_ON:
            flag = 1
        else:
            raise ProcessingError(f"Unknown pump-probe mode: {self.mode}")

        if processed.tid % 2 == 1 ^ flag:
            self._buffer_image(
                ('on', self._slice_pulses(handler, self.on_indices)))
        else:
            self._buffer_image(
                ('off', self._slice_pulses(handler, self.off_indices)))

        if len(self._buffer) == 2:
            processed.pp.off_image_mean = self._buffer.pop()[1]
            processed.pp.on_image_mean = self._buffer.pop()[1]

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

    def _slice_pulses(self, handler, indices):
        try:
            return handler(indices)
        except IndexError as e:
            raise ProcessingError(repr(e))
