"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray
from ..exceptions import ProcessingError
from ...database import Metadata as mt
from ...utils import profiler


class DigitizerProcessor(_BaseProcessor):
    """Digitizer data processor.

    TODO: whether to allow multiple digitizer channels have not been decided
          yet. For now, only one digitizer source is allowed to be selected
          in the data source tree.

    Process the Digitizer pipeline data.
    """

    _pulse_integral_a_ma = MovingAverageArray()
    _pulse_integral_b_ma = MovingAverageArray()
    _pulse_integral_c_ma = MovingAverageArray()
    _pulse_integral_d_ma = MovingAverageArray()
    _fast_adc_peaks_ma = MovingAverageArray()

    # AdqDigitizer has a single output channel for multiple digitizer channels,
    # while FastAdc has an output channel for each digitizer channel.
    _integ_channels = {
        'digitizers.channel_1_A.apd.pulseIntegral': 'A',
        'digitizers.channel_1_B.apd.pulseIntegral': 'B',
        'digitizers.channel_1_C.apd.pulseIntegral': 'C',
        'digitizers.channel_1_D.apd.pulseIntegral': 'D',
        'data.peaks': 'ADC'
    }

    _pulse_integral_channels = {
        'A': "_pulse_integral_a_ma",
        'B': "_pulse_integral_b_ma",
        'C': "_pulse_integral_c_ma",
        'D': "_pulse_integral_d_ma",
        'ADC': "_fast_adc_peaks_ma"
    }

    def __init__(self):
        super().__init__()

        self._ma_window = 1

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.GLOBAL_PROC)
        self._update_moving_average(cfg)

    def _update_moving_average(self, cfg):
        if 'reset_ma_digitizer' in cfg:
            # reset moving average
            del self._pulse_integral_a_ma
            del self._pulse_integral_b_ma
            del self._pulse_integral_c_ma
            del self._pulse_integral_d_ma
            del self._fast_adc_peaks_ma

            self._meta.hdel(mt.GLOBAL_PROC, 'reset_ma_digitizer')

        v = int(cfg['ma_window'])
        if self._ma_window != v:
            self.__class__._pulse_integral_a_ma.window = v
            self.__class__._pulse_integral_b_ma.window = v
            self.__class__._pulse_integral_c_ma.window = v
            self.__class__._pulse_integral_d_ma.window = v
            self.__class__._fast_adc_peaks_ma.window = v

        self._ma_window = v

    @profiler("Digitizer Processor")
    def process(self, data):
        """Override."""
        processed = data['processed']
        raw = data['raw']
        catalog = data['catalog']

        digitizer_srcs = catalog.from_category('Digitizer')
        for src in digitizer_srcs:
            arr = raw[src]
            device_id, ppt = src.split(' ')
            if ppt in self._integ_channels:
                channel = self._integ_channels[ppt]
                attr_name = self._pulse_integral_channels[channel]
                self.__class__.__dict__[attr_name].__set__(self, np.array(
                    arr[catalog.get_slicer(src)], dtype=np.float32))

                processed.pulse.digitizer[channel].pulse_integral = \
                    self.__class__.__dict__[attr_name].__get__(
                        self, self.__class__)

                # apply pulse filter
                self.filter_pulse_by_vrange(
                    self.__class__.__dict__[attr_name].__get__(
                        self, self.__class__),
                    catalog.get_vrange(src),
                    processed.pidx,
                    src)

                # It is allowed to select only one digitizer channel
                processed.pulse.digitizer.ch_normalizer = channel
            else:
                # This is not UnknownParameterError since the property input
                # by the user maybe a valid property for the digitizer device.
                raise ProcessingError(
                    f'[Digitizer] Unknown property: {ppt}')
