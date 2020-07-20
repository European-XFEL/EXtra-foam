"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import patch, MagicMock
import itertools

import numpy as np

from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.processors.digitizer import DigitizerProcessor
from extra_foam.database import SourceItem
from extra_foam.pipeline.exceptions import ProcessingError


class TestDigitizer(_TestDataMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._adq_channels = list(DigitizerProcessor._pulse_integral_channels.keys())
        cls._adq_channels.remove("ADC")
        cls._fastadc_channels = ["ADC"]

    def testPulseIntegral(self):
        data, processed = self.simple_data(1234, (2, 2))
        meta = data['meta']
        raw = data['raw']
        catalog = data['catalog']

        proc = DigitizerProcessor()
        proc._meta.hdel = MagicMock()

        category = 'Digitizer'

        # empty source
        self.assertNotIn(category, catalog)
        proc.process(data)

        # pipeline source with unknown property
        item = SourceItem(category, 'digitizer1:network', [], 'data.intensityTD',
                          slice(None, None), (0, 1000), 1)
        catalog.add_item(item)
        src = f"{item.name} {item.property}"
        meta[src] = {'train_id': 12346}
        raw[src] = [100, 200, 300]
        with self.assertRaises(ProcessingError):
            proc.process(data)
        catalog.remove_item(src)

        # pipeline source with valid property

        integral_vrange = (-1.0, 1.0)
        n_pulses = 5
        for ch in itertools.chain(self._adq_channels, self._fastadc_channels):
            if ch in self._adq_channels:
                item = SourceItem(category, 'digitizer1:network', [],
                                  f'digitizers.channel_1_{ch}.apd.pulseIntegral',
                                  slice(None, None), integral_vrange, 1)
            else:
                item = SourceItem(category, 'digitizer1:channel_2.output', [],
                                  f'data.peaks',
                                  slice(None, None), integral_vrange, 1)
            catalog.clear()

            catalog.add_item(item)
            src = f"{item.name} {item.property}"
            meta[src] = {'train_id': 12346}
            pulse_integral_gt = np.random.randn(n_pulses)
            raw[src] = pulse_integral_gt
            proc.process(data)
            np.testing.assert_array_almost_equal(
                pulse_integral_gt, processed.pulse.digitizer[ch].pulse_integral)
            self.assertEqual(ch, processed.pulse.digitizer.ch_normalizer)

            # test pulse filter
            np.testing.assert_array_equal(
                ((integral_vrange[0] <= pulse_integral_gt) & (pulse_integral_gt <= integral_vrange[1])).nonzero()[0],
                processed.pidx.kept_indices(n_pulses))

            self._reset_processed(processed)

            # test moving average

            # first reset
            proc._update_moving_average({
                'reset_ma': 1,
                'ma_window': 5
            })

            # 1st train
            raw[src] = [10, 20, 30]
            proc.process(data)
            self.assertListEqual([10, 20, 30], processed.pulse.digitizer[ch].pulse_integral.tolist())

            # 2nd train
            raw[src] = [30, 60, 90]
            proc.process(data)
            self.assertListEqual([20, 40, 60], processed.pulse.digitizer[ch].pulse_integral.tolist())

            self._reset_processed(processed)

    def _reset_processed(self, processed):
        for ch in self._adq_channels:
            processed.pulse.digitizer[ch].pulse_integral = None
        for ch in self._fastadc_channels:
            processed.pulse.digitizer[ch].pulse_integral = None
        processed.pidx.reset()
