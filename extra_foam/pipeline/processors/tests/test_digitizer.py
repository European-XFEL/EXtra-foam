"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import patch, MagicMock

from extra_foam.pipeline.processors.tests import _BaseProcessorTest
from extra_foam.pipeline.processors.digitizer import DigitizerProcessor
from extra_foam.database import SourceItem
from extra_foam.pipeline.exceptions import UnknownParameterError


class TestDigitizer(unittest.TestCase, _BaseProcessorTest):
    @classmethod
    def setUpClass(cls):
        cls._channels = DigitizerProcessor._pulse_integral_channels.keys()

    def testGeneral(self):
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
                          slice(None, None), (0, 1000))
        catalog.add_item(item)
        src = f"{item.name} {item.property}"
        meta[src] = {'tid': 12346}
        raw[src] = [100, 200, 300]
        with self.assertRaises(UnknownParameterError):
            proc.process(data)
        catalog.remove_item(src)

        # pipeline source with valid property

        for ch in self._channels:
            item = SourceItem(category, 'digitizer1:network', [],
                              f'digitizers.channel_1_{ch}.apd.pulseIntegral',
                              slice(None, None), (0, 1000))
            catalog.add_item(item)
            src = f"{item.name} {item.property}"
            meta[src] = {'tid': 12346}
            raw[src] = [100, 200, 300]
            proc.process(data)
            self.assertListEqual([100, 200, 300],
                                 processed.pulse.digitizer[ch].pulse_integral.tolist())
            self.assertEqual(ch, processed.pulse.digitizer.ch_normalizer)
            self._reset_processed(processed)

            # test moving average

            # first reset
            with patch.object(proc._meta, "hdel") as patched:
                proc._update_moving_average({
                    'reset_ma_digitizer': 1,
                    'ma_window': 5
                })
                patched.assert_called_once()

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
        for ch in self._channels:
            processed.pulse.digitizer[ch].pulse_integral = None
