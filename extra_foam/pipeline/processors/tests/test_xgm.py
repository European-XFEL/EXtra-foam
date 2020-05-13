"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import MagicMock, patch

from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.processors.xgm import XgmProcessor
from extra_foam.database import SourceItem
from extra_foam.pipeline.exceptions import ProcessingError


class TestXgmProcessor(_TestDataMixin, unittest.TestCase):
    def testGeneral(self):
        data, processed = self.simple_data(1234, (2, 2))
        meta = data['meta']
        raw = data['raw']
        catalog = data['catalog']

        proc = XgmProcessor()

        # empty source
        self.assertNotIn('XGM', catalog)
        proc.process(data)

        # invalid control source
        item = SourceItem('XGM', 'xgm1', [], 'some_property', None, None)
        catalog.add_item(item)
        src = f"{item.name} {item.property}"
        meta[src] = {'train_id': 12346}
        raw[src] = [100, 200, 300]
        with self.assertRaises(ProcessingError):
            proc.process(data)
        catalog.remove_item(src)

        # valid control sources
        src_pf = 'xgm1 pulseEnergy.photonFlux'
        src_bpx = 'xgm1 beamPosition.ixPos'
        src_bpy = 'xgm1 beamPosition.iyPos'
        catalog.add_item(SourceItem('XGM', 'xgm1', [], 'pulseEnergy.photonFlux', None, None))
        catalog.add_item(SourceItem('XGM', 'xgm1', [], 'beamPosition.ixPos', None, None))
        catalog.add_item(SourceItem('XGM', 'xgm1', [], 'beamPosition.iyPos', None, None))

        meta.update({
            src_pf: {'train_id': 12345}, src_bpx: {'train_id': 12345}, src_bpy: {'train_id': 12345}
        })
        raw.update({src_pf: 0.02, src_bpx: 1e-5, src_bpy: -2e-5})
        proc.process(data)
        self.assertEqual(0.02, processed.xgm.intensity)
        self.assertEqual(1e-5, processed.xgm.x)
        self.assertEqual(-2e-5, processed.xgm.y)
        self.assertIsNone(processed.pulse.xgm.intensity)
        self._reset_processed(processed)

        # invalid pipeline source
        item = SourceItem('XGM', 'xgm1:output', [], 'some_property', None, None)
        catalog.add_item(item)
        src = f"{item.name} {item.property}"
        meta[src] = {'train_id': 12346}
        raw[src] = [100, 200, 300]
        with self.assertRaises(ProcessingError):
            proc.process(data)
        catalog.remove_item(src)

        # valid pipeline source
        src_it = 'xgm1:output data.intensityTD'
        catalog.add_item(SourceItem(
            'XGM', 'xgm1:output', [], 'data.intensityTD', slice(None, None), (0, 1000)))
        meta[src_it] = {'train_id': 12346}
        raw[src_it] = [100, 200, 300]
        proc.process(data)
        self.assertListEqual([100, 200, 300], processed.pulse.xgm.intensity.tolist())
        self._reset_processed(processed)

        # same pipeline source with a different slice
        catalog.add_item(SourceItem(
            'XGM', 'xgm1:output', [], 'data.intensityTD', slice(1, 3), (0, 1000)))
        proc.process(data)
        self.assertListEqual([200, 300], processed.pulse.xgm.intensity.tolist())
        self._reset_processed(processed)

        # if the same source has different "intensity" properties, the value of
        # the last one will finally be set in the processed data
        src_it1 = 'xgm1:output data.intensitySa1TD'
        src_it2 = 'xgm1:output data.intensitySa2TD'
        src_it3 = 'xgm1:output data.intensitySa3TD'

        catalog.add_item(SourceItem(
            'XGM', 'xgm1:output', [], 'data.intensitySa1TD', slice(None, None), (0, 1000)))
        catalog.add_item(SourceItem(
            'XGM', 'xgm1:output', [], 'data.intensitySa2TD', slice(1, 4), (0, 100)))
        catalog.add_item(SourceItem(
            'XGM', 'xgm1:output', [], 'data.intensitySa3TD', slice(2, 3), (0, 10)))

        meta.update({
            src_it1: {'train_id': 54321}, src_it2: {'train_id': 54321}, src_it3: {'train_id': 54321}
        })
        raw.update({
            src_it1: [10, 20, 30], src_it2: [1, 2, 3], src_it3: [1000, 2000, 3000],
        })
        with patch("extra_foam.pipeline.processors.xgm.logger.warning") as mocked_warning:
            proc.process(data)
            mocked_warning.assert_called()
        self.assertListEqual([3000], processed.pulse.xgm.intensity.tolist())
        self._reset_processed(processed)

        # remove instrument source
        catalog.remove_item(src_pf)
        catalog.remove_item(src_bpx)
        with patch("extra_foam.pipeline.processors.xgm.logger.warning") as mocked_warning:
            proc.process(data)
            mocked_warning.assert_called()
        self.assertIsNone(processed.xgm.intensity)
        self.assertIsNone(processed.xgm.x)
        self.assertEqual(-2e-5, processed.xgm.y)
        self.assertListEqual([3000], processed.pulse.xgm.intensity.tolist())
        self._reset_processed(processed)

        # remove one pipeline source
        catalog.remove_item(src_it3)
        with patch("extra_foam.pipeline.processors.xgm.logger.warning") as mocked_warning:
            proc.process(data)
            mocked_warning.assert_called()
        # result from data.intensitySa2TD
        self.assertListEqual([2, 3], processed.pulse.xgm.intensity.tolist())
        self._reset_processed(processed)

        # remove all pipeline sources
        catalog.clear()
        proc.process(data)
        self.assertIsNone(processed.pulse.xgm.intensity)
        self._reset_processed(processed)

    def testMovingAverage(self):
        data, processed = self.simple_data(1234, (2, 2))
        meta = data['meta']
        raw = data['raw']
        catalog = data['catalog']

        src_pf = 'xgm1 pulseEnergy.photonFlux'
        catalog.add_item(SourceItem(
            'XGM', 'xgm1', [], 'pulseEnergy.photonFlux', None, None))
        meta[src_pf] = {'train_id': 12345}
        raw[src_pf] = 0.02

        src_it = 'xgm1:output data.intensityTD'
        catalog.add_item(SourceItem(
            'XGM', 'xgm1:output', [], 'data.intensityTD', slice(None, None), None))
        meta[src_it] = {'train_id': 12345}
        raw[src_it] = [100, 200, 300]

        proc = XgmProcessor()
        proc._meta.hdel = MagicMock()
        proc._update_moving_average({
            'reset_ma_xgm': 1,
            'ma_window': 5
        })

        # 1st train
        proc.process(data)
        self.assertAlmostEqual(0.02, processed.xgm.intensity)
        self.assertListEqual([100, 200, 300], processed.pulse.xgm.intensity.tolist())

        data['raw'][src_pf] = 0.04
        data['raw'][src_it] = [10, 20, 30]

        # 2nd train
        proc.process(data)
        self.assertAlmostEqual(0.03, processed.xgm.intensity)
        self.assertListEqual([55, 110, 165], processed.pulse.xgm.intensity.tolist())

    def _reset_processed(self, processed):
        processed.xgm.intensity = None
        processed.xgm.x = None
        processed.xgm.y = None
        processed.pulse.xgm.intensity = None
        processed.pulse.xgm.x = None
        processed.pulse.xgm.y = None
