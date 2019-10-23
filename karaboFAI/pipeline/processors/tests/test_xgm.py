"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from karaboFAI.database import SourceItem
from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.pipeline.processors.tests import _BaseProcessorTest
from karaboFAI.pipeline.processors.xgm import (
    XgmProcessor
)


class TestXgmProcessor(_BaseProcessorTest):
    def testGeneral(self):
        data, processed = self.simple_data(1234, (2, 2))

        # empty source
        proc = XgmProcessor()
        proc.process(data)

        # new instrument source
        proc._sources = [SourceItem('XGM', 'xgm1', "pulseEnergy.photonFlux")]
        with self.assertRaises(ProcessingError):
            proc.process(data)
        # set correct source
        data['raw']['xgm1'] = {
            "pulseEnergy.photonFlux": 0.02,
            "beamPosition.ixPos": 1e-5,
            "beamPosition.iyPos": -2e-5,
        }
        proc.process(data)
        self.assertEqual(0.02, processed.xgm.intensity)
        self.assertIsNone(processed.xgm.x)
        self.assertIsNone(processed.pulse.xgm.intensity)
        self._reset_processed(processed)

        # one more instrument source
        proc._sources.append(SourceItem('XGM', 'xgm1', "beamPosition.ixPos"))
        proc.process(data)
        self.assertEqual(0.02, processed.xgm.intensity)
        self.assertEqual(1e-5, processed.xgm.x)
        self.assertIsNone(processed.pulse.xgm.intensity)
        self._reset_processed(processed)

        # new pipeline source
        proc._sources.append(SourceItem('XGM', 'xgm1:output', "data.intensityTD", slice(None, None)))
        with self.assertRaises(ProcessingError):
            proc.process(data)
        # set correct source
        data['raw']['xgm1:output'] = {
            "data.intensityTD": [100, 200, 300]
        }
        proc.process(data)
        self.assertEqual(0.02, processed.xgm.intensity)
        self.assertEqual(1e-5, processed.xgm.x)
        self.assertListEqual([100, 200, 300], processed.pulse.xgm.intensity.tolist())
        self._reset_processed(processed)

        # same pipeline source with a different slice
        proc._sources.pop(-1)
        proc._sources.append(SourceItem('XGM', 'xgm1:output', "data.intensityTD", slice(1, 3)))
        proc.process(data)
        self.assertEqual(0.02, processed.xgm.intensity)
        self.assertEqual(1e-5, processed.xgm.x)
        self.assertListEqual([200, 300], processed.pulse.xgm.intensity.tolist())
        self._reset_processed(processed)

        # remove instrument source
        proc._sources.pop(0)
        proc.process(data)
        self.assertIsNone(processed.xgm.intensity)
        self.assertEqual(1e-5, processed.xgm.x)
        self.assertListEqual([200, 300], processed.pulse.xgm.intensity.tolist())
        self._reset_processed(processed)

        # remove the other instrument source
        proc._sources.pop(0)
        proc.process(data)
        self.assertIsNone(processed.xgm.intensity)
        self.assertIsNone(processed.xgm.x)
        self.assertListEqual([200, 300], processed.pulse.xgm.intensity.tolist())
        self._reset_processed(processed)

        # remove pipeline source
        proc._sources.pop(0)
        proc.process(data)
        self.assertIsNone(processed.xgm.intensity)
        self.assertIsNone(processed.xgm.x)
        self.assertIsNone(processed.pulse.xgm.intensity)
        self._reset_processed(processed)

    def testMovingAverage(self):
        data, processed = self.simple_data(1234, (2, 2))
        data['raw']['xgm1'] = {
            "pulseEnergy.photonFlux": 0.02,
            "beamPosition.ixPos": 1e-5,
            "beamPosition.iyPos": -2e-5,
        }
        data['raw']['xgm1:output'] = {
            "data.intensityTD": [100, 200, 300]
        }

        proc = XgmProcessor()
        proc._update_moving_average({
            'reset_ma_xgm': 1,
            'ma_window': 5
        })
        proc._sources = [
            SourceItem('XGM', 'xgm1', "pulseEnergy.photonFlux"),
            SourceItem('XGM', 'xgm1', "beamPosition.ixPos"),
            SourceItem('XGM', 'xgm1', "beamPosition.iyPos"),
            SourceItem('XGM', 'xgm1:output', "data.intensityTD", slice(None, None)),
        ]

        # 1st train
        proc.process(data)
        self.assertAlmostEqual(0.02, processed.xgm.intensity)
        self.assertAlmostEqual(1e-5, processed.xgm.x)
        self.assertAlmostEqual(-2e-5, processed.xgm.y)
        self.assertListEqual([100, 200, 300], processed.pulse.xgm.intensity.tolist())

        data['raw']['xgm1'] = {
            "pulseEnergy.photonFlux": 0.04,
            "beamPosition.ixPos": 3e-5,
            "beamPosition.iyPos": -4e-5,
        }
        data['raw']['xgm1:output'] = {
            "data.intensityTD": [10, 20, 30]
        }

        # 2nd train
        proc.process(data)
        self.assertAlmostEqual(0.03, processed.xgm.intensity)
        self.assertAlmostEqual(2e-5, processed.xgm.x)
        self.assertAlmostEqual(-3e-5, processed.xgm.y)
        self.assertListEqual([55, 110, 165], processed.pulse.xgm.intensity.tolist())

    def _reset_processed(self, processed):
        processed.xgm.intensity = None
        processed.xgm.x = None
        processed.xgm.y = None
        processed.pulse.xgm.intensity = None
        processed.pulse.xgm.x = None
        processed.pulse.xgm.y = None
