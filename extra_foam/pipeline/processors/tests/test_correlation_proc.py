"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import MagicMock

import numpy as np

from extra_foam.pipeline.data_model import ImageData, ProcessedData
from extra_foam.pipeline.processors import CorrelationProcessor
from extra_foam.pipeline.exceptions import ProcessingError
from extra_foam.config import AnalysisType

from extra_foam.pipeline.processors.tests import _BaseProcessorTest


class TestCorrelationProcessor(unittest.TestCase, _BaseProcessorTest):
    def testGeneral(self):
        proc = CorrelationProcessor()
        proc._reset = True

        data, processed = self.simple_data(1001, (2, 2))

        proc.process(data)

        self.assertTrue(processed.corr.correlation1.reset)
        self.assertTrue(processed.corr.correlation2.reset)

    def testFomExtraction(self):
        proc = CorrelationProcessor()

        data, processed = self.simple_data(1001, (2, 2))

        # PUMP_PROBE_FOM
        proc.analysis_type = AnalysisType.PUMP_PROBE
        proc.process(data)
        with self.assertRaises(ProcessingError):
            proc.process(data)
        self.assertIsNone(processed.corr.correlation1.y)
        processed.pp.fom = 10
        proc.process(data)
        self.assertEqual(10, processed.corr.correlation1.y)

        # ROI FOM
        proc.analysis_type = AnalysisType.ROI_FOM
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation1.y)
        processed.roi.fom = 11
        proc.process(data)
        self.assertEqual(11, processed.corr.correlation1.y)

        # ROI projection
        proc.analysis_type = AnalysisType.ROI_PROJ
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation1.y)
        processed.roi.proj.fom = 15
        proc.process(data)
        self.assertEqual(15, processed.corr.correlation1.y)

        # AZIMUTHAL_INTEG
        proc.analysis_type = AnalysisType.AZIMUTHAL_INTEG
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation2.y)
        processed.ai.fom = 19
        proc.process(data)
        self.assertEqual(19, processed.corr.correlation2.y)

        # Nothing happens with unknown FOM type
        proc.analysis_type = AnalysisType.UNDEFINED
        proc.process(data)

    def testCorrelatorExtraction(self):
        proc = CorrelationProcessor()

        processed = ProcessedData(1001)
        processed.image = ImageData.from_array(np.random.randn(2, 2))
        data = {'processed': processed,
                'raw': {
                    'A': {'e': 1},
                    # 'B': {'f': 2},
                }
        }

        proc._device_ids = ['A', 'B']
        proc._properties = ['e', 'f']
        proc._resolutions = [0.0, 1.0]

        # set any analysis type
        proc.analysis_type = AnalysisType.PUMP_PROBE
        processed.pp.fom = 10

        with self.assertRaisesRegex(ProcessingError, 'Correlation'):
            proc.process(data)

        self.assertEqual('A', processed.corr.correlation1.device_id)
        self.assertEqual('e', processed.corr.correlation1.property)
        self.assertEqual(0, processed.corr.correlation1.resolution)
        self.assertEqual(1, processed.corr.correlation1.x)
        self.assertEqual(10, processed.corr.correlation1.y)

        self.assertEqual('B', processed.corr.correlation2.device_id)
        self.assertEqual('f', processed.corr.correlation2.property)
        self.assertEqual(1.0, processed.corr.correlation2.resolution)
        self.assertEqual(None, processed.corr.correlation2.x)
        self.assertEqual(10, processed.corr.correlation2.y)
