"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Unittest for CorrelationProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import MagicMock

import numpy as np

from karaboFAI.pipeline.data_model import ImageData, ProcessedData
from karaboFAI.pipeline.processors import CorrelationProcessor
from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.config import AnalysisType

from karaboFAI.pipeline.processors.tests import _BaseProcessorTest


class TestCorrelationProcessor(_BaseProcessorTest):
    def testGeneral(self):
        proc = CorrelationProcessor()
        proc._reset = True

        data, processed = self.simple_data(1001, (2, 2))

        proc.process(data)

        self.assertTrue(processed.corr.correlation1.reset)
        self.assertTrue(processed.corr.correlation2.reset)
        self.assertTrue(processed.corr.correlation3.reset)
        self.assertTrue(processed.corr.correlation4.reset)

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

        # ROI1
        proc.analysis_type = AnalysisType.ROI1
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation1.y)
        processed.roi.roi1.fom = 11
        proc.process(data)
        self.assertEqual(11, processed.corr.correlation1.y)

        # ROI2
        proc.analysis_type = AnalysisType.ROI2
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation2.y)
        processed.roi.roi2.fom = 12
        proc.process(data)
        self.assertEqual(12, processed.corr.correlation2.y)

        # ROI1 - ROI2
        proc.analysis_type = AnalysisType.ROI1_SUB_ROI2
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation2.y)
        processed.roi.roi1_sub_roi2.fom = 13
        proc.process(data)
        self.assertEqual(13, processed.corr.correlation2.y)

        # ROI1 + ROI2
        proc.analysis_type = AnalysisType.ROI1_ADD_ROI2
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation3.y)
        processed.roi.roi1_add_roi2.fom = 14
        proc.process(data)
        self.assertEqual(14, processed.corr.correlation3.y)

        # ROI1 projection
        proc.analysis_type = AnalysisType.PROJ_ROI1
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation3.y)
        processed.roi.proj1.fom = 15
        proc.process(data)
        self.assertEqual(15, processed.corr.correlation3.y)

        # ROI2 projection
        proc.analysis_type = AnalysisType.PROJ_ROI2
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation4.y)
        processed.roi.proj2.fom = 16
        proc.process(data)
        self.assertEqual(16, processed.corr.correlation4.y)

        # ROI1_ADD_ROI2 projection
        proc.analysis_type = AnalysisType.PROJ_ROI1_ADD_ROI2
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation1.y)
        processed.roi.proj1_add_proj2.fom = 17
        proc.process(data)
        self.assertEqual(17, processed.corr.correlation1.y)

        # ROI1_SUB_ROI2 projection
        proc.analysis_type = AnalysisType.PROJ_ROI1_SUB_ROI2
        with self.assertRaises(ProcessingError):
            proc.process(data)  # FOM is not available
        self.assertIsNone(processed.corr.correlation1.y)
        processed.roi.proj1_sub_proj2.fom = 18
        proc.process(data)
        self.assertEqual(18, processed.corr.correlation1.y)

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
                    'B': {'f': 2},
                    # 'C': {'g': 3},
                    'D': {'h': 4}
                }
        }

        proc._device_ids = ['A', 'B', 'C', 'D']
        proc._properties = ['e', 'f', 'g', 'h']
        proc._resolutions = [0.0, 1.0, 2.0, 3.0]

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
        self.assertEqual(2, processed.corr.correlation2.x)
        self.assertEqual(10, processed.corr.correlation2.y)

        self.assertEqual('C', processed.corr.correlation3.device_id)
        self.assertEqual('g', processed.corr.correlation3.property)
        self.assertEqual(2.0, processed.corr.correlation3.resolution)
        self.assertEqual(None, processed.corr.correlation3.x)
        self.assertEqual(10, processed.corr.correlation3.y)

        self.assertEqual('D', processed.corr.correlation4.device_id)
        self.assertEqual('h', processed.corr.correlation4.property)
        self.assertEqual(3.0, processed.corr.correlation4.resolution)
        self.assertEqual(4, processed.corr.correlation4.x)
        self.assertEqual(10, processed.corr.correlation4.y)
