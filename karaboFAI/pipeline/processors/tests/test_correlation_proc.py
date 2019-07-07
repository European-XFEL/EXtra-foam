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

from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.processors import CorrelationProcessor
from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.config import AnalysisType


class TestCorrelationProcessor(unittest.TestCase):
    def testFomExtraction(self):
        proc = CorrelationProcessor()

        data = {'tid': 1001,
                'processed': ProcessedData(1001, np.random.randn(2, 2)),
                'raw': dict()}

        processed = data['processed']

        # PUMP_PROBE_FOM
        proc.analysis_type = AnalysisType.PUMP_PROBE
        processed.pp.fom = 10
        proc.process(data)
        self.assertEqual(10, processed.correlation.fom)

        # ROI1_SUM_ROI2
        proc.analysis_type = AnalysisType.ROI1_ADD_ROI2
        processed.roi.roi1_fom = 30
        processed.roi.roi2_fom = None
        proc.process(data)
        self.assertEqual(30, processed.correlation.fom)
        processed.roi.roi1_fom = 30
        processed.roi.roi2_fom = -40
        proc.process(data)
        self.assertEqual(-10, processed.correlation.fom)

        # ROI1_SUB_ROI2
        proc.analysis_type = AnalysisType.ROI1_SUB_ROI2
        processed.roi.roi1_fom = None
        processed.roi.roi2_fom = 1.2
        proc.process(data)
        self.assertEqual(-1.2, processed.correlation.fom)
        processed.roi.roi1_fom = 3.2
        processed.roi.roi2_fom = 1.2
        proc.process(data)
        self.assertEqual(2.0, processed.correlation.fom)

        # Nothing happens with unknown FOM type
        proc.analysis_type = 'unknown'
        proc.process(data)

    def testCorrelatorExtraction(self):
        proc = CorrelationProcessor()

        data = {'tid': 1001,
                'processed': ProcessedData(1001, np.random.randn(2, 2)),
                'raw': {
                    'A': {'e': 1},
                    'B': {'f': 2},
                    # 'C': {'g': 3},
                    'D': {'h': 4}
                }
                }
        processed = data['processed']

        proc._device_ids = ['A', 'B', 'C', 'D']
        proc._properties = ['e', 'f', 'g', 'h']

        _get_slow_data = MagicMock(return_value=1.0)

        # set any analysis type
        proc.analysis_type = AnalysisType.PUMP_PROBE
        processed.pp.fom = 10

        with self.assertRaisesRegex(ProcessingError, 'Correlation'):
            proc.process(data)

        self.assertEqual(1, processed.correlation.correlator1)
        self.assertEqual(2, processed.correlation.correlator2)
        self.assertEqual(None, processed.correlation.correlator3)
        self.assertEqual(4, processed.correlation.correlator4)
