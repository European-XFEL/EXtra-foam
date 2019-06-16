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
from karaboFAI.config import CorrelationFom


class TestCorrelationProcessor(unittest.TestCase):
    def setUp(self):
        self._proc = CorrelationProcessor()

    def testFomExtraction(self):
        data = {'tid': 1001,
                'processed': ProcessedData(1001, np.random.randn(2, 2)),
                'raw': dict()}

        processed = data['processed']

        # PUMP_PROBE_FOM
        self._proc.fom_type = CorrelationFom.PUMP_PROBE
        processed.pp.fom = None
        with self.assertRaisesRegex(ProcessingError, 'Correlation'):
            self._proc.process(data)
        processed.pp.fom = 10
        self._proc.process(data)
        self.assertEqual(10, processed.correlation.fom)

        # ROI1
        self._proc.fom_type = CorrelationFom.ROI1
        processed.roi.roi1_fom = None
        with self.assertRaisesRegex(ProcessingError, 'Correlation'):
            self._proc.process(data)
        processed.roi.roi1_fom = 20
        self._proc.process(data)
        self.assertEqual(20, processed.correlation.fom)

        # ROI2
        self._proc.fom_type = CorrelationFom.ROI2
        processed.roi.roi2_fom = None
        with self.assertRaisesRegex(ProcessingError, 'Correlation'):
            self._proc.process(data)
        processed.roi.roi2_fom = 30
        self._proc.process(data)
        self.assertEqual(30, processed.correlation.fom)

        # ROI_SUM
        self._proc.fom_type = CorrelationFom.ROI_SUM
        processed.roi.roi1_fom = 30
        processed.roi.roi2_fom = None
        with self.assertRaisesRegex(ProcessingError, 'Correlation'):
            self._proc.process(data)
        processed.roi.roi1_fom = 30
        processed.roi.roi2_fom = -40
        self._proc.process(data)
        self.assertEqual(-10, processed.correlation.fom)

        # ROI_SUB
        self._proc.fom_type = CorrelationFom.ROI_SUB
        processed.roi.roi1_fom = None
        processed.roi.roi2_fom = 1.2
        with self.assertRaisesRegex(ProcessingError, 'Correlation'):
            self._proc.process(data)
        processed.roi.roi1_fom = 3.2
        processed.roi.roi2_fom = 1.2
        self._proc.process(data)
        self.assertEqual(2.0, processed.correlation.fom)

        # Unknown FOM type
        self._proc.fom_type = 'unknown'
        with self.assertRaisesRegex(ProcessingError, 'Correlation'):
            self._proc.process(data)

    def testCorrelatorExtraction(self):
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

        self._proc.device_ids = ['A', 'B', 'C', 'D']
        self._proc.properties = ['e', 'f', 'g', 'h']

        _get_slow_data = MagicMock(return_value=1.0)

        # set any FOM type
        self._proc.fom_type = CorrelationFom.PUMP_PROBE
        processed.pp.fom = 10

        with self.assertRaisesRegex(ProcessingError, 'Correlation'):
            self._proc.process(data)

        self.assertEqual(1, processed.correlation.correlator1)
        self.assertEqual(2, processed.correlation.correlator2)
        self.assertEqual(None, processed.correlation.correlator3)
        # correlator4 is not assigned because processing was stopped when
        # processing correlator3
        self.assertEqual(None, processed.correlation.correlator4)
