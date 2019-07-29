"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Unittest for StatisticsProcessor.

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
import numpy as np

from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.pipeline.processors.statistics import StatisticsProcessor
from karaboFAI.config import AnalysisType


class TestStatisticsProcessor(unittest.TestCase):
    """TestStatisticsProcessor class"""
    def setUp(self):
        self._proc = StatisticsProcessor()
        self._proc._num_bins = 5
        self._proc._pulse_resolved = True

    def _get_data(self, pulse_resolved=True):
        processed = ProcessedData(1001)

        if pulse_resolved:
            processed.pulse.roi.roi1.fom = [10, 20, 30, 40, 20,
                                            30, 20, 10, 40, 10]
        else:
            processed.roi.roi1.fom = 10

        data = {'tid': 1001,
                'processed': processed,
                'raw': dict()}
        return data, processed

    def testFomHistogram(self):
        # Test Undefined analysis type
        self._proc.analysis_type = AnalysisType.UNDEFINED

        data, processed = self._get_data()
        self._proc.process(data)

        self.assertIsNone(processed.st.fom_hist)
        self.assertIsNone(processed.st.fom_bin_center)
        self.assertIsNone(processed.st.fom_counts)

        # Test pulse resolved ROI1 SUM
        self._proc.analysis_type = AnalysisType.ROI1_PULSE
        data, processed = self._get_data()
        self._proc.process(data)

        np.testing.assert_array_almost_equal(
            np.array([ 13.,  19.,  25.,  31.,  37.]),
                     processed.st.fom_bin_center)
        np.testing.assert_array_almost_equal(
            np.array([3, 3, 0, 2, 2]), processed.st.fom_counts)

        self._proc.process(data)
        np.testing.assert_array_almost_equal(
            np.array([ 13.,  19.,  25.,  31.,  37.]),
                     processed.st.fom_bin_center)
        np.testing.assert_array_almost_equal(
            np.array([6, 6, 0, 4, 4]), processed.st.fom_counts)

        self._proc.analysis_type = AnalysisType.UNDEFINED
        data, processed = self._get_data()
        self._proc.process(data)

        self.assertIsNone(processed.st.fom_hist)
        self.assertIsNone(processed.st.fom_bin_center)
        self.assertIsNone(processed.st.fom_counts)

        # Test raise of processing error
        self._proc.analysis_type = AnalysisType.AZIMUTHAL_INTEG_PULSE
        self._proc._reset = True
        data, processed = self._get_data()
        with self.assertRaises(ProcessingError):
            self._proc.process(data)

        # Test train resolved ROI1 SUM
        self._proc._pulse_resolved = False
        self._proc.analysis_type = AnalysisType.ROI1
        self._proc._reset = True
        data, processed = self._get_data(pulse_resolved=False)
        self._proc.process(data)

        np.testing.assert_array_almost_equal(
            np.array([9.6, 9.8, 10., 10.2, 10.4]),
                     processed.st.fom_bin_center)

        np.testing.assert_array_almost_equal(
            np.array([0, 0, 1, 0, 0]), processed.st.fom_counts)

        self._proc.process(data)
        np.testing.assert_array_almost_equal(
            np.array([9.6, 9.8, 10., 10.2, 10.4]),
                     processed.st.fom_bin_center)

        np.testing.assert_array_almost_equal(
            np.array([0, 0, 2, 0, 0]), processed.st.fom_counts)

