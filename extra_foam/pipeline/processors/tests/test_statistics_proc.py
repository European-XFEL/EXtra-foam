"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Ebad Kamil <ebad.kamil@xfel.eu>, Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest

import numpy as np

from extra_foam.pipeline.data_model import ProcessedData
from extra_foam.pipeline.exceptions import ProcessingError
from extra_foam.pipeline.processors.statistics import StatisticsProcessor
from extra_foam.pipeline.processors.tests import _BaseProcessorTest
from extra_foam.config import AnalysisType


class TestStatisticsProcessor(unittest.TestCase, _BaseProcessorTest):
    """TestStatisticsProcessor class"""
    def setUp(self):
        self._proc = StatisticsProcessor()
        self._proc._num_bins = 5

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

    def testGeneral(self):
        # Test Undefined analysis type
        self._proc.analysis_type = AnalysisType.UNDEFINED

        data, processed = self.simple_data(1001, (4, 2, 2))
        self._proc.process(data)

        self.assertIsNone(processed.st.fom_hist)
        self.assertIsNone(processed.st.fom_bin_center)
        self.assertIsNone(processed.st.fom_count)
        self.assertIsNone(processed.st.poi_fom_bin_center)
        self.assertIsNone(processed.st.poi_fom_count)

    def testFomHistogramPr(self):
        proc = self._proc
        proc._pulse_resolved = True

        # ROI FOM
        proc.analysis_type = AnalysisType.ROI_FOM_PULSE
        data, processed = self.simple_data(1001, (5, 2, 2))

        with self.assertRaisesRegex(ProcessingError, r".Statistics. Pulse .* ROI"):
            proc.process(data)

        fom_gt = [10, 20, 30, 40, 20, 30, 20, 10, 40, 10]
        processed.pulse.roi.fom = fom_gt.copy()
        proc.process(data)
        np.testing.assert_array_equal(fom_gt, proc._fom)
        np.testing.assert_array_almost_equal(
            [13.,  19.,  25.,  31.,  37.], processed.st.fom_bin_center)
        np.testing.assert_array_almost_equal(
            [3, 3, 0, 2, 2], processed.st.fom_count)
        # test POI statistics
        self.assertEqual(5, processed.n_pulses)
        self.assertListEqual([0, 0], processed.image.poi_indices)
        np.testing.assert_array_equal([1, 0, 0, 0, 1], processed.st.poi_fom_count[0])
        np.testing.assert_array_equal([12, 16, 20, 24, 28], processed.st.poi_fom_bin_center[0])
        for i in range(1, 5):
            self.assertIsNone(processed.st.poi_fom_count[i])
            self.assertIsNone(processed.st.poi_fom_bin_center[i])

        # the same data come again
        proc.process(data)
        np.testing.assert_array_equal(fom_gt * 2, proc._fom)
        np.testing.assert_array_almost_equal(
            [13.,  19.,  25.,  31.,  37.], processed.st.fom_bin_center)
        np.testing.assert_array_almost_equal(
            [6, 6, 0, 4, 4], processed.st.fom_count)

    def testFomHistogramTr(self):
        proc = self._proc
        proc._pulse_resolved = False

        # ROI FOM
        proc.analysis_type = AnalysisType.ROI_FOM
        data, processed = self.simple_data(1001, (2, 2))

        with self.assertRaisesRegex(ProcessingError, r'.Statistics. ROI'):
            proc.process(data)

        fom_gt = [10, 20, 30, 40, 20, 30, 20, 10, 40, 10]
        for item in fom_gt:
            processed.roi.fom = item
            proc.process(data)

        np.testing.assert_array_equal(fom_gt, proc._fom)
        np.testing.assert_array_almost_equal(
            [13.,  19.,  25.,  31.,  37.], processed.st.fom_bin_center)
        np.testing.assert_array_almost_equal(
            [3, 3, 0, 2, 2], processed.st.fom_count)

        # the same data comes again
        for item in fom_gt:
            processed.roi.fom = item
            proc.process(data)

        np.testing.assert_array_equal(fom_gt * 2, proc._fom)
        np.testing.assert_array_almost_equal(
            [13.,  19.,  25.,  31.,  37.], processed.st.fom_bin_center)
        np.testing.assert_array_almost_equal(
            [6, 6, 0, 4, 4], processed.st.fom_count)
