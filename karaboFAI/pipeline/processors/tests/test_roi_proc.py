"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Unittest for RoiProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
import numpy as np

from karaboFAI.pipeline.data_model import ProcessedData, RoiData
from karaboFAI.pipeline.processors.roi import RoiProcessor


class TestRoiProcessor(unittest.TestCase):
    def testFom(self):
        proc = RoiProcessor()

        default_roi = (0, 0, -1, -1)
        data = {'tid': 1001,
                'processed': ProcessedData(1001, np.ones((100, 100))),
                'raw': dict()}
        processed = data['processed']

        proc._roi_fom_handler = np.sum

        # test default values
        self.assertEqual([default_roi] * 4, proc._regions)
        self.assertListEqual([False] * 4, proc._visibilities)

        # set ROI1 and ROI4
        proc._visibilities[0] = True
        proc._regions[0] = [0, 0, 2, 3]
        proc._visibilities[3] = True
        proc._regions[3] = [1, 1, 4, 3]

        proc.process(data)

        # ROI regions
        self.assertEqual(processed.roi.roi1, [0, 0, 2, 3])
        self.assertEqual(processed.roi.roi2, None)
        self.assertEqual(processed.roi.roi3, None)
        self.assertEqual(processed.roi.roi4, [1, 1, 4, 3])
        # ROI FOMs
        self.assertEqual(6, processed.roi.roi1_fom)
        self.assertEqual(None, processed.roi.roi2_fom)
        self.assertEqual(None, processed.roi.roi3_fom)
        self.assertEqual(12, processed.roi.roi4_fom)
        # ROI 1D projection x
        np.testing.assert_array_almost_equal(np.array([3, 3]),
                                             processed.roi.roi1_proj_x)
        self.assertEqual(None, processed.roi.roi2_proj_x)
        self.assertEqual(None, processed.roi.roi3_proj_x)
        np.testing.assert_array_almost_equal(np.array([3, 3, 3, 3]),
                                             processed.roi.roi4_proj_x)
        # ROI 1D projection y
        np.testing.assert_array_almost_equal(np.array([2, 2, 2]),
                                             processed.roi.roi1_proj_y)
        self.assertEqual(None, processed.roi.roi2_proj_y)
        self.assertEqual(None, processed.roi.roi3_proj_y)
        np.testing.assert_array_almost_equal(np.array([4, 4, 4]),
                                             processed.roi.roi4_proj_y)
