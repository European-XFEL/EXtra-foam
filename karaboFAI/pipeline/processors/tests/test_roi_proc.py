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
from unittest.mock import MagicMock
import numpy as np

from karaboFAI.pipeline.data_model import ImageData, ProcessedData
from karaboFAI.pipeline.exceptions import ProcessingError
from karaboFAI.pipeline.processors.roi import (
    _RectROI, RoiProcessorTrain, RoiProcessorPulse
)
from karaboFAI.config import AnalysisType, VFomNormalizer
from karaboFAI.pipeline.processors.tests import _BaseProcessorTest


class TestRectROI(unittest.TestCase):
    def testGeneral(self):
        roi = _RectROI()

        roi.rect = (0, 0, 2, 2)

        img = np.arange(16).reshape(4, 4)
        img_on = 3 * np.ones((4, 4))
        img_off = np.ones((4, 4))

        # not activated
        self.assertIsNone(roi.get_image(img))
        self.assertTupleEqual((None, None), roi.get_images_pp(img_on, img_off))

        # activate the _RectROI
        roi.activated = True
        np.testing.assert_array_equal(np.array([[0, 1], [4, 5]]), roi.get_image(img))
        on, off = roi.get_images_pp(img_on, img_off)
        np.testing.assert_array_equal(np.array([[3, 3], [3, 3]]), on)
        np.testing.assert_array_equal(np.array([[1, 1], [1, 1]]), off)

        # partial intersected
        roi.rect = (-1, -1, 3, 3)
        self.assertListEqual([0, 0, 2, 2], roi.intersect(img))
        np.testing.assert_array_equal(np.array([[0, 1], [4, 5]]), roi.get_image(img))

        # test if there no intersection
        roi.rect = (0, 0, -5, -5)
        self.assertListEqual([0, 0, -5, -5], roi.intersect(img))
        self.assertIsNone(roi.get_image(img))


class TestRoiProcessorTrain(unittest.TestCase):

    def setUp(self):
        self._proc = RoiProcessorTrain()
        self._proc._direction = 'x'
        self._proc._auc_range = (0, 1000)
        self._proc._fom_integ_range = (0, 1000)

    def _get_data(self):
        processed = ProcessedData(1001)
        processed.image = ImageData.from_array(np.ones((20, 20)))
        return {'processed': processed, 'raw': dict()}, processed

    def testRoiFom(self):
        proc = self._proc
        # set ROI1 and ROI4
        proc._roi1.activated = True
        proc._roi1.rect = [0, 0, 2, 3]
        proc._roi2.activated = True
        proc._roi2.rect = [0, 0, 1, 3]

        data, processed = self._get_data()
        proc.process(data)

        # ROI rects
        self.assertEqual([0, 0, 2, 3], processed.roi.rect1)
        self.assertEqual([0, 0, 1, 3], processed.roi.rect2)
        self.assertEqual([0, 0, -1, -1], processed.roi.rect3)
        self.assertEqual([0, 0, -1, -1], processed.roi.rect4)

        self.assertEqual(6.0, processed.roi.roi1.fom)
        self.assertEqual(3.0, processed.roi.roi2.fom)
        # ROI1 and ROI2 have different shapes
        self.assertEqual(None, processed.roi.roi1_sub_roi2.fom)
        self.assertEqual(None, processed.roi.roi1_add_roi2.fom)

        proc._roi2.rect = [0, 0, 2, 3]

        data, processed = self._get_data()
        proc.process(data)

        self.assertEqual(6.0, processed.roi.roi1.fom)
        self.assertEqual(6.0, processed.roi.roi2.fom)
        # ROI1 and ROI2 have different shapes
        self.assertEqual(0.0, processed.roi.roi1_sub_roi2.fom)
        self.assertEqual(12.0, processed.roi.roi1_add_roi2.fom)

    def testProjFom(self):
        proc = self._proc
        # set ROI1 and ROI4
        proc._roi1.activated = True
        proc._roi1.rect = [0, 0, 2, 3]
        proc._roi2.activated = True
        proc._roi2.rect = [0, 0, 2, 3]

        data, processed = self._get_data()
        proc.process(data)

        # ROI projection FOMs are all None because of AnalysisType
        self.assertEqual(None, processed.roi.proj1.fom)
        self.assertEqual(None, processed.roi.proj2.fom)
        self.assertEqual(None, processed.roi.proj1_sub_proj2.fom)
        self.assertEqual(None, processed.roi.proj1_add_proj2.fom)

        # only 'register' PROJ_ROI1
        proc._has_analysis = MagicMock(
            side_effect=lambda x: True if x == AnalysisType.PROJ_ROI1 else False)
        data, processed = self._get_data()
        proc.process(data)
        self.assertEqual(2.0, processed.roi.proj1.fom)
        self.assertEqual(None, processed.roi.proj2.fom)

        # only 'register' PROJ_ROI2
        proc._has_analysis = MagicMock(
            side_effect=lambda x: True if x == AnalysisType.PROJ_ROI2 else False)
        data, processed = self._get_data()
        proc.process(data)
        self.assertEqual(None, processed.roi.proj1.fom)
        self.assertEqual(2.0, processed.roi.proj2.fom)

        # only 'register' ROI1_SUB_ROI2 / ROI1_ADD_ROI2
        proc._has_analysis = MagicMock(return_value=False)
        proc._has_any_analysis = MagicMock(return_value=True)
        data, processed = self._get_data()
        proc.process(data)
        self.assertEqual(2.0, processed.roi.proj1.fom)
        self.assertEqual(2.0, processed.roi.proj2.fom)
        self.assertEqual(0.0, processed.roi.proj1_sub_proj2.fom)
        self.assertEqual(4.0, processed.roi.proj1_add_proj2.fom)

        # ROI1 and ROI2 have different shapes
        proc._roi2.activated = True
        proc._roi2.rect = [0, 0, 4, 3]
        data, processed = self._get_data()
        proc.process(data)
        self.assertEqual(2.0, processed.roi.proj1.fom)
        self.assertAlmostEqual(1.3333334, processed.roi.proj2.fom)
        self.assertEqual(None, processed.roi.proj1_sub_proj2.fom)
        self.assertEqual(None, processed.roi.proj1_add_proj2.fom)

        # ROI2 not activated
        proc._roi2.activated = False
        proc._roi2.rect = [0, 0, 4, 3]
        data, processed = self._get_data()
        proc.process(data)
        self.assertEqual(2.0, processed.roi.proj1.fom)
        self.assertEqual(None, processed.roi.proj2.fom)
        self.assertEqual(None, processed.roi.proj1_sub_proj2.fom)
        self.assertEqual(None, processed.roi.proj1_add_proj2.fom)

    def testProjNormalizer(self):
        proc = self._proc
        # set ROI1 and ROI4
        proc._roi1.activated = True
        proc._roi1.rect = [0, 0, 2, 3]
        proc._roi2.activated = True
        proc._roi2.rect = [0, 0, 2, 3]
        proc._roi3.activated = True
        proc._roi3.rect = [0, 0, 4, 3]
        proc._roi4.activated = True
        proc._roi4.rect = [1, 1, 4, 3]

        # normalized by VFomNormalizer.ROI3_ADD_ROI4
        proc._normalizer = VFomNormalizer.ROI3_ADD_ROI4

        proc._has_analysis = MagicMock(return_value=True)
        proc._has_any_analysis = MagicMock(return_value=True)

        data, processed = self._get_data()
        proc.process(data)
        self.assertEqual(0.25, processed.roi.proj1.fom)
        self.assertEqual(0.25, processed.roi.proj2.fom)
        self.assertEqual(0.0, processed.roi.proj1_sub_proj2.fom)
        self.assertEqual(0.5, processed.roi.proj1_add_proj2.fom)

        # normalized by VFomNormalizer.ROI3_SUB_ROI4
        proc._normalizer = VFomNormalizer.ROI3_SUB_ROI4

        proc._roi4.activated = False
        with self.assertRaises(ProcessingError):
            # normalizer is None
            proc.process(data)

        proc._roi3.activated = True
        proc._roi3.rect = [0, 0, 4, 3]
        proc._roi4.activated = True
        with self.assertRaises(ProcessingError):
            data, processed = self._get_data()
            # normalizer is 0
            proc.process(data)

        proc._roi3.rect = [10, 10, 5, 3]
        with self.assertRaises(ProcessingError):
            data, processed = self._get_data()
            # ROI3 and ROI4 have different shapes
            proc.process(data)

    def testProjectionPp(self):
        proc = self._proc
        proc._roi1.activated = True
        proc._roi1.rect = [0, 0, 2, 3]
        proc._roi2.activated = True
        proc._roi2.rect = [0, 0, 2, 3]

        data, processed = self._get_data()
        processed.pp.analysis_type = AnalysisType.PROJ_ROI1_SUB_ROI2
        proc.process(data)
        # no on/off images
        self.assertIsNone(processed.pp.fom)

        # FIXME: after normalization ON-OFF is always 0. Better test?

        # ROI1
        data, processed = self._get_data()
        processed.pp.analysis_type = AnalysisType.PROJ_ROI1
        processed.pp.image_on = 4.0 * np.ones((20, 20), dtype=np.float32)
        processed.pp.image_off = 2.0 * np.ones((20, 20), dtype=np.float32)
        proc.process(data)
        self.assertEqual(0, processed.pp.fom)

        # ROI2
        data, processed = self._get_data()
        processed.pp.analysis_type = AnalysisType.PROJ_ROI2
        processed.pp.image_on = 4.0 * np.ones((20, 20), dtype=np.float32)
        processed.pp.image_off = 2.0 * np.ones((20, 20), dtype=np.float32)
        proc.process(data)
        self.assertEqual(0, processed.pp.fom)

        # ROI1 - ROI2
        data, processed = self._get_data()
        processed.pp.analysis_type = AnalysisType.PROJ_ROI1_SUB_ROI2
        processed.pp.image_on = 4.0 * np.ones((20, 20), dtype=np.float32)
        processed.pp.image_off = 2.0 * np.ones((20, 20), dtype=np.float32)
        proc.process(data)
        self.assertEqual(0, processed.pp.fom)

        # ROI1 + ROI2
        data, processed = self._get_data()
        processed.pp.analysis_type = AnalysisType.PROJ_ROI1_ADD_ROI2
        processed.pp.image_on = 4.0 * np.ones((20, 20), dtype=np.float32)
        processed.pp.image_off = 2.0 * np.ones((20, 20), dtype=np.float32)
        proc.process(data)
        self.assertEqual(0, processed.pp.fom)


class TestRoiProcessorPulse(_BaseProcessorTest):
    def setUp(self):
        self._proc = RoiProcessorPulse()

    def testRoiFom(self):
        proc = self._proc
        # set ROI1 and ROI2
        proc._roi1.activated = True
        proc._roi1.rect = [0, 0, 2, 3]
        proc._roi2.activated = True
        proc._roi2.rect = [0, 0, 1, 3]

        data, processed = self.data_with_assembled(1001, (4, 20, 20))

        # only ROI1_PULSE is registered

        proc._has_analysis = MagicMock(
            side_effect=lambda x: True if x == AnalysisType.ROI1_PULSE else False)
        proc.process(data)

        roi = processed.pulse.roi
        fom1_gt = list(np.sum(data['assembled'][:, 0:3, 0:2], axis=(-1, -2)))
        self.assertListEqual(fom1_gt, roi.roi1.fom)
        self.assertIsNone(roi.roi2.fom)

        # only ROI2_PULSE is registered

        roi.roi1.fom = None  # clear previous result
        proc._has_analysis = MagicMock(
            side_effect=lambda x: True if x == AnalysisType.ROI2_PULSE else False)
        proc.process(data)

        roi = processed.pulse.roi
        fom2_gt = list(np.sum(data['assembled'][:, 0:3, 0:1], axis=(-1, -2)))
        self.assertIsNone(roi.roi1.fom)
        self.assertListEqual(fom2_gt, roi.roi2.fom)

        # only ROI1_PULSE is registered (with image mask)

        data, processed = self.data_with_assembled(1001, (4, 20, 20),
                                                   with_image_mask=True)

        proc._has_analysis = MagicMock(
            side_effect=lambda x: True if x == AnalysisType.ROI1_PULSE else False)
        proc.process(data)

        roi = processed.pulse.roi
        masked_assembled = data['assembled'].copy()
        masked_assembled[:, processed.image.image_mask] = 0
        fom1_gt = list(np.sum(masked_assembled[:, 0:3, 0:2], axis=(-1, -2)))
        self.assertListEqual(fom1_gt, roi.roi1.fom)
        self.assertIsNone(roi.roi2.fom)

        # only ROI2_PULSE is registered (with image mask)

        roi.roi1.fom = None  # clear previous result
        proc._has_analysis = MagicMock(
            side_effect=lambda x: True if x == AnalysisType.ROI2_PULSE else False)
        proc.process(data)

        roi = processed.pulse.roi
        masked_assembled = data['assembled'].copy()
        masked_assembled[:, processed.image.image_mask] = 0
        fom2_gt = list(np.sum(masked_assembled[:, 0:3, 0:1], axis=(-1, -2)))
        self.assertIsNone(roi.roi1.fom)
        self.assertListEqual(fom2_gt, roi.roi2.fom)