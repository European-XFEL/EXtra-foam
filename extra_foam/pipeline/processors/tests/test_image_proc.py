"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from extra_foam.pipeline.processors.image_processor import ImageProcessor
from extra_foam.pipeline.exceptions import ImageProcessingError, ProcessingError
from extra_foam.pipeline.processors.tests import _BaseProcessorTest


class TestImageProcessorTr(unittest.TestCase, _BaseProcessorTest):
    """Test pulse-resolved ImageProcessor.

    For train-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessor()

        self._proc._background = -10
        self._proc._threshold_mask = (-100, 100)

        del self._proc._dark_run

    def testPulseSlice(self):
        # The sliced_indices for train-resolved data should always be [0]

        data, processed = self.data_with_assembled(1, (2, 2))

        self._proc.process(data)
        # FIXME
        # np.testing.assert_array_equal(data['detector']['assembled'], processed.image.images)
        self.assertIsInstance(processed.image.images, list)
        self.assertListEqual([0], processed.image.sliced_indices)

        # set a slicer
        self._proc._pulse_slicer = slice(0, 2)
        self._proc.process(data)
        # FIXME
        # np.testing.assert_array_equal(data['detector']['assembled'], processed.image.images)
        self.assertListEqual([0], processed.image.sliced_indices)

    def testImageShapeChangeOnTheFly(self):
        proc = self._proc
        proc._image_mask = np.ones((2, 2), dtype=np.bool)

        data, _ = self.data_with_assembled(1, (2, 2))
        proc.process(data)

        # image shape changes
        with self.assertRaisesRegex(ImageProcessingError, 'image mask'):
            data, _ = self.data_with_assembled(2, (4, 2))
            proc.process(data)

        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((2, 2), dtype=np.bool), proc._image_mask)
        proc._image_mask = None

        # assign a reference image
        proc._reference = np.ones((4, 2), dtype=np.float32)
        # image shape changes
        with self.assertRaisesRegex(ImageProcessingError, 'reference'):
            data, _ = self.data_with_assembled(3, (2, 2))
            proc.process(data)

        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((4, 2), dtype=np.float32), proc._reference)
        proc._reference = None

    def testDarkRecordingAndSubtraction(self):
        pass


class TestImageProcessorPr(unittest.TestCase, _BaseProcessorTest):
    """Test pulse-resolved ImageProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessor()

        del self._proc._dark_run

        self._proc._background = -10
        self._proc._threshold_mask = (-100, 100)

    def testDarkRecordingAndSubtraction(self):
        self._proc._recording = True
        self._proc._dark_subtraction = False

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        dark_run_gt = data['detector']['assembled'].copy()
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark_run)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)

        # test moving average is going on
        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled_gt = data['detector']['assembled'].copy()
        dark_run_gt = (dark_run_gt + assembled_gt) / 2.0
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark_run)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)
        # test 'assembled' is not subtracted by dark
        np.testing.assert_array_almost_equal(data['detector']['assembled'], assembled_gt)

        # --------------------------
        # test with dark subtraction
        # --------------------------

        self._proc._dark_subtraction = True  # with subtraction

        del self._proc._dark_run
        self._proc._dark_mean = None

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        dark_run_gt = data['detector']['assembled'].copy()
        assembled_gt = dark_run_gt
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark_run)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)
        # test 'assembled' is dark run subtracted
        np.testing.assert_array_almost_equal(
            data['detector']['assembled'], assembled_gt - self._proc._dark_run)

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled_gt = data['detector']['assembled'].copy()
        dark_run_gt = (dark_run_gt + assembled_gt) / 2.0
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark_run)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)
        # test 'assembled' is subtracted by dark
        np.testing.assert_array_almost_equal(
            data['detector']['assembled'], assembled_gt - self._proc._dark_run)
        self._proc._recording = False

        # test image has different shape from the dark
        # (this test should use the env from the above test

        # when recording, the shape inconsistency will be covered
        data, processed = self.data_with_assembled(1, (4, 3, 2))
        with self.assertRaisesRegex(ImageProcessingError, "Shape of the dark"):
            self._proc.process(data)

    def testPulseSlicing(self):
        proc = self._proc

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled_gt = data['detector']['assembled'].copy()

        proc.process(data)
        self.assertEqual(4, processed.image.n_images)
        self.assertListEqual([0, 1, 2, 3], processed.image.sliced_indices)

        # ---------------------------------------------------------------------
        # Test slice to list of indices.
        # ---------------------------------------------------------------------

        data['detector']['pulse_slicer'] = slice(0, 2)
        proc.process(data)
        # Note: this test ensures that POI and on/off pulse indices are all
        # based on the assembled data after pulse slicing.
        np.testing.assert_array_equal(assembled_gt[0:2], data['detector']['assembled'])
        self.assertEqual(2, processed.image.n_images)
        self.assertListEqual([0, 1], processed.image.sliced_indices)

        # ----------------------------------------------------------------------
        # Test the whole train will be recorded even if pulse slicer is applied.
        #
        # Test if the pulse slicer is applied, it will apply to the dark run
        # when performaing dark subtraction.
        # ---------------------------------------------------------------------

        proc._recording = True
        data, processed = self.data_with_assembled(1, (4, 2, 2))
        slicer = slice(0, 2)
        # ground truth of the dark run is unsliced
        dark_run_gt = data['detector']['assembled'].copy()
        data['detector']['pulse_slicer'] = slicer
        # ground truth of assembled is sliced
        assembled_gt = dark_run_gt[slicer]
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark_run)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)
        # Important: Test 'assembled' is sliced and is also subtracted by the sliced dark.
        #            This test ensures that the the pump-probe processor will use the sliced data
        np.testing.assert_array_almost_equal(
            data['detector']['assembled'], assembled_gt - self._proc._dark_run[slicer])
        proc._recording = False

    def testPOI(self):
        proc = self._proc
        data, processed = self.data_with_assembled(1, (4, 2, 2))
        imgs_gt_unsliced = data['detector']['assembled'].copy()

        proc.process(data)
        imgs = processed.image.images
        self.assertIsInstance(imgs, list)
        self.assertIsNone(proc._poi_indices)
        for img in imgs:
            self.assertIsNone(img)

        # ----------------------------
        # Test non-default POI indices
        # ----------------------------

        proc._poi_indices = [2, 3]
        proc.process(data)
        imgs = processed.image.images
        self.assertIsNone(imgs[0])
        self.assertIsNone(imgs[1])
        np.testing.assert_array_equal(imgs_gt_unsliced[2], imgs[2])
        np.testing.assert_array_equal(imgs_gt_unsliced[3], imgs[3])

        # --------------------
        # Test invalid indices
        # --------------------

        proc._poi_indices = [3, 4]
        with self.assertRaises(ProcessingError):
            proc.process(data)

        # ---------------------------------------------
        # Test POI indices is based on the sliced data.
        # ---------------------------------------------

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        imgs_gt_unsliced = data['detector']['assembled'].copy()
        data['detector']['pulse_slicer'] = slice(0, 4, 2)
        proc._poi_indices = [0, 1]  # POI indices should based on the sliced data
        proc.process(data)
        imgs = processed.image.images
        self.assertEqual(2, len(imgs))
        np.testing.assert_array_equal(imgs_gt_unsliced[0], imgs[0])
        np.testing.assert_array_equal(imgs_gt_unsliced[2], imgs[1])

    def testImageShapeChangeOnTheFly(self):
        proc = self._proc
        proc._image_mask = np.ones((2, 2), dtype=np.bool)

        data, _ = self.data_with_assembled(1, (4, 2, 2))
        proc.process(data)

        # image shape changes
        with self.assertRaisesRegex(ImageProcessingError, 'image mask'):
            data, _ = self.data_with_assembled(2, (4, 4, 2))
            proc.process(data)

        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((2, 2), dtype=np.bool), proc._image_mask)
        proc._image_mask = None

        # assign a reference image
        proc._reference = np.ones((4, 2), dtype=np.float32)
        # image shape changes
        with self.assertRaisesRegex(ImageProcessingError, 'reference'):
            data, _ = self.data_with_assembled(3, (4, 2, 2))
            proc.process(data)

        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((4, 2), dtype=np.float32), proc._reference)
        proc._reference = None

        # Number of pulses per train changes, but no exception will be raised
        data, _ = self.data_with_assembled(4, (8, 2, 2))
        proc.process(data)
