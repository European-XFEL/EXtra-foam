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
from extra_foam.pipeline.tests import _TestDataMixin


class TestImageProcessorTr(_TestDataMixin, unittest.TestCase):
    """Test pulse-resolved ImageProcessor.

    For train-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessor()
        self._proc._ref_sub.update = MagicMock(side_effect=lambda x: x)   # no redis server
        self._proc._cal_sub.update = MagicMock(
            side_effect=lambda x, y: (False, x, False, y))   # no redis server
        self._proc._mask_sub.update = MagicMock(side_effect=lambda x, y: x)

        self._proc._threshold_mask = (-100, 100)

        del self._proc._dark

    def testPulseSlice(self):
        data, processed = self.data_with_assembled(1, (2, 2))
        self._proc.process(data)
        self.assertListEqual([None], processed.image.images)
        self.assertListEqual([0], processed.image.sliced_indices)

    def testGainOffsetCorrection(self):
        proc = self._proc

        # -------------------------
        # test not apply correction
        # -------------------------

        proc._correct_gain = False
        proc._correct_offset = False
        data, processed = self.data_with_assembled(1, (2, 2))
        assembled_gt = data['assembled']['data'].copy()
        self._proc.process(data)
        np.testing.assert_array_almost_equal(data['assembled']['data'], assembled_gt)

        # ---------------------------------
        # test apply offset correction only
        # ---------------------------------

        proc._correct_offset = True
        proc._dark_as_offset = False
        data, processed = self.data_with_assembled(1, (2, 2))
        assembled_gt = data['assembled']['data'].copy()
        offset_gt = np.random.randn(2, 2).astype(np.float32)
        proc._offset = offset_gt
        self._proc.process(data)
        np.testing.assert_array_almost_equal(
            data['assembled']['data'], assembled_gt - offset_gt)

        # ------------------------------------------
        # test apply both gain and offset correction
        # ------------------------------------------

        proc._correct_gain = True
        proc._dark_as_offset = True
        data, processed = self.data_with_assembled(1, (2, 2))
        assembled_gt = data['assembled']['data'].copy()
        proc._dark = data['assembled']['data'] / 2.0
        gain_gt = np.random.randn(2, 2).astype(np.float32)
        proc._gain = gain_gt
        self._proc.process(data)
        np.testing.assert_array_almost_equal(
            data['assembled']['data'], gain_gt * (assembled_gt - proc._dark))

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


class TestImageProcessorPr(_TestDataMixin, unittest.TestCase):
    """Test pulse-resolved ImageProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessor()
        self._proc._ref_sub.update = MagicMock(side_effect=lambda x: x)   # no redis server
        self._proc._cal_sub.update = MagicMock(
            side_effect=lambda x, y: (False, x, False, y))   # no redis server
        self._proc._mask_sub.update = MagicMock(side_effect=lambda x, y: x)

        del self._proc._dark

        self._proc._threshold_mask = (-100, 100)

    def testDarkRecordingAndSubtraction(self):
        self._proc._recording_dark = True

        # -----------------------------
        # test without dark subtraction
        # -----------------------------
        self._proc._dark_as_offset = False

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        dark_run_gt = data['assembled']['data'].copy()
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)

        # test moving average is going on
        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled_gt = data['assembled']['data'].copy()
        dark_run_gt = (dark_run_gt + assembled_gt) / 2.0
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)
        # test 'assembled' is not subtracted by dark
        np.testing.assert_array_almost_equal(data['assembled']['data'], assembled_gt)

        # --------------------------
        # test with dark subtraction
        # --------------------------

        self._proc._dark_as_offset = True

        del self._proc._dark
        self._proc._dark_mean = None

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        dark_run_gt = data['assembled']['data'].copy()
        assembled_gt = dark_run_gt
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)
        # test 'assembled' is dark run subtracted
        np.testing.assert_array_almost_equal(
            data['assembled']['data'], assembled_gt - self._proc._dark)

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled_gt = data['assembled']['data'].copy()
        dark_run_gt = (dark_run_gt + assembled_gt) / 2.0
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)
        # test 'assembled' is subtracted by dark
        np.testing.assert_array_almost_equal(
            data['assembled']['data'], assembled_gt - self._proc._dark)

        # test image has different shape from the dark
        # (this test should use the env from the above test)

        # caveat: stop recording first
        self._proc._recording_dark = False

        # when recording, the shape inconsistency will be covered
        data, processed = self.data_with_assembled(1, (4, 3, 2))
        with self.assertRaises(ImageProcessingError):
            self._proc.process(data)

    def testGainOffsetCorrection(self):
        proc = self._proc
        proc._gain_slicer = slice(None, None)
        proc._offset_slicer = slice(None, None)

        # -------------------------
        # test not apply correction
        # -------------------------

        proc._correct_gain = False
        proc._correct_offset = False
        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled_gt = data['assembled']['data'].copy()
        self._proc.process(data)
        np.testing.assert_array_almost_equal(data['assembled']['data'], assembled_gt)

        # ---------------------------------
        # test apply offset correction only
        # ---------------------------------

        proc._correct_offset = True
        proc._dark_as_offset = False
        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled_gt = data['assembled']['data'].copy()
        offset_gt = np.random.randn(4, 2, 2).astype(np.float32)
        proc._offset = offset_gt
        self._proc.process(data)
        np.testing.assert_array_almost_equal(data['assembled']['data'], assembled_gt - offset_gt)

        # ------------------------------------------
        # test apply both gain and offset correction
        # ------------------------------------------

        proc._correct_gain = True
        proc._dark_as_offset = True
        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled_gt = data['assembled']['data'].copy()
        proc._dark = data['assembled']['data'] / 2.0
        gain_gt = np.random.randn(4, 2, 2).astype(np.float32)
        proc._gain = gain_gt
        self._proc.process(data)
        np.testing.assert_array_almost_equal(data['assembled']['data'],
                                             proc._gain * (assembled_gt - proc._dark))

    def testPulseSlicing(self):
        proc = self._proc

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        proc.process(data)
        self.assertEqual(4, processed.image.n_images)
        self.assertListEqual([0, 1, 2, 3], processed.image.sliced_indices)

        # ---------------------------------------------------------------------
        # Test slice to list of indices.
        # ---------------------------------------------------------------------

        data, processed = self.data_with_assembled(1, (4, 2, 2), slicer=slice(0, 2))
        assembled_gt = data['assembled']['data'].copy()
        proc.process(data)
        # Note: this test ensures that POI and on/off pulse indices are all
        # based on the assembled data after pulse slicing.
        np.testing.assert_array_equal(assembled_gt[0:2], data['assembled']['sliced'])
        self.assertEqual(2, processed.image.n_images)
        self.assertListEqual([0, 1], processed.image.sliced_indices)

        # ----------------------------------------------------------------------
        # Test the whole train will be recorded even if pulse slicer is applied.
        #
        # Test if the pulse slicer is applied, it will apply to the dark run
        # when performing dark subtraction.
        # ---------------------------------------------------------------------

        proc._recording_dark = True
        proc._dark_as_offset = True
        slicer = slice(0, 2)
        data, processed = self.data_with_assembled(1, (4, 2, 2), slicer=slicer)
        # ground truth of the dark run is unsliced
        dark_run_gt = data['assembled']['data'].copy()
        # ground truth of assembled is sliced
        assembled_gt = dark_run_gt[slicer]
        self._proc.process(data)
        np.testing.assert_array_almost_equal(dark_run_gt, self._proc._dark)
        np.testing.assert_array_almost_equal(
            np.nanmean(dark_run_gt, axis=0), self._proc._dark_mean)
        # Important: Test 'assembled' is sliced and is also subtracted by the sliced dark.
        #            This test ensures that the the pump-probe processor will use the sliced data
        np.testing.assert_array_almost_equal(
            data['assembled']['sliced'], assembled_gt - self._proc._dark[slicer])
        proc._recording = False

    def testGainOffsetUpdate(self):
        proc = self._proc

        data, processed = self.data_with_assembled(1, (4, 2, 2))

        # test gain with wrong shape
        with self.assertRaises(ImageProcessingError):
            gain = np.random.randn(3, 2, 2).astype(np.float32)
            proc._cal_sub.update = MagicMock(return_value=(True, gain, False, None))
            proc.process(data)
        np.testing.assert_array_equal(gain, proc._gain)
        np.testing.assert_array_almost_equal(np.mean(gain, axis=0), proc._gain_mean)

        # test offset with wrong shape
        proc._dark_as_offset = False
        with self.assertRaises(ImageProcessingError):
            offset = np.random.randn(3, 2, 2).astype(np.float32)
            proc._cal_sub.update = MagicMock(return_value=(False, None, True, offset))
            proc.process(data)
        np.testing.assert_array_equal(offset, proc._offset)
        np.testing.assert_array_almost_equal(np.mean(offset, axis=0), proc._offset_mean)

        # test offset with wrong shape but "dark as offset" is set
        proc._dark_as_offset = True
        proc._dark_mean = np.random.randn(2, 2).astype(np.float32)
        offset = np.random.randn(3, 2, 2).astype(np.float32)
        proc._cal_sub.update = MagicMock(return_value=(False, None, True, offset))
        proc.process(data)  # not raise
        np.testing.assert_array_equal(offset, proc._offset)  # offset will not change
        np.testing.assert_array_almost_equal(proc._dark_mean, proc._offset_mean)

    def testMaskUpdate(self):
        proc = self._proc

        data, processed = self.data_with_assembled(1, (4, 2, 2))

        # test setting mask but the mask shape is different
        # from the image shape
        with self.assertRaises(ImageProcessingError):
            image_mask = np.ones([3, 2, 2])
            proc._mask_sub.update = MagicMock(return_value=image_mask)
            proc.process(data)
        self.assertIsNone(proc._image_mask)

    def testReferenceUpdate(self):
        proc = self._proc

        data, processed = self.data_with_assembled(1, (4, 2, 2))

        # test setting reference but the reference shape is different
        # from the image shape
        with self.assertRaises(ImageProcessingError):
            ref_gt = np.ones([3, 2, 2])
            proc._ref_sub.update = MagicMock(return_value=ref_gt)
            proc.process(data)
        # test the reference is set even if the shape is not correct
        np.testing.assert_array_equal(ref_gt, proc._reference)

    def testPOI(self):
        proc = self._proc
        data, processed = self.data_with_assembled(1, (4, 2, 2))
        imgs_gt_unsliced = data['assembled']['data'].copy()

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

        data, processed = self.data_with_assembled(1, (4, 2, 2), slicer=slice(0, 4, 2))
        imgs_gt_unsliced = data['assembled']['data'].copy()
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
