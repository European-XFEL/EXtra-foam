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

from extra_foam.config import CalibrationOffsetPolicy
from extra_foam.pipeline.processors.image_processor import ImageProcessor, _IMAGE_DTYPE
from extra_foam.pipeline.exceptions import ImageProcessingError, ProcessingError
from extra_foam.pipeline.tests import _TestDataMixin
from extra_foam.pipeline.processors.image_processor import config


class _ImageProcessorTestBase(_TestDataMixin, unittest.TestCase):
    def _gain_offset_update_test_imp(self, ndim, info):
        proc = self._proc

        if ndim == 2:
            shape = (2, 2)
            false_shape = (3, 2)
            slicer_gt = slice(None, None)
        else:
            shape = (4, 2, 2)
            false_shape = (4, 3, 2)
            slicer_gt = slice(None, None, 2)

        data, processed = self.data_with_assembled(1, shape, slicer=slicer_gt)

        # test gain with wrong shape
        with self.assertRaisesRegex(ImageProcessingError, "shape"):
            gain_gt = np.random.randn(*false_shape).astype(np.float32)
            proc._cal_sub.update = MagicMock(return_value=(True, gain_gt, False, None))
            proc.process(data)
        np.testing.assert_array_equal(gain_gt, proc._full_gain)
        np.testing.assert_array_equal(gain_gt, proc._gain)
        if ndim == 2:
            np.testing.assert_array_almost_equal(gain_gt, proc._gain_mean)
        else:
            np.testing.assert_array_almost_equal(np.mean(gain_gt, axis=0), proc._gain_mean)
        info.assert_called_once_with(f"[Image processor] Loaded gain constants with shape = {false_shape}")
        info.reset_mock()

        # test offset with wrong shape
        proc._dark_as_offset = True
        proc._dark = np.random.randn(*shape).astype(np.float32)
        proc._dark_mean = proc._dark
        offset_gt = np.random.randn(*false_shape).astype(np.float32)
        # remove the invalid gain
        proc._cal_sub.update = MagicMock(return_value=(True, None, True, offset_gt))
        proc.process(data)  # not raise
        self.assertIsNone(proc._full_gain)
        self.assertIsNone(proc._gain)
        self.assertIsNone(proc._gain_mean)
        # _full_offset is updated and it has nothing to do with dark
        np.testing.assert_array_equal(offset_gt, proc._full_offset)
        np.testing.assert_array_almost_equal(offset_gt, proc._offset)
        if ndim == 2:
            np.testing.assert_array_almost_equal(offset_gt, proc._offset_mean)
        else:
            np.testing.assert_array_almost_equal(np.mean(offset_gt, axis=0), proc._offset_mean)
        assert info.mock_calls == [
            unittest.mock.call("[Image processor] Gain constants removed"),
            unittest.mock.call(f"[Image processor] Loaded offset constants with shape = {false_shape}")
        ]
        info.reset_mock()

        # test offset with wrong shape but "dark as offset" is set
        proc._dark_as_offset = False
        proc._cal_sub.update = MagicMock(return_value=(False, None, False, None))
        with self.assertRaisesRegex(ImageProcessingError, "shape"):
            proc.process(data)
        np.testing.assert_array_equal(offset_gt, proc._full_offset)
        np.testing.assert_array_almost_equal(offset_gt, proc._offset)
        if ndim == 2:
            np.testing.assert_array_almost_equal(offset_gt, proc._offset_mean)
        else:
            np.testing.assert_array_almost_equal(np.mean(offset_gt, axis=0), proc._offset_mean)
        info.assert_not_called()

        # test remove offset
        proc._dark_as_offset = True
        proc._cal_sub.update = MagicMock(return_value=(False, None, True, None))
        proc.process(data)
        self.assertIsNone(proc._full_gain)
        self.assertIsNone(proc._offset)
        self.assertIsNone(proc._offset_mean)
        info.assert_called_once_with("[Image processor] Offset constants removed")
        info.reset_mock()

    def _gain_offset_correction_test_imp(self, ndim):
        proc = self._proc

        if ndim == 2:
            shape = (2, 2)
            slicer_gt = slice(None, None)
            gain_gt = np.array([[1, 4], [6, 12]], dtype=np.float32)
            offset_gt = gain_gt - 1.
            dark_gt = offset_gt / 2.
        else:
            shape = (4, 2, 2)
            slicer_gt = slice(None, None, 2)
            gain_gt = np.array([[[ 1, 4], [ 6, 12]],
                                [[ 2, 3], [ 5,  6]],
                                [[ 0, 2], [ 8,  9]],
                                [[-5, 1], [12, 14]]], dtype=np.float32)
            offset_gt = gain_gt - 2.
            dark_gt = offset_gt / 4.

        proc._gain = gain_gt
        proc._offset = offset_gt
        proc._dark = dark_gt

        # test not apply correction
        proc._correct_gain = False
        proc._correct_offset = False
        proc._dark_as_offset = False
        data, processed = self.data_with_assembled(1, shape, gen='range', slicer=slicer_gt)
        assembled_gt = data['assembled']['data'].copy()
        proc.process(data)
        np.testing.assert_array_almost_equal(data['assembled']['data'], assembled_gt)

        # test apply offset correction only
        proc._correct_offset = True
        data, processed = self.data_with_assembled(1, shape, gen='range', slicer=slicer_gt)
        assembled_gt = data['assembled']['data'].copy()
        proc.process(data)
        np.testing.assert_array_almost_equal(
            data['assembled']['data'], (assembled_gt - offset_gt), decimal=3)

        # test apply both gain and offset correction
        proc._correct_gain = True
        data, processed = self.data_with_assembled(1, shape, gen='range', slicer=slicer_gt)
        assembled_gt = data['assembled']['data'].copy()
        proc.process(data)
        np.testing.assert_array_almost_equal(
            data['assembled']['data'], (gain_gt * (assembled_gt - offset_gt)))

        # test intra-dark correction
        if ndim == 3:
            proc._offset_policy = CalibrationOffsetPolicy.INTRA_DARK
            data, processed = self.data_with_assembled(1, shape, gen='range', slicer=slicer_gt)
            assembled_gt = data['assembled']['data'].copy()
            proc.process(data)
            corrected_gt = (gain_gt * (assembled_gt - offset_gt))
            corrected_gt[::2] -= corrected_gt[1::2]
            np.testing.assert_array_almost_equal(data['assembled']['data'], corrected_gt)
            proc._offset_policy = CalibrationOffsetPolicy.UNDEFINED

        # test dark is used as offset
        proc._dark_as_offset = True
        data, processed = self.data_with_assembled(1, shape, gen='range', slicer=slicer_gt)
        assembled_gt = data['assembled']['data'].copy()
        proc.process(data)
        np.testing.assert_array_almost_equal(
            data['assembled']['data'], (gain_gt * (assembled_gt - dark_gt)))

        proc._correct_gain = False
        proc._correct_offset = False
        data, processed = self.data_with_assembled(1, shape, gen='range', slicer=slicer_gt)
        assembled_gt = data['assembled']['data'].copy()
        proc.process(data)
        np.testing.assert_array_almost_equal(data['assembled']['data'], assembled_gt)

    def _reference_update_test_imp(self, ndim, info):
        proc = self._proc

        if ndim == 3:
            data, processed = self.data_with_assembled(1, (4, 2, 2))
        else:
            data, processed = self.data_with_assembled(1, (2, 2))

        with patch.object(proc._ref_sub, "update",
                          return_value=(True, np.ones((2, 2), dtype=_IMAGE_DTYPE))):
            proc.process(data)
            info.assert_called_once_with(
                "[Image processor] Loaded reference image with shape = (2, 2)")
            info.reset_mock()

        with patch.object(proc._ref_sub, "update",
                          return_value=(True, np.zeros((2, 2), dtype=bool))):
            proc.process(data)
            self.assertEqual(_IMAGE_DTYPE, proc._reference.dtype)
            np.testing.assert_array_equal(np.zeros((2, 2)), proc._reference)
            info.reset_mock()

        with patch.object(proc._ref_sub, "update", return_value=(True, None)):
            proc.process(data)
            self.assertIsNone(proc._reference)
            info.assert_called_once_with(
                "[Image processor] Reference image removed")
            info.reset_mock()

        with patch.object(proc._ref_sub, "update", return_value=(False, np.ones((2, 2)))):
            proc.process(data)
            self.assertIsNone(proc._reference)
            info.assert_not_called()


class TestImageProcessorTr(_ImageProcessorTestBase):
    """Test pulse-resolved ImageProcessor.

    For train-resolved data.
    """
    def setUp(self):
        with patch.dict(config._data, {"DETECTOR": "LPD"}):
            self._proc = ImageProcessor()

        self._proc._gain_cells = slice(None, None)
        self._proc._offset_cells = slice(None, None)

        self._proc._assembler.process = MagicMock(side_effect=lambda x: x)
        self._proc._ref_sub.update = MagicMock(return_value=(False, None))   # no redis server
        self._proc._cal_sub.update = MagicMock(
            side_effect=lambda: (False, None, False, None))   # no redis server
        self._proc._mask_sub.update = MagicMock(
            side_effect=lambda x, y: (False, np.zeros(y, dtype=bool) if x is None else x))

        self._proc._threshold_mask = (-100, 100)

        del self._proc._dark
        del self._proc._raw

    def testPulseSlice(self):
        data, processed = self.data_with_assembled(1, (2, 2))
        self._proc.process(data)
        self.assertListEqual([None], processed.image.images)
        self.assertListEqual([0], processed.image.sliced_indices)

    @patch("extra_foam.ipc.ProcessLogger.info")
    def testGainOffsetUpdate(self, info):
        self._gain_offset_update_test_imp(2, info)

    def testGainOffsetCorrection(self):
        self._gain_offset_correction_test_imp(2)

    def testMovingAverage(self):
        self._proc._set_ma_window(3)
        self.assertEqual(3, self._proc.__class__._raw.window)

        data, processed = self.data_with_assembled(1, (2, 2))
        assembled = data['assembled']['data']
        raw_gt = assembled.copy()
        self._proc.process(data)
        np.testing.assert_array_almost_equal(raw_gt, self._proc._raw)
        np.testing.assert_array_almost_equal(raw_gt, data['assembled']['data'])
        assembled[0, 0] = 999
        self.assertNotEqual(999, self._proc._raw[0, 0])  # test the first data is copied

        data, processed = self.data_with_assembled(1, (2, 2))
        raw_gt = (raw_gt + data['assembled']['data']) / 2.
        self._proc.process(data)
        np.testing.assert_array_almost_equal(raw_gt, self._proc._raw)
        np.testing.assert_array_almost_equal(raw_gt, data['assembled']['data'])

        # reset moving average to 1
        self._proc._set_ma_window(1)
        data, processed = self.data_with_assembled(1, (2, 2))
        self._proc.process(data)
        self.assertIsNone(self._proc._raw)

    @patch("extra_foam.ipc.ProcessLogger.info")
    def testReferenceUpdate(self, info):
        self._reference_update_test_imp(2, info)

    def testImageShapeChangeOnTheFly(self):
        proc = self._proc
        proc._image_mask = np.ones((2, 2), dtype=bool)

        data, _ = self.data_with_assembled(1, (2, 2))
        proc.process(data)

        # image shape changes
        with self.assertRaisesRegex(ImageProcessingError, 'image mask'):
            data, _ = self.data_with_assembled(2, (4, 2))
            proc.process(data)

        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((2, 2), dtype=bool), proc._image_mask)
        proc._image_mask = None


class TestImageProcessorPr(_ImageProcessorTestBase):
    """Test pulse-resolved ImageProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        with patch.dict(config._data, {"DETECTOR": "LPD"}):
            self._proc = ImageProcessor()

        self._proc._gain_cells = slice(None, None)
        self._proc._offset_cells = slice(None, None)

        self._proc._assembler.process = MagicMock(side_effect=lambda x: x)
        self._proc._ref_sub.update = MagicMock(return_value=(False, None))   # no redis server
        self._proc._cal_sub.update = MagicMock(
            side_effect=lambda: (False, None, False, None))   # no redis server
        self._proc._mask_sub.update = MagicMock(
            side_effect=lambda x, y: (False, np.zeros(y, dtype=bool) if x is None else x))

        del self._proc._dark
        del self._proc._raw

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

    @patch("extra_foam.ipc.ProcessLogger.info")
    def testGainOffsetUpdate(self, info):
        self._gain_offset_update_test_imp(3, info)

    def testGainOffsetCorrection(self):
        self._gain_offset_correction_test_imp(3)

    def testMovingAverage(self):
        self._proc._set_ma_window(3)
        self.assertEqual(3, self._proc.__class__._raw.window)

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled = data['assembled']['data']
        raw_gt = assembled.copy()
        self._proc.process(data)
        np.testing.assert_array_almost_equal(raw_gt, self._proc._raw)
        np.testing.assert_array_almost_equal(raw_gt, data['assembled']['data'])
        assembled[0, 0, 0] = 999
        self.assertNotEqual(999, self._proc._raw[0, 0, 0])  # test the first data is copied

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        raw_gt = (raw_gt + data['assembled']['data']) / 2.
        self._proc.process(data)
        np.testing.assert_array_almost_equal(raw_gt, self._proc._raw)
        np.testing.assert_array_almost_equal(raw_gt, data['assembled']['data'])

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        raw_gt = (2 * raw_gt + data['assembled']['data']) / 3.
        self._proc.process(data)
        np.testing.assert_array_almost_equal(raw_gt, self._proc._raw)
        np.testing.assert_array_almost_equal(raw_gt, data['assembled']['data'])

        # reset moving average to 1
        self._proc._set_ma_window(1)
        data, processed = self.data_with_assembled(1, (2, 2))
        self._proc.process(data)
        self.assertIsNone(self._proc._raw)

    def testMaskUpdate(self):
        proc = self._proc

        data, processed = self.data_with_assembled(1, (4, 2, 2))

        def dismantle_side_effect(*args, **kwargs):
            raise ValueError
        with patch.object(proc._assembler, "_geom") as geom:
            with patch.object(proc._mask_sub, "update") as update:
                # test generation of image_mask_in_modules
                proc._require_geom = True
                update.return_value = (True, np.ones([2, 2], dtype=bool))
                proc.process(data)
                geom.output_array_for_dismantle_fast.assert_called_once()
                geom.dismantle_all_modules.assert_called_once()

                with patch.object(geom, "dismantle_all_modules",
                                  side_effect=dismantle_side_effect):
                    with self.assertRaisesRegex(ImageProcessingError,
                                                "Geometry changed during updating image mask"):
                        proc.process(data)

                # for invalid mask with all zeros, a new mask will be generated automatically
                update.return_value = (True, np.zeros([3, 2], dtype=bool))
                proc.process(data)
                np.testing.assert_array_equal(np.zeros([2, 2], dtype=bool), proc._image_mask)

                # test setting mask but the mask shape is different from the image shape

                # reassemble if there is a geometry
                update.return_value = (True, np.ones([4, 2], dtype=bool))
                proc.process(data)
                geom.output_array_for_position_fast.assert_called_once()
                geom.position_all_modules.assert_called_once()
                # otherwise, exception will be raised
                proc._require_geom = False
                with self.assertRaises(ImageProcessingError):
                    update.return_value = (True, np.ones([4, 2], dtype=bool))
                    proc.process(data)

    @patch("extra_foam.ipc.ProcessLogger.info")
    def testReferenceUpdate(self, info):
        self._reference_update_test_imp(3, info)

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
        proc._image_mask = np.ones((2, 2), dtype=bool)

        data, _ = self.data_with_assembled(1, (4, 2, 2))
        proc.process(data)

        # image shape changes
        with self.assertRaisesRegex(ImageProcessingError, 'image mask'):
            data, _ = self.data_with_assembled(2, (4, 4, 2))
            proc.process(data)

        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((2, 2), dtype=bool), proc._image_mask)
        proc._image_mask = None

        # Number of pulses per train changes, but no exception will be raised
        data, _ = self.data_with_assembled(4, (8, 2, 2))
        proc.process(data)
