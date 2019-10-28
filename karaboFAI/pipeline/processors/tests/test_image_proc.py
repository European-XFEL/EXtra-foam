"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from unittest.mock import MagicMock, patch

import numpy as np

from karaboFAI.pipeline.processors.image_processor import ImageProcessor
from karaboFAI.pipeline.exceptions import ImageProcessingError, ProcessingError
from karaboFAI.pipeline.processors.tests import _BaseProcessorTest


class TestImageProcessorTr(_BaseProcessorTest):
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


class TestImageProcessorPr(_BaseProcessorTest):
    """Test pulse-resolved ImageProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessor()

        del self._proc._dark_run

        self._proc._background = -10
        self._proc._threshold_mask = (-100, 100)

    def testDarkRun(self):
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

        # test image has different shape from the dark
        # (this test should use the env from the above test

        # when recording, the shape inconsistency will be covered
        self._proc._recording = False
        data, processed = self.data_with_assembled(1, (4, 3, 2))
        with self.assertRaisesRegex(ImageProcessingError, "Shape of the dark"):
            self._proc.process(data)

    def testPulseSlicing(self):
        data, processed = self.data_with_assembled(1, (4, 2, 2))
        assembled_gt = data['detector']['assembled'].copy()

        self._proc.process(data)
        self.assertEqual(4, processed.image.n_images)
        self.assertListEqual([0, 1, 2, 3], processed.image.sliced_indices)

        # test slice to list of indices
        data['detector']['pulse_slicer'] = slice(0, 2)
        self._proc.process(data)
        # Note: this test ensures that POI and on/off pulse indices are all
        # based on the assembled data after pulse slicing.
        np.testing.assert_array_equal(assembled_gt[0:2], data['detector']['assembled'])
        self.assertEqual(2, processed.image.n_images)
        self.assertListEqual([0, 1], processed.image.sliced_indices)

    def testPOI(self):
        proc = self._proc
        data, processed = self.data_with_assembled(1, (4, 2, 2))

        proc.process(data)
        imgs = processed.image.images
        self.assertIsInstance(imgs, list)
        self.assertListEqual([0, 0], proc._poi_indices)
        np.testing.assert_array_equal(imgs[0], data['detector']['assembled'][0])
        self.assertIsNone(imgs[1])
        self.assertIsNone(imgs[3])

        # change POI indices
        proc._poi_indices = [2, 3]
        proc.process(data)
        imgs = processed.image.images
        self.assertIsNone(imgs[0])
        self.assertIsNone(imgs[1])
        np.testing.assert_array_equal(imgs[2], data['detector']['assembled'][2])
        np.testing.assert_array_equal(imgs[3], data['detector']['assembled'][3])

        # test invalid indices
        proc._poi_indices = [3, 4]
        with self.assertRaises(ProcessingError):
            proc.process(data)

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
