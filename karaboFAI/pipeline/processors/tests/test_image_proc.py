"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Unittest for ImageProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from karaboFAI.pipeline.processors.image_processor import (
    ImageProcessorTrain, ImageProcessorPulse
)
from karaboFAI.config import PumpProbeMode
from karaboFAI.pipeline.exceptions import (
    DropAllPulsesError, ImageProcessingError, PumpProbeIndexError,
    ProcessingError
)
from karaboFAI.pipeline.processors.tests import _BaseProcessorTest


class TestImageProcessorPulseTr(_BaseProcessorTest):
    """Test pulse-resolved ImageProcessor.

    For train-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessorPulse()

        self._proc._background = -10
        self._proc._threshold_mask = (-100, 100)

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


class TestImageProcessorPulsePr(_BaseProcessorTest):
    """Test pulse-resolved ImageProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessorPulse()

        del self._proc._dark_run

        self._proc._background = -10
        self._proc._threshold_mask = (-100, 100)

    def testDarkRun(self):
        self._proc._recording = True
        self._proc._process_dark = False  # no dark subtraction

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
        np.testing.assert_array_almost_equal(data['detector']['assembled'], assembled_gt)

        # ------------------------------
        # test self._process_dark = True
        # ------------------------------

        self._proc._process_dark = True  # with subtraction

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
        # test 'assembled' is dark run subtracted
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


class TestImageProcessorTrainTr(_BaseProcessorTest):
    """Test train-resolved ImageProcessor.

    For train-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessorTrain()
        self._proc._on_indices = [0]
        self._proc._off_indices = [0]

    def _gen_data(self, tid):
        return self.data_with_assembled(tid, (2, 2),
                                        threshold_mask=(-100, 100),
                                        background=0,
                                        poi_indices=[0, 0])

    def testPpUndefined(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.UNDEFINED

        data, processed = self._gen_data(1001)
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

    def testPpPredefinedOff(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF

        data, processed = self._gen_data(1001)

        proc.process(data)
        np.testing.assert_array_almost_equal(processed.pp.image_on, data['detector']['assembled'])
        np.testing.assert_array_almost_equal(processed.pp.image_off, np.zeros((2, 2)))

    def testPpOddOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.ODD_TRAIN_ON

        # test off will not be acknowledged without on
        data, processed = self._gen_data(1002)  # off
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

        data, processed = self._gen_data(1003)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

        np.testing.assert_array_almost_equal(data['detector']['assembled'], proc._prev_unmasked_on)

        data, processed = self._gen_data(1005)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        np.testing.assert_array_almost_equal(data['detector']['assembled'], proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data, processed = self._gen_data(1006)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_off, data['detector']['assembled'])

    def testPpEvenOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.EVEN_TRAIN_ON

        # test off will not be acknowledged without on
        data, processed = self._gen_data(1001)  # off
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

        data, processed = self._gen_data(1002)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        np.testing.assert_array_almost_equal(data['detector']['assembled'], proc._prev_unmasked_on)

        # test when two 'on' are received successively
        data, processed = self._gen_data(1004)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        np.testing.assert_array_almost_equal(data['detector']['assembled'], proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data, processed = self._gen_data(1005)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_off, data['detector']['assembled'])


class TestImageProcessorTrainPr(_BaseProcessorTest):
    """Test train-resolved ImageProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessorTrain()
        self._proc._on_indices = [0]
        self._proc._off_indices = [0]

    def _gen_data(self, tid):
        return self.data_with_assembled(tid, (4, 2, 2),
                                        threshold_mask=(-100, 100),
                                        background=0,
                                        poi_indices=[0, 0])

    def testPulseFilter(self):
        proc = self._proc

        data, processed = self._gen_data(1001)
        image_data = processed.image
        self.assertListEqual([], image_data.dropped_indices)
        image_data.dropped_indices = [0, 2]
        proc.process(data)
        # test calculating the average image after pulse filtering
        np.testing.assert_array_equal(
            np.nanmean(data['detector']['assembled'][[1, 3]], axis=0),
            image_data.mean
        )

    def testInvalidPulseIndices(self):
        proc = self._proc
        proc._on_indices = [0, 1, 5]
        proc._off_indices = [1]

        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF
        with self.assertRaises(PumpProbeIndexError):
            # the maximum index is 4
            data, _ = self._gen_data(1001)
            proc.process(data)

        proc._on_indices = [0, 1, 5]
        proc._off_indices = [1, 3]
        proc._pp_mode = PumpProbeMode.EVEN_TRAIN_ON
        with self.assertRaises(PumpProbeIndexError):
            data, _ = self._gen_data(1001)
            proc.process(data)

        # raises when the same pulse index was found in both
        # on- and off- indices
        proc._on_indices = [0, 1]
        proc._off_indices = [1, 3]
        proc._pp_mode = PumpProbeMode.SAME_TRAIN
        with self.assertRaises(PumpProbeIndexError):
            data, _ = self._gen_data(1001)
            proc.process(data)

        # off-indices check is not trigger in PRE_DEFINED_OFF mode
        proc._on_indices = [0, 1]
        proc._off_indices = [5]
        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF
        data, _ = self._gen_data(1001)
        proc.process(data)

    def testUndefined(self):
        proc = self._proc
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]
        proc._threshold_mask = (-np.inf, np.inf)

        proc._pp_mode = PumpProbeMode.UNDEFINED

        data, processed = self._gen_data(1001)
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

    def testPredefinedOff(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]

        data, processed = self._gen_data(1001)
        proc.process(data)
        np.testing.assert_array_almost_equal(
            processed.pp.image_on, np.mean(data['detector']['assembled'][::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(processed.pp.image_off, np.zeros((2, 2)))

        # --------------------
        # test pulse filtering
        # --------------------

        data, processed = self._gen_data(1002)
        image_data = processed.image
        image_data.dropped_indices = [0, 2]
        with self.assertRaises(DropAllPulsesError):
            proc.process(data)

        image_data.dropped_indices = [1, 3]
        # no Exception
        proc.process(data)

        # test image_on correctness
        image_data.dropped_indices = [0]
        proc.process(data)
        np.testing.assert_array_equal(processed.pp.image_on, data['detector']['assembled'][2])

    def testSameTrain(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.SAME_TRAIN
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]

        data, processed = self._gen_data(1001)
        proc.process(data)
        np.testing.assert_array_almost_equal(
            processed.pp.image_on, np.mean(data['detector']['assembled'][::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(
            processed.pp.image_off, np.mean(data['detector']['assembled'][1::2, :, :], axis=0))

        # --------------------
        # test pulse filtering
        # --------------------

        data, processed = self._gen_data(1002)
        image_data = processed.image
        image_data.dropped_indices = [0, 2]
        with self.assertRaises(DropAllPulsesError):
            proc.process(data)
        image_data.dropped_indices = [1, 3]
        with self.assertRaises(DropAllPulsesError):
            proc.process(data)

        # test image_on correctness
        image_data.dropped_indices = [0, 1]
        proc.process(data)
        np.testing.assert_array_equal(processed.pp.image_on, data['detector']['assembled'][2])
        np.testing.assert_array_equal(processed.pp.image_off, data['detector']['assembled'][3])

    def testEvenOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.EVEN_TRAIN_ON
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]

        # test off will not be acknowledged without on
        data, processed = self._gen_data(1001)  # off
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

        data, processed = self._gen_data(1002)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        np.testing.assert_array_almost_equal(
            np.mean(data['detector']['assembled'][::2, :, :], axis=0), proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data, processed = self._gen_data(1003)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            processed.pp.image_off, np.mean(data['detector']['assembled'][1::2, :, :], axis=0))

        # --------------------
        # test pulse filtering
        # --------------------

        data, processed = self._gen_data(1002)
        image_data = processed.image
        image_data.dropped_indices = [0, 2]
        with self.assertRaises(DropAllPulsesError):
            proc.process(data)
        image_data.dropped_indices = [1, 3]
        # no Exception since this is an ON pulse
        proc.process(data)
        # drop one on/off indices each
        image_data.dropped_indices = [0, 1]
        proc.process(data)
        np.testing.assert_array_equal(proc._prev_unmasked_on, data['detector']['assembled'][2])

        data, processed = self._gen_data(1003)
        image_data = processed.image
        # drop all off indices
        image_data.dropped_indices = [1, 3]
        with self.assertRaises(DropAllPulsesError):
            self.assertIsNotNone(proc._prev_unmasked_on)
            proc.process(data)
        # drop all on indices
        image_data.dropped_indices = [0, 2]
        # no Exception since this is an OFF pulse
        proc.process(data)
        # drop one on/off indices each
        image_data.dropped_indices = [0, 1]
        proc._prev_unmasked_on = np.ones((2, 2), np.float32)  # any value except None
        proc.process(data)
        np.testing.assert_array_equal(processed.pp.image_off, data['detector']['assembled'][3])

    def testOddOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.ODD_TRAIN_ON
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]

        # test off will not be acknowledged without on
        data, processed = self._gen_data(1002)  # off
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

        data, processed = self._gen_data(1003)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        np.testing.assert_array_almost_equal(
            np.mean(data['detector']['assembled'][::2, :, :], axis=0), proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data, processed = self._gen_data(1004)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            processed.pp.image_off, np.mean(data['detector']['assembled'][1::2, :, :], axis=0))

        # --------------------
        # test pulse filtering
        # --------------------
        # not necessary according to the implementation
