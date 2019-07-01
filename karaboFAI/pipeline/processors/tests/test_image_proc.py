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
from unittest.mock import MagicMock

import numpy as np

from karaboFAI.pipeline.processors.image_processor import (
    RawImageData, ImageProcessor
)
from karaboFAI.config import PumpProbeMode


class TestRawImageData(unittest.TestCase):

    def testInvalidInput(self):
        with self.assertRaises(TypeError):
            RawImageData()

        with self.assertRaises(TypeError):
            RawImageData([1, 2, 3])

        with self.assertRaises(ValueError):
            RawImageData(np.arange(2))

        with self.assertRaises(ValueError):
            RawImageData(np.arange(16).reshape((2, 2, 2, 2)))

    def testTrainResolved(self):
        data = RawImageData(np.ones((3, 3), dtype=np.float32))
        self.assertEqual(1, data.n_images)

        data.ma_window = 5
        # ma_window will not change until the next data arrive
        self.assertEqual(1, data.ma_window)
        self.assertEqual(1, data.ma_count)
        data.images = 3 * np.ones((3, 3), dtype=np.float32)
        self.assertEqual(5, data.ma_window)
        self.assertEqual(2, data.ma_count)
        np.testing.assert_array_equal(2*np.ones((3, 3), dtype=np.float32),
                                      data.images)

        # set a ma window which is smaller than the current window
        data.ma_window = 3
        self.assertEqual(5, data.ma_window)
        self.assertEqual(2, data.ma_count)
        new_data = 2*np.ones((3, 3), dtype=np.float32)
        data.images = new_data
        self.assertEqual(3, data.ma_window)
        self.assertEqual(1, data.ma_count)
        np.testing.assert_array_equal(new_data, data.images)

        # set an image with a different shape
        new_data = 2*np.ones((3, 1), dtype=np.float32)
        data.images = new_data
        self.assertEqual(3, data.ma_window)
        self.assertEqual(1, data.ma_count)
        np.testing.assert_array_equal(new_data, data.images)

    def testPulseResolved(self):
        data = RawImageData(np.ones((3, 4, 4), dtype=np.float32))

        self.assertEqual(3, data.n_images)

        data.ma_window = 10
        # ma_window will not change until the next data arrive
        self.assertEqual(1, data.ma_window)
        self.assertEqual(1, data.ma_count)
        new_data = 5*np.ones((3, 4, 4), dtype=np.float32)
        data.images = new_data
        self.assertEqual(10, data.ma_window)
        self.assertEqual(2, data.ma_count)
        np.testing.assert_array_equal(3*np.ones((3, 4, 4), dtype=np.float32),
                                      data.images)

        # set a ma window which is smaller than the current window
        data.ma_window = 2
        self.assertEqual(10, data.ma_window)
        self.assertEqual(2, data.ma_count)
        new_data = 0.1*np.ones((3, 4, 4), dtype=np.float32)
        data.images = new_data
        self.assertEqual(2, data.ma_window)
        self.assertEqual(1, data.ma_count)
        np.testing.assert_array_equal(new_data, data.images)

        # set a data with a different number of images
        new_data = 5*np.ones((5, 4, 4))
        data.images = new_data
        self.assertEqual(2, data.ma_window)
        self.assertEqual(1, data.ma_count)
        np.testing.assert_array_equal(new_data, data.images)

        new_data = 1*np.ones((5, 3, 4))
        data.images = new_data
        self.assertEqual(2, data.ma_window)
        self.assertEqual(1, data.ma_count)
        np.testing.assert_array_equal(new_data, data.images)


class TestImageProcessorTr(unittest.TestCase):
    """Test train-resolved ImageProcessor."""
    def setUp(self):
        self._proc = ImageProcessor()
        self._proc.on_indices = [0]
        self._proc.off_indices = [0]
        self._proc.threshold_mask = (-np.inf, np.inf)

    def _gen_data(self, tid):
        imgs = np.random.randn(2, 2)
        data = {'tid': tid,
                'assembled': imgs}
        return data

    def testGeneral(self):
        proc = self._proc

        proc._ma_window = 2
        proc._background = 0
        proc._threshold_mask = (-100, 100)
        proc._pulse_index_filter = [-1]
        proc._vip_pulse_indices = [0, 0]

        imgs1 = np.random.randn(2, 2)
        imgs1_gt = imgs1.copy()
        data = {
            'tid': 1,
            'assembled': imgs1,
        }

        proc.process(data)
        processed = data['processed']

        np.testing.assert_array_almost_equal(imgs1_gt,
                                             processed.image.images)
        np.testing.assert_array_almost_equal(imgs1_gt,
                                             proc._raw_data.images)

        imgs2 = np.random.randn(2, 2)
        imgs2_gt = imgs2.copy()
        data = {
            'tid': 2,
            'assembled': imgs2,
        }

        proc.process(data)
        processed = data['processed']

        ma_gt = (imgs1_gt + imgs2_gt) / 2.0
        np.testing.assert_array_almost_equal(ma_gt,
                                             processed.image.images)
        np.testing.assert_array_almost_equal(ma_gt,
                                             proc._raw_data.images)

        # test the internal data of _raw_data shares memory with the first data
        self.assertIs(imgs1, proc._raw_data.images)

    def testPpUndefined(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.UNDEFINED

        data = self._gen_data(1001)

        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)

    def testPpPredefinedOff(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF

        data = self._gen_data(1001)
        assembled = data['assembled']

        proc.process(data)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.on_image_mean, assembled)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.off_image_mean, np.zeros((2, 2)))

    def testPpOddOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.ODD_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1002)  # off
        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)

        data = self._gen_data(1003)  # on
        assembled = data['assembled']
        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)

        np.testing.assert_array_almost_equal(assembled, proc._prev_unmasked_on)

        data = self._gen_data(1005)  # on
        assembled = data['assembled']
        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)
        np.testing.assert_array_almost_equal(assembled, proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data = self._gen_data(1006)  # off
        assembled = data['assembled']
        proc.process(data)

        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.on_image_mean, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.off_image_mean, assembled)

    def testPpEvenOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.EVEN_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1001)  # off

        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)

        data = self._gen_data(1002)  # on
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)
        np.testing.assert_array_almost_equal(assembled, proc._prev_unmasked_on)

        # test when two 'on' are received successively
        data = self._gen_data(1004)  # on
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)
        np.testing.assert_array_almost_equal(assembled, proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data = self._gen_data(1005)  # off
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.on_image_mean, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.off_image_mean, assembled)


class TestImageProcessorPr(unittest.TestCase):
    """Test pulse-resolved ImageProcessor."""
    def setUp(self):
        self._proc = ImageProcessor()
        self._proc._ma_window = 3
        self._proc._background = -10
        self._proc._threshold_mask = (-100, 100)

        self._proc._pulse_index_filter = [-1]
        self._proc._vip_pulse_indices = [0, 2]
        self._proc._on_indices = [0, 2]
        self._proc._off_indices = [1, 3]

    def testGeneral(self):
        proc = self._proc

        imgs1 = np.random.randn(4, 2, 2)
        imgs1_gt = imgs1.copy()
        data = {
            'tid': 1,
            'assembled': imgs1,
        }

        proc.process(data)

        np.testing.assert_array_equal(imgs1_gt, proc._raw_data.images)

        imgs2 = np.random.randn(4, 2, 2)
        imgs2_gt = imgs2.copy()
        data = {
            'tid': 2,
            'assembled': imgs2,
        }

        proc.process(data)

        processed = data['processed']
        self.assertEqual(proc._background, processed.image.background)
        self.assertEqual(proc._ma_window, processed.image.ma_window)
        self.assertTupleEqual(proc._threshold_mask,
                              processed.image.threshold_mask)
        self.assertEqual(2, processed.image.ma_count)
        # test only VIP pulses are kept
        ma_gt = (imgs1_gt + imgs2_gt) / 2.0
        np.testing.assert_array_almost_equal(ma_gt[0],
                                             processed.image.images[0])
        self.assertIsNone(processed.image.images[1])
        np.testing.assert_array_almost_equal(ma_gt[2],
                                             processed.image.images[2])
        self.assertIsNone(processed.image.images[3])

        np.testing.assert_array_almost_equal(ma_gt, proc._raw_data.images)

        # test the internal data of _raw_data shares memory with the first data
        self.assertIs(imgs1, proc._raw_data.images)

        # test keep all pulse images
        proc._has_analysis = MagicMock(return_value=True)

        imgs3 = np.random.randn(4, 2, 2)
        imgs3_gt = imgs3.copy()
        data = {
            'tid': 3,
            'assembled': imgs3,
        }

        proc.process(data)
        processed = data['processed']

        ma_gt = (imgs1_gt + imgs2_gt + imgs3_gt) / 3.0
        for i in range(4):
            np.testing.assert_array_almost_equal(ma_gt[i],
                                                 processed.image.images[i])

    def _gen_data(self, tid):
        imgs = np.random.randn(4, 2, 2)
        data = {'tid': tid,
                'assembled': imgs.copy()}
        return data

    def testUndefined(self):
        proc = self._proc
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]
        proc._threshold_mask = (-np.inf, np.inf)

        proc._pp_mode = PumpProbeMode.UNDEFINED

        data = self._gen_data(1001)
        proc.process(data)

        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)

    def testPredefinedOff(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF

        data = self._gen_data(1001)
        assembled = data['assembled']

        proc.process(data)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.on_image_mean,
            np.mean(assembled[::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(
            data['processed'].pp.off_image_mean, np.zeros((2, 2)))

    def testSameTrain(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.SAME_TRAIN

        data = self._gen_data(1001)
        assembled = data['assembled']

        proc.process(data)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.on_image_mean,
            np.mean(assembled[::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(
            data['processed'].pp.off_image_mean,
            np.mean(assembled[1::2, :, :], axis=0))

    def testEvenOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.EVEN_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1001)  # off

        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)

        data = self._gen_data(1002)  # on
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)
        np.testing.assert_array_almost_equal(
            np.mean(assembled[::2, :, :], axis=0), proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data = self._gen_data(1003)  # off
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.on_image_mean, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.off_image_mean,
            np.mean(assembled[1::2, :, :], axis=0))

    def testOddOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.ODD_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1002)  # off

        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)

        data = self._gen_data(1003)  # on
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(data['processed'].pp.on_image_mean)
        self.assertIsNone(data['processed'].pp.off_image_mean)
        np.testing.assert_array_almost_equal(
            np.mean(assembled[::2, :, :], axis=0), proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data = self._gen_data(1004)  # off
        assembled = data['assembled']
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.on_image_mean, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.off_image_mean,
            np.mean(assembled[1::2, :, :], axis=0))
