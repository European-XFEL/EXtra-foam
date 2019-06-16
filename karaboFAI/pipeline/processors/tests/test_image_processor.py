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

from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.processors.image_processor import (
    _RawImageData, GeneralImageProcessor, ImageProcessor,
    PumpProbeImageProcessor
)
from karaboFAI.config import PumpProbeMode


class TestRawImageData(unittest.TestCase):

    def testInvalidInput(self):
        with self.assertRaises(TypeError):
            _RawImageData([1, 2, 3])

        with self.assertRaises(ValueError):
            _RawImageData(np.arange(2))

        with self.assertRaises(ValueError):
            _RawImageData(np.arange(16).reshape((2, 2, 2, 2)))

    def test_trainresolved_ma(self):
        """Test the case with moving average of image."""
        imgs_orig = np.arange(16, dtype=np.float).reshape(4, 4)

        img_data = _RawImageData()
        self.assertEqual(0, img_data.n_images)
        img_data.images = np.copy(imgs_orig)
        self.assertEqual(1, img_data.n_images)

        img_data.ma_window = 3
        img_data.images = imgs_orig - 2
        self.assertEqual(2, img_data.ma_count)
        self.assertEqual(1, img_data.n_images)

        img_data.images = imgs_orig + 2
        self.assertEqual(3, img_data.ma_count)

        img_data.images = imgs_orig
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig, img_data.images)

        # test moving average window size change
        img_data.ma_window = 4
        img_data.images = imgs_orig - 4
        self.assertEqual(4, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig - 1, img_data.images)

        img_data.ma_window = 2
        self.assertEqual(2, img_data.ma_window)
        self.assertEqual(0, img_data.ma_count)
        self.assertIsNone(img_data.images)

        img_data.clear()
        self.assertEqual(1, img_data.ma_window)

    def test_pulseresolved_ma(self):
        imgs_orig = np.arange(32, dtype=np.float).reshape((2, 4, 4))

        # test if images' shapes are different
        img_data = _RawImageData(np.copy(imgs_orig[0, ...]))
        img_data.ma_window = 3
        img_data.images = np.copy(imgs_orig)
        self.assertEqual(1, img_data.ma_count)

        img_data.images = imgs_orig - 2
        self.assertEqual(2, img_data.ma_count)

        img_data.images = imgs_orig + 2
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig, img_data.images)

        img_data.images = imgs_orig + 3
        self.assertEqual(3, img_data.ma_count)
        np.testing.assert_array_equal(imgs_orig+1, img_data.images)

        img_data.clear()
        self.assertEqual(1, img_data.ma_window)
        self.assertEqual(0, img_data.ma_count)
        self.assertIsNone(img_data.images)


class TestImageProcessor(unittest.TestCase):
    def testImageProcessor(self):
        proc = ImageProcessor()

        self.assertIsInstance(proc._children[0], GeneralImageProcessor)
        self.assertIsInstance(proc._children[1], PumpProbeImageProcessor)

    def testGeneralProcPulseResolved(self):
        proc = GeneralImageProcessor()

        proc.ma_window = 3
        proc.background = -10
        proc.threshold_mask = (-100, 100)

        proc.pulse_index_filter = [-1]
        proc.vip_pulse_indices = [0, 2]

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
        self.assertEqual(proc.background, processed.image.background)
        self.assertEqual(proc.ma_window, processed.image.ma_window)
        self.assertTupleEqual(proc.threshold_mask,
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

    def testGeneralProcTrainResolved(self):
        proc = GeneralImageProcessor()

        proc.ma_window = 2
        proc.background = 0
        proc.threshold_mask = (-100, 100)

        proc.pulse_index_filter = [-1]
        proc.vip_pulse_indices = [0, 1]

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


class TestPumpProbeImageProcessorPs(unittest.TestCase):
    def setUp(self):
        self.proc = PumpProbeImageProcessor()
        self.proc.on_indices = [0, 2]
        self.proc.off_indices = [1, 3]
        self.proc.threshold_mask = (-np.inf, np.inf)

    def _gen_data(self, tid):
        imgs = np.random.randn(4, 2, 2)
        data = {'tid': tid,
                'processed': ProcessedData(tid, imgs),
                'assembled': imgs.copy()}
        return data

    def testPsUndefined(self):
        self.proc.pp_mode = PumpProbeMode.UNDEFINED

        data = self._gen_data(1001)
        processed = data['processed']

        self.proc.process(processed)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)

    def testPsPredefinedOff(self):
        self.proc.pp_mode = PumpProbeMode.PRE_DEFINED_OFF

        data = self._gen_data(1001)
        processed = data['processed']
        assembled = data['assembled']

        self.proc.process(data)
        np.testing.assert_array_almost_equal(
            processed.pp.on_image_mean, np.mean(assembled[::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(
            processed.pp.off_image_mean, np.zeros((2, 2)))

    def testPPsSameTrain(self):
        self.proc.pp_mode = PumpProbeMode.SAME_TRAIN

        data = self._gen_data(1001)
        processed = data['processed']
        assembled = data['assembled']

        self.proc.process(data)
        np.testing.assert_array_almost_equal(
            processed.pp.on_image_mean, np.mean(assembled[::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(
            processed.pp.off_image_mean, np.mean(assembled[1::2, :, :], axis=0))

    def testPsEvenOn(self):
        self.proc.pp_mode = PumpProbeMode.EVEN_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1001)  # off
        processed = data['processed']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)

        data = self._gen_data(1002)  # on
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)
        np.testing.assert_array_almost_equal(
            np.mean(assembled[::2, :, :], axis=0), self.proc._prev_on)
        prev_on = self.proc._prev_on

        data = self._gen_data(1003)  # off
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(self.proc._prev_on)
        np.testing.assert_array_almost_equal(
            processed.pp.on_image_mean, prev_on)
        np.testing.assert_array_almost_equal(
            processed.pp.off_image_mean, np.mean(assembled[1::2, :, :], axis=0))

    def testPsOddOn(self):
        self.proc.pp_mode = PumpProbeMode.ODD_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1002)  # off
        processed = data['processed']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)

        data = self._gen_data(1003)  # on
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)
        np.testing.assert_array_almost_equal(
            np.mean(assembled[::2, :, :], axis=0), self.proc._prev_on)
        prev_on = self.proc._prev_on

        data = self._gen_data(1004)  # off
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(self.proc._prev_on)
        np.testing.assert_array_almost_equal(
            processed.pp.on_image_mean, prev_on)
        np.testing.assert_array_almost_equal(
            processed.pp.off_image_mean, np.mean(assembled[1::2, :, :], axis=0))


class TestPumpProbeImageProcessorTs(unittest.TestCase):
    def setUp(self):
        self.proc = PumpProbeImageProcessor()
        self.proc.on_indices = [0]
        self.proc.off_indices = [0]
        self.proc.threshold_mask = (-np.inf, np.inf)

    def _gen_data(self, tid):
        imgs = np.random.randn(2, 2)
        data = {'tid': tid,
                'processed': ProcessedData(tid, imgs),
                'assembled': imgs}
        return data

    def testPsUndefined(self):
        self.proc.pp_mode = PumpProbeMode.UNDEFINED

        data = self._gen_data(1001)
        processed = data['processed']

        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)

    def testPsPredefinedOff(self):
        self.proc.pp_mode = PumpProbeMode.PRE_DEFINED_OFF

        data = self._gen_data(1001)
        processed = data['processed']
        assembled = data['assembled']

        self.proc.process(data)
        np.testing.assert_array_almost_equal(
            processed.pp.on_image_mean, assembled)
        np.testing.assert_array_almost_equal(
            processed.pp.off_image_mean, np.zeros((2, 2)))

    def testPsEvenOn(self):
        self.proc.pp_mode = PumpProbeMode.EVEN_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1001)  # off
        processed = data['processed']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)

        data = self._gen_data(1002)  # on
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)
        np.testing.assert_array_almost_equal(assembled, self.proc._prev_on)

        # test when two 'on' are received successively
        data = self._gen_data(1004)  # on
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)
        np.testing.assert_array_almost_equal(assembled, self.proc._prev_on)
        prev_on = self.proc._prev_on

        data = self._gen_data(1005)  # off
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(self.proc._prev_on)
        np.testing.assert_array_almost_equal(
            processed.pp.on_image_mean, prev_on)
        np.testing.assert_array_almost_equal(
            processed.pp.off_image_mean, assembled)

    def testTsOddOn(self):
        self.proc.pp_mode = PumpProbeMode.ODD_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1002)  # off
        processed = data['processed']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)

        data = self._gen_data(1003)  # on
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)
        np.testing.assert_array_almost_equal(assembled, self.proc._prev_on)

        data = self._gen_data(1005)  # on
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(processed.pp.on_image_mean)
        self.assertIsNone(processed.pp.off_image_mean)
        np.testing.assert_array_almost_equal(assembled, self.proc._prev_on)
        prev_on = self.proc._prev_on

        data = self._gen_data(1006)  # off
        processed = data['processed']
        assembled = data['assembled']
        self.proc.process(data)
        self.assertIsNone(self.proc._prev_on)
        np.testing.assert_array_almost_equal(
            processed.pp.on_image_mean, prev_on)
        np.testing.assert_array_almost_equal(
            processed.pp.off_image_mean, assembled)
