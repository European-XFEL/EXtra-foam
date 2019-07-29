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
from karaboFAI.pipeline.data_model import ImageData, ProcessedData
from karaboFAI.config import PumpProbeMode
from karaboFAI.pipeline.exceptions import (
    PumpProbeIndexError, ProcessingError
)


class TestImageProcessorPulseTr(unittest.TestCase):
    """Test pulse-resolved ImageProcessor.

    For train-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessorPulse()

        del self._proc._raw_data

        ImageProcessorPulse._raw_data.window = 3
        self._proc._background = -10
        self._proc._threshold_mask = (-100, 100)

    def testMovingAverage(self):
        proc = self._proc

        imgs1 = np.random.randn(2, 2)
        imgs1_gt = imgs1.copy()
        data = {
            'processed': ProcessedData(1),
            'assembled': imgs1,
        }

        proc.process(data)

        np.testing.assert_array_equal(imgs1_gt, proc._raw_data)

        imgs2 = np.random.randn(2, 2)
        imgs2_gt = imgs2.copy()
        data = {
            'processed': ProcessedData(1),
            'assembled': imgs2,
        }

        proc.process(data)

        processed = data['processed']
        self.assertEqual(proc._background, processed.image.background)
        self.assertTupleEqual(proc._threshold_mask,
                              processed.image.threshold_mask)
        # The moving average test is redundant for now since pulse-resolved
        # detector is not allow to set moving average on images on ImageToolWindow.
        self.assertEqual(2, processed.image.ma_count)
        ma_gt = (imgs1_gt + imgs2_gt) / 2.0
        np.testing.assert_array_almost_equal(ma_gt, proc._raw_data)

        # test the internal data of _raw_data shares memory with the first data
        # FIXME: This not true with the c++ code. But will be fixed when
        #        xtensor-python has a new release.
        # self.assertIs(imgs1, proc._raw_data)

    def testImageShapeChangeOnTheFly(self):
        proc = self._proc
        proc._image_mask = np.ones((2, 2), dtype=np.bool)

        proc.process({
            'processed': ProcessedData(1),
            'assembled': np.random.randn(2, 2)
        })

        # image shape changes
        with self.assertRaisesRegex(ProcessingError, 'image mask'):
            proc.process({
                'processed': ProcessedData(2),
                'assembled': np.random.randn(4, 2)
            })
        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((2, 2), dtype=np.bool), proc._image_mask)
        proc._image_mask = None

        # assign a reference image
        proc._reference = np.ones((4, 2), dtype=np.float32)
        # image shape changes
        with self.assertRaisesRegex(ProcessingError, 'reference'):
            proc.process({
                'processed': ProcessedData(3),
                'assembled': np.random.randn(2, 2)
            })
        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((4, 2), dtype=np.float32), proc._reference)
        proc._reference = None


class TestImageProcessorPulsePr(unittest.TestCase):
    """Test pulse-resolved ImageProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessorPulse()

        del self._proc._raw_data

        ImageProcessorPulse._raw_data.window = 3
        self._proc._background = -10
        self._proc._threshold_mask = (-100, 100)

    def testMovingAverage(self):
        # The moving average test is redundant for now since pulse-resolved
        # detector is not allow to set moving average on images on ImageToolWindow.
        proc = self._proc

        imgs1 = np.random.randn(4, 2, 2)
        imgs1_gt = imgs1.copy()
        data = {
            'processed': ProcessedData(1),
            'assembled': imgs1,
        }

        proc.process(data)

        np.testing.assert_array_equal(imgs1_gt, proc._raw_data)

        imgs2 = np.random.randn(4, 2, 2)
        imgs2_gt = imgs2.copy()
        data = {
            'processed': ProcessedData(2),
            'assembled': imgs2,
        }

        proc.process(data)

        processed = data['processed']
        self.assertEqual(proc._background, processed.image.background)
        self.assertTupleEqual(proc._threshold_mask,
                              processed.image.threshold_mask)
        self.assertEqual(2, processed.image.ma_count)
        ma_gt = (imgs1_gt + imgs2_gt) / 2.0
        np.testing.assert_array_almost_equal(ma_gt, proc._raw_data)

        # test the internal data of _raw_data shares memory with the first data
        # FIXME: This not true with the c++ code. But will be fixed when
        #        xtensor-python has a new release.
        # self.assertIs(imgs1, proc._raw_data)

    def testImageShapeChangeOnTheFly(self):
        proc = self._proc
        proc._image_mask = np.ones((2, 2), dtype=np.bool)

        proc.process({
            'processed': ProcessedData(1),
            'assembled': np.random.randn(4, 2, 2)
        })

        # image shape changes
        with self.assertRaisesRegex(ProcessingError, 'image mask'):
            proc.process({
                'processed': ProcessedData(2),
                'assembled': np.random.randn(4, 4, 2)
            })
        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((2, 2), dtype=np.bool), proc._image_mask)
        proc._image_mask = None

        # assign a reference image
        proc._reference = np.ones((4, 2), dtype=np.float32)
        # image shape changes
        with self.assertRaisesRegex(ProcessingError, 'reference'):
            proc.process({
                'processed': ProcessedData(3),
                'assembled': np.random.randn(4, 2, 2)
            })
        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((4, 2), dtype=np.float32), proc._reference)
        proc._reference = None

        # Number of pulses per train changes, but no exception will be raised
        proc.process({
            'processed': ProcessedData(4),
            'assembled': np.random.randn(8, 2, 2)
        })


class TestImageProcessorTrainTr(unittest.TestCase):
    """Test train-resolved ImageProcessor.

    For train-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessorTrain()
        self._proc._on_indices = [0]
        self._proc._off_indices = [0]

    def _gen_data(self, tid):
        data = {'processed': ProcessedData(tid),
                'assembled': np.random.randn(2, 2)}
        image_data = ImageData()
        image_data._background = 0
        image_data._threshold_mask = (-100, 100)
        image_data._pulse_index_filter = [-1]
        image_data._poi_indices = [0, 0]
        return data

    def testPpUndefined(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.UNDEFINED

        data = self._gen_data(1001)

        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)

    def testPpPredefinedOff(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF

        data = self._gen_data(1001)
        assembled = data['assembled']

        proc.process(data)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_on, assembled)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_off, np.zeros((2, 2)))

    def testPpOddOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.ODD_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1002)  # off
        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)

        data = self._gen_data(1003)  # on
        assembled = data['assembled']
        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)

        np.testing.assert_array_almost_equal(assembled, proc._prev_unmasked_on)

        data = self._gen_data(1005)  # on
        assembled = data['assembled']
        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)
        np.testing.assert_array_almost_equal(assembled, proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data = self._gen_data(1006)  # off
        assembled = data['assembled']
        proc.process(data)

        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_off, assembled)

    def testPpEvenOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.EVEN_TRAIN_ON

        # test off will not be acknowledged without on
        data = self._gen_data(1001)  # off

        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)

        data = self._gen_data(1002)  # on
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)
        np.testing.assert_array_almost_equal(assembled, proc._prev_unmasked_on)

        # test when two 'on' are received successively
        data = self._gen_data(1004)  # on
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)
        np.testing.assert_array_almost_equal(assembled, proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data = self._gen_data(1005)  # off
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_off, assembled)


class TestImageProcessorTrainPr(unittest.TestCase):
    """Test train-resolved ImageProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        self._proc = ImageProcessorTrain()
        self._proc._on_indices = [0]
        self._proc._off_indices = [0]

    def _gen_data(self, tid):
        data = {'processed': ProcessedData(tid),
                'assembled': np.random.randn(4, 2, 2)}
        image_data = ImageData()
        image_data._background = 0
        image_data._threshold_mask = (-100, 100)
        image_data._pulse_index_filter = [-1]
        image_data._poi_indices = [0, 0]
        return data

    def testInvalidPulseIndices(self):
        proc = self._proc
        proc._on_indices = [0, 1, 5]
        proc._off_indices = [1]

        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF
        with self.assertRaises(PumpProbeIndexError):
            # the maximum index is 4
            proc.process(self._gen_data(1001))

        proc._off_indices = [1, 3]
        proc._pp_mode = PumpProbeMode.EVEN_TRAIN_ON
        with self.assertRaises(PumpProbeIndexError):
            proc.process(self._gen_data(1001))

        # raises when the same pulse index was found in both
        # on- and off- indices
        proc._on_indices = [0, 1]
        proc._off_indices = [1, 3]
        proc._pp_mode = PumpProbeMode.SAME_TRAIN
        with self.assertRaises(PumpProbeIndexError):
            proc.process(self._gen_data(1001))

        # off-indices check is not trigger in PRE_DEFINED_OFF mode
        proc._off_indices = [5]
        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF
        proc.process(self._gen_data(1001))

    def testUndefined(self):
        proc = self._proc
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]
        proc._threshold_mask = (-np.inf, np.inf)

        proc._pp_mode = PumpProbeMode.UNDEFINED

        data = self._gen_data(1001)
        proc.process(data)

        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)

    def testPredefinedOff(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]

        data = self._gen_data(1001)
        assembled = data['assembled']

        proc.process(data)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_on,
            np.mean(assembled[::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_off, np.zeros((2, 2)))

    def testSameTrain(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.SAME_TRAIN
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]

        data = self._gen_data(1001)
        assembled = data['assembled']

        proc.process(data)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_on,
            np.mean(assembled[::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_off,
            np.mean(assembled[1::2, :, :], axis=0))

    def testEvenOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.EVEN_TRAIN_ON
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]

        # test off will not be acknowledged without on
        data = self._gen_data(1001)  # off

        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)

        data = self._gen_data(1002)  # on
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)
        np.testing.assert_array_almost_equal(
            np.mean(assembled[::2, :, :], axis=0), proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data = self._gen_data(1003)  # off
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_off,
            np.mean(assembled[1::2, :, :], axis=0))

    def testOddOn(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.ODD_TRAIN_ON
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]

        # test off will not be acknowledged without on
        data = self._gen_data(1002)  # off

        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)

        data = self._gen_data(1003)  # on
        assembled = data['assembled']

        proc.process(data)
        self.assertIsNone(data['processed'].pp.image_on)
        self.assertIsNone(data['processed'].pp.image_off)
        np.testing.assert_array_almost_equal(
            np.mean(assembled[::2, :, :], axis=0), proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data = self._gen_data(1004)  # off
        assembled = data['assembled']
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            data['processed'].pp.image_off,
            np.mean(assembled[1::2, :, :], axis=0))
