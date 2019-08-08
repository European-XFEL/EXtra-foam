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
from karaboFAI.pipeline.processors.tests import _BaseProcessorTest


class TestImageProcessorPulseTr(_BaseProcessorTest):
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

        data, _ = self.data_with_assembled(1, (2, 2))
        imgs1 = data['assembled']
        imgs1_gt = imgs1.copy()

        proc.process(data)

        np.testing.assert_array_equal(imgs1_gt, proc._raw_data)

        data, processed = self.data_with_assembled(2, (2, 2))
        imgs2 = data['assembled']
        imgs2_gt = imgs2.copy()

        proc.process(data)

        self.assertEqual(proc._background, processed.image.background)
        self.assertTupleEqual(proc._threshold_mask, processed.image.threshold_mask)
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

        data, _ = self.data_with_assembled(1, (2, 2))
        proc.process(data)

        # image shape changes
        with self.assertRaisesRegex(ProcessingError, 'image mask'):
            data, _ = self.data_with_assembled(2, (4, 2))
            proc.process(data)

        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((2, 2), dtype=np.bool), proc._image_mask)
        proc._image_mask = None

        # assign a reference image
        proc._reference = np.ones((4, 2), dtype=np.float32)
        # image shape changes
        with self.assertRaisesRegex(ProcessingError, 'reference'):
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

        del self._proc._raw_data

        ImageProcessorPulse._raw_data.window = 3
        self._proc._background = -10
        self._proc._threshold_mask = (-100, 100)

    def testGeneral(self):
        data, processed = self.data_with_assembled(1, (4, 2, 2))

        self._proc.process(data)
        self.assertIsInstance(processed.image.images, list)
        self.assertListEqual([0, 1, 2, 3], processed.image.sliced_indices)

        # test slice to list of indices
        self._proc._pulse_slicer = slice(0, 2)
        self._proc.process(data)
        self.assertListEqual([0, 1], processed.image.sliced_indices)

    def testMovingAverage(self):
        # The moving average test is redundant for now since pulse-resolved
        # detector is not allow to set moving average on images on ImageToolWindow.
        proc = self._proc

        data, _ = self.data_with_assembled(1, (4, 2, 2))
        imgs1 = data['assembled']
        imgs1_gt = imgs1.copy()

        proc.process(data)

        np.testing.assert_array_equal(imgs1_gt, proc._raw_data)

        data, processed = self.data_with_assembled(1, (4, 2, 2))
        imgs2 = data['assembled']
        imgs2_gt = imgs2.copy()

        proc.process(data)

        self.assertEqual(proc._background, processed.image.background)
        self.assertTupleEqual(proc._threshold_mask, processed.image.threshold_mask)
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

        data, _ = self.data_with_assembled(1, (4, 2, 2))
        proc.process(data)

        # image shape changes
        with self.assertRaisesRegex(ProcessingError, 'image mask'):
            data, _ = self.data_with_assembled(2, (4, 4, 2))
            proc.process(data)

        # image mask remains the same, one needs to clear it by hand
        np.testing.assert_array_equal(np.ones((2, 2), dtype=np.bool), proc._image_mask)
        proc._image_mask = None

        # assign a reference image
        proc._reference = np.ones((4, 2), dtype=np.float32)
        # image shape changes
        with self.assertRaisesRegex(ProcessingError, 'reference'):
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
        np.testing.assert_array_almost_equal(processed.pp.image_on, data['assembled'])
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

        np.testing.assert_array_almost_equal(data['assembled'], proc._prev_unmasked_on)

        data, processed = self._gen_data(1005)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        np.testing.assert_array_almost_equal(data['assembled'], proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data, processed = self._gen_data(1006)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_off, data['assembled'])

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
        np.testing.assert_array_almost_equal(data['assembled'], proc._prev_unmasked_on)

        # test when two 'on' are received successively
        data, processed = self._gen_data(1004)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        np.testing.assert_array_almost_equal(data['assembled'], proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data, processed = self._gen_data(1005)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_off, data['assembled'])


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

    def testInvalidPulseIndices(self):
        proc = self._proc
        proc._on_indices = [0, 1, 5]
        proc._off_indices = [1]

        proc._pp_mode = PumpProbeMode.PRE_DEFINED_OFF
        with self.assertRaises(PumpProbeIndexError):
            # the maximum index is 4
            data, _ = self._gen_data(1001)
            proc.process(data)

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
            processed.pp.image_on, np.mean(data['assembled'][::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(processed.pp.image_off, np.zeros((2, 2)))

    def testSameTrain(self):
        proc = self._proc
        proc._pp_mode = PumpProbeMode.SAME_TRAIN
        proc._on_indices = [0, 2]
        proc._off_indices = [1, 3]

        data, processed = self._gen_data(1001)
        proc.process(data)
        np.testing.assert_array_almost_equal(
            processed.pp.image_on, np.mean(data['assembled'][::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(
            processed.pp.image_off, np.mean(data['assembled'][1::2, :, :], axis=0))

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
            np.mean(data['assembled'][::2, :, :], axis=0), proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data, processed = self._gen_data(1003)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            processed.pp.image_off, np.mean(data['assembled'][1::2, :, :], axis=0))

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
            np.mean(data['assembled'][::2, :, :], axis=0), proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on

        data, processed = self._gen_data(1004)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        np.testing.assert_array_almost_equal(
            processed.pp.image_off, np.mean(data['assembled'][1::2, :, :], axis=0))
