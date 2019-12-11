"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import unittest

import numpy as np

from extra_foam.pipeline.processors.pump_probe import PumpProbeProcessor
from extra_foam.config import PumpProbeMode
from extra_foam.pipeline.exceptions import DropAllPulsesError,PumpProbeIndexError
from extra_foam.pipeline.processors.tests import _BaseProcessorTest


class TestPumpProbeProcessorTr(unittest.TestCase, _BaseProcessorTest):
    """Test train-resolved ImageProcessor.

    For train-resolved data.
    """
    def setUp(self):
        self._proc = PumpProbeProcessor()
        self._proc._indices_on = [0]
        self._proc._indices_off = [0]

    def _gen_data(self, tid):
        return self.data_with_assembled(tid, (2, 2),
                                        threshold_mask=(-100, 100),
                                        background=0,
                                        poi_indices=[0, 0])

    def testPpUndefined(self):
        proc = self._proc
        proc._mode = PumpProbeMode.UNDEFINED

        data, processed = self._gen_data(1001)
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

        self._check_pp_params_in_data_model(processed)

    def testPpPredefinedOff(self):
        proc = self._proc
        proc._mode = PumpProbeMode.PRE_DEFINED_OFF

        data, processed = self._gen_data(1001)

        proc.process(data)
        np.testing.assert_array_almost_equal(processed.pp.image_on, data['detector']['assembled'])
        np.testing.assert_array_almost_equal(processed.pp.image_off, np.zeros((2, 2)))

        self._check_pp_params_in_data_model(processed)

    def testPpOddOn(self):
        proc = self._proc
        proc._mode = PumpProbeMode.ODD_TRAIN_ON

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

        self._check_pp_params_in_data_model(processed)

    def testPpEvenOn(self):
        proc = self._proc
        proc._mode = PumpProbeMode.EVEN_TRAIN_ON

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

        self._check_pp_params_in_data_model(processed)

    def _check_pp_params_in_data_model(self, data):
        self.assertEqual(self._proc._mode, data.pp.mode)
        self.assertListEqual(self._proc._indices_on, data.pp.indices_on)
        self.assertListEqual(self._proc._indices_off, data.pp.indices_off)


class TestPumpProbeProcessorPr(unittest.TestCase, _BaseProcessorTest):
    """Test train-resolved PumpProbeProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        self._proc = PumpProbeProcessor()
        self._proc._indices_on = [0]
        self._proc._indices_off = [0]

    def _gen_data(self, tid):
        return self.data_with_assembled(tid, (4, 2, 2),
                                        threshold_mask=(-100, 100),
                                        background=0,
                                        poi_indices=[0, 0])

    def testPulseFilter(self):
        proc = self._proc

        data, processed = self._gen_data(1001)
        image_data = processed.image
        processed.pidx.mask([0, 2])
        proc.process(data)
        # test calculating the average image after pulse filtering
        np.testing.assert_array_equal(
            np.nanmean(data['detector']['assembled'][[1, 3]], axis=0),
            image_data.mean
        )

    def testInvalidPulseIndices(self):
        proc = self._proc
        proc._indices_on = [0, 1, 5]
        proc._indices_off = [1]

        proc._mode = PumpProbeMode.PRE_DEFINED_OFF
        with self.assertRaises(PumpProbeIndexError):
            # the maximum index is 4
            data, _ = self._gen_data(1001)
            proc.process(data)

        proc._indices_on = [0, 1, 5]
        proc._indices_off = [1, 3]
        proc._mode = PumpProbeMode.EVEN_TRAIN_ON
        with self.assertRaises(PumpProbeIndexError):
            data, _ = self._gen_data(1001)
            proc.process(data)

        # raises when the same pulse index was found in both
        # on- and off- indices
        proc._indices_on = [0, 1]
        proc._indices_off = [1, 3]
        proc._mode = PumpProbeMode.SAME_TRAIN
        with self.assertRaises(PumpProbeIndexError):
            data, _ = self._gen_data(1001)
            proc.process(data)

        # off-indices check is not trigger in PRE_DEFINED_OFF mode
        proc._indices_on = [0, 1]
        proc._indices_off = [5]
        proc._mode = PumpProbeMode.PRE_DEFINED_OFF
        data, _ = self._gen_data(1001)
        proc.process(data)

    def testUndefined(self):
        proc = self._proc
        proc._indices_on = [0, 2]
        proc._indices_off = [1, 3]
        proc._threshold_mask = (-np.inf, np.inf)

        proc._mode = PumpProbeMode.UNDEFINED

        data, processed = self._gen_data(1001)
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

        self._check_pp_params_in_data_model(processed)

    def testPredefinedOff(self):
        proc = self._proc
        proc._mode = PumpProbeMode.PRE_DEFINED_OFF
        proc._indices_on = [0, 2]
        proc._indices_off = [1, 3]

        data, processed = self._gen_data(1001)
        proc.process(data)
        np.testing.assert_array_almost_equal(
            processed.pp.image_on, np.mean(data['detector']['assembled'][::2, :, :], axis=0))
        np.testing.assert_array_almost_equal(processed.pp.image_off, np.zeros((2, 2)))

        # --------------------
        # test pulse filtering
        # --------------------

        data, processed = self._gen_data(1002)
        processed.pidx.mask([0, 2])
        with self.assertRaises(DropAllPulsesError):
            proc.process(data)

        data, processed = self._gen_data(1002)
        processed.pidx.mask([1, 3])
        # no Exception
        proc.process(data)

        # test image_on correctness
        processed.pidx.mask([0])
        proc.process(data)
        np.testing.assert_array_equal(processed.pp.image_on, data['detector']['assembled'][2])

        self._check_pp_params_in_data_model(processed)

    def testSameTrain(self):
        proc = self._proc
        proc._mode = PumpProbeMode.SAME_TRAIN
        proc._indices_on = [0, 2]
        proc._indices_off = [1, 3]

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
        processed.pidx.mask([0, 2])
        with self.assertRaises(DropAllPulsesError):
            proc.process(data)

        data, processed = self._gen_data(1002)
        processed.pidx.mask([1, 3])
        with self.assertRaises(DropAllPulsesError):
            proc.process(data)

        # test image_on correctness
        data, processed = self._gen_data(1002)
        processed.pidx.mask([0, 1])
        proc.process(data)
        np.testing.assert_array_equal(processed.pp.image_on, data['detector']['assembled'][2])
        np.testing.assert_array_equal(processed.pp.image_off, data['detector']['assembled'][3])

        self._check_pp_params_in_data_model(processed)

    def testEvenOn(self):
        proc = self._proc
        proc._mode = PumpProbeMode.EVEN_TRAIN_ON
        proc._indices_on = [0, 2]
        proc._indices_off = [1, 3]

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
        processed.pidx.mask([0, 2])
        with self.assertRaises(DropAllPulsesError):
            proc.process(data)
        data, processed = self._gen_data(1002)
        processed.pidx.mask([1, 3])
        # no Exception since this is an ON pulse
        proc.process(data)
        # drop one on/off indices each
        processed.pidx.mask([0, 1])
        proc.process(data)
        np.testing.assert_array_equal(proc._prev_unmasked_on, data['detector']['assembled'][2])

        data, processed = self._gen_data(1003)
        processed.pidx.mask([1, 3])  # drop all off indices
        with self.assertRaises(DropAllPulsesError):
            self.assertIsNotNone(proc._prev_unmasked_on)
            proc.process(data)

        # drop all on indices
        data, processed = self._gen_data(1003)
        processed.pidx.mask([0, 2])
        # no Exception since this is an OFF pulse
        proc.process(data)

        # drop one on/off indices each
        data, processed = self._gen_data(1003)
        processed.pidx.mask([0, 1])
        proc._prev_unmasked_on = np.ones((2, 2), np.float32)  # any value except None
        proc.process(data)
        np.testing.assert_array_equal(processed.pp.image_off, data['detector']['assembled'][3])

        self._check_pp_params_in_data_model(processed)

    def testOddOn(self):
        proc = self._proc
        proc._mode = PumpProbeMode.ODD_TRAIN_ON
        proc._indices_on = [0, 2]
        proc._indices_off = [1, 3]

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

        self._check_pp_params_in_data_model(processed)

        # --------------------
        # test pulse filtering
        # --------------------
        # not necessary according to the implementation

    def _check_pp_params_in_data_model(self, data):
        self.assertEqual(self._proc._mode, data.pp.mode)
        self.assertListEqual(self._proc._indices_on, data.pp.indices_on)
        self.assertListEqual(self._proc._indices_off, data.pp.indices_off)
