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
from extra_foam.pipeline.tests import _TestDataMixin


class _PumpProbeTestMixin:
    def _check_pp_params_in_data_model(self, data):
        self.assertEqual(self._proc._mode, data.pp.mode)
        self.assertEqual(self._proc._indices_on, data.pp.indices_on)
        self.assertEqual(self._proc._indices_off, data.pp.indices_off)

    def check_other_none(self, processed):
        pp = processed.pp
        self.assertIsNone(pp.on.xgm_intensity)
        self.assertIsNone(pp.off.xgm_intensity)
        self.assertIsNone(pp.on.digitizer_pulse_integral)
        self.assertIsNone(pp.off.digitizer_pulse_integral)

    def check_xgm(self, processed, onoff, indices):
        self.assertEqual(np.mean(processed.pulse.xgm.intensity[indices]),
                         processed.pp.__dict__[onoff].xgm_intensity)

    def check_digitizer(self, processed, onoff, indices):
        self.assertEqual(np.mean(processed.pulse.digitizer['B'].pulse_integral[indices]),
                         processed.pp.__dict__[onoff].digitizer_pulse_integral)


class TestPumpProbeProcessorTr(_PumpProbeTestMixin, _TestDataMixin, unittest.TestCase):
    """Test train-resolved ImageProcessor.

    For train-resolved data.
    """
    def setUp(self):
        self._proc = PumpProbeProcessor()
        self._proc._indices_on = slice(None, None)
        self._proc._indices_off = slice(None, None)

    def _gen_data(self, tid, with_xgm=True, with_digitizer=True):
        shape = (3, 2)
        image_mask = np.zeros(shape, dtype=np.bool)
        image_mask[1, 1] = True
        return self.data_with_assembled(tid, shape,
                                        image_mask=image_mask,
                                        threshold_mask=(-100, 100),
                                        poi_indices=[0, 0],
                                        with_xgm=with_xgm,
                                        with_digitizer=with_digitizer)

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
        proc._mode = PumpProbeMode.REFERENCE_AS_OFF

        data, processed = self._gen_data(1001)

        # test invalid shape of the reference
        processed.image.reference = np.zeros((9, 9), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "Shape"):
            proc.process(data)

        processed.image.reference = None
        image_on_gt = data['assembled']['sliced'].copy()
        image_on_gt[1, 1] = np.nan
        image_off_gt = np.zeros_like(image_on_gt)
        image_off_gt[1, 1] = np.nan
        proc.process(data)
        np.testing.assert_array_almost_equal(processed.pp.image_on, image_on_gt)
        np.testing.assert_array_almost_equal(processed.pp.image_off, image_off_gt)
        self.check_xgm(processed, "on", [0])
        self.check_digitizer(processed, "on", [0])
        self.assertEqual(1, processed.pp.off.xgm_intensity)
        self.assertEqual(1, processed.pp.off.digitizer_pulse_integral)

        self._check_pp_params_in_data_model(processed)
        data, processed = self._gen_data(1001, with_xgm=False, with_digitizer=False)
        proc.process(data)
        self.check_other_none(processed)

    def testPpOddOn(self):
        proc = self._proc
        proc._mode = PumpProbeMode.ODD_TRAIN_ON

        # test off will not be acknowledged without on
        data, processed = self._gen_data(1002)  # off
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)

        data, processed = self._gen_data(1003)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)

        np.testing.assert_array_almost_equal(data['assembled']['sliced'], proc._prev_unmasked_on)

        data, processed = self._gen_data(1005)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)
        np.testing.assert_array_almost_equal(data['assembled']['sliced'], proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on
        self.assertEqual(processed.pulse.xgm.intensity[0], proc._prev_xi_on)
        prev_xi_on = proc._prev_xi_on
        self.assertEqual(processed.pulse.digitizer['B'].pulse_integral[0], proc._prev_dpi_on)
        prev_dpi_on = proc._prev_dpi_on

        data, processed = self._gen_data(1006)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        image_off_gt = data['assembled']['sliced'].copy()
        image_off_gt[1, 1] = np.nan
        np.testing.assert_array_almost_equal(processed.pp.image_off, image_off_gt)
        self.assertEqual(processed.pp.on.xgm_intensity, prev_xi_on)
        self.check_xgm(processed, 'off', [0])
        self.assertEqual(processed.pp.on.digitizer_pulse_integral, prev_dpi_on)
        self.check_digitizer(processed, 'off', [0])

        self._check_pp_params_in_data_model(processed)
        data, processed = self._gen_data(1001, with_xgm=False, with_digitizer=False)
        proc.process(data)
        self.check_other_none(processed)

    def testPpEvenOn(self):
        proc = self._proc
        proc._mode = PumpProbeMode.EVEN_TRAIN_ON

        # test off will not be acknowledged without on
        data, processed = self._gen_data(1001)  # off
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)

        data, processed = self._gen_data(1002)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)
        np.testing.assert_array_almost_equal(data['assembled']['sliced'], proc._prev_unmasked_on)

        # test when two 'on' are received successively
        data, processed = self._gen_data(1004)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)
        np.testing.assert_array_almost_equal(data['assembled']['sliced'], proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on
        self.assertEqual(processed.pulse.xgm.intensity[0], proc._prev_xi_on)
        prev_xi_on = proc._prev_xi_on
        self.assertEqual(processed.pulse.digitizer['B'].pulse_integral[0], proc._prev_dpi_on)
        prev_dpi_on = proc._prev_dpi_on

        data, processed = self._gen_data(1005)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        image_off_gt = data['assembled']['sliced'].copy()
        image_off_gt[1, 1] = np.nan
        np.testing.assert_array_almost_equal(processed.pp.image_off, image_off_gt)
        self.assertEqual(processed.pp.on.xgm_intensity, prev_xi_on)
        self.check_xgm(processed, 'off', [0])
        self.assertEqual(processed.pp.on.digitizer_pulse_integral, prev_dpi_on)
        self.check_digitizer(processed, 'off', [0])

        self._check_pp_params_in_data_model(processed)


class TestPumpProbeProcessorPr(_PumpProbeTestMixin, _TestDataMixin, unittest.TestCase):
    """Test train-resolved PumpProbeProcessor.

    For pulse-resolved data.
    """
    def setUp(self):
        self._proc = PumpProbeProcessor()
        self._proc._indices_on = slice(0, 1, 1)
        self._proc._indices_off = slice(0, 1, 1)

    def _gen_data(self, tid, with_xgm=True, with_digitizer=True):
        shape = (3, 2)
        image_mask = np.zeros(shape, dtype=np.bool)
        image_mask[1, 1] = True
        return self.data_with_assembled(tid, (4, 3, 2),
                                        image_mask=image_mask,
                                        threshold_mask=(-100, 100),
                                        poi_indices=[0, 0],
                                        with_xgm=with_xgm,
                                        with_digitizer=with_digitizer)

    def testFomPulseFilter(self):
        proc = self._proc

        data, processed = self._gen_data(1001)
        image_data = processed.image
        processed.pidx.mask([0, 2])
        proc.process(data)
        # test calculating the average image after pulse filtering
        np.testing.assert_array_equal(
            np.nanmean(data['assembled']['sliced'][[1, 3]], axis=0),
            image_data.mean
        )

    def testInvalidPulseIndices(self):
        proc = self._proc

        # raises when the same pulse index was found in both
        # on- and off- indices
        proc._indices_on = slice(None, None)
        proc._indices_off = slice(None, None)
        proc._mode = PumpProbeMode.SAME_TRAIN
        with self.assertRaises(PumpProbeIndexError):
            data, _ = self._gen_data(1001)
            proc.process(data)

    def testUndefined(self):
        proc = self._proc
        proc._indices_on = slice(0, None, 2)
        proc._indices_off = slice(1, None, 2)
        proc._threshold_mask = (-np.inf, np.inf)

        proc._mode = PumpProbeMode.UNDEFINED

        data, processed = self._gen_data(1001)
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)

        self._check_pp_params_in_data_model(processed)

    def testPredefinedOff(self):
        proc = self._proc
        proc._mode = PumpProbeMode.REFERENCE_AS_OFF
        proc._indices_on = slice(0, None, 2)
        proc._indices_off = slice(1, None, 2)

        data, processed = self._gen_data(1001)

        # test invalid shape of the reference
        processed.image.reference = np.zeros((9, 9), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "Shape"):
            proc.process(data)

        processed.image.reference = None
        image_on_gt = np.mean(data['assembled']['sliced'][::2, :, :], axis=0)
        image_on_gt[1, 1] = np.nan
        image_off_gt = np.zeros_like(image_on_gt)
        image_off_gt[1, 1] = np.nan
        proc.process(data)
        np.testing.assert_array_almost_equal(processed.pp.image_on, image_on_gt)
        np.testing.assert_array_almost_equal(processed.pp.image_off, image_off_gt)

        # XGM and digitizer
        self.check_xgm(processed, "on", [0, 2])
        self.check_digitizer(processed, "on", [0, 2])
        self.assertEqual(1, processed.pp.off.xgm_intensity)
        self.assertEqual(1, processed.pp.off.digitizer_pulse_integral)

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
        image_on_gt = data['assembled']['sliced'][2]
        image_on_gt[1, 1] = np.nan
        np.testing.assert_array_equal(processed.pp.image_on, image_on_gt)
        # XGM and digitizer
        self.check_xgm(processed, "on", [2])
        self.check_digitizer(processed, "on", [2])

        self._check_pp_params_in_data_model(processed)
        data, processed = self._gen_data(1001, with_xgm=False, with_digitizer=False)
        proc.process(data)
        self.check_other_none(processed)

    def testSameTrain(self):
        proc = self._proc
        proc._mode = PumpProbeMode.SAME_TRAIN
        proc._indices_on = slice(0, None, 2)
        proc._indices_off = slice(1, None, 2)

        data, processed = self._gen_data(1001)
        proc.process(data)
        image_on_gt = np.mean(data['assembled']['sliced'][::2, :, :], axis=0)
        image_on_gt[1, 1] = np.nan
        np.testing.assert_array_almost_equal(processed.pp.image_on, image_on_gt)
        image_off_gt = np.mean(data['assembled']['sliced'][1::2, :, :], axis=0)
        image_off_gt[1, 1] = np.nan
        np.testing.assert_array_almost_equal(processed.pp.image_off, image_off_gt)
        # XGM and digitizer
        self.check_xgm(processed, 'on', [0, 2])
        self.check_xgm(processed, 'off', [1, 3])
        self.check_digitizer(processed, 'on', [0, 2])
        self.check_digitizer(processed, 'off', [1, 3])

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
        image_on_gt = data['assembled']['sliced'][2]
        image_on_gt[1, 1] = np.nan
        np.testing.assert_array_equal(processed.pp.image_on, image_on_gt)
        image_off_gt = data['assembled']['sliced'][3]
        image_off_gt[1, 1] = np.nan
        np.testing.assert_array_equal(processed.pp.image_off, image_off_gt)
        # XGM and digitizer
        self.check_xgm(processed, 'on', [2])
        self.check_xgm(processed, 'off', [3])
        self.check_digitizer(processed, 'on', [2])
        self.check_digitizer(processed, 'off', [3])

        self._check_pp_params_in_data_model(processed)
        data, processed = self._gen_data(1001, with_xgm=False, with_digitizer=False)
        proc.process(data)
        self.check_other_none(processed)

    def testEvenOn(self):
        proc = self._proc
        proc._mode = PumpProbeMode.EVEN_TRAIN_ON
        proc._indices_on = slice(0, None, 2)
        proc._indices_off = slice(1, None, 2)

        # test off will not be acknowledged without on
        data, processed = self._gen_data(1001)  # off
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)

        data, processed = self._gen_data(1002)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)
        np.testing.assert_array_almost_equal(
            np.nanmean(data['assembled']['sliced'][::2, :, :], axis=0), proc._prev_unmasked_on)
        prev_unmasked_on = proc._prev_unmasked_on
        # XGM and digitizer
        self.assertEqual(np.mean(processed.pulse.xgm.intensity[::2]), proc._prev_xi_on)
        prev_xi_on = proc._prev_xi_on
        self.assertEqual(np.mean(processed.pulse.digitizer['B'].pulse_integral[::2]), proc._prev_dpi_on)
        prev_dpi_on = proc._prev_dpi_on

        data, processed = self._gen_data(1003)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        image_off_gt = np.mean(data['assembled']['sliced'][1::2, :, :], axis=0)
        image_off_gt[1, 1] = np.nan
        np.testing.assert_array_almost_equal(processed.pp.image_off, image_off_gt)
        # XGM and digitizer
        self.assertEqual(processed.pp.on.xgm_intensity, prev_xi_on)
        self.check_xgm(processed, 'off', [1, 3])
        self.assertEqual(processed.pp.on.digitizer_pulse_integral, prev_dpi_on)
        self.check_digitizer(processed, 'off', [1, 3])

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
        np.testing.assert_array_equal(data['assembled']['sliced'][2], proc._prev_unmasked_on)
        # XGM and digitizer
        self.assertEqual(processed.pulse.xgm.intensity[2], proc._prev_xi_on)
        self.assertEqual(processed.pulse.digitizer['B'].pulse_integral[2], proc._prev_dpi_on)

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
        proc._prev_unmasked_on = np.zeros_like(data['assembled']['sliced'][0])  # any value except None
        proc.process(data)
        image_off_gt = data['assembled']['sliced'][3]
        image_off_gt[1, 1] = np.nan
        np.testing.assert_array_equal(processed.pp.image_off, image_off_gt)
        # XGM and digitizer
        self.check_xgm(processed, 'off', [3])
        self.check_digitizer(processed, 'off', [3])

        self._check_pp_params_in_data_model(processed)
        data, processed = self._gen_data(1001, with_xgm=False, with_digitizer=False)
        proc.process(data)
        self.check_other_none(processed)

    def testOddOn(self):
        proc = self._proc
        proc._mode = PumpProbeMode.ODD_TRAIN_ON
        proc._indices_on = slice(0, None, 2)
        proc._indices_off = slice(1, None, 2)

        # test off will not be acknowledged without on
        data, processed = self._gen_data(1002)  # off
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)

        data, processed = self._gen_data(1003)  # on
        proc.process(data)
        self.assertIsNone(processed.pp.image_on)
        self.assertIsNone(processed.pp.image_off)
        self.check_other_none(processed)
        np.testing.assert_array_almost_equal(
            np.nanmean(data['assembled']['sliced'][::2, :, :], axis=0), proc._prev_unmasked_on)
        # XGM and digitizer
        prev_unmasked_on = proc._prev_unmasked_on
        self.assertEqual(np.mean(processed.pulse.xgm.intensity[::2]), proc._prev_xi_on)
        prev_xi_on = proc._prev_xi_on
        self.assertEqual(np.mean(processed.pulse.digitizer['B'].pulse_integral[::2]), proc._prev_dpi_on)
        prev_dpi_on = proc._prev_dpi_on

        data, processed = self._gen_data(1004)  # off
        proc.process(data)
        self.assertIsNone(proc._prev_unmasked_on)
        np.testing.assert_array_almost_equal(processed.pp.image_on, prev_unmasked_on)
        image_off_gt = np.mean(data['assembled']['sliced'][1::2, :, :], axis=0)
        image_off_gt[1, 1] = np.nan
        np.testing.assert_array_almost_equal(processed.pp.image_off, image_off_gt)
        # XGM and digitizer
        self.assertEqual(processed.pp.on.xgm_intensity, prev_xi_on)
        self.check_xgm(processed, 'off', [1, 3])
        self.assertEqual(processed.pp.on.digitizer_pulse_integral, prev_dpi_on)
        self.check_digitizer(processed, 'off', [1, 3])

        self._check_pp_params_in_data_model(processed)
        data, processed = self._gen_data(1001, with_xgm=False, with_digitizer=False)
        proc.process(data)
        self.check_other_none(processed)

        # --------------------
        # test pulse filtering
        # --------------------
        # not necessary according to the implementation
