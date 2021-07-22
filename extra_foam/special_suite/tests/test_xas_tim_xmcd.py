from unittest.mock import MagicMock, patch
from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtTest import QSignalSpy, QTest

from extra_foam.pipeline.tests import _RawDataMixin

from extra_foam.special_suite import logger, mkQApp
from extra_foam.special_suite.xas_tim_xmcd_proc import XasTimXmcdProcessor
from extra_foam.special_suite.xas_tim_proc import ProcessingError
from extra_foam.special_suite.xas_tim_xmcd_w import (
    XasTimXmcdWindow, XasTimXgmPulsePlot, XasTimDigitizerPulsePlot, XasTimXmcdSlowScanPlot,
    XasTimCorrelationPlot, XasTimXmcdAbsorpPnSpectraPlot,
    XasTimXmcdSpectraPlot, XasTimXgmSpectrumPlot, _DEFAULT_CURRENT_THRESHOLD
)

from . import _SpecialSuiteWindowTestBase, _SpecialSuiteProcessorTestBase


app = mkQApp()

logger.setLevel('INFO')


class TestXasTimXmcdWindow(_SpecialSuiteWindowTestBase):
    _window_type = XasTimXmcdWindow

    @staticmethod
    def data4visualization(n_trains=10, n_pulses_per_train=10, n_bins=6):
        stats = [(np.arange(n_bins), np.arange(n_bins))] * 4
        stats.append(np.arange(n_bins))
        return {
            "xgm_intensity": np.arange(n_pulses_per_train),
            "digitizer_apds": [np.arange(n_pulses_per_train)] * 3 + [None],
            "energy_scan": (np.arange(n_trains), np.arange(n_trains)),
            "current_scan": (np.arange(n_trains), np.arange(n_trains)),
            "correlation_length": 20,
            "i0": np.arange(n_trains * n_pulses_per_train),
            "i1": [np.arange(n_trains * n_pulses_per_train)] * 3 + [None],
            "spectra": (stats, np.arange(n_bins), np.arange(n_bins)),
        }

    def testWindow(self):
        win = self._win

        self.assertEqual(10, len(win._plot_widgets_st))
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[XasTimXgmPulsePlot])
        self.assertEqual(1, counter[XasTimDigitizerPulsePlot])
        self.assertEqual(1, counter[XasTimXmcdSlowScanPlot])
        self.assertEqual(4, counter[XasTimCorrelationPlot])
        self.assertEqual(1, counter[XasTimXmcdAbsorpPnSpectraPlot])
        self.assertEqual(1, counter[XasTimXmcdSpectraPlot])
        self.assertEqual(1, counter[XasTimXgmSpectrumPlot])

        self._check_update_plots()

    def testCtrl(self):
        win = self._win
        ctrl_widget = win._ctrl_widget_st
        proc = win._worker_st

        # test default values
        self.assertTrue(proc._magnet_device_id)
        self.assertEqual(_DEFAULT_CURRENT_THRESHOLD, proc._current_threshold)

        # test set new values

        widget = ctrl_widget.magnet_device_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("new/output/channel", proc._magnet_device_id)

        widget = ctrl_widget.current_threshold_le
        widget.clear()
        QTest.keyClicks(widget, "0.1")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(0.1, proc._current_threshold)

        # only allow to display the spectra of one channel
        for i, widget in enumerate(ctrl_widget.spectra_displayed.buttons()):
            QTest.mouseClick(widget, Qt.LeftButton, pos=QPoint(2, int(widget.height() / 2)))
            self.assertEqual(i, win._pn_spectra._displayed)
            self.assertEqual(i, win._xas_xmcd_spectra._displayed)


class TestXasTimXmcdProcessor(_RawDataMixin, _SpecialSuiteProcessorTestBase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = XasTimXmcdProcessor(object(), object())

    def testGeneral(self):
        proc = self._proc
        proc._n_pulses_per_train = 10
        proc._apd_stride = 1
        proc._xgm_output_channel = "xgm:output"
        proc._digitizer_output_channel = "digitizer:output"
        proc._mono_device_id = "softmono"
        proc._magnet_device_id = "magnet"
        proc._i0_threshold = 7
        proc._current_threshold = 50
        proc._correlation_window = 500
        proc._n_bins = 4

        def _get_raw(current):
            return {
                "xgm:output": [("data.intensitySa3TD", np.arange(10))],
                "digitizer:output": [
                    ("digitizers.channel_1_A.apd.pulseIntegral", np.ones(10)),
                    ("digitizers.channel_1_B.apd.pulseIntegral", np.ones(10)),
                    ("digitizers.channel_1_C.apd.pulseIntegral", np.ones(10)),
                    ("digitizers.channel_1_D.apd.pulseIntegral", np.ones(10))
                ],
                "softmono": [("actualEnergy", 600.)],
                "magnet": [("value", current)]
            }

        # test no digitizer channel is selected
        with pytest.raises(ProcessingError, match="At least one"):
            proc.process(self._gen_data(1234, _get_raw(100)))

        proc._digitizer_channels = [True, False, True, False]

        for i, tid in enumerate((1234, 1235, 1236)):
            ret = proc.process(self._gen_data(tid, _get_raw(100 * i)))

        self._check_processed_data_structure(ret)

        np.testing.assert_array_equal(np.arange(10), ret["xgm_intensity"])
        for i, apd in enumerate(ret["digitizer_apds"]):
            if proc._digitizer_channels[i]:
                np.testing.assert_array_equal(np.ones(10), apd)
            else:
                assert apd is None

        np.testing.assert_array_equal(ret['energy_scan'][0], np.array([1234, 1235, 1236]))
        np.testing.assert_array_equal(ret['energy_scan'][1], np.array([600, 600, 600]))

        np.testing.assert_array_equal(ret['current_scan'][0], np.array([1234, 1235, 1236]))
        np.testing.assert_array_equal(ret['current_scan'][1], np.array([0, 100, 200]))

        assert ret['correlation_length'] == 500

        # data from train 1234 is not included because of the current threshold
        np.testing.assert_array_equal([8, 9, 8, 9], ret["i0"])

        for i, item in enumerate(ret["i1"]):
            if proc._digitizer_channels[i]:
                np.testing.assert_array_equal([1, 1, 1, 1], item)
            else:
                assert item is None

        self._check_reset(proc)

    def _check_processed_data_structure(self, ret):
        """Override."""
        data_gt = TestXasTimXmcdWindow.data4visualization().keys()
        assert set(ret.keys()) == set(data_gt)

    def _check_reset(self, proc):
        """Override."""
        assert len(proc._i0) != 0

        proc.reset()

        assert len(proc._i0) == 0
        for i1 in proc._i1:
            assert len(i1) == 0
        assert len(proc._energy) == 0
        assert len(proc._current) == 0

        assert len(proc._energy_scan) == 0
        assert len(proc._current_scan) == 0
