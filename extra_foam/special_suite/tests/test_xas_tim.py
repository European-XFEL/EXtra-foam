from unittest.mock import patch
from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtTest import QTest

from extra_foam.pipeline.tests import _RawDataMixin

from extra_foam.special_suite import logger, mkQApp
from extra_foam.special_suite.xas_tim_proc import ProcessingError, XasTimProcessor
from extra_foam.special_suite.xas_tim_w import (
    XasTimWindow, XasTimXgmPulsePlot, XasTimDigitizerPulsePlot, XasTimMonoScanPlot,
    XasTimCorrelationPlot, XasTimSpectraPlot, XasTimXgmSpectrumPlot,
    _DEFAULT_N_PULSES_PER_TRAIN, _DEFAULT_I0_THRESHOLD, _MAX_WINDOW,
    _MAX_CORRELATION_WINDOW, _DEFAULT_N_BINS
)

from . import _SpecialSuiteWindowTestBase, _SpecialSuiteProcessorTestBase


app = mkQApp()
window_type = XasTimWindow

logger.setLevel('INFO')


class TestXasTimWindow(_SpecialSuiteWindowTestBase):
    @staticmethod
    def data4visualization(n_trains=10, n_pulses_per_train=10, n_bins=6):
        """Override."""
        return {
            "xgm_intensity": np.arange(n_pulses_per_train),
            "digitizer_apds": [np.arange(n_pulses_per_train)] * 3 + [None],
            "energy_scan": (np.arange(n_trains), np.arange(n_trains)),
            "correlation_length": 20,
            "i0": np.arange(n_trains * n_pulses_per_train),
            "i1": [np.arange(n_trains * n_pulses_per_train)] * 3 + [None],
            "spectra": ([np.arange(n_bins)] * 5, np.arange(n_bins), np.arange(n_bins)),
        }

    def testWindow(self, win, check_update_plots):
        assert 9 == len(win._plot_widgets_st)
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        assert 1 == counter[XasTimXgmPulsePlot]
        assert 1 == counter[XasTimDigitizerPulsePlot]
        assert 1 == counter[XasTimMonoScanPlot]
        assert 4 == counter[XasTimCorrelationPlot]
        assert 1 == counter[XasTimSpectraPlot]
        assert 1 == counter[XasTimXgmSpectrumPlot]

        check_update_plots()

    def testCtrl(self, win):
        ctrl_widget = win._ctrl_widget_st
        proc = win._worker_st

        # test default values
        assert proc._xgm_output_channel
        assert proc._digitizer_output_channel
        assert proc._mono_device_id
        assert all(proc._digitizer_channels)
        assert _DEFAULT_N_PULSES_PER_TRAIN == proc._n_pulses_per_train
        assert 1 == proc._apd_stride
        assert _DEFAULT_I0_THRESHOLD == proc._i0_threshold
        assert _MAX_CORRELATION_WINDOW == proc._correlation_window
        assert _MAX_WINDOW == proc._window
        assert _DEFAULT_N_BINS == proc._n_bins

        # test set new values

        widget = ctrl_widget.xgm_output_ch_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert "new/output/channel" == proc._xgm_output_channel

        widget = ctrl_widget.digitizer_output_ch_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert "new/output/channel" == proc._digitizer_output_channel

        widget = ctrl_widget.mono_device_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert "new/output/channel" == proc._mono_device_id

        with patch.object(proc, "reset") as mocked_reset:
            mask_gt = [True] * 4
            for i, widget in enumerate(ctrl_widget.digitizer_channels.buttons()):
                mask_gt[i] = False
                QTest.mouseClick(widget, Qt.LeftButton, pos=QPoint(2, int(widget.height() / 2)))
                assert mask_gt == proc._digitizer_channels
            # test "reset" is only called when new digitizer channel is checked
            mocked_reset.assert_not_called()
            QTest.mouseClick(widget, Qt.LeftButton, pos=QPoint(2, int(widget.height() / 2)))
            mocked_reset.assert_called_once()

        widget = ctrl_widget.n_pulses_per_train_le
        widget.clear()
        QTest.keyClicks(widget, "300")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert 300 == proc._n_pulses_per_train

        widget = ctrl_widget.apd_stride_le
        widget.clear()
        QTest.keyClicks(widget, "2")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert 2 == proc._apd_stride

        widget = ctrl_widget.i0_threshold_le
        widget.clear()
        QTest.keyClicks(widget, "-0.1")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert -0.1 == proc._i0_threshold

        widget = ctrl_widget.correlation_window_le
        widget.clear()
        QTest.keyClicks(widget, "500")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert 500 == proc._correlation_window

        widget = ctrl_widget.pulse_window_le
        widget.clear()
        QTest.keyClicks(widget, "9999")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert 9999 == proc._window

        widget = ctrl_widget.n_bins_le
        widget.clear()
        QTest.keyClicks(widget, "200")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert 200 == proc._n_bins

        displayed_gt = [True] * 4
        for i, widget in enumerate(ctrl_widget.spectra_displayed.buttons()):
            displayed_gt[i] = False
            QTest.mouseClick(widget, Qt.LeftButton, pos=QPoint(2, int(widget.height() / 2)))
            assert displayed_gt == win._spectra._displayed


class TestXasTimProcessor(_RawDataMixin, _SpecialSuiteProcessorTestBase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = XasTimProcessor(object(), object())

    def testGeneral(self):
        proc = self._proc
        proc._n_pulses_per_train = 10
        proc._apd_stride = 1
        proc._xgm_output_channel = "xgm:output"
        proc._digitizer_output_channel = "digitizer:output"
        proc._mono_device_id = "softmono"
        proc._i0_threshold = 7
        proc._correlation_window = 500
        proc._n_bins = 4

        _raw = {
            "xgm:output": [("data.intensitySa3TD", np.arange(10))],
            "digitizer:output": [
                ("digitizers.channel_1_A.apd.pulseIntegral", np.ones(10)),
                ("digitizers.channel_1_B.apd.pulseIntegral", np.ones(10)),
                ("digitizers.channel_1_C.apd.pulseIntegral", np.ones(10)),
                ("digitizers.channel_1_D.apd.pulseIntegral", np.ones(10))
            ],
            "softmono": [("actualEnergy", 600.)]
        }

        # test no digitizer channel is selected
        with pytest.raises(ProcessingError, match="At least one"):
            proc.process(self._gen_data(1234, _raw))

        proc._digitizer_channels = [True, False, True, False]

        for tid in (1234, 1235):
            ret = proc.process(self._gen_data(tid, _raw))

        self._check_processed_data_structure(ret)

        np.testing.assert_array_equal(np.arange(10), ret["xgm_intensity"])
        for i, apd in enumerate(ret["digitizer_apds"]):
            if proc._digitizer_channels[i]:
                np.testing.assert_array_equal(np.ones(10), apd)
            else:
                assert apd is None

        np.testing.assert_array_equal(ret['energy_scan'][0], np.array([1234, 1235]))
        np.testing.assert_array_equal(ret['energy_scan'][1], np.array([600, 600]))

        assert ret['correlation_length'] == 500

        np.testing.assert_array_equal([8, 9, 8, 9], ret["i0"])

        for i, item in enumerate(ret["i1"]):
            if proc._digitizer_channels[i]:
                np.testing.assert_array_equal([1, 1, 1, 1], item)
            else:
                assert item is None

        self._check_reset(proc)

    def _check_processed_data_structure(self, ret):
        """Override."""
        data_gt = TestXasTimWindow.data4visualization().keys()
        assert set(ret.keys()) == set(data_gt)

    def _check_reset(self, proc):
        """Override."""
        assert len(proc._i0) != 0

        proc.reset()

        assert len(proc._i0) == 0
        for i1 in proc._i1:
            assert len(i1) == 0
        assert len(proc._energy) == 0
        assert len(proc._energy_scan) == 0
