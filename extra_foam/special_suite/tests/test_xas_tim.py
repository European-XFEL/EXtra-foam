import unittest
from unittest.mock import MagicMock, patch
from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtTest import QSignalSpy, QTest

from extra_foam.logger import logger_suite as logger
from extra_foam.gui import mkQApp
from extra_foam.special_suite.xas_tim_proc import XasTimProcessor
from extra_foam.special_suite.xas_tim_w import (
    XasTimWindow, XasTimXgmPulsePlot, XasTimDigitizerPulsePlot, XasTimMonoScanPlot,
    XasTimCorrelationPlot, XasTimAbsorpSpectraPlot, XasTimXgmSpectrumPlot
)
from extra_foam.pipeline.tests import _RawDataMixin


app = mkQApp()

logger.setLevel('CRITICAL')


class TestXasTimWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("extra_foam.special_suite.special_analysis_base._SpecialAnalysisBase.startWorker"):
            cls._win = XasTimWindow('SCS')

    @classmethod
    def tearDown(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._win.close()

    def testWindow(self):
        win = self._win

        self.assertEqual(9, len(win._plot_widgets_st))
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[XasTimXgmPulsePlot])
        self.assertEqual(1, counter[XasTimDigitizerPulsePlot])
        self.assertEqual(1, counter[XasTimMonoScanPlot])
        self.assertEqual(4, counter[XasTimCorrelationPlot])
        self.assertEqual(1, counter[XasTimAbsorpSpectraPlot])
        self.assertEqual(1, counter[XasTimXgmSpectrumPlot])

        win.updateWidgetsST()

    def testCtrl(self):
        from extra_foam.special_suite.xas_tim_w import (
            _DEFAULT_N_PULSES_PER_TRAIN, _DEFAULT_XGM_THRESHOLD, _MAX_WINDOW,
            _MAX_CORRELATION_WINDOW, _DEFAULT_N_BINS
        )

        win = self._win
        ctrl_widget = win._ctrl_widget_st
        proc = win._worker_st

        # test default values
        self.assertTrue(proc._xgm_output_channel)
        self.assertTrue(proc._digitizer_output_channel)
        self.assertTrue(proc._mono_device_id)
        self.assertEqual(0x0F, proc._digitizer_channel_mask)
        self.assertEqual(_DEFAULT_N_PULSES_PER_TRAIN, proc._n_pulses_per_train)
        self.assertEqual(1, proc._apd_stride)
        self.assertEqual(_DEFAULT_XGM_THRESHOLD, proc._xgm_threshold)
        self.assertEqual(_MAX_CORRELATION_WINDOW, proc._correlation_window)
        self.assertEqual(_MAX_WINDOW, proc._window)
        self.assertEqual(_DEFAULT_N_BINS, proc._n_bins)

        # test set new values

        widget = ctrl_widget.xgm_output_ch_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("new/output/channel", proc._xgm_output_channel)

        widget = ctrl_widget.digitizer_output_ch_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("new/output/channel", proc._digitizer_output_channel)

        widget = ctrl_widget.mono_device_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("new/output/channel", proc._mono_device_id)

        expected_masks = [0x0E, 0x0C, 0x08, 0x00]
        for widget, mask_gt in zip(ctrl_widget.spectra_displayed, expected_masks):
            QTest.mouseClick(widget, Qt.LeftButton, pos=QPoint(2, widget.height()/2))
            self.assertEqual(mask_gt, proc._digitizer_channel_mask)

        widget = ctrl_widget.n_pulses_per_train_le
        widget.clear()
        QTest.keyClicks(widget, "300")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(300, proc._n_pulses_per_train)

        widget = ctrl_widget.apd_stride_le
        widget.clear()
        QTest.keyClicks(widget, "2")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(2, proc._apd_stride)

        widget = ctrl_widget.xgm_threshold_le
        widget.clear()
        QTest.keyClicks(widget, "-0.1")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(-0.1, proc._xgm_threshold)

        widget = ctrl_widget.correlation_window_le
        widget.clear()
        QTest.keyClicks(widget, "500")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(500, proc._correlation_window)

        widget = ctrl_widget.pulse_window_le
        widget.clear()
        QTest.keyClicks(widget, "9999")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(9999, proc._window)

        widget = ctrl_widget.n_bins_le
        widget.clear()
        QTest.keyClicks(widget, "200")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(200, proc._n_bins)


class TestXasTimProcessor(_RawDataMixin):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = XasTimProcessor(object(), object())

    def testGeneral(self):
        proc = self._proc
        proc._xgm_output_channel = "xgm:output"
        proc._digitizer_output_channel = "digitizer:output"
        proc._mono_device_id = "softmono"

        data = self._gen_data(1234, {
            "xgm:output": [("data.intensitySa3TD", np.ones(10))],
            "digitizer:output": [
                ("digitizers.channel_1_A.apd.pulseIntegral", np.ones(10)),
                ("digitizers.channel_1_B.apd.pulseIntegral", np.ones(10)),
                ("digitizers.channel_1_C.apd.pulseIntegral", np.ones(10)),
                ("digitizers.channel_1_D.apd.pulseIntegral", np.ones(10))
            ],
            "softmono": [("actualEnergy", 600.)]
        })

        ret = proc.process(data)
        assert ret is not None

        # test reset
        proc.reset()
        assert len(proc._i0) == 0
        for i1 in proc._i1:
            assert len(i1) == 0
        assert len(proc._energy) == 0
        assert len(proc._energy_scan) == 0
