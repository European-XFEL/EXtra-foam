import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy, QTest

from extra_foam.logger import logger_suite as logger
from extra_foam.gui import mkQApp
from extra_foam.special_suite.gotthard_pump_probe_proc import GotthardPumpProbeProcessor
from extra_foam.special_suite.gotthard_pump_probe_w import (
    GotthardPumpProbeWindow, GotthardPumpProbeImageView, GotthardPumpProbeAvgPlot,
    GotthardPumpProbePulsePlot,
)
from extra_foam.special_suite.special_analysis_base import (
    ProcessingError
)
from extra_foam.pipeline.tests import _RawDataMixin

app = mkQApp()

logger.setLevel('CRITICAL')


class TestGotthardPumpProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("extra_foam.special_suite.special_analysis_base._SpecialAnalysisBase.startWorker"):
            cls._win = GotthardPumpProbeWindow('SCS')

    @classmethod
    def tearDown(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._win.close()

    def testWindow(self):
        win = self._win

        self.assertEqual(3, len(win._plot_widgets_st))
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[GotthardPumpProbeImageView])
        self.assertEqual(1, counter[GotthardPumpProbeAvgPlot])
        self.assertEqual(1, counter[GotthardPumpProbePulsePlot])

        win.updateWidgetsST()

    def testCtrl(self):
        win = self._win
        ctrl_widget = win._ctrl_widget_st
        proc = win._worker_st

        # test default values
        self.assertTrue(proc._output_channel)
        self.assertEqual(0, proc._poi_index)
        self.assertEqual(1, proc.__class__._vfom_ma.window)

        # test set new values
        widget = ctrl_widget.output_ch_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("new/output/channel", proc._output_channel)

        widget = ctrl_widget.on_slicer_le
        widget.clear()
        QTest.keyClicks(widget, ":20:2")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(slice(None, 20, 2), proc._on_slicer)

        widget = ctrl_widget.off_slicer_le
        widget.clear()
        QTest.keyClicks(widget, "1:20:2")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(slice(1, 20, 2), proc._off_slicer)

        widget = ctrl_widget.poi_index_le
        widget.clear()
        QTest.keyClicks(widget, "121")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(0, proc._poi_index)  # maximum is 119 and one can still type "121"
        widget.clear()
        QTest.keyClicks(widget, "119")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(119, proc._poi_index)

        widget = ctrl_widget.dark_slicer_le
        widget.clear()
        QTest.keyClicks(widget, "30:40")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(slice(30, 40), proc._dark_slicer)

        widget = ctrl_widget.ma_window_le
        widget.clear()
        QTest.keyClicks(widget, "9")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(9, proc.__class__._vfom_ma.window)


class TestGotthardPumpProbeProcessor(_RawDataMixin):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = GotthardPumpProbeProcessor(object(), object())

        self._proc._output_channel = "gotthard:output"
        self._adc = np.random.randint(0, 100, size=(10, 4), dtype=np.uint16)

        self._proc._on_slicer = slice(0, 4, 2)
        self._proc._off_slicer = slice(1, 4, 2)
        self._proc._dark_slicer = slice(4, None)

    def _get_data(self, tid, times=1):
        # data, meta
        return self._gen_data(tid, {
            "gotthard:output": [
                ("data.adc", times * self._adc),
                ("data.3d", np.ones((4, 2, 2)))
            ]})

    def testPreProcessing(self):
        proc = self._proc
        data = self._get_data(12345)

        with pytest.raises(ProcessingError, match="actual 3D"):
            with patch.object(GotthardPumpProbeProcessor, "_ppt",
                              new_callable=PropertyMock, create=True, return_value="data.3d"):
                proc.process(data)

        with pytest.raises(ProcessingError, match="out of boundary"):
            proc._poi_index = 2
            proc.process(data)

        # test not raise
        proc._poi_index = 1
        proc.process(data)

        with pytest.raises(ProcessingError, match="on and off pulses are different"):
            proc._off_slicer = slice(0, None)
            proc.process(data)

    def testProcessing(self):
        from extra_foam.special_suite.gotthard_proc import _PIXEL_DTYPE

        proc = self._proc
        proc._poi_index = 1

        adc_gt = self._adc.astype(_PIXEL_DTYPE)

        processed = proc.process(self._get_data(12345))

        np.testing.assert_array_almost_equal(adc_gt, processed["spectrum"])
