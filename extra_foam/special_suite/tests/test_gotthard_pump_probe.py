from unittest.mock import patch, PropertyMock
from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest

from extra_foam.pipeline.tests import _RawDataMixin

from extra_foam.special_suite import logger, mkQApp
from extra_foam.special_suite.gotthard_pump_probe_proc import GotthardPpProcessor
from extra_foam.special_suite.gotthard_pump_probe_w import (
    GotthardPumpProbeWindow, GotthardPpImageView, GotthardPpFomMeanPlot,
    GotthardPpFomPulsePlot, GotthardPpRawPulsePlot, GotthardPpDarkPulsePlot
)
from extra_foam.special_suite.special_analysis_base import ProcessingError

from . import _SpecialSuiteWindowTestBase, _SpecialSuiteProcessorTestBase

app = mkQApp()
window_type = GotthardPumpProbeWindow

logger.setLevel('INFO')


class TestGotthardPpWindow(_SpecialSuiteWindowTestBase):
    @staticmethod
    def data4visualization(n_pulses=5, n_on=2, n_off=2):
        """Override."""
        return {
            "raw": np.arange(8*n_pulses).reshape(n_pulses, 8),
            "corrected": np.arange(8 * n_pulses).reshape(n_pulses, 8),
            "on_slicer": slice(0, n_on),
            "off_slicer": slice(n_on, n_on + n_off),
            "dark_slicer": slice(n_on + n_off, None),
            "poi_index": n_on - 1,
            "dark_poi_index": 0,
            "vfom": np.arange(40).reshape(5, 8),
            "vfom_ma": np.arange(40).reshape(5, 8),
            "vfom_mean": np.arange(8),
            "vfom_ma_mean": np.arange(8),
        }

    def testWindow(self, win, check_update_plots):
        assert 5 == len(win._plot_widgets_st)
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        assert 1 == counter[GotthardPpImageView]
        assert 1 == counter[GotthardPpFomMeanPlot]
        assert 1 == counter[GotthardPpFomPulsePlot]
        assert 1 == counter[GotthardPpRawPulsePlot]
        assert 1 == counter[GotthardPpDarkPulsePlot]

        check_update_plots()

    def testCtrl(self, win):
        ctrl_widget = win._ctrl_widget_st
        proc = win._worker_st

        # test default values
        assert proc._output_channel
        assert 0 == proc._poi_index
        assert 0 == proc._dark_poi_index
        assert 1 == proc.__class__._vfom_ma.window

        # test set new values
        widget = ctrl_widget.output_ch_le
        widget.clear()
        QTest.keyClicks(widget, "new/output/channel")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert "new/output/channel" == proc._output_channel

        widget = ctrl_widget.on_slicer_le
        widget.clear()
        QTest.keyClicks(widget, ":20:2")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert slice(None, 20, 2) == proc._on_slicer

        widget = ctrl_widget.off_slicer_le
        widget.clear()
        QTest.keyClicks(widget, "1:20:2")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert slice(1, 20, 2) == proc._off_slicer

        for widget, target in zip((ctrl_widget.poi_index_le, ctrl_widget.dark_poi_index_le),
                                  ("_poi_index", "_dark_poi_index")):
            widget.clear()
            QTest.keyClicks(widget, "121")
            QTest.keyPress(widget, Qt.Key_Enter)
            assert 0 == getattr(proc, target)  # maximum is 119 and one can still type "121"
            widget.clear()
            QTest.keyClicks(widget, "119")
            QTest.keyPress(widget, Qt.Key_Enter)
            assert 119 == getattr(proc, target)

        widget = ctrl_widget.dark_slicer_le
        widget.clear()
        QTest.keyClicks(widget, "30:40")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert slice(30, 40) == proc._dark_slicer

        widget = ctrl_widget.ma_window_le
        widget.clear()
        QTest.keyClicks(widget, "9")
        QTest.keyPress(widget, Qt.Key_Enter)
        assert 9 == proc.__class__._vfom_ma.window


class TestGotthardPpProcessor(_RawDataMixin, _SpecialSuiteProcessorTestBase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = GotthardPpProcessor(object(), object())

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
            with patch.object(GotthardPpProcessor, "_ppt",
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
        self._check_processed_data_structure(processed)

        np.testing.assert_array_almost_equal(adc_gt, processed["raw"])
        corrected_gt = adc_gt - np.mean(adc_gt[proc._dark_slicer], axis=0)
        np.testing.assert_array_almost_equal(corrected_gt, processed["corrected"])
        vfom_gt = corrected_gt[proc._on_slicer] - corrected_gt[proc._off_slicer]
        np.testing.assert_array_almost_equal(vfom_gt, processed["vfom"])

    def _check_processed_data_structure(self, ret):
        """Override."""
        data_gt = TestGotthardPpWindow.data4visualization().keys()
        assert set(ret.keys()) == set(data_gt)
