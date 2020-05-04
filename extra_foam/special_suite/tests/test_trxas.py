from unittest.mock import MagicMock, patch
from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy, QTest

from extra_foam.logger import logger_suite as logger
from extra_foam.gui import mkQApp
from extra_foam.special_suite.trxas_proc import TrXasProcessor
from extra_foam.special_suite.trxas_w import (
    TrXasWindow, TrXasSpectraPlot, TrXasRoiImageView, TrXasHeatmap
)
from extra_foam.pipeline.tests import _TestDataMixin

from . import _SpecialSuiteWindowTestBase, _SpecialSuiteProcessorTestBase


app = mkQApp()

logger.setLevel('CRITICAL')


class TestTrXasWindow(_SpecialSuiteWindowTestBase):
    @classmethod
    def setUpClass(cls):
        cls._win = TrXasWindow('SCS')

    @classmethod
    def tearDownClass(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._win.close()

    @staticmethod
    def data4visualization(n_bins1=6, n_bins2=4):
        """Override."""
        return {
            "roi1": np.ones((2, 2)),
            "roi2": np.ones((2, 2)),
            "roi3": np.ones((2, 2)),
            "centers1": np.arange(n_bins1),
            "counts1": np.arange(n_bins1),
            "centers2": np.arange(n_bins2),
            "a13_stats": np.arange(n_bins1),
            "a23_stats": np.arange(n_bins1),
            "a21_stats": np.arange(n_bins1),
            "a21_heat": np.ones((n_bins2, n_bins1)),
            "a21_heat_count": np.ones((n_bins2, n_bins1)),
        }

    def testWindow(self):
        win = self._win

        self.assertEqual(6, len(win._plot_widgets_st))
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        self.assertEqual(3, counter[TrXasRoiImageView])
        self.assertEqual(2, counter[TrXasSpectraPlot])
        self.assertEqual(1, counter[TrXasHeatmap])

        self._check_update_plots()

    def testCtrl(self):
        from extra_foam.special_suite.trxas_w import _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE

        win = self._win
        ctrl_widget = win._ctrl_widget_st
        proc = win._worker_st

        # test default values
        self.assertTrue(proc._device_id1)
        self.assertTrue(proc._ppt1)
        self.assertTrue(win._a21._plot_item.getAxis("bottom").label.toPlainText())
        self.assertTrue(win._a13_a23._plot_item.getAxis("bottom").label.toPlainText())
        self.assertTrue(win._a21_heatmap._plot_widget._plot_item.getAxis("left").label.toPlainText())
        self.assertTrue(win._a21_heatmap._plot_widget._plot_item.getAxis("bottom").label.toPlainText())
        self.assertTrue(proc._device_id2)
        self.assertTrue(proc._ppt2)
        default_bin_range = tuple(float(v) for v in _DEFAULT_BIN_RANGE.split(','))
        self.assertTupleEqual(default_bin_range, proc._bin_range1)
        self.assertTupleEqual(default_bin_range, proc._bin_range2)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_bins1)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_bins2)

        # test set new values

        widget = ctrl_widget.device_id1_le
        widget.clear()
        QTest.keyClicks(widget, "phase/shifter")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("phase/shifter", proc._device_id1)

        widget = ctrl_widget.ppt1_le
        widget.clear()
        QTest.keyClicks(widget, "phase/shifter/ppt")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("phase/shifter/ppt", proc._ppt1)

        with patch.object(win._a21, "setLabel") as mocked_setLabel1:
            with patch.object(win._a13_a23, "setLabel") as mocked_setLabel2:
                with patch.object(win._a21_heatmap, "setLabel") as mocked_setLabel3:
                    widget = ctrl_widget.label1_le
                    widget.clear()
                    QTest.keyClicks(widget, "faked delay (s)")
                    QTest.keyPress(widget, Qt.Key_Enter)
                    mocked_setLabel1.assert_called_once_with("bottom", 'faked delay (s)')
                    mocked_setLabel2.assert_called_once_with("bottom", 'faked delay (s)')
                    mocked_setLabel3.assert_called_once_with("left", 'faked delay (s)')
                    mocked_setLabel3.reset_mock()

                    widget = ctrl_widget.label2_le
                    widget.clear()
                    QTest.keyClicks(widget, "faked energy (eV)")
                    QTest.keyPress(widget, Qt.Key_Enter)
                    mocked_setLabel3.assert_called_once_with("bottom", 'faked energy (eV)')

        widget = ctrl_widget.device_id2_le
        widget.clear()
        QTest.keyClicks(widget, "mono")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("mono", proc._device_id2)

        widget = ctrl_widget.ppt2_le
        widget.clear()
        QTest.keyClicks(widget, "mono/ppt")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("mono/ppt", proc._ppt2)

        widget = ctrl_widget.bin_range2_le
        widget.clear()
        QTest.keyClicks(widget, "-1.0, 1.0")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertTupleEqual((-1.0, 1.0), proc._bin_range2)

        widget = ctrl_widget.n_bins2_le
        widget.clear()
        QTest.keyClicks(widget, "1000")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(100, proc._n_bins2)  # maximum is 999 and one can not put the 3rd 0 in
        widget.clear()
        QTest.keyClicks(widget, "999")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(999, proc._n_bins2)

        widget = ctrl_widget.bin_range1_le
        widget.clear()
        QTest.keyClicks(widget, "-1, 1")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertTupleEqual((-1, 1), proc._bin_range1)

        widget = ctrl_widget.n_bins1_le
        widget.clear()
        QTest.keyClicks(widget, "1000")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(100, proc._n_bins1)  # maximum is 999 and one can not put the 3rd 0 in
        widget.clear()
        QTest.keyClicks(widget, "999")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(999, proc._n_bins1)

        QTest.mouseClick(ctrl_widget.swap_btn, Qt.LeftButton)
        self.assertEqual("phase/shifter", proc._device_id2)
        self.assertEqual("phase/shifter/ppt", proc._ppt2)
        self.assertEqual("faked delay (s)", ctrl_widget.label2_le.text())
        self.assertEqual("mono", proc._device_id1)
        self.assertEqual("mono/ppt", proc._ppt1)
        self.assertEqual("faked energy (eV)", ctrl_widget.label1_le.text())


class TestTrXasProcessor(_TestDataMixin, _SpecialSuiteProcessorTestBase):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = TrXasProcessor(object(), object())

    def test1dBinning(self):
        proc = self._proc

        n = 10
        proc._a13.extend(np.random.randn(n))
        proc._a23.extend(np.random.randn(n))
        proc._a21.extend(np.random.randn(n))

        proc._slow1.extend(np.arange(10))
        proc._n_bins1 = 4
        proc._bin_range1 = [-np.inf, np.inf]
        proc._actual_range1 = [1, 3]  # this is the range used in binning

        proc._new_1d_binning()
        assert len(proc._a13_stats) == 4
        assert len(proc._a23_stats) == 4
        assert len(proc._a21_stats) == 4
        assert proc._counts1.tolist() == [1, 0, 1, 1]
        assert proc._edges1.tolist() == [1., 1.5, 2., 2.5, 3.]

        # new outsider data point
        proc._update_1d_binning(0.1, 0.2, 0.3, 3.5)  # index 3
        assert proc._counts1.tolist() == [1, 0, 1, 1]

        # new valid data point
        proc._update_1d_binning(0.1, 0.2, 0.3, 2)  # index 2
        assert proc._counts1.tolist() == [1, 0, 2, 1]

    def test2dBinning(self):
        proc = self._proc

        n = 10
        proc._a21.extend(np.random.randn(n))
        proc._a13.extend(np.random.randn(n))
        proc._a23.extend(np.random.randn(n))

        proc._slow1.extend(np.arange(10))
        proc._slow2.extend(np.arange(10) + 1)
        proc._n_bins1 = 4
        proc._bin_range1 = [-np.inf, np.inf]
        proc._actual_range1 = [1, 8]  # this is the range used in binning
        proc._n_bins2 = 2
        proc._bin_range2 = [-np.inf, np.inf]
        proc._actual_range2 = [2, 6]  # this is the range used in binning

        proc._new_2d_binning()

        assert (4, 2) == proc._a21_heat.shape
        assert (4, 2) == proc._a21_heat_count.shape
        assert [[2, 0], [0, 2], [0, 1], [0, 0]], proc._a21_heat_count.tolist()
        assert [2, 4, 6] == proc._edges2.tolist()
        assert proc._edges1 is None  # calculated in _new_1d_binning

        proc._new_1d_binning()
        assert [1.0, 2.75, 4.5, 6.25, 8.0] == proc._edges1.tolist()

        # new outsider data point
        proc._update_2d_binning(0.1, 6.5, 7)  # index (2, 3)
        assert [[2, 0], [0, 2], [0, 1], [0, 0]] == proc._a21_heat_count.tolist()

        # new valid data point
        proc._update_2d_binning(0.1, 2, 1)  # index (0, 0)
        assert [[3, 0], [0, 2], [0, 1], [0, 0]] == proc._a21_heat_count.tolist()

    def testProcess(self):
        proc = self._proc
        data, processed = self.simple_data(1001, (4, 2, 2))

        # test auto range
        proc._n_bins1 = 5
        proc._bin_range1 = [-np.inf, np.inf]
        proc._auto_range1 = [True, True]
        proc._n_bins2 = 2
        proc._bin_range2 = [-np.inf, np.inf]
        proc._auto_range2 = [True, True]
        n = 10
        proc._slow1.extend(np.arange(n))
        proc._slow2.extend(np.arange(n))
        proc._a13.extend(np.random.randn(n))
        proc._a23.extend(np.random.randn(n))
        proc._a21.extend(np.random.randn(n))
        ret = proc.process(data)

        np.testing.assert_array_almost_equal([0.9, 2.7, 4.5, 6.3, 8.1], ret["centers1"])
        np.testing.assert_array_almost_equal([2.25, 6.75], ret["centers2"])

        self._check_processed_data_structure(ret)

        self._check_reset(proc)

    def _check_processed_data_structure(self, ret):
        """Override."""
        data_gt = TestTrXasWindow.data4visualization().keys()
        assert set(ret.keys()) == set(data_gt)

    def _check_reset(self, proc):
        assert len(proc._slow1) != 0

        proc.reset()
        assert len(proc._slow1) == 0
        assert len(proc._slow2) == 0
        assert len(proc._a13) == 0
        assert len(proc._a23) == 0
        assert len(proc._a21) == 0
