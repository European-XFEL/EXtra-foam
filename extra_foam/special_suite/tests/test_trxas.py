import unittest
from unittest.mock import MagicMock, patch
from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy, QTest

from extra_foam.logger import logger_suite as logger
from extra_foam.gui import mkQApp
from extra_foam.special_suite.trxas_proc import TrxasProcessor
from extra_foam.special_suite.trxas_w import (
    TrxasWindow, TrxasAbsorptionPlot, TrxasRoiImageView, TrxasHeatmap
)
from extra_foam.pipeline.tests import _TestDataMixin


app = mkQApp()

logger.setLevel('CRITICAL')


class TestTrxasWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("extra_foam.special_suite.special_analysis_base._SpecialAnalysisBase.startWorker"):
            cls._win = TrxasWindow('SCS')

    @classmethod
    def tearDown(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._win.close()

    def testWindow(self):
        win = self._win

        self.assertEqual(6, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(3, counter[TrxasRoiImageView])
        self.assertEqual(2, counter[TrxasAbsorptionPlot])
        self.assertEqual(1, counter[TrxasHeatmap])

        win.updateWidgetsF()

    def testCtrl(self):
        from extra_foam.special_suite.trxas_w import (
            _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE, _DEFAULT_DEVICE_ID, _DEFAULT_PROPERTY
        )

        default_bin_range = tuple(float(v) for v in _DEFAULT_BIN_RANGE.split(','))

        win = self._win
        ctrl_widget = win._ctrl_widget
        proc = win._worker

        # test default values
        self.assertTupleEqual(default_bin_range, proc._delay_range)
        self.assertTupleEqual(default_bin_range, proc._energy_range)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_delay_bins)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_energy_bins)
        self.assertEqual(_DEFAULT_DEVICE_ID, proc._energy_device)
        self.assertEqual(_DEFAULT_PROPERTY, proc._energy_ppt)
        self.assertEqual(_DEFAULT_DEVICE_ID, proc._delay_device)
        self.assertEqual(_DEFAULT_PROPERTY, proc._delay_ppt)

        # test set new values

        widget = ctrl_widget.energy_device_le
        widget.clear()
        QTest.keyClicks(widget, "mono")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("mono", proc._energy_device)

        widget = ctrl_widget.energy_ppt_le
        widget.clear()
        QTest.keyClicks(widget, "mono/ppt")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("mono/ppt", proc._energy_ppt)

        widget = ctrl_widget.delay_device_le
        widget.clear()
        QTest.keyClicks(widget, "phase/shifter")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("phase/shifter", proc._delay_device)

        widget = ctrl_widget.delay_ppt_le
        widget.clear()
        QTest.keyClicks(widget, "phase/shifter/ppt")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual("phase/shifter/ppt", proc._delay_ppt)

        widget = ctrl_widget.energy_range_le
        widget.clear()
        QTest.keyClicks(widget, "-1.0, 1.0")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertTupleEqual((-1.0, 1.0), proc._energy_range)

        widget = ctrl_widget.n_energy_bins_le
        widget.clear()
        QTest.keyClicks(widget, "1000")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(100, proc._n_energy_bins)  # maximum is 999 and one can not put the 3rd 0 in
        widget.clear()
        QTest.keyClicks(widget, "999")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(999, proc._n_energy_bins)

        widget = ctrl_widget.delay_range_le
        widget.clear()
        QTest.keyClicks(widget, "-1, 1")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertTupleEqual((-1, 1), proc._delay_range)

        widget = ctrl_widget.n_delay_bins_le
        widget.clear()
        QTest.keyClicks(widget, "1000")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(100, proc._n_delay_bins)  # maximum is 999 and one can not put the 3rd 0 in
        widget.clear()
        QTest.keyClicks(widget, "999")
        QTest.keyPress(widget, Qt.Key_Enter)
        self.assertEqual(999, proc._n_delay_bins)

        # test reset button
        proc._reset = False
        win.reset_sgn.emit()
        self.assertTrue(proc._reset)

    def testAbsorptionPlot(self):
        from extra_foam.special_suite.trxas_w import TrxasAbsorptionPlot

        widget = TrxasAbsorptionPlot()

    def testHeatmap(self):
        from extra_foam.special_suite.trxas_w import TrxasHeatmap

        widget = TrxasHeatmap()


class TestTrXasProcessor(_TestDataMixin):
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = TrxasProcessor(object(), object())
        yield
        self._proc._clear_history()

    def test1dBinning(self):
        proc = self._proc

        n = 10
        proc._a13.extend(np.random.randn(n))
        proc._a23.extend(np.random.randn(n))
        proc._a21.extend(np.random.randn(n))

        proc._delays.extend(np.arange(10))
        proc._n_delay_bins = 4
        proc._delay_range = [1, 3]

        proc._new_1d_binning()
        assert not proc._bin1d
        assert proc._bin2d
        assert len(proc._a13_stats) == 4
        assert len(proc._a23_stats) == 4
        assert len(proc._a21_stats) == 4
        assert proc._delay_bin_counts.tolist() == [1, 0, 1, 1]
        assert proc._delay_bin_edges.tolist() == [1., 1.5, 2., 2.5, 3.]

        # new outsider data point
        proc._update_1d_binning(0.1, 0.2, 0.3, 3.5)  # index 3
        assert proc._delay_bin_counts.tolist() == [1, 0, 1, 1]

        # new valid data point
        proc._update_1d_binning(0.1, 0.2, 0.3, 2)  # index 2
        assert proc._delay_bin_counts.tolist() == [1, 0, 2, 1]

        # TODO: test moving average calculation

    def test2dBinning(self):
        proc = self._proc

        n = 10
        proc._a21.extend(np.random.randn(n))
        proc._a13.extend(np.random.randn(n))
        proc._a23.extend(np.random.randn(n))

        proc._delays.extend(np.arange(10))
        proc._energies.extend(np.arange(10) + 1)
        proc._n_delay_bins = 4
        proc._delay_range = [1, 8]
        proc._n_energy_bins = 2
        proc._energy_range = [2, 6]

        proc._new_2d_binning()
        assert proc._bin1d
        assert not proc._bin2d

        assert (4, 2) == proc._a21_heat.shape
        assert (4, 2) == proc._a21_heatcount.shape
        assert [[2, 0], [0, 2], [0, 1], [0, 0]], proc._a21_heatcount.tolist()
        assert [2, 4, 6] == proc._energy_bin_edges.tolist()
        assert proc._delay_bin_edges is None  # calculated in _new_1d_binning
        proc._new_1d_binning()
        assert [1.0, 2.75, 4.5, 6.25, 8.0] == proc._delay_bin_edges.tolist()

        # new outsider data point
        proc._update_2d_binning(0.1, 6.5, 7)  # index (2, 3)
        assert [[2, 0], [0, 2], [0, 1], [0, 0]] == proc._a21_heatcount.tolist()

        # new valid data point
        proc._update_2d_binning(0.1, 2, 1)  # index (0, 0)
        assert [[3, 0], [0, 2], [0, 1], [0, 0]] == proc._a21_heatcount.tolist()

        # TODO: test moving average calculation

    def testProcess(self):
        proc = self._proc
        data, processed = self.simple_data(1001, (4, 2, 2))

        proc._n_delay_bins = 4
        proc._delay_range = [-1, 1]
        proc._n_energy_bins = 2
        proc._energy_range = [-1, 1]
        n = 10
        proc._delays.extend(np.random.randn(n))
        proc._energies.extend(np.random.randn(n))
        proc._a13.extend(np.random.randn(n))
        proc._a23.extend(np.random.randn(n))
        proc._a21.extend(np.random.randn(n))
        processed = proc.process(data)

        np.testing.assert_array_almost_equal([-.75, -0.25, 0.25, .75],
                                             processed["delay_bin_centers"])
        np.testing.assert_array_almost_equal([-.5, .5],
                                             processed["energy_bin_centers"])
