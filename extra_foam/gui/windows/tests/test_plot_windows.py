from collections import Counter
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import os
import random
import tempfile

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest, QSignalSpy

from extra_foam.config import AnalysisType, BinMode, config
from extra_foam.database import Metadata as mt
from extra_foam.logger import logger
from extra_foam.gui import mkQApp
from extra_foam.gui.windows import (
    BinningWindow, CorrelationWindow, HistogramWindow,
    PulseOfInterestWindow, PumpProbeWindow,
    FileStreamWindow, AboutWindow
)
from extra_foam.processes import wait_until_redis_shutdown
from extra_foam.services import Foam

app = mkQApp()

logger.setLevel('CRITICAL')

_tmp_cfg_dir = tempfile.mkdtemp()


def setup_module(module):
    from extra_foam import config
    module._backup_ROOT_PATH = config.ROOT_PATH
    config.ROOT_PATH = _tmp_cfg_dir


def teardown_module(module):
    os.rmdir(_tmp_cfg_dir)
    from extra_foam import config
    config.ROOT_PATH = module._backup_ROOT_PATH


class TestPlotWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load(*random.choice([('LPD', 'FXE'), ('DSSC', 'SCS')]))

        cls.foam = Foam().init()

        cls.gui = cls.foam._gui
        cls.train_worker = cls.foam.train_worker

        actions = cls.gui._tool_bar.actions()

        cls.poi_action = actions[4]
        assert "Pulse-of-interest" == cls.poi_action.text()
        cls.pp_action = actions[5]
        assert "Pump-probe" == cls.pp_action.text()
        cls.correlation_action = actions[6]
        assert "Correlation" == cls.correlation_action.text()
        cls.histogram_action = actions[7]
        assert "Histogram" == cls.histogram_action.text()
        cls.binning_action = actions[8]
        assert "Binning" == cls.binning_action.text()

    @classmethod
    def tearDownClass(cls):
        cls.foam.terminate()

        wait_until_redis_shutdown()

        os.remove(config.config_file)

    def setUp(self):
        self.assertTrue(self.gui.updateMetaData())

    def testOpenCloseWindows(self):
        pp_window = self._check_open_window(self.pp_action)
        self.assertIsInstance(pp_window, PumpProbeWindow)
        self._checkPumpProbeWindow(pp_window)

        correlation_window = self._check_open_window(self.correlation_action)
        self.assertIsInstance(correlation_window, CorrelationWindow)
        self._checkCorrelationWindow(correlation_window)
        self._checkCorrelationCtrlWidget(correlation_window)

        binning_window = self._check_open_window(self.binning_action)
        self.assertIsInstance(binning_window, BinningWindow)
        self._checkBinningWindow(binning_window)
        self._checkBinCtrlWidget(binning_window)

        histogram_window = self._check_open_window(self.histogram_action)
        self.assertIsInstance(histogram_window, HistogramWindow)
        self._checkHistogramWindow(histogram_window)
        self._checkHistogramCtrlWidget(histogram_window)

        poi_window = self._check_open_window(self.poi_action)
        self.assertIsInstance(poi_window, PulseOfInterestWindow)
        self._checkPulseOfInterestWindow(poi_window)
        # open one window twice
        self._check_open_window(self.poi_action, registered=False)

        correlation_window._ctrl_widget._analysis_type_cb.setCurrentIndex(1)
        binning_window._ctrl_widget._analysis_type_cb.setCurrentIndex(1)
        histogram_window._ctrl_widget._analysis_type_cb.setCurrentIndex(1)
        self._checkAnalysisTypeRegistration(True)
        self._check_close_window(pp_window)
        self._check_close_window(correlation_window)
        self._check_close_window(binning_window)
        self._check_close_window(histogram_window)
        self._check_close_window(poi_window)
        # test analysis type is unregistered
        self._checkAnalysisTypeRegistration(False)
        self.assertEqual("", correlation_window._ctrl_widget._analysis_type_cb.currentText())
        self.assertEqual("", binning_window._ctrl_widget._analysis_type_cb.currentText())
        self.assertEqual("", histogram_window._ctrl_widget._analysis_type_cb.currentText())

        # if a plot window is closed, it can be re-openned and a new instance
        # will be created
        pp_window_new = self._check_open_window(self.pp_action)
        self.assertIsInstance(pp_window_new, PumpProbeWindow)
        self.assertIsNot(pp_window_new, pp_window)

    def testOpenCloseWindowTs(self):
        with patch("extra_foam.gui.MainGUI._pulse_resolved",
                   new_callable=PropertyMock, create=True, return_value=False):
            histogram_window = self._check_open_window(self.histogram_action)
            self._checkHistogramCtrlWidgetTs(histogram_window._ctrl_widget)
            self._check_close_window(histogram_window)

    def testOpenCloseSatelliteWindows(self):
        actions = self.gui._tool_bar.actions()
        about_action = actions[-1]
        streamer_action = actions[-2]

        about_window = self._check_open_satellite_window(about_action)
        self.assertIsInstance(about_window, AboutWindow)

        streamer_window = self._check_open_satellite_window(streamer_action)
        self.assertIsInstance(streamer_window, FileStreamWindow)

        # open one window twice
        self._check_open_satellite_window(about_action, registered=False)

        self._check_close_satellite_window(about_window)
        self._check_close_satellite_window(streamer_window)

        # if a window is closed, it can be re-opened and a new instance
        # will be created
        about_window_new = self._check_open_satellite_window(about_action)
        self.assertIsInstance(about_window_new, AboutWindow)
        self.assertIsNot(about_window_new, about_window)

    def _check_open_window(self, action, registered=True):
        """Check triggering action about opening a window.

        :param bool registered: True for the new window is expected to be
            registered; False for the old window will be activate and thus
            no new window will be registered.
        """
        n_registered = len(self.gui._plot_windows)
        action.trigger()
        if registered:
            window = list(self.gui._plot_windows.keys())[-1]
            self.assertEqual(n_registered+1, len(self.gui._plot_windows))
            return window

        self.assertEqual(n_registered, len(self.gui._plot_windows))

    def _check_close_window(self, window):
        n_registered = len(self.gui._plot_windows)
        window.close()
        self.assertEqual(n_registered-1, len(self.gui._plot_windows))

    def _check_open_satellite_window(self, action, registered=True):
        """Check triggering action about opening a satellite window.

        :param bool registered: True for the new window is expected to be
            registered; False for the old window will be activate and thus
            no new window will be registered.
        """
        n_registered = len(self.gui._satellite_windows)
        action.trigger()
        if registered:
            window = list(self.gui._satellite_windows.keys())[-1]
            self.assertEqual(n_registered+1, len(self.gui._satellite_windows))
            return window

        self.assertEqual(n_registered, len(self.gui._satellite_windows))

    def _check_close_satellite_window(self, window):
        n_registered = len(self.gui._satellite_windows)
        window.close()
        self.assertEqual(n_registered-1, len(self.gui._satellite_windows))

    def _checkAnalysisTypeRegistration(self, registered):
        meta = self.train_worker._binning_proc._meta
        for key in [mt.BIN_PROC, mt.CORRELATION_PROC, mt.HISTOGRAM_PROC]:
            cfg = meta.hget_all(key)
            self.assertEqual(registered, bool(int(cfg["analysis_type"])))

    def _checkPumpProbeWindow(self, win):
        from extra_foam.gui.windows.pump_probe_w import (
            PumpProbeImageView, PumpProbeVFomPlot, PumpProbeFomPlot
        )

        self.assertEqual(5, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[PumpProbeImageView])
        self.assertEqual(2, counter[PumpProbeVFomPlot])
        self.assertEqual(1, counter[PumpProbeFomPlot])

        win.updateWidgetsF()

    def _checkCorrelationWindow(self, win):
        from extra_foam.gui.windows.correlation_w import CorrelationPlot

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[CorrelationPlot])

        win.updateWidgetsF()

    def _checkBinningWindow(self, win):
        from extra_foam.gui.windows.binning_w import Bin1dHeatmap, Bin1dHist, Bin2dHeatmap

        self.assertEqual(4, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[Bin1dHeatmap])
        self.assertEqual(1, counter[Bin1dHist])
        self.assertEqual(2, counter[Bin2dHeatmap])

        win.updateWidgetsF()

    def _checkHistogramWindow(self, win):
        from extra_foam.gui.windows.histogram_w import FomHist, InTrainFomPlot

        self.assertEqual(2, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(1, counter[InTrainFomPlot])
        self.assertEqual(1, counter[FomHist])

        win.updateWidgetsF()

    def _checkPulseOfInterestWindow(self, win):
        from extra_foam.gui.windows.pulse_of_interest_w import (
            PulseOfInterestWindow, PoiImageView, PoiFomHist, PoiRoiHist
        )

        self.assertEqual(6, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[PoiImageView])
        self.assertEqual(2, counter[PoiFomHist])
        self.assertEqual(2, counter[PoiRoiHist])

        win.updateWidgetsF()

    def _checkCorrelationCtrlWidget(self, win):
        from extra_foam.gui.ctrl_widgets.correlation_ctrl_widget import (
            _N_PARAMS, _DEFAULT_RESOLUTION)
        from extra_foam.pipeline.processors.base_processor import (
            SimplePairSequence, OneWayAccuPairSequence
        )
        USER_DEFINED_KEY = config["SOURCE_USER_DEFINED_CATEGORY"]

        widget = win._ctrl_widget
        analysis_types = {value: key for key, value in widget._analysis_types.items()}

        for i in range(_N_PARAMS):
            # test category list
            c_widget = widget._table.cellWidget(0, i)
            combo_lst = [c_widget.itemText(j) for j in range(c_widget.count())]
            self.assertListEqual(["", "Metadata"]
                                 + [k for k, v in config.control_sources.items() if v]
                                 + [USER_DEFINED_KEY],
                                 combo_lst)

        train_worker = self.train_worker
        processors = [train_worker._correlation1_proc, train_worker._correlation2_proc]

        # test default
        for proc in processors:
            proc.update()
            self.assertEqual(AnalysisType(0), proc.analysis_type)
            self.assertEqual("", proc._source)
            self.assertEqual(_DEFAULT_RESOLUTION, proc._resolution)

        # set new FOM
        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.ROI_PROJ])
        for proc in processors:
            proc._reset = False
            proc.update()
            self.assertEqual(AnalysisType.ROI_PROJ, proc.analysis_type)
            self.assertTrue(proc._reset)

        for idx, proc in enumerate(processors):
            # change source
            proc._reset = False
            ctg, device_id, ppt = 'Metadata', "META", "timestamp.tid"
            widget._table.cellWidget(0, idx).setCurrentText(ctg)
            self.assertEqual(device_id, widget._table.cellWidget(1, idx).currentText())
            self.assertEqual(ppt, widget._table.cellWidget(2, idx).currentText())
            proc.update()
            src = f"{device_id} {ppt}" if device_id and ppt else ""
            self.assertEqual(src, proc._source)
            self.assertTrue(proc._reset)

            # just test we can set a motor source
            proc._reset = False
            widget._table.cellWidget(0, idx).setCurrentText("Motor")
            proc.update()
            self.assertTrue(proc._reset)

            proc._reset = False
            ctg, device_id, ppt = USER_DEFINED_KEY, "ABC", "efg"
            widget._table.cellWidget(0, idx).setCurrentText(ctg)
            self.assertEqual('', widget._table.cellWidget(1, idx).text())
            self.assertEqual('', widget._table.cellWidget(2, idx).text())
            widget._table.cellWidget(1, idx).setText(device_id)
            widget._table.cellWidget(2, idx).setText(ppt)
            self.assertEqual(device_id, widget._table.cellWidget(1, 0).text())
            self.assertEqual(ppt, widget._table.cellWidget(2, 0).text())
            proc.update()
            src = f"{device_id} {ppt}" if device_id and ppt else ""
            self.assertEqual(src, proc._source)
            self.assertTrue(proc._reset)

            # change resolution
            proc._reset = False
            self.assertIsInstance(proc._corr, SimplePairSequence)
            self.assertIsInstance(proc._corr_slave, SimplePairSequence)
            widget._table.cellWidget(3, idx).setText(str(1.0))
            proc.update()
            self.assertEqual(1.0, proc._resolution)
            self.assertIsInstance(proc._corr, OneWayAccuPairSequence)
            self.assertIsInstance(proc._corr_slave, OneWayAccuPairSequence)
            # sequence type change will not have 'reset'
            self.assertFalse(proc._reset)
            widget._table.cellWidget(3, idx).setText(str(2.0))
            proc.update()
            self.assertEqual(2.0, proc._resolution)

            # test reset button
            proc._reset = False
            widget._reset_btn.clicked.emit()
            proc.update()
            self.assertTrue(proc._reset)

        # test loading meta data
        mediator = widget._mediator
        mediator.onCorrelationAnalysisTypeChange(AnalysisType.UNDEFINED)
        if config["TOPIC"] == "FXE":
            motor_id = 'FXE_SMS_USR/MOTOR/UM01'
        else:
            motor_id = 'SCS_ILH_LAS/MOTOR/LT3'
        mediator.onCorrelationParamChange((1, f'{motor_id} actualPosition', 0.0))
        mediator.onCorrelationParamChange((2, 'ABC abc', 2.0))
        widget.loadMetaData()
        self.assertEqual("", widget._analysis_type_cb.currentText())
        self.assertEqual('Motor', widget._table.cellWidget(0, 0).currentText())
        self.assertEqual(motor_id, widget._table.cellWidget(1, 0).currentText())
        self.assertEqual('actualPosition', widget._table.cellWidget(2, 0).currentText())
        self.assertEqual('0.0', widget._table.cellWidget(3, 0).text())
        self.assertEqual(widget._user_defined_key, widget._table.cellWidget(0, 1).currentText())
        self.assertEqual('ABC', widget._table.cellWidget(1, 1).text())
        self.assertEqual('abc', widget._table.cellWidget(2, 1).text())
        self.assertEqual('2.0', widget._table.cellWidget(3, 1).text())

        mediator.onCorrelationParamChange((1, f'', 0.0))
        mediator.onCorrelationParamChange((2, f'', 1.0))
        widget.loadMetaData()
        self.assertEqual('', widget._table.cellWidget(0, 0).currentText())
        self.assertEqual('', widget._table.cellWidget(1, 0).text())
        self.assertEqual('', widget._table.cellWidget(2, 0).text())
        self.assertEqual('', widget._table.cellWidget(0, 1).currentText())
        self.assertEqual('', widget._table.cellWidget(1, 1).text())
        self.assertEqual('', widget._table.cellWidget(2, 1).text())

    def _checkBinCtrlWidget(self, win):
        from extra_foam.gui.ctrl_widgets.bin_ctrl_widget import (
            _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE, _N_PARAMS
        )

        _DEFAULT_BIN_RANGE = tuple([float(v) for v in _DEFAULT_BIN_RANGE.split(",")])
        USER_DEFINED_KEY = config["SOURCE_USER_DEFINED_CATEGORY"]

        widget = win._ctrl_widget
        analysis_types_inv = widget._analysis_types_inv
        bin_modes_inv = widget._bin_modes_inv

        for i in range(_N_PARAMS):
            c_widget = widget._table.cellWidget(0, i)
            combo_lst = [c_widget.itemText(j) for j in range(c_widget.count())]
            self.assertListEqual(["", "Metadata"]
                                 + [k for k, v in config.control_sources.items() if v]
                                 + [USER_DEFINED_KEY],
                                 combo_lst)

        train_worker = self.train_worker
        proc = train_worker._binning_proc
        proc.update()

        # test default
        self.assertEqual(AnalysisType.UNDEFINED, proc.analysis_type)
        self.assertEqual(BinMode.AVERAGE, proc._mode)
        self.assertEqual("", proc._source1)
        self.assertEqual(_DEFAULT_BIN_RANGE, proc._bin_range1)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_bins1)
        self.assertEqual("", proc._source2)
        self.assertEqual(_DEFAULT_BIN_RANGE, proc._bin_range2)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_bins2)

        # test analysis type and mode change
        widget._analysis_type_cb.setCurrentText(analysis_types_inv[AnalysisType.PUMP_PROBE])
        widget._mode_cb.setCurrentText(bin_modes_inv[BinMode.ACCUMULATE])
        proc.update()
        self.assertEqual(AnalysisType.PUMP_PROBE, proc.analysis_type)
        self.assertEqual(BinMode.ACCUMULATE, proc._mode)

        # test source change
        for i in range(_N_PARAMS):
            ctg, device_id, ppt = 'Metadata', "META", "timestamp.tid"
            widget._table.cellWidget(0, i).setCurrentText(ctg)
            self.assertEqual(device_id, widget._table.cellWidget(1, i).currentText())
            self.assertEqual(ppt, widget._table.cellWidget(2, i).currentText())
            proc.update()
            src = f"{device_id} {ppt}" if device_id and ppt else ""
            self.assertEqual(src, getattr(proc, f"_source{i+1}"))

            # just test we can set a motor source
            widget._table.cellWidget(0, i).setCurrentText("Motor")
            proc.update()

            ctg, device_id, ppt = USER_DEFINED_KEY, "ABC", "efg"
            widget._table.cellWidget(0, i).setCurrentText(ctg)
            self.assertEqual('', widget._table.cellWidget(1, i).text())
            self.assertEqual('', widget._table.cellWidget(2, i).text())
            widget._table.cellWidget(1, i).setText(device_id)
            widget._table.cellWidget(2, i).setText(ppt)
            self.assertEqual(device_id, widget._table.cellWidget(1, 0).text())
            self.assertEqual(ppt, widget._table.cellWidget(2, 0).text())
            proc.update()
            src = f"{device_id} {ppt}" if device_id and ppt else ""
            self.assertEqual(src, getattr(proc, f"_source{i+1}"))

        # test bin range and number of bins change

        # bin parameter 1
        widget._table.cellWidget(3, 0).setText("0, 10")  # range
        widget._table.cellWidget(4, 0).setText("5")  # n_bins
        widget._table.cellWidget(3, 1).setText("-4, inf")  # range
        widget._table.cellWidget(4, 1).setText("2")  # n_bins
        proc.update()
        self.assertEqual(5, proc._n_bins1)
        self.assertTupleEqual((0, 10), proc._bin_range1)
        self.assertEqual(2, proc._n_bins2)
        self.assertTupleEqual((-4, np.inf), proc._bin_range2)

        # test "reset" button
        proc._reset = False
        widget._reset_btn.clicked.emit()
        proc.update()
        self.assertTrue(proc._reset)

        # test "Auto level" button
        win._bin1d_vfom._auto_level = False
        win._bin2d_value._auto_level = False
        win._bin2d_count._auto_level = False
        QTest.mouseClick(widget._auto_level_btn, Qt.LeftButton)
        self.assertTrue(win._bin1d_vfom._auto_level)
        self.assertTrue(win._bin2d_value._auto_level)
        self.assertTrue(win._bin2d_count._auto_level)

        # test loading meta data
        mediator = widget._mediator
        mediator.onBinAnalysisTypeChange(AnalysisType.UNDEFINED)
        mediator.onBinModeChange(BinMode.AVERAGE)
        if config["TOPIC"] == "FXE":
            motor_id = 'FXE_SMS_USR/MOTOR/UM01'
        else:
            motor_id = 'SCS_ILH_LAS/MOTOR/LT3'
        mediator.onBinParamChange((1, f'{motor_id} actualPosition', (-9, 9), 5))
        mediator.onBinParamChange((2, 'ABC abc', (-19, 19), 15))
        widget.loadMetaData()
        self.assertEqual("", widget._analysis_type_cb.currentText())
        self.assertEqual("average", widget._mode_cb.currentText())
        self.assertEqual('Motor', widget._table.cellWidget(0, 0).currentText())
        self.assertEqual(motor_id, widget._table.cellWidget(1, 0).currentText())
        self.assertEqual('actualPosition', widget._table.cellWidget(2, 0).currentText())
        self.assertEqual('-9, 9', widget._table.cellWidget(3, 0).text())
        self.assertEqual('5', widget._table.cellWidget(4, 0).text())
        self.assertEqual(widget._user_defined_key, widget._table.cellWidget(0, 1).currentText())
        self.assertEqual('ABC', widget._table.cellWidget(1, 1).text())
        self.assertEqual('abc', widget._table.cellWidget(2, 1).text())
        self.assertEqual('-19, 19', widget._table.cellWidget(3, 1).text())
        self.assertEqual('15', widget._table.cellWidget(4, 1).text())

        mediator.onBinParamChange((1, f'', (-9, 9), 5))
        mediator.onBinParamChange((2, f'', (-1, 1), 10))
        widget.loadMetaData()
        self.assertEqual('', widget._table.cellWidget(0, 0).currentText())
        self.assertEqual('', widget._table.cellWidget(1, 0).text())
        self.assertEqual('', widget._table.cellWidget(2, 0).text())
        self.assertEqual('', widget._table.cellWidget(0, 1).currentText())
        self.assertEqual('', widget._table.cellWidget(1, 1).text())
        self.assertEqual('', widget._table.cellWidget(2, 1).text())

    def _checkHistogramCtrlWidget(self, win):
        widget = win._ctrl_widget

        train_worker = self.train_worker
        proc = train_worker._histogram
        proc.update()

        analysis_types = {value: key for key, value in
                          widget._analysis_types.items()}

        self.assertEqual(AnalysisType.UNDEFINED, proc.analysis_type)
        self.assertTrue(proc._pulse_resolved)
        self.assertEqual(10, proc._n_bins)
        self.assertEqual((-np.inf, np.inf), proc._bin_range)

        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.ROI_FOM])
        proc.update()
        self.assertTrue(proc._reset)
        self.assertEqual(AnalysisType.ROI_FOM_PULSE, proc.analysis_type)

        proc._reset = False
        widget._pulse_resolved_cb.setChecked(False)  # switch to train-resolved
        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.ROI_FOM])
        proc.update()
        self.assertTrue(proc._reset)
        self.assertEqual(AnalysisType.ROI_FOM, proc.analysis_type)

        widget._n_bins_le.setText("100")
        widget._bin_range_le.setText("-1, 2")
        proc.update()
        self.assertEqual((-1, 2), proc._bin_range)
        self.assertEqual(100, proc._n_bins)

        proc._reset = False
        widget._reset_btn.clicked.emit()
        proc.update()
        self.assertTrue(proc._reset)

        # test loading meta data
        mediator = widget._mediator
        mediator.onHistAnalysisTypeChange(AnalysisType.UNDEFINED)
        mediator.onHistBinRangeChange((-10, 10))
        mediator.onHistNumBinsChange(55)
        mediator.onHistPulseResolvedChange(True)
        widget.loadMetaData()
        self.assertEqual("", widget._analysis_type_cb.currentText())
        self.assertEqual("-10, 10", widget._bin_range_le.text())
        self.assertEqual("55", widget._n_bins_le.text())
        self.assertEqual(True, widget._pulse_resolved_cb.isChecked())

    def _checkHistogramCtrlWidgetTs(self, widget):
        # test default
        self.assertFalse(widget._pulse_resolved_cb.isChecked())
        self.assertFalse(widget._pulse_resolved_cb.isEnabled())

        # test loading meta data
        # test if the meta data is invalid
        mediator = widget._mediator
        mediator.onHistPulseResolvedChange(True)
        widget.loadMetaData()
        self.assertEqual(False, widget._pulse_resolved_cb.isChecked())
