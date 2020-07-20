from collections import Counter
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import os
import random
import tempfile

import numpy as np

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest, QSignalSpy

from extra_foam.config import AnalysisType, BinMode, config, PumpProbeMode
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
        cls.pulse_worker = cls.foam.pulse_worker

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

    def tearDown(self):
        self.gui._analysis_setup_manager._resetToDefault()

    def testOpenCloseWindows(self):
        pp_window = self._check_open_window(self.pp_action)
        self.assertIsInstance(pp_window, PumpProbeWindow)
        self._checkPumpProbeWindow(pp_window)
        self._checkPumpProbeCtrlWidget(pp_window)

        correlation_window = self._check_open_window(self.correlation_action)
        self.assertIsInstance(correlation_window, CorrelationWindow)
        self._checkCorrelationWindow(correlation_window)
        self._checkCorrelationCtrlWidget(correlation_window)
        self._checkCorrelationCurveFitting(correlation_window)

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
        self._check_close_window(pp_window_new)

    def testOpenCloseWindowTs(self):
        with patch("extra_foam.gui.MainGUI._pulse_resolved",
                   new_callable=PropertyMock, create=True, return_value=False):
            histogram_window = self._check_open_window(self.histogram_action)
            self._checkHistogramCtrlWidgetTs(histogram_window)
            self._check_close_window(histogram_window)

            pp_window = self._check_open_window(self.pp_action)
            self._checkPumpProbeCtrlWidgetTs(pp_window)
            self._check_close_window(pp_window)

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
        for key in [mt.BINNING_PROC, mt.CORRELATION_PROC, mt.HISTOGRAM_PROC]:
            cfg = meta.hget_all(key)
            self.assertEqual(registered, bool(int(cfg["analysis_type"])))

    def _checkPumpProbeWindow(self, win):
        from extra_foam.gui.windows.pump_probe_w import (
            PumpProbeVFomPlot, PumpProbeFomPlot
        )

        self.assertEqual(3, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

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
        from extra_foam.gui.windows.pulse_of_interest_w import PoiImageView, PoiFomHist, PoiRoiHist

        self.assertEqual(6, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        self.assertEqual(2, counter[PoiImageView])
        self.assertEqual(2, counter[PoiFomHist])
        self.assertEqual(2, counter[PoiRoiHist])

        win.updateWidgetsF()

    def _checkPumpProbeCtrlWidget(self, win):
        widget = win._ctrl_widget
        pp_proc = self.pulse_worker._pp_proc

        all_modes = {value: key for key, value in widget._available_modes.items()}

        # check default reconfigurable params
        pp_proc.update()
        self.assertTrue(pp_proc._abs_difference)
        self.assertEqual(AnalysisType.UNDEFINED, pp_proc.analysis_type)
        self.assertEqual(PumpProbeMode.UNDEFINED, pp_proc._mode)
        self.assertEqual(slice(None, None), pp_proc._indices_on)
        self.assertEqual(slice(None, None), pp_proc._indices_off)

        # change analysis type
        pp_proc._reset = False
        widget._analysis_type_cb.setCurrentText('ROI proj')
        pp_proc.update()
        self.assertEqual(AnalysisType.ROI_PROJ, pp_proc.analysis_type)
        self.assertTrue(pp_proc._reset)

        # change pump-probe mode
        pp_proc._reset = False
        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.EVEN_TRAIN_ON])
        pp_proc.update()
        self.assertTrue(pp_proc._reset)

        # off_pulse_le will be disabled when the mode is REFERENCE_AS_OFF
        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.REFERENCE_AS_OFF])
        self.assertTrue(widget._on_pulse_le.isEnabled())
        self.assertFalse(widget._off_pulse_le.isEnabled())
        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.EVEN_TRAIN_ON])
        self.assertTrue(widget._on_pulse_le.isEnabled())
        self.assertTrue(widget._off_pulse_le.isEnabled())

        # change abs_difference
        pp_proc._reset = False
        QTest.mouseClick(widget._abs_difference_cb, Qt.LeftButton,
                         pos=QPoint(2, widget._abs_difference_cb.height()/2))
        pp_proc.update()
        self.assertFalse(pp_proc._abs_difference)
        self.assertTrue(pp_proc._reset)

        # change on/off pulse indices
        widget._on_pulse_le.setText('0:10:2')
        widget._off_pulse_le.setText('1:10:2')
        pp_proc.update()
        self.assertEqual(PumpProbeMode.EVEN_TRAIN_ON, pp_proc._mode)
        self.assertEqual(slice(0, 10, 2), pp_proc._indices_on)
        self.assertEqual(slice(1, 10, 2), pp_proc._indices_off)

        # test reset button
        pp_proc._reset = False
        widget._reset_btn.clicked.emit()
        pp_proc.update()
        self.assertTrue(pp_proc._reset)
        pp_proc._reset = False
        self.gui.analysis_ctrl_widget._reset_pp_btn.clicked.emit()
        pp_proc.update()
        self.assertTrue(pp_proc._reset)
        pp_proc._reset = False

        # test loading meta data
        mediator = widget._mediator
        mediator.onPpAnalysisTypeChange(AnalysisType.AZIMUTHAL_INTEG)
        mediator.onPpModeChange(PumpProbeMode.ODD_TRAIN_ON)
        mediator.onPpOnPulseSlicerChange([0, None, 2])
        mediator.onPpOffPulseSlicerChange([1, None, 2])
        mediator.onPpAbsDifferenceChange(True)
        widget.loadMetaData()
        self.assertEqual("azimuthal integ", widget._analysis_type_cb.currentText())
        self.assertEqual("odd/even train", widget._mode_cb.currentText())
        self.assertEqual(True, widget._abs_difference_cb.isChecked())
        self.assertEqual("0::2", widget._on_pulse_le.text())
        self.assertEqual("1::2", widget._off_pulse_le.text())

    def _checkPumpProbeCtrlWidgetTs(self, win):
        widget = win._ctrl_widget
        pp_proc = self.pulse_worker._pp_proc

        self.assertFalse(widget._on_pulse_le.isEnabled())
        self.assertFalse(widget._off_pulse_le.isEnabled())

        all_modes = widget._available_modes_inv

        # we only test train-resolved detector specific configuration

        pp_proc.update()
        self.assertEqual(PumpProbeMode.UNDEFINED, pp_proc._mode)
        self.assertEqual(slice(None, None), pp_proc._indices_on)
        self.assertEqual(slice(None, None), pp_proc._indices_off)

        spy = QSignalSpy(widget._mode_cb.currentTextChanged)

        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.EVEN_TRAIN_ON])
        self.assertEqual(1, len(spy))

        pp_proc.update()
        self.assertEqual(PumpProbeMode(PumpProbeMode.EVEN_TRAIN_ON), pp_proc._mode)
        self.assertEqual(slice(None, None), pp_proc._indices_on)
        self.assertEqual(slice(None, None), pp_proc._indices_off)

        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.REFERENCE_AS_OFF])
        self.assertEqual(2, len(spy))
        # test on_pulse_le is still disabled, which will become enabled if the
        # detector is pulse-resolved
        self.assertFalse(widget._on_pulse_le.isEnabled())

        # PumpProbeMode.SAME_TRAIN is not available
        self.assertNotIn(PumpProbeMode.SAME_TRAIN, all_modes)

        # test loading meta data
        # test if the meta data is invalid
        mediator = widget._mediator
        mediator.onPpOnPulseSlicerChange([0, None, 2])
        mediator.onPpOffPulseSlicerChange([0, None, 2])
        widget.loadMetaData()
        self.assertEqual(":", widget._on_pulse_le.text())
        self.assertEqual(":", widget._off_pulse_le.text())

        mediator.onPpModeChange(PumpProbeMode.SAME_TRAIN)
        with self.assertRaises(KeyError):
            widget.loadMetaData()

    def _checkCorrelationCtrlWidget(self, win):
        from extra_foam.gui.ctrl_widgets.correlation_ctrl_widget import (
            _N_PARAMS, _DEFAULT_RESOLUTION)

        USER_DEFINED_KEY = config["SOURCE_USER_DEFINED_CATEGORY"]

        widget = win._ctrl_widget
        analysis_ctrl_widget = self.gui.analysis_ctrl_widget
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
        for i, proc in enumerate(processors):
            proc.update()
            self.assertEqual(AnalysisType(0), proc.analysis_type)
            self.assertEqual("", proc._source)
            self.assertEqual(_DEFAULT_RESOLUTION, proc._resolution)
            self.assertFalse(proc._auto_reset_ma)

        for i, proc in enumerate(processors):
            analysis_ctrl_widget._ma_window_le.setText("2")
            proc.update()
            if i == 0:
                self.assertTrue(proc._auto_reset_ma)
            else:
                self.assertFalse(proc._auto_reset_ma)

        # set new values
        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.ROI_PROJ])
        widget._auto_reset_ma_cb.setChecked(False)
        for i, proc in enumerate(processors):
            proc._reset = False
            proc.update()
            self.assertEqual(AnalysisType.ROI_PROJ, proc.analysis_type)
            self.assertTrue(proc._reset)
            self.assertFalse(proc._auto_reset_ma)

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

            # set user defined
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

            # set resolution
            widget._table.cellWidget(3, idx).setText(str(1.0))
            proc.update()
            self.assertEqual(1.0, proc._resolution)

            # test reset button
            proc._reset = False
            widget._reset_btn.clicked.emit()
            proc.update()
            self.assertTrue(proc._reset)
            proc._reset = False
            self.gui.analysis_ctrl_widget._reset_correlation_btn.clicked.emit()
            proc.update()
            self.assertTrue(proc._reset)

        # test loading meta data
        mediator = widget._mediator
        mediator.onCorrelationAnalysisTypeChange(AnalysisType.UNDEFINED)
        mediator.onCorrelationAutoResetMaChange('True')
        if config["TOPIC"] == "FXE":
            motor_id = 'FXE_SMS_USR/MOTOR/UM01'
        else:
            motor_id = 'SCS_ILH_LAS/MOTOR/LT3'
        mediator.onCorrelationParamChange((1, f'{motor_id} actualPosition', 0.0))
        mediator.onCorrelationParamChange((2, 'ABC abc', 2.0))
        widget.loadMetaData()
        self.assertEqual("", widget._analysis_type_cb.currentText())
        self.assertTrue(widget._auto_reset_ma_cb.isChecked())
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

    def _checkCorrelationCurveFitting(self, win):
        widget = win._ctrl_widget
        fitting = widget._fitting

        # test correlation1 and correlation2 checkbox
        fitting.corr2_cb.setChecked(True)
        self.assertFalse(fitting.corr1_cb.isChecked())
        fitting.corr1_cb.setChecked(True)
        self.assertFalse(fitting.corr2_cb.isChecked())

        # test fit
        x1, y1 = np.random.rand(10), np.random.rand(10)
        win._corr1._plot.setData(x1, y1)
        x1_slave, y1_slave = np.random.rand(10), np.random.rand(10)
        win._corr1._plot_slave.setData(x1_slave, y1_slave)
        x2, y2 = np.random.rand(10), np.random.rand(10)
        win._corr2._plot.setData(x2, y2)
        x2_slave, y2_slave = np.random.rand(10), np.random.rand(10)
        win._corr2._plot_slave.setData(x2_slave, y2_slave)

        self.assertTrue(fitting.corr1_cb.isChecked())
        with patch.object(fitting, "fit") as mocked_fit:
            mocked_fit.return_value = ([], [])
            QTest.mouseClick(fitting.fit_btn, Qt.LeftButton)
            self.assertEqual(2, len(mocked_fit.call_args_list))
            self.assertTupleEqual(mocked_fit.call_args_list[0][0], (x1, y1, True))
            self.assertTupleEqual(mocked_fit.call_args_list[1][0], (x1_slave, y1_slave, False))

        fitting.corr2_cb.setChecked(True)
        with patch.object(fitting, "fit") as mocked_fit:
            mocked_fit.return_value = ([], [])
            QTest.mouseClick(fitting.fit_btn, Qt.LeftButton)
            self.assertEqual(2, len(mocked_fit.call_args_list))
            self.assertTupleEqual(mocked_fit.call_args_list[0][0], (x2, y2, True))
            self.assertTupleEqual(mocked_fit.call_args_list[1][0], (x2_slave, y2_slave, False))

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
        proc._reset = False
        self.gui.analysis_ctrl_widget._reset_binning_btn.clicked.emit()
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
        proc._reset = False
        self.gui.analysis_ctrl_widget._reset_histogram_btn.clicked.emit()
        proc.update()
        self.assertTrue(proc._reset)

        # test loading meta data
        mediator = widget._mediator
        mediator.onHistAnalysisTypeChange(AnalysisType.PUMP_PROBE)
        mediator.onHistBinRangeChange((-10, 10))
        mediator.onHistNumBinsChange(55)
        mediator.onHistPulseResolvedChange(True)
        widget.loadMetaData()
        self.assertEqual("pump-probe", widget._analysis_type_cb.currentText())
        self.assertEqual("-10, 10", widget._bin_range_le.text())
        self.assertEqual("55", widget._n_bins_le.text())
        self.assertEqual(True, widget._pulse_resolved_cb.isChecked())

    def _checkHistogramCtrlWidgetTs(self, win):
        widget = win._ctrl_widget

        # test default
        self.assertFalse(widget._pulse_resolved_cb.isChecked())
        self.assertFalse(widget._pulse_resolved_cb.isEnabled())

        # test loading meta data
        # test if the meta data is invalid
        mediator = widget._mediator
        mediator.onHistPulseResolvedChange(True)
        widget.loadMetaData()
        self.assertEqual(False, widget._pulse_resolved_cb.isChecked())

