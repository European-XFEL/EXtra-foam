import unittest
from unittest.mock import patch, MagicMock, Mock
import random
import tempfile
import os

import numpy as np

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest, QSignalSpy

from extra_foam.database import Metadata as mt
from extra_foam.logger import logger
from extra_foam.services import Foam
from extra_foam.gui import mkQApp
from extra_foam.gui.windows import (
    BinningWindow, CorrelationWindow, HistogramWindow,
    PulseOfInterestWindow, PumpProbeWindow,
    FileStreamControllerWindow, AboutWindow,
)
from extra_foam.config import (
    config, AnalysisType, BinMode, PumpProbeMode,
)
from extra_foam.processes import wait_until_redis_shutdown

app = mkQApp()

logger.setLevel("CRITICAL")


_tmp_cfg_dir = tempfile.mkdtemp()


def setup_module(module):
    from extra_foam import config
    module._backup_ROOT_PATH = config.ROOT_PATH
    config.ROOT_PATH = _tmp_cfg_dir


def teardown_module(module):
    os.rmdir(_tmp_cfg_dir)
    from extra_foam import config
    config.ROOT_PATH = module._backup_ROOT_PATH


class TestMainGuiCtrl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load(*random.choice([('LPD', 'FXE'), ('DSSC', 'SCS')]))

        cls.foam = Foam().init()

        cls.gui = cls.foam._gui
        cls.train_worker = cls.foam.train_worker
        cls.pulse_worker = cls.foam.pulse_worker

    @classmethod
    def tearDownClass(cls):
        cls.foam.terminate()

        wait_until_redis_shutdown()

        os.remove(config.config_file)

    def setUp(self):
        self.assertTrue(self.gui.updateMetaData())

    def testAnalysisCtrlWidget(self):
        widget = self.gui.analysis_ctrl_widget
        train_worker = self.train_worker
        pulse_worker = self.pulse_worker

        xgm_proc = pulse_worker._xgm_proc
        digitizer_proc = pulse_worker._digitizer_proc
        roi_proc = train_worker._image_roi
        ai_proc = train_worker._ai_proc

        meta = xgm_proc._meta  # any meta is OK
        # test "Reset moving average" button
        widget._reset_ma_btn.clicked.emit()
        self.assertEqual('1', meta.hget(mt.GLOBAL_PROC, 'reset_ma_ai'))
        self.assertEqual('1', meta.hget(mt.GLOBAL_PROC, 'reset_ma_roi'))
        self.assertEqual('1', meta.hget(mt.GLOBAL_PROC, 'reset_ma_xgm'))
        self.assertEqual('1', meta.hget(mt.GLOBAL_PROC, 'reset_ma_digitizer'))
        roi_proc.update()
        self.assertIsNone(meta.hget(mt.GLOBAL_PROC, 'reset_ma_roi'))
        ai_proc.update()
        self.assertIsNone(meta.hget(mt.GLOBAL_PROC, 'reset_ma_ai'))
        xgm_proc.update()
        self.assertIsNone(meta.hget(mt.GLOBAL_PROC, 'reset_ma_xgm'))
        digitizer_proc.update()
        self.assertIsNone(meta.hget(mt.GLOBAL_PROC, 'reset_ma_digitizer'))

        # ----------------
        # Test POI indices
        # ----------------

        image_proc = self.pulse_worker._image_proc

        # default values
        image_proc.update()
        for idx in image_proc._poi_indices:
            self.assertEqual(0, idx)

        new_indices = [10, 20]
        for i in range(len(new_indices)):
            widget._poi_index_les[i].setText(str(new_indices[i]))
        image_proc.update()
        for i in range(len(new_indices)):
            self.assertEqual(new_indices[i], image_proc._poi_indices[i])

        # the PoiWindow will be informed when opened
        self.assertEqual(0, len(self.gui._plot_windows))
        poi_action = self.gui._tool_bar.actions()[4]
        self.assertEqual("Pulse-of-interest", poi_action.text())
        poi_action.trigger()
        win = list(self.gui._plot_windows.keys())[-1]
        self.assertIsInstance(win, PulseOfInterestWindow)
        for i, index in enumerate(new_indices):
            self.assertEqual(index, win._poi_imgs[i]._index)
            self.assertEqual(index, win._poi_fom_hists[i]._index)
            self.assertEqual(index, win._poi_roi_hists[i]._index)
        win.close()

    def testPumpProbeCtrlWidget(self):
        widget = self.gui.pump_probe_ctrl_widget
        pp_proc = self.pulse_worker._pp_proc

        all_modes = {value: key for key, value in
                     widget._available_modes.items()}

        # check default reconfigurable params
        pp_proc.update()
        self.assertTrue(pp_proc._abs_difference)
        self.assertEqual(AnalysisType(0), pp_proc.analysis_type)

        pp_proc.update()
        self.assertEqual(PumpProbeMode.UNDEFINED, pp_proc._mode)
        self.assertListEqual([-1], pp_proc._indices_on)
        self.assertIsInstance(pp_proc._indices_on[0], int)
        self.assertListEqual([-1], pp_proc._indices_off)
        self.assertIsInstance(pp_proc._indices_off[0], int)

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
        self.assertListEqual([0, 2, 4, 6, 8], pp_proc._indices_on)
        self.assertListEqual([1, 3, 5, 7, 9], pp_proc._indices_off)

        # test reset button
        pp_proc._reset = False
        widget._reset_btn.clicked.emit()
        pp_proc.update()
        self.assertTrue(pp_proc._reset)

    def testDataSourceWidget(self):
        from extra_foam.gui.ctrl_widgets.data_source_widget import DataSourceWidget

        widget = self.gui._source_cw
        self.assertIsInstance(widget, DataSourceWidget)

    def testFomFilterCtrlWidget(self):
        widget = self.gui.fom_filter_ctrl_widget
        filter_pulse = self.pulse_worker._filter
        filter_train = self.train_worker._filter

        analysis_types = {value: key for key, value in widget._analysis_types.items()}

        # test default

        self.assertTrue(widget._pulse_resolved_cb.isChecked())

        filter_pulse.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_pulse.analysis_type)
        self.assertTupleEqual((-np.inf, np.inf), filter_pulse._fom_range)
        filter_train.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_pulse.analysis_type)
        self.assertTupleEqual((-np.inf, np.inf), filter_pulse._fom_range)

        # test set new

        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.ROI_FOM])
        widget._fom_range_le.setText("-1, 1")
        filter_pulse.update()
        self.assertEqual(AnalysisType.ROI_FOM_PULSE, filter_pulse.analysis_type)
        self.assertEqual((-1, 1), filter_pulse._fom_range)
        filter_train.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_train.analysis_type)
        self.assertTupleEqual((-np.inf, np.inf), filter_train._fom_range)

        widget._fom_range_le.setText("-2, 2")
        widget._pulse_resolved_cb.setChecked(False)
        filter_pulse.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_pulse.analysis_type)
        self.assertEqual((-1, 1), filter_pulse._fom_range)
        filter_train.update()
        self.assertEqual(AnalysisType.ROI_FOM, filter_train.analysis_type)
        self.assertTupleEqual((-2, 2), filter_train._fom_range)

    def testCorrelationCtrlWidget(self):
        from extra_foam.gui.ctrl_widgets.correlation_ctrl_widget import (
            _N_PARAMS, _DEFAULT_RESOLUTION)
        from extra_foam.pipeline.processors.base_processor import (
            SimplePairSequence, OneWayAccuPairSequence
        )
        USER_DEFINED_KEY = config["SOURCE_USER_DEFINED_CATEGORY"]

        widget = self.gui.correlation_ctrl_widget
        analysis_types = {value: key for key, value in widget._analysis_types.items()}

        for i in range(_N_PARAMS):
            # test category list
            combo_lst = [widget._table.cellWidget(i, 0).itemText(j)
                         for j in range(widget._table.cellWidget(i, 0).count())]
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
            widget._table.cellWidget(idx, 0).setCurrentText(ctg)
            self.assertEqual(device_id, widget._table.cellWidget(idx, 1).currentText())
            self.assertEqual(ppt, widget._table.cellWidget(idx, 2).currentText())
            proc.update()
            src = f"{device_id} {ppt}" if device_id and ppt else ""
            self.assertEqual(src, proc._source)
            self.assertTrue(proc._reset)

            # just test we can set a motor source
            proc._reset = False
            widget._table.cellWidget(idx, 0).setCurrentText("Motor")
            proc.update()
            self.assertTrue(proc._reset)

            proc._reset = False
            ctg, device_id, ppt = USER_DEFINED_KEY, "ABC", "efg"
            widget._table.cellWidget(idx, 0).setCurrentText(ctg)
            self.assertEqual('', widget._table.cellWidget(idx, 1).text())
            self.assertEqual('', widget._table.cellWidget(idx, 2).text())
            widget._table.cellWidget(idx, 1).setText(device_id)
            widget._table.cellWidget(idx, 2).setText(ppt)
            self.assertEqual(device_id, widget._table.cellWidget(0, 1).text())
            self.assertEqual(ppt, widget._table.cellWidget(0, 2).text())
            proc.update()
            src = f"{device_id} {ppt}" if device_id and ppt else ""
            self.assertEqual(src, proc._source)
            self.assertTrue(proc._reset)

            # change resolution
            proc._reset = False
            self.assertIsInstance(proc._correlation, SimplePairSequence)
            self.assertIsInstance(proc._correlation_slave, SimplePairSequence)
            widget._table.cellWidget(idx, 3).setText(str(1.0))
            proc.update()
            self.assertEqual(1.0, proc._resolution)
            self.assertIsInstance(proc._correlation, OneWayAccuPairSequence)
            self.assertIsInstance(proc._correlation_slave, OneWayAccuPairSequence)
            # sequence type change will not have 'reset'
            self.assertFalse(proc._reset)
            widget._table.cellWidget(idx, 3).setText(str(2.0))
            proc.update()
            self.assertEqual(2.0, proc._resolution)
            self.assertTrue(proc._reset)

            # test reset button
            proc._reset = False
            widget._reset_btn.clicked.emit()
            proc.update()
            self.assertTrue(proc._reset)

    def testBinCtrlWidget(self):
        from extra_foam.gui.ctrl_widgets.bin_ctrl_widget import (
            _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE, _N_PARAMS
        )
        _DEFAULT_BIN_RANGE = tuple([float(v) for v in _DEFAULT_BIN_RANGE.split(",")])
        USER_DEFINED_KEY = config["SOURCE_USER_DEFINED_CATEGORY"]

        widget = self.gui.bin_ctrl_widget

        analysis_types = {value: key for key, value in widget._analysis_types.items()}
        bin_modes = {value: key for key, value in widget._bin_modes.items()}

        for i in range(_N_PARAMS):
            combo_lst = [widget._table.cellWidget(i, 0).itemText(j)
                         for j in range(widget._table.cellWidget(i, 0).count())]
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
        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.PUMP_PROBE])
        widget._mode_cb.setCurrentText(bin_modes[BinMode.ACCUMULATE])
        proc.update()
        self.assertEqual(AnalysisType.PUMP_PROBE, proc.analysis_type)
        self.assertEqual(BinMode.ACCUMULATE, proc._mode)

        # test source change
        for i in range(_N_PARAMS):
            ctg, device_id, ppt = 'Metadata', "META", "timestamp.tid"
            widget._table.cellWidget(i, 0).setCurrentText(ctg)
            self.assertEqual(device_id, widget._table.cellWidget(i, 1).currentText())
            self.assertEqual(ppt, widget._table.cellWidget(i, 2).currentText())
            proc.update()
            src = f"{device_id} {ppt}" if device_id and ppt else ""
            self.assertEqual(src, getattr(proc, f"_source{i+1}"))

            # just test we can set a motor source
            widget._table.cellWidget(i, 0).setCurrentText("Motor")
            proc.update()

            ctg, device_id, ppt = USER_DEFINED_KEY, "ABC", "efg"
            widget._table.cellWidget(i, 0).setCurrentText(ctg)
            self.assertEqual('', widget._table.cellWidget(i, 1).text())
            self.assertEqual('', widget._table.cellWidget(i, 2).text())
            widget._table.cellWidget(i, 1).setText(device_id)
            widget._table.cellWidget(i, 2).setText(ppt)
            self.assertEqual(device_id, widget._table.cellWidget(0, 1).text())
            self.assertEqual(ppt, widget._table.cellWidget(0, 2).text())
            proc.update()
            src = f"{device_id} {ppt}" if device_id and ppt else ""
            self.assertEqual(src, getattr(proc, f"_source{i+1}"))

        # test bin range and number of bins change

        # bin parameter 1
        widget._table.cellWidget(0, 3).setText("0, 10")  # range
        widget._table.cellWidget(0, 4).setText("5")  # n_bins
        widget._table.cellWidget(1, 3).setText("-4, inf")  # range
        widget._table.cellWidget(1, 4).setText("2")  # n_bins
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
        binning_action = self.gui._tool_bar.actions()[8]
        self.assertEqual("Binning", binning_action.text())
        binning_action.trigger()
        win = list(self.gui._plot_windows.keys())[-1]
        win._bin1d_vfom._auto_level = False
        win._bin2d_value._auto_level = False
        win._bin2d_count._auto_level = False
        QTest.mouseClick(widget._auto_level_btn, Qt.LeftButton)
        self.assertTrue(win._bin1d_vfom._auto_level)
        self.assertTrue(win._bin2d_value._auto_level)
        self.assertTrue(win._bin2d_count._auto_level)
        win.close()

    def testHistogramCtrlWidget(self):
        widget = self.gui.histogram_ctrl_widget
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

    @patch('extra_foam.gui.ctrl_widgets.PumpProbeCtrlWidget.'
           'updateMetaData', MagicMock(return_value=True))
    @patch('extra_foam.gui.ctrl_widgets.HistogramCtrlWidget.'
           'updateMetaData', MagicMock(return_value=True))
    @patch('extra_foam.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.'
           'updateMetaData', MagicMock(return_value=True))
    @patch('extra_foam.gui.ctrl_widgets.DataSourceWidget.'
           'updateMetaData', MagicMock(return_value=True))
    @patch('extra_foam.gui.ctrl_widgets.PumpProbeCtrlWidget.onStart', Mock())
    @patch('extra_foam.gui.ctrl_widgets.HistogramCtrlWidget.onStart', Mock())
    @patch('extra_foam.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.onStart', Mock())
    @patch('extra_foam.gui.ctrl_widgets.PumpProbeCtrlWidget.onStop', Mock())
    @patch('extra_foam.gui.ctrl_widgets.HistogramCtrlWidget.onStop', Mock())
    @patch('extra_foam.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.onStop', Mock())
    def testStartStop(self):
        start_spy = QSignalSpy(self.gui.start_sgn)
        stop_spy = QSignalSpy(self.gui.stop_sgn)

        # -------------------------------------------------------------
        # test when the start action button is clicked
        # -------------------------------------------------------------

        start_action = self.gui._tool_bar.actions()[0]
        stop_action = self.gui._tool_bar.actions()[1]

        start_action.trigger()

        # test a ctrl widget own by the ImageToolWindow
        azimuthal_integ_ctrl_widget = self.gui._image_tool._azimuthal_integ_1d_view._ctrl_widget
        geometry_ctrl_widget = self.gui._image_tool._geometry_view._ctrl_widget
        pump_probe_ctrl_widget = self.gui.pump_probe_ctrl_widget
        histogram_ctrl_widget = self.gui.histogram_ctrl_widget
        source_ctrl_widget = self.gui._source_cw

        azimuthal_integ_ctrl_widget.updateMetaData.assert_called_once()
        pump_probe_ctrl_widget.updateMetaData.assert_called_once()
        histogram_ctrl_widget.updateMetaData.assert_called_once()
        source_ctrl_widget.updateMetaData.assert_called_once()

        self.assertEqual(1, len(start_spy))

        pump_probe_ctrl_widget.onStart.assert_called_once()
        histogram_ctrl_widget.onStart.assert_called_once()
        azimuthal_integ_ctrl_widget.onStart.assert_called_once()

        self.assertFalse(start_action.isEnabled())
        self.assertTrue(stop_action.isEnabled())
        self.assertFalse(source_ctrl_widget._con_view.isEnabled())
        self.assertFalse(geometry_ctrl_widget.isEnabled())

        self.assertTrue(self.train_worker.running)
        self.assertTrue(self.pulse_worker.running)

        # -------------------------------------------------------------
        # test when the stop action button is clicked
        # -------------------------------------------------------------

        stop_action.trigger()

        pump_probe_ctrl_widget.onStop.assert_called_once()
        histogram_ctrl_widget.onStop.assert_called_once()
        azimuthal_integ_ctrl_widget.onStop.assert_called_once()

        self.assertEqual(1, len(stop_spy))

        self.assertTrue(start_action.isEnabled())
        self.assertFalse(stop_action.isEnabled())
        self.assertTrue(source_ctrl_widget._con_view.isEnabled())
        self.assertTrue(geometry_ctrl_widget.isEnabled())

        self.assertFalse(self.train_worker.running)
        self.assertFalse(self.pulse_worker.running)

    def testTrXasCtrl(self):
        from extra_foam.gui.ctrl_widgets.trxas_ctrl_widget import (
            _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE,
        )
        default_bin_range = tuple(float(v) for v in _DEFAULT_BIN_RANGE.split(','))

        widget = self.gui._trxas_ctrl_widget
        proc = self.train_worker._tr_xas

        # test default values
        proc.update()
        self.assertTupleEqual(default_bin_range, proc._delay_range)
        self.assertTupleEqual(default_bin_range, proc._energy_range)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_delay_bins)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_energy_bins)

        widget._energy_device_le.setText("new/mono")
        widget._energy_ppt_le.setText("new/mono/ppt")
        widget._delay_device_le.setText("new/phase/shifter")
        widget._delay_ppt_le.setText("new/phase/shifter/ppt")
        widget._delay_range_le.setText("-1, 1")
        widget._energy_range_le.setText("-1.0, 1.0")
        widget._n_delay_bins_le.setText("100")
        widget._n_energy_bins_le.setText("1000")
        proc.update()
        self.assertEqual("new/mono new/mono/ppt", proc._energy_src)
        self.assertEqual("new/phase/shifter new/phase/shifter/ppt", proc._delay_src)
        self.assertTupleEqual((-1, 1), proc._delay_range)
        self.assertTupleEqual((-1.0, 1.0), proc._energy_range)
        self.assertEqual(100, proc._n_delay_bins)
        self.assertEqual(1000, proc._n_energy_bins)

        # test reset button
        proc._reset = False
        widget._scan_btn_set.reset_sgn.emit()
        proc.update()
        self.assertTrue(proc._reset)

    def testOpenCloseWindows(self):
        actions = self.gui._tool_bar.actions()

        poi_action = actions[4]
        self.assertEqual("Pulse-of-interest", poi_action.text())
        pp_action = actions[5]
        self.assertEqual("Pump-probe", pp_action.text())
        correlation_action = actions[6]
        self.assertEqual("Correlation", correlation_action.text())
        histogram_action = actions[7]
        self.assertEqual("Histogram", histogram_action.text())
        binning_action = actions[8]
        self.assertEqual("Binning", binning_action.text())

        pp_window = self._check_open_window(pp_action)
        self.assertIsInstance(pp_window, PumpProbeWindow)

        correlation_window = self._check_open_window(correlation_action)
        self.assertIsInstance(correlation_window, CorrelationWindow)

        binning_window = self._check_open_window(binning_action)
        self.assertIsInstance(binning_window, BinningWindow)

        histogram_window = self._check_open_window(histogram_action)
        self.assertIsInstance(histogram_window, HistogramWindow)

        poi_window = self._check_open_window(poi_action)
        self.assertIsInstance(poi_window, PulseOfInterestWindow)
        # open one window twice
        self._check_open_window(poi_action, registered=False)

        self._check_close_window(pp_window)
        self._check_close_window(correlation_window)
        self._check_close_window(binning_window)
        self._check_close_window(histogram_window)
        self._check_close_window(poi_window)

        # if a plot window is closed, it can be re-openned and a new instance
        # will be created
        pp_window_new = self._check_open_window(pp_action)
        self.assertIsInstance(pp_window_new, PumpProbeWindow)
        self.assertIsNot(pp_window_new, pp_window)

    def testOpenCloseSatelliteWindows(self):
        actions = self.gui._tool_bar.actions()
        about_action = actions[-1]
        streamer_action = actions[-2]

        about_window = self._check_open_satellite_window(about_action)
        self.assertIsInstance(about_window, AboutWindow)

        streamer_window = self._check_open_satellite_window(streamer_action)
        self.assertIsInstance(streamer_window, FileStreamControllerWindow)

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


class TestJungFrauMainGuiCtrl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('JungFrau', 'FXE')

        cls.foam = Foam().init()

        cls.gui = cls.foam._gui
        cls.pulse_worker = cls.foam.pulse_worker
        cls.train_worker = cls.foam.train_worker

    @classmethod
    def tearDownClass(cls):
        cls.foam.terminate()

        wait_until_redis_shutdown()

        os.remove(config.config_file)

    def setUp(self):
        self.assertTrue(self.gui.updateMetaData())

    def testAnalysisCtrlWidget(self):
        widget = self.gui.analysis_ctrl_widget
        image_proc = self.pulse_worker._image_proc

        image_proc.update()
        for i, idx in enumerate(image_proc._poi_indices):
            self.assertFalse(widget._poi_index_les[i].isEnabled())
            self.assertEqual(0, idx)  # test default values

    def testPumpProbeCtrlWidget(self):
        widget = self.gui.pump_probe_ctrl_widget
        pp_proc = self.pulse_worker._pp_proc

        self.assertFalse(widget._on_pulse_le.isEnabled())
        self.assertFalse(widget._off_pulse_le.isEnabled())

        all_modes = {value: key for key, value in
                     widget._available_modes.items()}

        # we only test train-resolved detector specific configuration

        pp_proc.update()
        self.assertEqual(PumpProbeMode.UNDEFINED, pp_proc._mode)
        self.assertListEqual([-1], pp_proc._indices_on)
        self.assertListEqual([-1], pp_proc._indices_off)

        spy = QSignalSpy(widget._mode_cb.currentTextChanged)

        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.EVEN_TRAIN_ON])
        self.assertEqual(1, len(spy))

        pp_proc.update()
        self.assertEqual(PumpProbeMode(PumpProbeMode.EVEN_TRAIN_ON), pp_proc._mode)
        self.assertListEqual([-1], pp_proc._indices_on)
        self.assertListEqual([-1], pp_proc._indices_off)

        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.REFERENCE_AS_OFF])
        self.assertEqual(2, len(spy))
        # test on_pulse_le is still disabled, which will become enabled if the
        # detector is pulse-resolved
        self.assertFalse(widget._on_pulse_le.isEnabled())

        # PumpProbeMode.SAME_TRAIN is not available
        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.SAME_TRAIN])
        self.assertEqual(2, len(spy))

    def testFomFilterCtrlWidget(self):
        widget = self.gui.fom_filter_ctrl_widget
        filter_pulse = self.pulse_worker._filter
        filter_train = self.train_worker._filter

        analysis_types = {value: key for key, value in widget._analysis_types.items()}

        # test default

        self.assertFalse(widget._pulse_resolved_cb.isChecked())

        filter_pulse.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_pulse.analysis_type)
        self.assertTupleEqual((-np.inf, np.inf), filter_pulse._fom_range)
        filter_train.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_pulse.analysis_type)
        self.assertTupleEqual((-np.inf, np.inf), filter_pulse._fom_range)

        # test set new

        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.ROI_FOM])
        widget._fom_range_le.setText("-2, 2")
        filter_pulse.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_pulse.analysis_type)
        self.assertEqual((-np.inf, np.inf), filter_pulse._fom_range)
        filter_train.update()
        self.assertEqual(AnalysisType.ROI_FOM, filter_train.analysis_type)
        self.assertTupleEqual((-2, 2), filter_train._fom_range)

    def testHistogramCtrlWidget(self):
        # TODO
        pass
