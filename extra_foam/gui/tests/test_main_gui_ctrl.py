import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os

import numpy as np

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest, QSignalSpy

from extra_foam.database import Metadata as mt
from extra_foam.logger import logger
from extra_foam.services import Foam
from extra_foam.gui import mkQApp
from extra_foam.gui.windows import PulseOfInterestWindow
from extra_foam.config import (
    _Config, ConfigWrapper, config, AnalysisType, BinMode, PumpProbeMode,
    RoiCombo
)
from extra_foam.processes import wait_until_redis_shutdown

app = mkQApp()

logger.setLevel("CRITICAL")


class TestLpdMainGuiCtrl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()   # ensure file
        config.load('LPD')
        config.set_topic("FXE")

        cls.foam = Foam().init()

        cls.gui = cls.foam._gui
        cls.train_worker = cls.foam.train_worker
        cls.pulse_worker = cls.foam.pulse_worker

    @classmethod
    def tearDownClass(cls):
        cls.foam.terminate()

        wait_until_redis_shutdown()

    def setUp(self):
        self.assertTrue(self.gui.updateMetaData())

    def testAnalysisCtrlWidget(self):
        widget = self.gui.analysis_ctrl_widget
        train_worker = self.train_worker
        pulse_worker = self.pulse_worker

        xgm_proc = pulse_worker._xgm_proc
        ai_proc = train_worker._ai_proc
        roi_proc = train_worker._roi_proc

        # test "Moving average window"
        widget._ma_window_le.setText("5")
        xgm_proc.update()
        self.assertEqual(5, xgm_proc.__class__._pulse_intensity_ma.window)
        self.assertEqual(5, xgm_proc.__class__._intensity_ma.window)
        self.assertEqual(5, xgm_proc.__class__._x_ma.window)
        self.assertEqual(5, xgm_proc.__class__._y_ma.window)

        # test "Reset moving average" button
        widget._reset_ma_btn.clicked.emit()
        self.assertEqual('1', xgm_proc._meta.hget(mt.GLOBAL_PROC, 'reset_ma_xgm'))
        self.assertEqual('1', ai_proc._meta.hget(mt.GLOBAL_PROC, 'reset_ma_ai'))
        self.assertEqual('1', roi_proc._meta.hget(mt.GLOBAL_PROC, 'reset_ma_roi'))

        xgm_proc.update()
        self.assertIsNone(xgm_proc._meta.hget(mt.GLOBAL_PROC, 'reset_ma_xgm'))
        ai_proc.update()
        self.assertIsNone(ai_proc._meta.hget(mt.GLOBAL_PROC, 'reset_ma_ai'))
        roi_proc.update()
        self.assertIsNone(roi_proc._meta.hget(mt.GLOBAL_PROC, 'reset_ma_roi'))

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

        # off_pulse_le will be disabled when the mode is PRE_DEFINED_OFF
        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.PRE_DEFINED_OFF])
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

    @patch.dict(config._data, {"SOURCE_NAME_BRIDGE": ["E", "F", "G"],
                               "SOURCE_NAME_FILE": ["A", "B"]})
    def testConnectionCtrlWidget(self):
        from extra_foam.gui.ctrl_widgets.data_source_widget import ConnectionCtrlWidget

        for widget in self.gui._ctrl_widgets:
            if isinstance(widget, ConnectionCtrlWidget):
                break

        train_worker = self.train_worker
        pulse_worker = self.pulse_worker
        assembler = pulse_worker._assembler

        # test passing tcp hostname and port

        # TODO: testit
        # hostname = config['SERVER_ADDR']
        # port = config['SERVER_PORT']
        # self.assertEqual(f"tcp://{hostname}:{port}", bridge._endpoint)

        #
        # widget._hostname_le.setText('127.0.0.1')
        # widget._port_le.setText('12345')
        # self.assertEqual(f"tcp://127.0.0.1:12345", bridge._endpoint)

        # self.assertEqual("A", assembler._source_name)
        # items = []
        # for i in range(widget._detector_src_cb.count()):
        #     items.append(widget._detector_src_cb.itemText(i))
        # self.assertListEqual(["A", "B"], items)

        # self.assertEqual("E", assembler._source_name)
        # items = []
        # for i in range(widget._detector_src_cb.count()):
        #     items.append(widget._detector_src_cb.itemText(i))
        # self.assertListEqual(["E", "F", "G"], items)

    def testPulseFilterCtrlWidget(self):
        widget = self.gui.pulse_filter_ctrl_widget
        pulse_worker = self.pulse_worker
        post_pulse_filter = pulse_worker._post_pulse_filter

        analysis_types = {value: key for key, value in
                          widget._analysis_types.items()}
        post_pulse_filter.update()
        self.assertEqual(AnalysisType.UNDEFINED, post_pulse_filter.analysis_type)
        self.assertTupleEqual((-np.inf, np.inf), post_pulse_filter._fom_range)

        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.ROI_FOM_PULSE])
        widget._fom_range_le.setText("-1, 1")
        post_pulse_filter.update()
        self.assertEqual(AnalysisType.ROI_FOM_PULSE, post_pulse_filter.analysis_type)
        self.assertEqual((-1, 1), post_pulse_filter._fom_range)

    def testCorrelationCtrlWidget(self):
        from extra_foam.gui.ctrl_widgets.correlation_ctrl_widget import (
            _N_PARAMS, _DEFAULT_RESOLUTION)

        widget = self.gui.correlation_ctrl_widget

        for i in range(_N_PARAMS):
            combo_lst = [widget._table.cellWidget(i, 0).itemText(j)
                         for j in range(widget._table.cellWidget(i, 0).count())]
            self.assertEqual(
                list(widget._TOPIC_DATA_CATEGORIES[config["TOPIC"]]), combo_lst)

        train_worker = self.train_worker
        proc = train_worker._correlation_proc

        proc.update()

        # test default
        self.assertEqual(AnalysisType(0), proc.analysis_type)
        self.assertEqual([""] * _N_PARAMS, proc._device_ids)
        self.assertEqual([""] * _N_PARAMS, proc._properties)

        # set new FOM
        widget._analysis_type_cb.setCurrentText('ROI proj')
        proc.update()
        self.assertEqual(AnalysisType.ROI_PROJ, proc.analysis_type)

        # test the correlation param table
        for i in range(_N_PARAMS):
            # change category
            widget._table.cellWidget(i, 0).setCurrentIndex(1)

            # change device id
            widget._table.cellWidget(i, 1).setCurrentIndex(1)

            resolution = (i+1)*5 if i < 2 else 0.0
            resolution_le = widget._table.cellWidget(i, 3)
            resolution_le.setText(str(resolution))

        proc.update()
        for i in range(_N_PARAMS):
            device_id = widget._table.cellWidget(i, 1).currentText()
            self.assertEqual(device_id, proc._device_ids[i])
            ppt = widget._table.cellWidget(i, 1).currentText()
            self.assertEqual(ppt, proc._device_ids[i])

        # test reset button
        proc._reset = False
        widget._reset_btn.clicked.emit()
        proc.update()
        self.assertTrue(proc._reset)

    def testBinCtrlWidget(self):
        from extra_foam.gui.ctrl_widgets.bin_ctrl_widget import (
            _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE, _N_PARAMS
        )

        widget = self.gui.bin_ctrl_widget
        for i in range(_N_PARAMS):
            combo_lst = [widget._table.cellWidget(i, 0).itemText(j)
                         for j in range(widget._table.cellWidget(i, 0).count())]
            self.assertEqual(
                list(widget._TOPIC_DATA_CATEGORIES[config["TOPIC"]]), combo_lst)

        train_worker = self.train_worker
        proc = train_worker._bin_proc
        proc.update()

        default_bin_range = tuple(float(v) for v in _DEFAULT_BIN_RANGE.split(','))

        # test default
        self.assertEqual(AnalysisType.UNDEFINED, proc.analysis_type)
        self.assertEqual(BinMode.AVERAGE, proc._mode)
        self.assertEqual("", proc._device_id1)
        self.assertEqual("", proc._property1)
        self.assertTupleEqual(default_bin_range, proc._range1)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_bins1)
        self.assertEqual("", proc._device_id2)
        self.assertEqual("", proc._property2)
        self.assertEqual(default_bin_range, proc._range2)
        self.assertEqual(int(_DEFAULT_N_BINS), proc._n_bins2)

        # test analysis type change
        proc._reset1 = False
        proc._reset2 = False
        widget._analysis_type_cb.setCurrentIndex(1)
        proc.update()
        self.assertEqual(AnalysisType(AnalysisType.PUMP_PROBE),
                         proc.analysis_type)
        self.assertTrue(proc._reset1)
        self.assertTrue(proc._reset2)

        # test device id and property change
        proc._reset1 = False
        proc._reset2 = False
        # bin parameter 1
        widget._table.cellWidget(0, 0).setCurrentText('Train ID')
        self.assertEqual("Train ID", widget._table.cellWidget(0, 0).currentText())
        widget._table.cellWidget(0, 1).setCurrentText('Any')
        self.assertEqual("Any", widget._table.cellWidget(0, 1).currentText())
        self.assertEqual("timestamp.tid", widget._table.cellWidget(0, 2).currentText())
        proc.update()
        self.assertTrue(proc._reset1)
        self.assertFalse(proc._reset2)
        # bin parameter 2
        widget._table.cellWidget(1, 0).setCurrentText('User defined')
        self.assertEqual("User defined", widget._table.cellWidget(1, 0).currentText())
        widget._table.cellWidget(1, 1).setText('Any')
        self.assertEqual("Any", widget._table.cellWidget(1, 1).text())
        widget._table.cellWidget(1, 2).setText('timestamp.tid')
        self.assertEqual("timestamp.tid", widget._table.cellWidget(1, 2).text())
        proc.update()
        self.assertTrue(proc._reset1)
        self.assertTrue(proc._reset2)

        # test bin range and number of bins change
        proc._reset1 = False
        proc._reset2 = False
        # bin parameter 1
        widget._table.cellWidget(0, 3).setText("0, 10")  # range
        widget._table.cellWidget(0, 4).setText("5")  # n_bins
        proc.update()
        self.assertEqual(5, proc._n_bins1)
        self.assertTupleEqual((0, 10), proc._range1)
        np.testing.assert_array_equal(np.array([0, 2, 4, 6, 8, 10]), proc._edge1)
        np.testing.assert_array_equal(np.array([1, 3, 5, 7, 9]), proc._center1)
        self.assertTrue(proc._reset1)
        self.assertFalse(proc._reset2)
        # bin parameter 2
        widget._table.cellWidget(1, 3).setText("-4, 4")  # range
        widget._table.cellWidget(1, 4).setText("2")  # n_bins
        proc.update()
        self.assertEqual(2, proc._n_bins2)
        self.assertTupleEqual((-4, 4), proc._range2)
        np.testing.assert_array_equal(np.array([-4, 0, 4]), proc._edge2)
        np.testing.assert_array_equal(np.array([-2, 2]), proc._center2)
        self.assertTrue(proc._reset1)
        self.assertTrue(proc._reset2)

        # test reset button
        proc._reset1 = False
        proc._reset2 = False
        widget._reset_btn.clicked.emit()
        proc.update()
        self.assertTrue(proc._reset1)
        self.assertTrue(proc._reset2)

    def testStatisticsCtrlWidget(self):
        widget = self.gui.statistics_ctrl_widget
        train_worker = self.train_worker
        proc = train_worker._statistics
        proc.update()

        analysis_types = {value: key for key, value in
                          widget._analysis_types.items()}

        self.assertEqual(AnalysisType.UNDEFINED, proc.analysis_type)
        self.assertTrue(proc._pulse_resolved)
        self.assertEqual(10, proc._num_bins)

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

        widget._num_bins_le.setText("100")
        proc.update()
        self.assertEqual(100, proc._num_bins)

        proc._reset = False
        widget._reset_btn.clicked.emit()
        proc.update()
        self.assertTrue(proc._reset)

    @patch('extra_foam.gui.ctrl_widgets.PumpProbeCtrlWidget.'
           'updateMetaData', MagicMock(return_value=True))
    @patch('extra_foam.gui.ctrl_widgets.StatisticsCtrlWidget.'
           'updateMetaData', MagicMock(return_value=True))
    @patch('extra_foam.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.'
           'updateMetaData', MagicMock(return_value=True))
    @patch('extra_foam.gui.ctrl_widgets.PumpProbeCtrlWidget.onStart', Mock())
    @patch('extra_foam.gui.ctrl_widgets.StatisticsCtrlWidget.onStart', Mock())
    @patch('extra_foam.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.onStart', Mock())
    @patch('extra_foam.gui.ctrl_widgets.PumpProbeCtrlWidget.onStop', Mock())
    @patch('extra_foam.gui.ctrl_widgets.StatisticsCtrlWidget.onStop', Mock())
    @patch('extra_foam.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.onStop', Mock())
    @patch('extra_foam.pipeline.TrainWorker.resume', Mock())
    @patch('extra_foam.pipeline.TrainWorker.pause', Mock())
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

        self.gui.pump_probe_ctrl_widget.updateMetaData. \
            assert_called_once()
        self.gui.statistics_ctrl_widget.updateMetaData. \
            assert_called_once()
        azimuthal_integ_ctrl_widget.updateMetaData. \
            assert_called_once()

        self.assertEqual(1, len(start_spy))

        self.gui.pump_probe_ctrl_widget.onStart.assert_called_once()
        self.gui.statistics_ctrl_widget.onStart.assert_called_once()
        azimuthal_integ_ctrl_widget.onStart.assert_called_once()

        self.assertFalse(start_action.isEnabled())
        self.assertTrue(stop_action.isEnabled())

        # -------------------------------------------------------------
        # test when the start action button is clicked
        # -------------------------------------------------------------

        stop_action.trigger()

        self.gui.pump_probe_ctrl_widget.onStop.assert_called_once()
        self.gui.statistics_ctrl_widget.onStop.assert_called_once()
        azimuthal_integ_ctrl_widget.onStop.assert_called_once()

        self.assertEqual(1, len(stop_spy))

        self.assertTrue(start_action.isEnabled())
        self.assertFalse(stop_action.isEnabled())

    def testPoiWindowCtrl(self):
        image_proc = self.pulse_worker._image_proc

        # --------------------------
        # test setting POI pulse indices
        # --------------------------
        poi_action = self.gui._tool_bar.actions()[4]
        self.assertEqual("Pulse-of-interest", poi_action.text())
        poi_action.trigger()

        window = [w for w in self.gui._windows
                  if isinstance(w, PulseOfInterestWindow)][0]

        # default values
        default_index1 = int(window._poi_widget1._index_le.text())
        self.assertEqual(default_index1, window._poi_widget1._poi_img._index)
        self.assertEqual(default_index1, window._poi_widget1._poi_statistics._index)

        default_index2 = int(window._poi_widget2._index_le.text())
        self.assertEqual(default_index2, window._poi_widget2._poi_img._index)
        self.assertEqual(default_index2, window._poi_widget2._poi_statistics._index)

        image_proc.update()
        self.assertEqual(default_index1, image_proc._poi_indices[0])
        self.assertEqual(default_index2, image_proc._poi_indices[1])

        # set new values
        poi_index1 = 10
        window._poi_widget1._index_le.setText(str(poi_index1))
        self.assertEqual(poi_index1, window._poi_widget1._poi_img._index)
        self.assertEqual(poi_index1, window._poi_widget1._poi_statistics._index)

        poi_index2 = 20
        window._poi_widget2._index_le.setText(str(poi_index2))
        self.assertEqual(poi_index2, window._poi_widget2._poi_img._index)
        self.assertEqual(poi_index2, window._poi_widget2._poi_statistics._index)

        image_proc.update()
        self.assertEqual(poi_index1, image_proc._poi_indices[0])
        self.assertEqual(poi_index2, image_proc._poi_indices[1])

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

        widget._energy_device_le.setText("new mono")
        widget._energy_ppt_le.setText("new mono ppt")
        widget._delay_device_le.setText("new phase shifter")
        widget._delay_ppt_le.setText("new phase shifter ppt")
        widget._delay_range_le.setText("-1, 1")
        widget._energy_range_le.setText("-1.0, 1.0")
        widget._n_delay_bins_le.setText("100")
        widget._n_energy_bins_le.setText("1000")
        proc.update()
        self.assertEqual("new mono", proc._energy_device)
        self.assertEqual("new mono ppt", proc._energy_ppt)
        self.assertEqual("new phase shifter", proc._delay_device)
        self.assertEqual("new phase shifter ppt", proc._delay_ppt)
        self.assertTupleEqual((-1, 1), proc._delay_range)
        self.assertTupleEqual((-1.0, 1.0), proc._energy_range)
        self.assertEqual(100, proc._n_delay_bins)
        self.assertEqual(1000, proc._n_energy_bins)

        # test reset button
        proc._reset = False
        widget._scan_btn_set.reset_sgn.emit()
        proc.update()
        self.assertTrue(proc._reset)


class TestJungFrauMainGuiCtrl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()   # ensure file
        config.load('JungFrau')
        config.set_topic("FXE")

        cls.foam = Foam().init()

        cls.gui = cls.foam._gui
        cls.pulse_worker = cls.foam.pulse_worker

    @classmethod
    def tearDownClass(cls):
        cls.foam.terminate()

        wait_until_redis_shutdown()

    def setUp(self):
        self.assertTrue(self.gui.updateMetaData())

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

        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.PRE_DEFINED_OFF])
        self.assertEqual(2, len(spy))
        # test on_pulse_le is still disabled, which will become enabled if the
        # detector is pulse-resolved
        self.assertFalse(widget._on_pulse_le.isEnabled())

        # PumpProbeMode.SAME_TRAIN is not available
        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.SAME_TRAIN])
        self.assertEqual(2, len(spy))

    def testStatisticsCtrlWidget(self):
        # TODO
        pass

    def testPoiWindowCtrl(self):
        image_proc = self.pulse_worker._image_proc

        # POI action is disabled
        poi_action = self.gui._tool_bar.actions()[4]
        self.assertEqual("Pulse-of-interest", poi_action.text())
        self.assertFalse(poi_action.isEnabled())

        image_proc.update()
        self.assertIsNone(image_proc._poi_indices)
