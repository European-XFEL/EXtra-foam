import unittest
from unittest.mock import patch, MagicMock, Mock
import math
import tempfile
import os

import numpy as np

from PyQt5 import QtCore
from PyQt5.QtTest import QTest, QSignalSpy
from PyQt5.QtCore import Qt

from karabo_data.geometry2 import LPD_1MGeometry

from karaboFAI.database import Metadata as mt
from karaboFAI.logger import logger
from karaboFAI.services import FAI
from karaboFAI.gui import mkQApp
from karaboFAI.gui.windows import PulseOfInterestWindow
from karaboFAI.config import (
    _Config, ConfigWrapper, config, AnalysisType, BinMode,
    DataSource, Normalizer, PumpProbeMode
)
from karaboFAI.processes import wait_until_redis_shutdown
from karaboFAI.pipeline.processors.azimuthal_integration import energy2wavelength
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

        cls.fai = FAI().init()

        cls.gui = cls.fai._gui
        cls.scheduler = cls.fai.scheduler
        cls.image_worker = cls.fai.image_worker

        cls._actions = cls.gui._tool_bar.actions()
        cls._start_action = cls._actions[0]
        cls._stop_action = cls._actions[1]
        cls._pp_action = cls._actions[3]
        cls._statistics_action = cls._actions[4]
        cls._correlation_action = cls._actions[5]
        cls._bin1d_action = cls._actions[6]
        cls._bin2d_action = cls._actions[7]
        cls._poi_action = cls._actions[8]
        cls._ai_action = cls._actions[9]

    @classmethod
    def tearDownClass(cls):
        cls.fai.terminate()

        wait_until_redis_shutdown()

    def setUp(self):
        self.assertTrue(self.gui.updateMetaData())

    def testAnalysisCtrlWidget(self):
        widget = self.gui.analysis_ctrl_widget
        scheduler = self.scheduler
        image_worker = self.image_worker

        image_proc = image_worker._image_proc
        xgm_proc = image_worker._xgm_proc
        ai_proc = scheduler._ai_proc_train
        roi_proc = scheduler._roi_proc_train

        # --------------------------
        # test setting POI pulse indices
        # --------------------------
        self._poi_action.trigger()
        window = [w for w in self.gui._windows
                  if isinstance(w, PulseOfInterestWindow)][0]

        # default values
        poi_index1 = int(widget._poi_index1_le.text())
        self.assertEqual(poi_index1, window._poi1_img.pulse_index)
        poi_index2 = int(widget._poi_index2_le.text())
        self.assertEqual(poi_index2, window._poi2_img.pulse_index)

        image_proc.update()
        self.assertEqual(poi_index1, image_proc._poi_indices[0])
        self.assertEqual(poi_index2, image_proc._poi_indices[1])

        # set new values
        poi_index1 = 10
        widget._poi_index1_le.setText(str(poi_index1))
        self.assertEqual(poi_index1, window._poi1_img.pulse_index)

        poi_index2 = 20
        widget._poi_index2_le.setText(str(poi_index2))
        self.assertEqual(poi_index2, window._poi2_img.pulse_index)

        image_proc.update()
        self.assertEqual(poi_index1, image_proc._poi_indices[0])
        self.assertEqual(poi_index2, image_proc._poi_indices[1])

        # test params sent to AzimuthalIntegrationProcessor
        ai_proc.update()
        self.assertAlmostEqual(config['SAMPLE_DISTANCE'],
                               ai_proc._sample_dist)
        self.assertAlmostEqual(energy2wavelength(config['PHOTON_ENERGY']),
                               ai_proc._wavelength)

        widget._photon_energy_le.setText("12.4")
        widget._sample_dist_le.setText("0.3")

        ai_proc.update()
        self.assertAlmostEqual(1e-10, ai_proc._wavelength)
        self.assertAlmostEqual(0.3, ai_proc._sample_dist)

        # test "Moving average window"
        widget._ma_window_le.setText("5")
        xgm_proc.update()
        self.assertEqual(5, xgm_proc.__class__._pulse_intensity_ma.window)
        self.assertEqual(5, xgm_proc.__class__._intensity_ma.window)
        self.assertEqual(5, xgm_proc.__class__._x_ma.window)
        self.assertEqual(5, xgm_proc.__class__._y_ma.window)

        # test "Reset moving average" button
        widget._reset_ma_btn.clicked.emit()
        self.assertEqual('1', xgm_proc._meta.get(mt.GLOBAL_PROC, 'reset_ma_xgm'))
        self.assertEqual('1', ai_proc._meta.get(mt.GLOBAL_PROC, 'reset_ma_ai'))
        self.assertEqual('1', roi_proc._meta.get(mt.GLOBAL_PROC, 'reset_ma_roi'))

        xgm_proc.update()
        self.assertIsNone(xgm_proc._meta.get(mt.GLOBAL_PROC, 'reset_ma_xgm'))
        ai_proc.update()
        self.assertIsNone(ai_proc._meta.get(mt.GLOBAL_PROC, 'reset_ma_ai'))
        roi_proc.update()
        self.assertIsNone(roi_proc._meta.get(mt.GLOBAL_PROC, 'reset_ma_roi'))

    def testAzimuthalIntegCtrlWidget(self):
        widget = self.gui.azimuthal_integ_ctrl_widget
        scheduler = self.scheduler
        proc = scheduler._ai_proc_train

        proc.update()

        self.assertEqual(AnalysisType.UNDEFINED, proc.analysis_type)
        default_integ_method = 'BBox'
        self.assertEqual(default_integ_method, proc._integ_method)
        default_normalizer = Normalizer.UNDEFINED
        self.assertEqual(default_normalizer, proc._normalizer)
        self.assertEqual(config["AZIMUTHAL_INTEG_POINTS"], proc._integ_points)
        default_integ_range = tuple(config["AZIMUTHAL_INTEG_RANGE"])
        self.assertTupleEqual(tuple(config["AZIMUTHAL_INTEG_RANGE"]), proc._integ_range)
        self.assertTupleEqual(default_integ_range, proc._auc_range)
        self.assertTupleEqual(default_integ_range, proc._fom_integ_range)
        pixel_size = config["PIXEL_SIZE"]
        self.assertEqual(config["CENTER_Y"] * pixel_size, proc._poni1)
        self.assertEqual(config["CENTER_X"] * pixel_size, proc._poni2)

        integ_method = 'nosplit_csr'
        widget._integ_method_cb.setCurrentText(integ_method)
        ai_normalizer = Normalizer.ROI3_SUB_ROI4
        widget._normalizers_cb.setCurrentIndex(ai_normalizer)
        widget._integ_pts_le.setText(str(1024))
        widget._integ_range_le.setText("0.1, 0.2")
        widget._auc_range_le.setText("0.2, 0.3")
        widget._fom_integ_range_le.setText("0.3, 0.4")
        widget._cx_le.setText("-1000")
        widget._cy_le.setText("1000")

        proc.update()
        self.assertEqual(integ_method, proc._integ_method)
        self.assertEqual(ai_normalizer, proc._normalizer)
        self.assertEqual(1024, proc._integ_points)
        self.assertTupleEqual((0.1, 0.2), proc._integ_range)
        self.assertTupleEqual((0.2, 0.3), proc._auc_range)
        self.assertTupleEqual((0.3, 0.4), proc._fom_integ_range)
        self.assertEqual(-1000*pixel_size, proc._poni2)
        self.assertEqual(1000*pixel_size, proc._poni1)

    def testProjection1DCtrlWidget(self):
        widget = self.gui.roi_ctrl_widget
        proc = self.scheduler._roi_proc_train
        proc.update()

        # test default reconfigurable values
        self.assertEqual('x', proc._direction)
        self.assertEqual(Normalizer.UNDEFINED, proc._normalizer)
        self.assertEqual((0, math.inf), proc._fom_integ_range)
        self.assertEqual((0, math.inf), proc._auc_range)

        # test setting new values
        widget._direct_cb.setCurrentText('y')
        widget._normalizers_cb.setCurrentText('ROI3 - ROI4')
        widget._fom_integ_range_le.setText("10, 20")
        widget._auc_range_le.setText("30, 40")
        proc.update()
        self.assertEqual('y', proc._direction)
        self.assertEqual(Normalizer.ROI3_SUB_ROI4, proc._normalizer)
        self.assertEqual((10, 20), proc._fom_integ_range)
        self.assertEqual((30, 40), proc._auc_range)

    def testPumpProbeCtrlWidget(self):
        widget = self.gui.pump_probe_ctrl_widget
        pp_proc = self.image_worker._pp_proc

        all_modes = {value: key for key, value in
                     widget._available_modes.items()}

        # check default reconfigurable params
        pp_proc.update()
        self.assertTrue(pp_proc._abs_difference)
        self.assertEqual(AnalysisType(0), pp_proc.analysis_type)

        pp_proc.update()
        self.assertEqual(PumpProbeMode.UNDEFINED, pp_proc._mode)
        self.assertListEqual([-1], pp_proc._on_indices)
        self.assertIsInstance(pp_proc._on_indices[0], int)
        self.assertListEqual([-1], pp_proc._off_indices)
        self.assertIsInstance(pp_proc._off_indices[0], int)

        # change analysis type
        pp_proc._reset = False
        widget._analysis_type_cb.setCurrentText('ROI1 - ROI2 (proj)')
        pp_proc.update()
        self.assertEqual(AnalysisType.PROJ_ROI1_SUB_ROI2, pp_proc.analysis_type)
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
                         pos=QtCore.QPoint(2, widget._abs_difference_cb.height()/2))
        pp_proc.update()
        self.assertFalse(pp_proc._abs_difference)
        self.assertTrue(pp_proc._reset)

        # change on/off pulse indices
        widget._on_pulse_le.setText('0:10:2')
        widget._off_pulse_le.setText('1:10:2')
        pp_proc.update()
        self.assertEqual(PumpProbeMode.EVEN_TRAIN_ON, pp_proc._mode)
        self.assertListEqual([0, 2, 4, 6, 8], pp_proc._on_indices)
        self.assertListEqual([1, 3, 5, 7, 9], pp_proc._off_indices)

        # test reset button
        pp_proc._reset = False
        widget._reset_btn.clicked.emit()
        pp_proc.update()
        self.assertTrue(pp_proc._reset)

    @patch.dict(config._data, {"SOURCE_NAME_BRIDGE": ["E", "F", "G"],
                               "SOURCE_NAME_FILE": ["A", "B"]})
    def testConnectionCtrlWidget(self):
        from karaboFAI.gui.ctrl_widgets.data_source_widget import ConnectionCtrlWidget

        for widget in self.gui._ctrl_widgets:
            if isinstance(widget, ConnectionCtrlWidget):
                break

        scheduler = self.scheduler
        image_worker = self.image_worker
        assembler = image_worker._assembler

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

    def testGeometryCtrlWidget(self):
        from karaboFAI.gui.ctrl_widgets import GeometryCtrlWidget

        for widget in self.gui._ctrl_widgets:
            if isinstance(widget, GeometryCtrlWidget):
                break

        image_worker = self.image_worker

        widget._geom_file_le.setText(config["GEOMETRY_FILE"])

        self.assertTrue(self.gui.updateMetaData())

        image_worker._assembler.update()
        self.assertIsInstance(image_worker._assembler._geom, LPD_1MGeometry)

    def testPulseFilterCtrlWidget(self):
        widget = self.gui.pulse_filter_ctrl_widget
        image_worker = self.image_worker
        post_pulse_filter = image_worker._post_pulse_filter

        analysis_types = {value: key for key, value in
                          widget._analysis_types.items()}
        post_pulse_filter.update()
        self.assertEqual(AnalysisType.UNDEFINED, post_pulse_filter.analysis_type)
        self.assertTupleEqual((-np.inf, np.inf), post_pulse_filter._fom_range)

        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.ROI1_PULSE])
        widget._fom_range_le.setText("-1, 1")
        post_pulse_filter.update()
        self.assertEqual(AnalysisType.ROI1_PULSE, post_pulse_filter.analysis_type)
        self.assertEqual((-1, 1), post_pulse_filter._fom_range)

    def testCorrelationCtrlWidget(self):
        from karaboFAI.gui.ctrl_widgets.correlation_ctrl_widget import (
            _N_PARAMS, _DEFAULT_RESOLUTION)

        widget = self.gui.correlation_ctrl_widget

        for i in range(_N_PARAMS):
            combo_lst = [widget._table.cellWidget(i, 0).itemText(j)
                         for j in range(widget._table.cellWidget(i, 0).count())]
            self.assertEqual(
                list(widget._TOPIC_DATA_CATEGORIES[config["TOPIC"]]), combo_lst)

        scheduler = self.scheduler
        proc = scheduler._correlation_proc
        self._correlation_action.trigger()
        window = list(self.gui._windows.keys())[-1]

        proc.update()

        # test default
        self.assertEqual(AnalysisType(0), proc.analysis_type)
        self.assertEqual([""] * 4, proc._device_ids)
        self.assertEqual([""] * 4, proc._properties)

        # set new FOM
        widget._analysis_type_cb.setCurrentText('ROI1 + ROI2 (proj)')
        proc.update()
        self.assertEqual(AnalysisType.PROJ_ROI1_ADD_ROI2, proc.analysis_type)

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
        from karaboFAI.gui.ctrl_widgets.bin_ctrl_widget import (
            _DEFAULT_N_BINS, _DEFAULT_BIN_RANGE, _N_PARAMS
        )

        widget = self.gui.bin_ctrl_widget
        for i in range(_N_PARAMS):
            combo_lst = [widget._table.cellWidget(i, 0).itemText(j)
                         for j in range(widget._table.cellWidget(i, 0).count())]
            self.assertEqual(
                list(widget._TOPIC_DATA_CATEGORIES[config["TOPIC"]]), combo_lst)

        scheduler = self.scheduler
        proc = scheduler._bin_proc
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
        scheduler = self.scheduler
        proc = scheduler._statistics
        proc.update()

        analysis_types = {value: key for key, value in
                          widget._analysis_types.items()}

        self.assertEqual(AnalysisType.UNDEFINED, proc.analysis_type)
        self.assertTrue(proc._pulse_resolved)
        self.assertEqual(10, proc._num_bins)

        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.ROI1])
        proc.update()
        self.assertTrue(proc._reset)
        self.assertEqual(AnalysisType.ROI1_PULSE, proc.analysis_type)

        proc._reset = False
        widget._analysis_type_cb.setCurrentText(analysis_types[AnalysisType.AZIMUTHAL_INTEG])
        proc.update()
        self.assertTrue(proc._reset)
        self.assertEqual(AnalysisType.AZIMUTHAL_INTEG_PULSE,
                         proc.analysis_type)

        proc._reset = False
        widget._pulse_resolved_cb.setChecked(False)
        proc.update()
        self.assertTrue(proc._reset)
        self.assertEqual(AnalysisType.AZIMUTHAL_INTEG,
                         proc.analysis_type)

        proc._reset = False
        widget._analysis_type_cb.setCurrentIndex(1)
        proc.update()
        self.assertTrue(proc._reset)
        self.assertEqual(AnalysisType.ROI1, proc.analysis_type)

        widget._num_bins_le.setText("100")
        proc.update()
        self.assertEqual(100, proc._num_bins)

        proc._reset = False
        widget._reset_btn.clicked.emit()
        proc.update()
        self.assertTrue(proc._reset)

    @patch('karaboFAI.gui.ctrl_widgets.PumpProbeCtrlWidget.'
           'updateMetaData', MagicMock(return_value=True))
    @patch('karaboFAI.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.'
           'updateMetaData', MagicMock(return_value=True))
    @patch('karaboFAI.gui.ctrl_widgets.PumpProbeCtrlWidget.onStart', Mock())
    @patch('karaboFAI.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.onStart', Mock())
    @patch('karaboFAI.gui.ctrl_widgets.PumpProbeCtrlWidget.onStop', Mock())
    @patch('karaboFAI.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.onStop', Mock())
    @patch('karaboFAI.pipeline.Scheduler.resume', Mock())
    @patch('karaboFAI.pipeline.Scheduler.pause', Mock())
    def testStartStop(self):
        start_spy = QSignalSpy(self.gui.start_sgn)
        stop_spy = QSignalSpy(self.gui.stop_sgn)

        # -------------------------------------------------------------
        # test when the start action button is clicked
        # -------------------------------------------------------------

        self._start_action.trigger()

        self.gui.pump_probe_ctrl_widget.updateMetaData. \
            assert_called_once()
        self.gui.azimuthal_integ_ctrl_widget.updateMetaData. \
            assert_called_once()

        self.assertEqual(1, len(start_spy))

        self.gui.azimuthal_integ_ctrl_widget.onStart.assert_called_once()
        self.gui.pump_probe_ctrl_widget.onStart.assert_called_once()

        self.assertFalse(self._start_action.isEnabled())
        self.assertTrue(self._stop_action.isEnabled())

        # FIXME
        # self.bridge.activate.assert_called_once()

        # -------------------------------------------------------------
        # test when the start action button is clicked
        # -------------------------------------------------------------

        self._stop_action.trigger()

        self.gui.azimuthal_integ_ctrl_widget.onStop.assert_called_once()
        self.gui.pump_probe_ctrl_widget.onStop.assert_called_once()

        self.assertEqual(1, len(stop_spy))

        self.assertTrue(self._start_action.isEnabled())
        self.assertFalse(self._stop_action.isEnabled())

        # FIXME
        # self.bridge.pause.assert_called_once()


class TestJungFrauMainGuiCtrl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()   # ensure file
        config.load('JungFrau')

        cls.fai = FAI().init()

        cls.gui = cls.fai._gui
        cls.image_worker = cls.fai.image_worker

    @classmethod
    def tearDownClass(cls):
        cls.fai.terminate()

        wait_until_redis_shutdown()

    def setUp(self):
        self.assertTrue(self.gui.updateMetaData())

    def testAnalysisCtrlWidget(self):
        widget = self.gui.analysis_ctrl_widget

        self.assertFalse(widget._poi_index1_le.isEnabled())
        self.assertFalse(widget._poi_index2_le.isEnabled())

        image_proc = self.image_worker._image_proc

        # Although the widgets are disabled, they still send values to
        # the processor - test default values
        self.assertEqual(0, int(widget._poi_index1_le.text()))
        self.assertEqual(0, int(widget._poi_index2_le.text()))
        image_proc.update()
        self.assertEqual(0, image_proc._poi_indices[0])
        self.assertEqual(0, image_proc._poi_indices[1])

    def testPumpProbeCtrlWidget(self):
        widget = self.gui.pump_probe_ctrl_widget
        pp_proc = self.image_worker._pp_proc

        self.assertFalse(widget._on_pulse_le.isEnabled())
        self.assertFalse(widget._off_pulse_le.isEnabled())

        all_modes = {value: key for key, value in
                     widget._available_modes.items()}

        # we only test train-resolved detector specific configuration

        pp_proc.update()
        self.assertEqual(PumpProbeMode.UNDEFINED, pp_proc._mode)
        self.assertListEqual([-1], pp_proc._on_indices)
        self.assertListEqual([-1], pp_proc._off_indices)

        spy = QSignalSpy(widget._mode_cb.currentTextChanged)

        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.EVEN_TRAIN_ON])
        self.assertEqual(1, len(spy))

        pp_proc.update()
        self.assertEqual(PumpProbeMode(PumpProbeMode.EVEN_TRAIN_ON), pp_proc._mode)
        self.assertListEqual([-1], pp_proc._on_indices)
        self.assertListEqual([-1], pp_proc._off_indices)

        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.PRE_DEFINED_OFF])
        self.assertEqual(2, len(spy))
        # test on_pulse_le is still disabled, which will become enabled if the
        # detector is pulse-resolved
        self.assertFalse(widget._on_pulse_le.isEnabled())

        # PumpProbeMode.SAME_TRAIN is not available
        widget._mode_cb.setCurrentText(all_modes[PumpProbeMode.SAME_TRAIN])
        self.assertEqual(2, len(spy))

    def testGeometryCtrlWidget(self):
        widget = self.gui.geometry_ctrl_widget
        self.assertFalse(widget.isEnabled())

    def testStatisticsCtrlWidget(self):
        pass
        # TODO
