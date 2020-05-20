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
from extra_foam.gui.windows import PulseOfInterestWindow
from extra_foam.config import config, AnalysisType, PumpProbeMode
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


class TestMainGui(unittest.TestCase):
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

        # test loading meta data
        mediator = widget._mediator
        mediator.onMaWindowChange(111)
        mediator.onPoiIndexChange(0, 22)
        mediator.onPoiIndexChange(1, 33)
        widget.loadMetaData()
        self.assertEqual("111", widget._ma_window_le.text())
        self.assertEqual("22", widget._poi_index_les[0].text())
        self.assertEqual("33", widget._poi_index_les[1].text())

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

        # test loading meta data
        mediator = widget._mediator
        mediator.onFomFilterAnalysisTypeChange(AnalysisType.UNDEFINED)
        mediator.onFomFilterRangeChange((-10, 10))
        mediator.onFomFilterPulseResolvedChange(True)
        widget.loadMetaData()
        self.assertEqual("", widget._analysis_type_cb.currentText())
        self.assertEqual("-10, 10", widget._fom_range_le.text())
        self.assertEqual(True, widget._pulse_resolved_cb.isChecked())

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

        actions = self.gui._tool_bar.actions()
        start_action = actions[0]
        stop_action = actions[1]

        histogram_action = actions[7]
        self.assertEqual("Histogram", histogram_action.text())
        histogram_action.trigger()
        histogram_ctrl_widget = list(self.gui._plot_windows.keys())[-1]._ctrl_widgets[0]
        histogram_ctrl_widget.updateMetaData.reset_mock()

        start_action.trigger()

        # test a ctrl widget own by the ImageToolWindow
        azimuthal_integ_ctrl_widget = self.gui._image_tool._azimuthal_integ_1d_view._ctrl_widget
        geometry_ctrl_widget = self.gui._image_tool._geometry_view._ctrl_widget
        pump_probe_ctrl_widget = self.gui.pump_probe_ctrl_widget
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


class TestMainGuiTs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('ePix100', 'MID')

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

        # test loading meta data
        # Test if the meta data is invalid.
        mediator = widget._mediator
        mediator.onPoiIndexChange(0, 22)
        mediator.onPoiIndexChange(1, 33)
        widget.loadMetaData()
        self.assertEqual("0", widget._poi_index_les[0].text())
        self.assertEqual("0", widget._poi_index_les[1].text())

    def testPumpProbeCtrlWidget(self):
        widget = self.gui.pump_probe_ctrl_widget
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

    def testFomFilterCtrlWidget(self):
        widget = self.gui.fom_filter_ctrl_widget
        filter_pulse = self.pulse_worker._filter
        filter_train = self.train_worker._filter

        # test default

        self.assertFalse(widget._pulse_resolved_cb.isChecked())
        self.assertFalse(widget._pulse_resolved_cb.isEnabled())

        filter_pulse.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_pulse.analysis_type)
        self.assertTupleEqual((-np.inf, np.inf), filter_pulse._fom_range)
        filter_train.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_pulse.analysis_type)
        self.assertTupleEqual((-np.inf, np.inf), filter_pulse._fom_range)

        # test set new

        widget._analysis_type_cb.setCurrentText(widget._analysis_types_inv[AnalysisType.ROI_FOM])
        widget._fom_range_le.setText("-2, 2")
        filter_pulse.update()
        self.assertEqual(AnalysisType.UNDEFINED, filter_pulse.analysis_type)
        self.assertEqual((-np.inf, np.inf), filter_pulse._fom_range)
        filter_train.update()
        self.assertEqual(AnalysisType.ROI_FOM, filter_train.analysis_type)
        self.assertTupleEqual((-2, 2), filter_train._fom_range)

        # test loading meta data
        # test if the meta data is invalid
        mediator = widget._mediator
        mediator.onFomFilterPulseResolvedChange(True)
        widget.loadMetaData()
        self.assertEqual(False, widget._pulse_resolved_cb.isChecked())
