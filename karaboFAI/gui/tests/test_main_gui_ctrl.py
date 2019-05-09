import unittest
from unittest.mock import patch
import tempfile
import os

import numpy as np

from PyQt5 import QtCore
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from karabo_data.geometry import LPDGeometry

from karaboFAI.services import FaiServer
from karaboFAI.pipeline.data_model import ImageData, ProcessedData
from karaboFAI.config import (
    config, AiNormalizer, FomName, DataSource, Projection1dNormalizer,
    PumpProbeMode, PumpProbeType
)


class TestMainGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from karaboFAI.config import _Config, ConfigWrapper

        # do not use the config file in the current computer
        cls.dir = tempfile.mkdtemp()
        _Config._filename = os.path.join(cls.dir, "config.json")
        ConfigWrapper()  # ensure file

        fai = FaiServer('LPD')
        cls.gui = fai._gui
        cls.app = fai.qt_app()
        cls.scheduler = fai._scheduler
        cls.bridge = fai._bridge

        cls._actions = cls.gui._tool_bar.actions()
        cls._imagetool_action = cls._actions[2]
        cls._pp_action = cls._actions[4]
        cls._correlation_action = cls._actions[5]
        cls._xas_action = cls._actions[6]
        cls._pulsed_ai_action = cls._actions[7]

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def setUp(self):
        ImageData.clear()

    def testAnalysisCtrlWidget(self):
        widget = self.gui.analysis_ctrl_widget
        scheduler = self.scheduler
        self._pulsed_ai_action.trigger()
        window = list(self.gui._windows.keys())[-1]

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertTrue(self.gui.updateSharedParameters())

        # --------------------------
        # test setting VIP pulse IDs
        # --------------------------

        vip_pulse_id1 = int(widget._vip_pulse_id1_le.text())
        self.assertEqual(vip_pulse_id1, window._vip1_ai.pulse_id)
        self.assertEqual(vip_pulse_id1, window._vip1_img.pulse_id)
        vip_pulse_id2 = int(widget._vip_pulse_id2_le.text())
        self.assertEqual(vip_pulse_id2, window._vip2_ai.pulse_id)
        self.assertEqual(vip_pulse_id2, window._vip2_img.pulse_id)

        vip_pulse_id1 = 10
        widget._vip_pulse_id1_le.setText(str(vip_pulse_id1))
        widget._vip_pulse_id1_le.returnPressed.emit()
        self.assertEqual(vip_pulse_id1, window._vip1_ai.pulse_id)
        self.assertEqual(vip_pulse_id1, window._vip1_img.pulse_id)
        vip_pulse_id2 = 20
        widget._vip_pulse_id2_le.setText(str(vip_pulse_id2))
        widget._vip_pulse_id2_le.returnPressed.emit()
        self.assertEqual(vip_pulse_id2, window._vip2_ai.pulse_id)
        self.assertEqual(vip_pulse_id2, window._vip2_img.pulse_id)

        # --------------------------
        # test setting max pulse ID
        # --------------------------
        scheduler = self.scheduler

        widget.updateSharedParameters()
        self.assertEqual((0, 2700), scheduler._image_assembler.pulse_id_range)

        widget._max_pulse_id_le.setText("1000")
        widget.updateSharedParameters()
        self.assertEqual((0, 1001), scheduler._image_assembler.pulse_id_range)

    def testAiCtrlWidget(self):
        widget = self.gui.ai_ctrl_widget
        scheduler = self.scheduler

        photon_energy = 12.4
        photon_wavelength = 1.0e-10
        sample_dist = 0.3
        cx = 1024
        cy = 512
        itgt_method = 'nosplit_csr'
        itgt_pts = 1024
        itgt_range = (0.1, 0.2)
        ai_normalizer = AiNormalizer.ROI2
        aux_x_range = (0.2, 0.3)
        fom_itgt_range = (0.3, 0.4)

        widget._photon_energy_le.setText(str(photon_energy))
        widget._sample_dist_le.setText(str(sample_dist))
        widget._cx_le.setText(str(cx))
        widget._cy_le.setText(str(cy))
        widget._itgt_method_cb.setCurrentText(itgt_method)
        widget._itgt_points_le.setText(str(itgt_pts))
        widget._itgt_range_le.setText(','.join([str(x) for x in itgt_range]))
        widget._normalizers_cb.setCurrentIndex(ai_normalizer)
        widget._auc_x_range_le.setText(','.join([str(x) for x in aux_x_range]))
        widget._fom_integ_range_le.setText(
            ','.join([str(x) for x in fom_itgt_range]))

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertFalse(scheduler._ai_proc.pulsed_ai)
        self.assertAlmostEqual(scheduler._ai_proc.wavelength, photon_wavelength, 13)
        self.assertAlmostEqual(scheduler._ai_proc.sample_distance, sample_dist)
        self.assertTupleEqual(scheduler._ai_proc.integration_center, (cx, cy))
        self.assertEqual(scheduler._ai_proc.integration_method, itgt_method)
        self.assertEqual(scheduler._ai_proc.integration_points, itgt_pts)
        self.assertTupleEqual(scheduler._ai_proc.integration_range, itgt_range)
        self.assertEqual(scheduler._ai_proc.normalizer, ai_normalizer)
        self.assertTupleEqual(scheduler._ai_proc.auc_x_range, aux_x_range)

        self.assertTupleEqual(scheduler._correlation_proc.fom_itgt_range,
                              fom_itgt_range)

        self.assertTupleEqual(scheduler._pp_proc.fom_itgt_range, fom_itgt_range)

        # activate "pulsed azimuthal integration"
        QTest.mouseClick(widget.pulsed_ai_cb, Qt.LeftButton,
                         pos=QtCore.QPoint(2, widget.pulsed_ai_cb.height()/2))
        self.assertTrue(self.gui.updateSharedParameters())
        self.assertTrue(scheduler._ai_proc.pulsed_ai)

    def testProject1dWidget(self):
        widget = self.gui.projection1d_ctrl_widget
        scheduler = self.scheduler

        self.assertEqual(Projection1dNormalizer.AUC,
                         scheduler._roi_proc.proj1d_normalizer)

        widget._fom_integ_range_le.setText("10, 20")
        widget._fom_integ_range_le.editingFinished.emit()
        self.assertEqual((10, 20), scheduler._roi_proc.proj1d_fom_integ_range)

        widget._auc_x_range_le.setText("30, 40")
        widget._auc_x_range_le.editingFinished.emit()
        self.assertEqual((30, 40), scheduler._roi_proc.proj1d_auc_x_range)

    def testPumpProbeCtrlWidget(self):
        widget = self.gui.pump_probe_ctrl_widget
        scheduler = self.scheduler

        self.assertEqual(1, scheduler._pp_proc.ma_window)

        on_pulse_ids = [0, 2, 4, 6, 8]
        off_pulse_ids = [1, 3, 5, 7, 9]

        self.assertTrue(scheduler._pp_proc.abs_difference)  # default is False
        QTest.mouseClick(widget._abs_difference_cb, Qt.LeftButton,
                         pos=QtCore.QPoint(2, widget._abs_difference_cb.height()/2))
        self.assertFalse(scheduler._pp_proc.abs_difference)

        widget._ma_window_le.setText(str(10))
        widget._ma_window_le.editingFinished.emit()
        self.assertEqual(10, scheduler._pp_proc.ma_window)

        self.assertEqual(PumpProbeType(0), scheduler._pp_proc.analysis_type)
        new_fom = PumpProbeType.ROI
        widget._analysis_type_cb.setCurrentIndex(new_fom)
        self.assertEqual(PumpProbeType(new_fom),
                         scheduler._pp_proc.analysis_type)

        # test default FOM name
        self.assertTrue(self.gui.updateSharedParameters())
        self.assertEqual(PumpProbeMode.UNDEFINED, scheduler._pp_proc.mode)

        # assign new values
        new_mode = PumpProbeMode.EVEN_TRAIN_ON
        widget._mode_cb.setCurrentIndex(new_mode)
        widget._on_pulse_le.setText('0:10:2')
        widget._off_pulse_le.setText('1:10:2')
        widget._ma_window_le.editingFinished.emit()

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertEqual(PumpProbeMode(new_mode), scheduler._pp_proc.mode)
        self.assertListEqual(on_pulse_ids, scheduler._pp_proc.on_pulse_ids)
        self.assertListEqual(off_pulse_ids, scheduler._pp_proc.off_pulse_ids)

    def testXasCtrlWidget(self):
        widget = self.gui.xas_ctrl_widget
        scheduler = self.scheduler

        # check initial value is set
        self.assertEqual(int(widget._nbins_le.text()), scheduler._xas_proc.n_bins)
        # set another value
        widget._nbins_le.setText("40")
        widget._nbins_le.editingFinished.emit()
        self.assertEqual(40, scheduler._xas_proc.n_bins)

    @patch.dict(config._data, {"SOURCE_NAME_BRIDGE": ["E", "F", "G"],
                               "SOURCE_NAME_FILE": ["A", "B"]})
    def testDataCtrlWidget(self):
        widget = self.gui.data_ctrl_widget
        scheduler = self.scheduler
        bridge = self.bridge

        # test passing tcp hostname and port
        hostname = config['SERVER_ADDR']
        port = config['SERVER_PORT']
        self.assertEqual(f"tcp://{hostname}:{port}", bridge._endpoint)
        widget._hostname_le.setText('127.0.0.1')
        self.assertEqual(f"tcp://127.0.0.1:{port}", bridge._endpoint)
        widget._port_le.setText('12345')
        self.assertEqual("tcp://127.0.0.1:12345", bridge._endpoint)

        # test passing data source types and detector source name

        source_type = DataSource.FILE
        widget._source_type_cb.setCurrentIndex(source_type)
        self.assertEqual(source_type, scheduler._image_assembler.source_type)
        self.assertEqual(source_type, scheduler._source_type)
        self.assertEqual(source_type, bridge._source_type)
        self.assertEqual("A", scheduler._image_assembler.source_name)
        items = []
        for i in range(widget._detector_src_cb.count()):
            items.append(widget._detector_src_cb.itemText(i))
        self.assertListEqual(["A", "B"], items)

        # change source_type from FILE to BRIDGE
        source_type = DataSource.BRIDGE
        widget._source_type_cb.setCurrentIndex(source_type)
        self.assertEqual(source_type, scheduler._image_assembler.source_type)
        self.assertEqual(source_type, scheduler._source_type)
        self.assertEqual(source_type, bridge._source_type)
        self.assertEqual("E", scheduler._image_assembler.source_name)
        items = []
        for i in range(widget._detector_src_cb.count()):
            items.append(widget._detector_src_cb.itemText(i))
        self.assertListEqual(["E", "F", "G"], items)

        # test mono source name
        mono_src_cb = widget._mono_src_cb
        # test default value is set
        self.assertEqual(mono_src_cb.currentText(), scheduler._data_aggregator.mono_src)
        mono_src_cb.setCurrentIndex(1)
        self.assertEqual(mono_src_cb.currentText(), scheduler._data_aggregator.mono_src)

        # test xgm source name
        xgm_src_cb = widget._xgm_src_cb
        # test default value is set
        self.assertEqual(xgm_src_cb.currentText(), scheduler._data_aggregator.xgm_src)
        xgm_src_cb.setCurrentIndex(1)
        self.assertEqual(xgm_src_cb.currentText(), scheduler._data_aggregator.xgm_src)

    def testGeometryCtrlWidget(self):
        widget = self.gui.geometry_ctrl_widget
        scheduler = self.scheduler

        widget._geom_file_le.setText(config["GEOMETRY_FILE"])

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertIsInstance(scheduler._image_assembler._geom, LPDGeometry)

    def testCorrelationCtrlWidget(self):
        widget = self.gui.correlation_ctrl_widget
        scheduler = self.scheduler
        self._correlation_action.trigger()
        window = list(self.gui._windows.keys())[-1]

        self.assertEqual(FomName(0), scheduler._correlation_proc.fom_name)
        new_fom = FomName.ROI1
        widget._figure_of_merit_cb.setCurrentIndex(new_fom)
        self.assertEqual(FomName(new_fom),
                         scheduler._correlation_proc.fom_name)

        # test default FOM name
        self.assertTrue(self.gui.updateSharedParameters())

        # test the correlation param table
        expected_params = []
        for i in range(widget._n_params):
            widget._table.cellWidget(i, 0).setCurrentIndex(1)
            self.assertListEqual(expected_params,
                                 ProcessedData(1).correlation.get_params())
            widget._table.cellWidget(i, 1).setCurrentIndex(1)
            param = f'param{i}'
            expected_params.append(param)

            resolution = (i+1)*5 if i < 2 else 0.0
            resolution_le = widget._table.cellWidget(i, 3)
            resolution_le.setText(str(resolution))
            resolution_le.returnPressed.emit()

            if resolution > 0:
                _, _, info = getattr(ProcessedData(1).correlation, param)
                self.assertEqual(resolution, info['resolution'])
            else:
                _, _, info = getattr(ProcessedData(1).correlation, param)
                self.assertNotIn('resolution', info)

        # test data visualization
        # the upper two plots have error bars
        data = ProcessedData(1, images=np.arange(480).reshape(120, 2, 2))
        for i in range(1000):
            data.correlation.param0 = (int(i/5), 100*i)
            data.correlation.param1 = (int(i/5), -100*i)
            data.correlation.param2 = (i, i+1)
            data.correlation.param3 = (i, -i)
        self.gui._data.set(data)
        window.update()
        self.app.processEvents()

        # change the resolutions
        for i in range(widget._n_params):
            resolution = (i+1)*5 if i >= 2 else 0.0
            resolution_le = widget._table.cellWidget(i, 3)
            resolution_le.setText(str(resolution))
            resolution_le.returnPressed.emit()

        # the data is cleared after the resolutions were changed
        # now the lower two plots have error bars but the upper ones do not
        for i in range(1000):
            data.correlation.param2 = (int(i/5), 100*i)
            data.correlation.param3 = (int(i/5), -100*i)
            data.correlation.param0 = (i, i+1)
            data.correlation.param1 = (i, -i)
        self.gui._data.set(data)
        window.update()
        self.app.processEvents()
