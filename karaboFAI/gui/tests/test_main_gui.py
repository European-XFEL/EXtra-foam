import unittest

import numpy as np

from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from karabo_data.geometry import LPDGeometry

from karaboFAI.gui.main_gui import MainGUI
from karaboFAI.gui.windows import (
    CorrelationWindow, OverviewWindow, PumpProbeWindow, XasWindow
)

from karaboFAI.pipeline.data_model import ImageData, ProcessedData
from karaboFAI.config import config, FomName, AiNormalizer, PumpProbeMode

from . import mkQApp
app = mkQApp()


class TestMainGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = MainGUI('LPD')
        cls.proc = cls.gui._proc_worker

        cls._actions = cls.gui._tool_bar.actions()
        cls._overview_action = cls._actions[3]
        cls._pp_action = cls._actions[4]
        cls._correlation_action = cls._actions[5]
        cls._xas_action = cls._actions[6]

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def setUp(self):
        ImageData.clear()

    def testAnalysisCtrlWidget(self):
        widget = self.gui.analysis_ctrl_widget
        worker = self.gui._proc_worker

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertFalse(worker._ai_proc.isEnabled())
        self.assertFalse(worker._sample_degradation_proc.isEnabled())
        self.assertFalse(worker._pp_proc.isEnabled())

        QTest.mouseClick(widget.enable_ai_cb, Qt.LeftButton)
        self.assertTrue(self.gui.updateSharedParameters())

        self.assertTrue(worker._ai_proc.isEnabled())
        self.assertTrue(worker._sample_degradation_proc.isEnabled())
        self.assertTrue(worker._pp_proc.isEnabled())

    def testAiCtrlWidget(self):
        widget = self.gui.ai_ctrl_widget
        worker = self.gui._proc_worker

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
        widget._fom_itgt_range_le.setText(
            ','.join([str(x) for x in fom_itgt_range]))

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertAlmostEqual(worker._ai_proc.wavelength, photon_wavelength, 13)
        self.assertAlmostEqual(worker._ai_proc.sample_distance, sample_dist)
        self.assertTupleEqual(worker._ai_proc.integration_center, (cx, cy))
        self.assertEqual(worker._ai_proc.integration_method, itgt_method)
        self.assertEqual(worker._ai_proc.integration_points, itgt_pts)
        self.assertTupleEqual(worker._ai_proc.integration_range, itgt_range)
        self.assertEqual(worker._ai_proc.normalizer, ai_normalizer)
        self.assertTupleEqual(worker._ai_proc.auc_x_range, aux_x_range)

        self.assertTupleEqual(worker._correlation_proc.fom_itgt_range,
                              fom_itgt_range)

        self.assertTupleEqual(worker._sample_degradation_proc.fom_itgt_range,
                              fom_itgt_range)

        self.assertTupleEqual(worker._pp_proc.fom_itgt_range, fom_itgt_range)

    def testPumpProbeCtrlWidget(self):
        widget = self.gui.pump_probe_ctrl_widget
        worker = self.gui._proc_worker

        on_pulse_ids = [0, 2, 4, 6, 8]
        off_pulse_ids = [1, 3, 5, 7, 9]

        widget._laser_mode_cb.setCurrentIndex(1)
        widget._on_pulse_le.setText('0:10:2')
        widget._off_pulse_le.setText('1:10:2')
        QTest.mouseClick(widget.abs_difference_cb, Qt.LeftButton)
        self.assertTrue(worker._pp_proc.abs_difference)

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertEqual(PumpProbeMode.SAME_TRAIN, worker._pp_proc.mode)
        self.assertListEqual(on_pulse_ids, worker._pp_proc.on_pulse_ids)
        self.assertListEqual(off_pulse_ids, worker._pp_proc.off_pulse_ids)
        self.assertFalse(worker._pp_proc.abs_difference)

    def testDataCtrlWidget(self):
        widget = self.gui.data_ctrl_widget
        worker = self.gui._proc_worker
        daq = self.gui._daq_worker

        # test passing tcp hostname and port
        tcp_addr = "localhost:56565"

        widget._hostname_le.setText(tcp_addr.split(":")[0])
        widget._port_le.setText(tcp_addr.split(":")[1])

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertEqual(daq.server_tcp_sp, "tcp://" + tcp_addr)

        # test detector source name
        src_name_cb = widget._source_name_cb
        self.assertEqual(src_name_cb.currentText(),
                         worker._image_assembler.source_name)

        # test mono source name
        mono_src_cb = widget._mono_src_name_cb
        # test default value is set
        self.assertEqual(mono_src_cb.currentText(), worker._data_aggregator.mono_src)
        mono_src_cb.setCurrentIndex(1)
        self.assertEqual(mono_src_cb.currentText(), worker._data_aggregator.mono_src)

        # test xgm source name
        xgm_src_cb = widget._xgm_src_name_cb
        # test default value is set
        self.assertEqual(xgm_src_cb.currentText(), worker._data_aggregator.xgm_src)
        xgm_src_cb.setCurrentIndex(1)
        self.assertEqual(xgm_src_cb.currentText(), worker._data_aggregator.xgm_src)

        # test passing data source types
        for rbt in widget._source_type_rbts:
            QTest.mouseClick(rbt, Qt.LeftButton)
            self.assertTrue(self.gui.updateSharedParameters())
            self.assertEqual(worker._image_assembler.source_type,
                             widget._available_sources[rbt.text()])
        # make source type available
        QTest.mouseClick(widget._source_type_rbts[0], Qt.LeftButton)

    def testGeometryCtrlWidget(self):
        widget = self.gui.geometry_ctrl_widget
        worker = self.gui._proc_worker

        widget._geom_file_le.setText(config["GEOMETRY_FILE"])

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertIsInstance(worker._image_assembler._geom, LPDGeometry)

    def testCorrelationCtrlWidget(self):
        widget =self.gui.correlation_ctrl_widget
        worker = self.gui._proc_worker

        n_registered = len(self.gui._windows)
        self._correlation_action.trigger()
        window = list(self.gui._windows.keys())[-1]
        self.assertIsInstance(window, CorrelationWindow)
        self.assertEqual(n_registered + 1, len(self.gui._windows))

        # test FOM name
        fom = FomName.ROI1
        widget._figure_of_merit_cb.setCurrentIndex(fom)

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertEqual(fom, worker._correlation_proc.fom_name)

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
        app.processEvents()

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
        app.processEvents()

        # test unregister
        window.close()
        self.assertEqual(n_registered, len(self.gui._windows))

    def testOverviewWindow(self):
        widget = self.gui.analysis_ctrl_widget

        n_registered = len(self.gui._windows)
        self._overview_action.trigger()
        window = list(self.gui._windows.keys())[-1]
        self.assertIsInstance(window, OverviewWindow)
        self.assertEqual(n_registered + 1, len(self.gui._windows))

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
        worker = self.gui._proc_worker

        widget.updateSharedParameters()
        self.assertEqual((0, 2700), worker._image_assembler.pulse_id_range)

        widget._max_pulse_id_le.setText("1000")
        widget.updateSharedParameters()
        self.assertEqual((0, 1001), worker._image_assembler.pulse_id_range)

        # test unregister
        window.close()
        self.assertEqual(n_registered, len(self.gui._windows))

    def testPumpProbeWindow(self):
        n_registered = len(self.gui._windows)
        self._pp_action.trigger()
        window = list(self.gui._windows.keys())[-1]
        self.assertIsInstance(window, PumpProbeWindow)
        self.assertEqual(n_registered + 1, len(self.gui._windows))

    def testXasWindow(self):
        n_registered = len(self.gui._windows)
        self._xas_action.trigger()
        window = list(self.gui._windows.keys())[-1]
        self.assertIsInstance(window, XasWindow)
        self.assertEqual(n_registered + 1, len(self.gui._windows))
