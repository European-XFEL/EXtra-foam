import unittest
from enum import IntEnum

from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

from karabo_data.geometry import LPDGeometry

from karaboFAI.gui.main_gui import MainGUI
from karaboFAI.config import config, FomName, OpLaserMode


class Win(IntEnum):
    ImageTool = 2
    Overview = 3
    Correlation = 4


class TestMainGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = MainGUI('LPD')
        cls.proc = cls.gui._proc_worker

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testOpenCloseWindows(self):
        actions = self.gui._tool_bar.actions()

        count = 0
        for idx in (Win.ImageTool, Win.Overview, Win.Correlation):
            count += 1
            actions[idx].trigger()
            self.assertEqual(count, len(self.gui._windows))

        # test Window instances will be unregistered after being closed
        with self.assertRaises(StopIteration):
            i = 0
            while i < 100 and self.gui._windows.keys():
                key = next(self.gui._windows.keys())
                key.close()
                i += 1

    def testAnalysisCtrlWidget(self):
        widget = self.gui.analysis_ctrl_widget
        worker = self.gui._proc_worker

        self.assertFalse(worker._enable_ai)

        QTest.mouseClick(widget.enable_ai_cb, Qt.LeftButton)
        self.assertTrue(worker._enable_ai)

    def testAiCtrlWidget(self):
        widget = self.gui.ai_ctrl_widget
        worker = self.gui._proc_worker

        photon_energy = 12.4
        photon_wavelength = 1.0e-10
        sample_dist = 0.3
        cx = 1024
        cy = 512
        # TODO: test integration method
        itgt_pts = 1024
        itgt_range = (0.1, 0.2)
        # TODO: test normalizer
        aux_x_range = (0.2, 0.3)
        fom_itgt_range = (0.3, 0.4)

        widget._photon_energy_le.setText(str(photon_energy))
        widget._sample_dist_le.setText(str(sample_dist))
        widget._cx_le.setText(str(cx))
        widget._cy_le.setText(str(cy))
        widget._itgt_points_le.setText(str(itgt_pts))
        widget._itgt_range_le.setText(','.join([str(x) for x in itgt_range]))
        widget._auc_x_range_le.setText(','.join([str(x) for x in aux_x_range]))
        widget._fom_itgt_range_le.setText(
            ','.join([str(x) for x in fom_itgt_range]))

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertAlmostEqual(worker.wavelength_sp, photon_wavelength, 13)
        self.assertAlmostEqual(worker.sample_distance_sp, sample_dist)
        self.assertTupleEqual(worker.poni_sp, (cy, cx))
        self.assertEqual(worker.integration_points_sp, itgt_pts)
        # self.assertTupleEqual(worker.integration_range_sp, itgt_range)
        self.assertTupleEqual(worker._correlation_proc.auc_x_range,
                              aux_x_range)
        self.assertTupleEqual(worker._sample_degradation_proc.auc_x_range,
                              aux_x_range)
        self.assertTupleEqual(worker._laser_on_off_proc.auc_x_range,
                              aux_x_range)
        self.assertTupleEqual(worker._correlation_proc.fom_itgt_range,
                              fom_itgt_range)
        self.assertTupleEqual(worker._sample_degradation_proc.fom_itgt_range,
                              fom_itgt_range)
        self.assertTupleEqual(worker._laser_on_off_proc.fom_itgt_range,
                              fom_itgt_range)

    def testPumpProbeCtrlWidget(self):
        widget = self.gui.pump_probe_ctrl_widget
        worker = self.gui._proc_worker

        on_pulse_ids = [0, 2, 4, 6, 8]
        off_pulse_ids = [1, 3, 5, 7, 9]
        moving_average = 10

        widget._laser_mode_cb.setCurrentIndex(1)
        widget._on_pulse_le.setText('0:10:2')
        widget._off_pulse_le.setText('1:10:2')
        widget._moving_avg_window_le.setText(str(moving_average))
        QTest.mouseClick(widget.abs_difference_cb, Qt.LeftButton)
        self.assertTrue(worker._laser_on_off_proc.abs_difference)

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertEqual(OpLaserMode.NORMAL,
                         worker._laser_on_off_proc.laser_mode)
        self.assertListEqual(on_pulse_ids,
                             worker._laser_on_off_proc.on_pulse_ids)
        self.assertListEqual(off_pulse_ids,
                             worker._laser_on_off_proc.off_pulse_ids)
        self.assertEqual(moving_average,
                         worker._laser_on_off_proc.moving_avg_window)
        self.assertFalse(worker._laser_on_off_proc.abs_difference)

    def testDataCtrlWidget(self):
        widget = self.gui.data_ctrl_widget
        daq = self.gui._daq_worker

        tcp_addr = "localhost:56565"

        widget._hostname_le.setText(tcp_addr.split(":")[0])
        widget._port_le.setText(tcp_addr.split(":")[1])

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertEqual(daq.server_tcp_sp, "tcp://" + tcp_addr)

    def testGeometryCtrlWidget(self):
        widget = self.gui.geometry_ctrl_widget
        worker = self.gui._proc_worker

        widget._geom_file_le.setText(config["GEOMETRY_FILE"])

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertIsInstance(worker.geom_sp, LPDGeometry)

    def testCorrelationCtrlWidget(self):
        widget =self.gui.correlation_ctrl_widget
        worker = self.gui._proc_worker

        fom = FomName.ROI1

        widget._figure_of_merit_cb.setCurrentIndex(fom)

        self.assertTrue(self.gui.updateSharedParameters())

        self.assertEqual(fom, worker._correlation_proc.fom_name)
