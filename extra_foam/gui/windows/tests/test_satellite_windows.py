import unittest
from unittest.mock import MagicMock, patch

from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QMainWindow

from extra_foam.logger import logger
from extra_foam.gui import mkQApp
from extra_foam.gui.windows.file_stream_controller_w import (
    FileStreamWindow
)

app = mkQApp()

logger.setLevel('CRITICAL')


class TestFileStreamWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = QMainWindow()  # dummy MainGUI
        cls.gui.registerSatelliteWindow = MagicMock()

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testWithParent(self):
        from extra_foam.gui.mediator import Mediator

        mediator = Mediator()
        spy = QSignalSpy(mediator.file_stream_initialized_sgn)
        win = FileStreamWindow(parent=self.gui)
        widget = win._ctrl_widget
        self.assertEqual('*', widget.port_le.text())
        self.assertTrue(widget.port_le.isReadOnly())
        self.assertEqual(1, len(spy))

        self.gui.registerSatelliteWindow.assert_called_once_with(win)
        self.gui.registerSatelliteWindow.reset_mock()

        mediator.connection_change_sgn.emit({
            "tcp://127.0.0.1:1234": 0,
            "tcp://127.0.0.1:1235": 1,
        })
        self.assertEqual(1234, win._port)

    def testStandAlone(self):
        with self.assertRaises(ValueError):
            FileStreamWindow(port=454522)

        win = FileStreamWindow(port=45452)
        widget = win._ctrl_widget
        self.assertEqual('45452', widget.port_le.text())
        self.assertIsNone(win._mediator)

        # test when win._rd_cal is None
        spy = QSignalSpy(win.file_server_started_sgn)
        widget.serve_start_btn.clicked.emit()
        self.assertEqual(0, len(spy))

        # test when win._rd_cal is not None
        win._rd_cal = MagicMock()
        widget.serve_start_btn.clicked.emit()

        # test populate sources
        with patch("extra_foam.gui.windows.file_stream_controller_w.load_runs") as lr:
            with patch("extra_foam.gui.windows.file_stream_controller_w.gather_sources") as gs:
                with patch.object(widget, "initProgressBar") as fpb:
                    with patch.object(widget, "fillSourceTables") as fst:
                        # test load_runs return (None, None)
                        win._rd_cal, win._rd_raw = object(), object()
                        gs.return_value = (set(), set(), set())
                        lr.return_value = (None, None)
                        widget.data_folder_le.setText("abc")
                        lr.assert_called_with("abc")
                        fpb.assert_called_once_with(0, 0)
                        fpb.reset_mock()
                        fst.assert_called_once_with(None, None)  # test _rd_cal and _rd_raw were reset
                        fst.reset_mock()

                        # test load_runs raises
                        lr.side_effect = ValueError
                        widget.data_folder_le.setText("efg")
                        fpb.assert_called_once_with(0, 0)
                        fpb.reset_mock()
                        fst.assert_called_once_with(None, None)
                        fst.reset_mock()

                    # test load_runs return
                    lr.side_effect = None
                    gs.return_value = ({"DET1": ["data.adc"]},
                                       {"output1": ["x", "y"]},
                                       {"motor1": ["actualPosition"], "motor2": ["actualCurrent"]})
                    widget.data_folder_le.setText("hij")
                    self.assertEqual("DET1", widget._detector_src_tb.item(0, 0).text())
                    cell_widget = widget._detector_src_tb.cellWidget(0, 1)
                    self.assertEqual(1, cell_widget.count())
                    self.assertEqual("data.adc", cell_widget.currentText())

                    self.assertEqual("output1", widget._instrument_src_tb.item(0, 0).text())
                    cell_widget = widget._instrument_src_tb.cellWidget(0, 1)
                    self.assertEqual(2, cell_widget.count())
                    self.assertEqual("x", cell_widget.currentText())

                    self.assertEqual("motor1", widget._control_src_tb.item(0, 0).text())
                    cell_widget = widget._control_src_tb.cellWidget(0, 1)
                    self.assertEqual(1, cell_widget.count())
                    self.assertEqual("actualPosition", cell_widget.currentText())

                    self.assertEqual("motor2", widget._control_src_tb.item(1, 0).text())
                    cell_widget = widget._control_src_tb.cellWidget(1, 1)
                    self.assertEqual(1, cell_widget.count())
                    self.assertEqual("actualCurrent", cell_widget.currentText())

                    # None is selected.
                    self.assertEqual(([], [], []), widget.getSourceLists())