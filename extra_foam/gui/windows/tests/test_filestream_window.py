import unittest
from unittest.mock import MagicMock, patch

from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QMainWindow

from extra_foam.logger import logger_stream as logger
from extra_foam.gui import mkQApp
from extra_foam.gui.windows.file_stream_w import FileStreamWindow, DataSelector, RunData

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

        # Disconnect all slots connected to this signal. We do this to prevent
        # any un-GC'ed objects connected to it from previous tests from
        # executing their slots.
        #
        # This came up when TestMainGuiCtrl::testFomFilterCtrlWidget() executed
        # immediately before testWithParent(). That test happens to create an
        # entire Foam() instance, which creates a DataSourceWidget somewhere in
        # the object tree, which connects to this signal. The slot,
        # DataSourceWidget.updateMetaData(), ends up making a call to Redis. So
        # when testWithParent() ran and emitted this signal, that slot was
        # called because the DataSourceWidget hadn't been GC'ed yet. This
        # particular test case doesn't spin up Redis, so the slot would fail and
        # raise an exception.
        try:
            mediator.file_stream_initialized_sgn.disconnect()
        except TypeError:
            # This call fails if there are no connections
            pass

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

        with patch.object(win._ctrl_widget, "close") as mocked_close:
            with patch.object(self.gui, "unregisterSatelliteWindow", create=True):
                win.close()
                mocked_close.assert_called_once()

    def testStandAlone(self):
        with self.assertRaises(ValueError):
            FileStreamWindow(port=454522)

        win = FileStreamWindow(port=45449)
        widget = win._ctrl_widget
        module = "extra_foam.gui.windows.file_stream_w"
        self.assertEqual('45449', widget.port_le.text())
        self.assertIsNone(win._mediator)

        with patch(f"{module}.Process.start") as mocked_start:
            # test when win._rd_cal is None
            spy = QSignalSpy(win.file_server_started_sgn)
            widget.serve_start_btn.clicked.emit()
            self.assertEqual(0, len(spy))
            mocked_start.assert_not_called()

            # test when win._rd is not None
            win._rd = MagicMock()
            widget.serve_start_btn.clicked.emit()
            self.assertEqual(1, len(spy))
            mocked_start.assert_called_once()

        # test populate sources
        with (patch(f"{module}.RunDirectory") as RunDirectory,
              patch(f"{module}.open_run") as open_run,
              patch(f"{module}.gather_sources") as gs,
              patch.object(widget, "initProgressControl") as fpc):
            with patch.object(widget, "fillSourceTables") as fst:
                win._rd = object()
                gs.return_value = (set(), set(), set())

                # Load from a proposal/run number. Loading should not occur
                # until all necessary fields have been set.
                widget.run_source_cb.setCurrentText(DataSelector.RUN_NUMBER.value)
                open_run.assert_not_called()
                widget.proposal_number_le.setText("1234")
                open_run.assert_not_called()
                widget.run_number_le.setText("314")
                open_run.assert_called_once()
                fst.assert_called_once()

                # Changing the run data should also trigger a reload
                open_run.reset_mock()
                fst.reset_mock()
                widget.run_data_cb.setCurrentText(RunData.RAW.value)
                open_run.assert_called_with(1234, 314, data=RunData.RAW.value)
                fst.assert_called_once()

                # Test loading from a run directory
                widget.run_source_cb.setCurrentText(DataSelector.RUN_DIR.value)
                widget.data_folder_le.setText("abc")
                RunDirectory.assert_called_with("abc")

                # Changing back to the proposal/run number selector should load
                # the run with the previously-set values.
                open_run.reset_mock()
                widget.run_source_cb.setCurrentText(DataSelector.RUN_NUMBER.value)
                open_run.assert_called_once()

            with patch("extra_foam.gui.windows.file_stream_w.run_info",
                       return_value=(100, 1001, 1100)):
                gs.return_value = ({"DET1": ["data.adc"]},
                                   {"output1": ["x", "y"]},
                                   {"motor1": ["actualPosition"], "motor2": ["actualCurrent"]})
                fpc.reset_mock()
                widget.data_folder_le.setText("hij")
                fpc.assert_called_once_with(1001, 1100)

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

    def testInitProgressControl(self):
        win = FileStreamWindow(port=45452)
        widget = win._ctrl_widget

        with patch.object(widget.tid_progress_br, "reset") as mocked_reset:
            widget.initProgressControl(1001, 1100)
            self.assertEqual(1001, widget.tid_start_sld.minimum())
            self.assertEqual(1100, widget.tid_start_sld.maximum())
            self.assertEqual(1001, widget.tid_start_sld.value())
            self.assertEqual("1001", widget.tid_start_lb.text())

            self.assertEqual(1001, widget.tid_end_sld.minimum())
            self.assertEqual(1100, widget.tid_end_sld.maximum())
            self.assertEqual(1100, widget.tid_end_sld.value())
            self.assertEqual("1100", widget.tid_end_lb.text())

            self.assertEqual(1001, widget.tid_progress_br.minimum())
            self.assertEqual(1100, widget.tid_progress_br.maximum())
            self.assertEqual(1000, widget.tid_progress_br.value())
            self.assertEqual(3, mocked_reset.call_count)
            mocked_reset.reset_mock()

            # test set individual sliders
            widget.tid_start_sld.setValue(1050)
            self.assertEqual(1050, widget.tid_progress_br.minimum())
            self.assertEqual("1050", widget.tid_start_lb.text())
            mocked_reset.assert_called_once()
            mocked_reset.reset_mock()
            # test last tid is set to be smaller than the first one
            widget.tid_end_sld.setValue(1049)
            self.assertEqual(1050, widget.tid_end_sld.value())
            self.assertEqual(1050, widget.tid_progress_br.maximum())
            self.assertEqual("1050", widget.tid_end_lb.text())
            mocked_reset.assert_called_once()
            mocked_reset.reset_mock()

            # test reset
            widget.initProgressControl(-1, -1)
            self.assertEqual(-1, widget.tid_start_sld.minimum())
            self.assertEqual(-1, widget.tid_start_sld.maximum())
            self.assertEqual(-1, widget.tid_start_sld.value())
            self.assertEqual("", widget.tid_start_lb.text())

            self.assertEqual(-1, widget.tid_end_sld.minimum())
            self.assertEqual(-1, widget.tid_end_sld.maximum())
            self.assertEqual(-1, widget.tid_end_sld.value())
            self.assertEqual("", widget.tid_end_lb.text())

            self.assertEqual(-1, widget.tid_progress_br.minimum())
            self.assertEqual(-1, widget.tid_progress_br.maximum())
            self.assertEqual(-2, widget.tid_progress_br.value())
            mocked_reset.assert_called()
