from unittest.mock import patch
import functools
import logging

import pytest
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QWidget

from extra_foam.pipeline.tests import _RawDataMixin

from extra_foam.special_suite import logger, mkQApp, special_suite_logger_name
from extra_foam.gui.plot_widgets import ImageViewF, PlotWidgetF
from extra_foam.special_suite.special_analysis_base import (
    _BaseAnalysisCtrlWidgetS, _SpecialAnalysisBase, create_special,
    ClientType, QThreadWorker, QThreadFoamClient, QThreadKbClient
)


app = mkQApp()

logger.setLevel('CRITICAL')


class TestSpecialAnalysisBase(_RawDataMixin):
    @pytest.fixture
    def win(self):
        class DummyCtrlWidget(_BaseAnalysisCtrlWidgetS):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.dummy_widget = QWidget()
                self._non_reconfigurable_widgets = [
                    self.dummy_widget
                ]

        class DummyProcessor(QThreadWorker):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self._dark_removed = False

            def process(self, data):
                """Override."""
                pass

            def onRemoveDark(self):
                """Override."""
                self._dark_removed = True

            def sources(self):
                return [
                    ("device1:output", "property1", 1),
                    ("device2", "property2", 0)
                ]

        class DummyImageView(ImageViewF):
            def __init__(self, *, parent=None):
                super().__init__(parent=parent)

            def updateF(self, data):
                """Override."""
                pass

        class DummyImageViewWithRoi(ImageViewF):
            def __init__(self, *, parent=None):
                super().__init__(has_roi=True, parent=parent)

            def updateF(self, data):
                """Override."""
                pass

        class DummyPlotWidget(PlotWidgetF):
            def __init__(self, *, parent=None):
                super().__init__(parent=parent)
                self._plot = self.plotCurve(name="dummy")

            def updateF(self, data):
                """Override."""
                pass

        @create_special(DummyCtrlWidget, DummyProcessor)
        class DummyWindow(_SpecialAnalysisBase):
            _title = "Dummy"
            _long_title = "Dummy analysis"
            _client_support = ClientType.BOTH

            def __init__(self, topic):
                super().__init__(topic)

                self._line = DummyPlotWidget(parent=self)
                self._view = DummyImageView(parent=self)
                self._view_with_roi = DummyImageViewWithRoi(parent=self)

                self.initUI()
                self.initConnections()
                self.startWorker()

            def initUI(self):
                """Override."""
                pass

            def initConnections(self):
                """Override."""
                pass

        # Note: startWorker is not patched as it is in other tests of
        #       concrete windows
        window = DummyWindow('DET')

        # See the win() fixture in conftest.py for an explanation of these steps
        window._worker_st._waiting_st.wait()
        yield window
        window.close()

    def testGeneral(self, win):
        assert 'DET' == win._ctrl_widget_st.topic

    def testPlotWidgets(self, win):
        assert 3 == len(win._plot_widgets_st)
        assert win._line in win._plot_widgets_st
        assert win._view in win._plot_widgets_st
        assert 2 == len(win._image_views_st)
        assert win._view in win._image_views_st
        assert win._view_with_roi in win._image_views_st

        with patch.object(win._view, "updateImage") as update_image:
            QTest.mouseClick(win._com_ctrl_st.auto_level_btn, Qt.LeftButton)
            update_image.assert_called_once()

        with patch.object(win._view, "updateF") as update_view:
            with patch.object(win._line, "updateF") as update_line:
                win.updateWidgetsST()
                # win._data is empty
                update_line.assert_not_called()
                update_view.assert_not_called()
                # patch win._worker_st.get()
                with patch.object(win._worker_st, "getOutputDataST"):
                    win.updateWidgetsST()
                    update_line.assert_called_once()
                    update_view.assert_called_once()

    def testCommonStartStopReset(self, win, caplog):
        com_ctrl_widget = win._com_ctrl_st
        ctrl_widget = win._ctrl_widget_st
        client = win._client_st
        worker = win._worker_st

        assert not com_ctrl_widget.stop_btn.isEnabled()

        assert client._endpoint_st is None
        with patch.object(client, "start") as client_start:
            with patch.object(win._plot_timer_st, "start") as timer_start:
                spy = QSignalSpy(win.started_sgn)
                QTest.mouseClick(com_ctrl_widget.start_btn, Qt.LeftButton)

                assert f"tcp://{com_ctrl_widget._hostname_le.text()}:" \
                       f"{com_ctrl_widget._port_le.text()}" == client._endpoint_st

                assert 2 == len(client._catalog_st)
                assert "device1:output property1" in client._catalog_st
                assert "device2 property2" in client._catalog_st

                assert 1 == len(spy)
                assert com_ctrl_widget.stop_btn.isEnabled()
                assert not com_ctrl_widget.start_btn.isEnabled()
                assert not com_ctrl_widget.load_dark_run_btn.isEnabled()

                assert not ctrl_widget.dummy_widget.isEnabled()

                client_start.assert_called_once()
                timer_start.assert_called_once()

        with patch.object(client, "terminateRunST") as client_stop:
            with patch.object(win._plot_timer_st, "stop") as timer_stop:
                spy = QSignalSpy(win.stopped_sgn)
                QTest.mouseClick(com_ctrl_widget.stop_btn, Qt.LeftButton)
                assert 1 == len(spy)
                assert not com_ctrl_widget.stop_btn.isEnabled()
                assert com_ctrl_widget.start_btn.isEnabled()
                assert com_ctrl_widget.load_dark_run_btn.isEnabled()

                assert ctrl_widget.dummy_widget.isEnabled()

                client_stop.assert_called_once()
                timer_stop.assert_called_once()

        caplog.set_level(logging.ERROR, logger=special_suite_logger_name)
        with patch.object(client, "start") as client_start:
            with patch.object(win._plot_timer_st, "start") as timer_start:
                with patch.object(worker, "sources") as mocked_sources:
                    mocked_sources.return_value = [("", "property1", 1)]
                    QTest.mouseClick(com_ctrl_widget.start_btn, Qt.LeftButton)
                    client_start.assert_not_called()
                    timer_start.assert_not_called()
                    assert "Empty source" in caplog.messages[-1]

                    mocked_sources.return_value = [("device", "", 0)]
                    QTest.mouseClick(com_ctrl_widget.start_btn, Qt.LeftButton)
                    client_start.assert_not_called()
                    timer_start.assert_not_called()
                    assert "Empty property" in caplog.messages[-1]

                    mocked_sources.return_value = [("device", "property", 2)]
                    QTest.mouseClick(com_ctrl_widget.start_btn, Qt.LeftButton)
                    client_start.assert_not_called()
                    timer_start.assert_not_called()
                    assert "Not understandable data type" in caplog.messages[-1]

        with patch.object(client, "onResetST") as client_reset:
            with patch.object(worker, "onResetST") as worker_reset:
                with patch.object(win._line, "reset") as line_reset:
                    with patch.object(win._view, "reset") as view_reset:
                        QTest.mouseClick(com_ctrl_widget.reset_btn, Qt.LeftButton)

                        client_reset.assert_called_once()
                        worker_reset.assert_called_once()
                        line_reset.assert_called_once()
                        view_reset.assert_called_once()

        with patch.object(worker._input_st, "clear") as input_clear:
            with patch.object(worker._output_st, "clear") as output_clear:
                worker._reset_st = False
                worker.onResetST()
                input_clear.assert_called_once()
                output_clear.assert_called_once()
                worker._reset_st = True

        with patch.object(client._transformer_st, "reset") as transformer_reset:
            with patch.object(client._output_st, "clear") as output_clear:
                client.onResetST()
                transformer_reset.assert_called_once()
                output_clear.assert_called_once()

    def testProcessFlow(self, win):
        worker = win._worker_st
        data = object()
        with patch.object(worker, "preprocess") as mocked_preprocess:
            with patch.object(worker, "process") as mocked_process:
                with patch.object(worker, "postprocess") as mocked_postprocess:
                    with patch.object(worker, "reset") as mocked_reset:
                        worker._reset_st = False
                        worker._processImpST(data)
                        mocked_preprocess.assert_called_once()
                        mocked_process.assert_called_once_with(data)
                        mocked_postprocess.assert_called_once()
                        mocked_reset.assert_not_called()

                        worker._reset_st = True
                        worker._processImpST(data)
                        mocked_reset.assert_called_once()
                        assert not worker._reset_st

    def testCommonDarkOperation(self, win):
        widget = win._com_ctrl_st
        worker = win._worker_st

        # recording dark
        assert not worker.recordingDark()  # default value
        QTest.mouseClick(widget.record_dark_btn, Qt.LeftButton)
        assert worker.recordingDark()
        assert widget.record_dark_btn.isChecked()
        QTest.mouseClick(widget.record_dark_btn, Qt.LeftButton)
        assert not worker.recordingDark()
        assert not widget.record_dark_btn.isChecked()

        # load dark run
        with patch.object(worker, "onLoadDarkRun") as load_dark_run:
            with patch('extra_foam.special_suite.special_analysis_base.QFileDialog.getExistingDirectory',
                       return_value=""):
                QTest.mouseClick(widget.load_dark_run_btn, Qt.LeftButton)
                load_dark_run.assert_not_called()

            with patch('extra_foam.special_suite.special_analysis_base.QFileDialog.getExistingDirectory',
                       return_value="/run/directory"):
                QTest.mouseClick(widget.load_dark_run_btn, Qt.LeftButton)
                load_dark_run.assert_called_with("/run/directory")

        # remove dark
        # patch.object does not work
        assert not worker._dark_removed
        QTest.mouseClick(widget.remove_dark_btn, Qt.LeftButton)
        assert worker._dark_removed

        # subtract dark
        assert worker.subtractDark()  # default value
        widget.dark_subtraction_cb.setChecked(False)
        assert not worker.subtractDark()

    def testRoiCtrl(self):
        pass

    def testSqueezeCameraImage(self, win):
        a1d = np.ones((4, ))
        a2d = np.ones((2, 1))
        a3d = np.ones((3, 3, 1))

        func = functools.partial(win._worker_st.squeezeToVector, 1234)

        assert func(None) is None
        assert func(a3d) is None

        ret_1d = func(a1d)
        np.testing.assert_array_equal(a1d, ret_1d)

        ret_2d = func(a2d)
        np.testing.assert_array_equal(a2d.squeeze(axis=-1), ret_2d)

    def testSqueezeToVector(self, win):
        a1d = np.ones((4, ))
        a2d = np.ones((2, 2))
        a3d = np.ones((3, 3, 1))
        a3d_f = np.ones((3, 3, 2))
        a4d = np.ones((2, 2, 2, 2))

        func = functools.partial(win._worker_st.squeezeToImage, 1234)

        assert func(None) is None
        assert func(a1d) is None
        assert func(a4d) is None

        ret_2d = func(a2d)
        np.testing.assert_array_equal(a2d, ret_2d)
        assert np.float32 == ret_2d.dtype

        ret_3d = func(a3d)
        np.testing.assert_array_equal(a3d.squeeze(axis=-1), ret_3d)
        assert np.float32 == ret_3d.dtype
        assert func(a3d_f) is None

    def testGetRoiData(self, win):
        worker = win._worker_st

        # test 2D array
        img = np.ones((4, 6))

        # test ROI geometry not specified
        worker._roi_geom_st = None
        roi = worker.getRoiData(img)
        assert img is roi
        roi = worker.getRoiData(img, copy=True)
        assert img is not roi
        np.testing.assert_array_equal(img, roi)

        # test with intersection
        worker._roi_geom_st = (1, 2, 2, 3)
        roi = worker.getRoiData(img)
        np.testing.assert_array_equal(img[2:5, 1:3], roi)

        # test without intersection
        worker._roi_geom_st = (-5, -6, 2, 3)
        roi = worker.getRoiData(img)
        np.testing.assert_array_equal(np.empty((0, 0)), roi)

        # test 3D array
        img = np.ones((3, 4, 6))

        # test with intersection
        worker._roi_geom_st = (1, 2, 2, 3)
        roi = worker.getRoiData(img)
        np.testing.assert_array_equal(img[:, 2:5, 1:3], roi)

        # test without intersection
        worker._roi_geom_st = (-5, -6, 2, 3)
        roi = worker.getRoiData(img)
        np.testing.assert_array_equal(np.empty((3, 0, 0)), roi)

    def testClientChange(self, win):
        ctrl_widget = win._com_ctrl_st
        worker = win._worker_st

        # Helper function to get the appropriate class for a client type
        def client_class(client_type):
            if client_type == ClientType.EXTRA_FOAM:
                return QThreadFoamClient
            elif client_type == ClientType.KARABO_BRIDGE:
                return QThreadKbClient
            else:
                raise RuntimeError("Unrecognized client type")

        # Check the client is initialized properly
        client_type = ClientType(ctrl_widget._client_type_cb.currentText())
        assert client_type == ctrl_widget.selected_client
        assert client_class(client_type) == type(win._client_st)
        assert worker.client_type == client_type

        # Which client is selected by default doesn't actually matter for
        # operation, but it's simpler to test if we know that this is the
        # default.
        assert client_type == ClientType.EXTRA_FOAM

        # Change the client type
        ctrl_widget._client_type_cb.setCurrentText(ClientType.KARABO_BRIDGE.value)
        client_type = ctrl_widget.selected_client
        assert client_class(client_type) == type(win._client_st)
        assert worker.client_type == client_type
