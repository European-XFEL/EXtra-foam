"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
import functools
from queue import Empty, Full
import sys
from threading import Condition
import traceback
from weakref import WeakKeyDictionary

import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, Qt, QThread, QTimer
from PyQt5.QtGui import QColor, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QGridLayout, QLabel, QMainWindow,
    QPushButton, QSplitter, QWidget
)

from extra_data import RunDirectory
from karabo_bridge import Client as KaraboBridgeClient

from .. import __version__
from ..config import config
from ..database import SourceCatalog
from ..gui.ctrl_widgets.smart_widgets import SmartLineEdit
from ..gui.plot_widgets import ImageViewF
from ..gui.misc_widgets import GuiLogger, set_button_color
from ..pipeline.f_queue import SimpleQueue
from ..pipeline.f_transformer import DataTransformer
from ..logger import logger_suite as logger
from ..pipeline.f_zmq import FoamZmqClient
from ..pipeline.exceptions import ProcessingError


_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']


class _SharedCtrlWidgetS(QFrame):
    """Control widget used in all special analysis window.

    It provides connection setup, start/stop/reset control, dark recording/
    subtraction control as well as other common GUI controls.
    """

    def __init__(self, parent=None, *, with_dark=True, with_levels=True):
        """Initialization.

        :param bool with_dark: whether the dark recording/subtraction control
            widgets are included. For special analysis which makes use of
            the processed data in EXtra-foam, dark recording/subtraction
            control is not needed since it is done in the ImageTool.
        :param bool with_levels: whether the image levels related control
            widgets are included.
        """
        super().__init__(parent=parent)

        self._with_dark = with_dark
        self._with_levels = with_levels

        self._hostname_le = SmartLineEdit("127.0.0.1")
        self._hostname_le.setMinimumWidth(100)
        self._port_le = SmartLineEdit(str(config["BRIDGE_PORT"]))
        self._port_le.setValidator(QIntValidator(0, 65535))

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset")

        self.record_dark_btn = QPushButton("Record dark")
        self.record_dark_btn.setCheckable(True)
        self.load_dark_run_btn = QPushButton("Load dark run")
        self.remove_dark_btn = QPushButton("Remove dark")
        self.dark_subtraction_cb = QCheckBox("Subtract dark")
        self.dark_subtraction_cb.setChecked(True)

        self.auto_level_btn = QPushButton("Auto level")

        self.initUI()

        self.setFrameStyle(QFrame.StyledPanel)

    def initUI(self):
        set_button_color(self.start_btn, QColor(Qt.green))
        set_button_color(self.stop_btn, QColor(Qt.red))
        set_button_color(self.reset_btn, QColor(Qt.yellow))

        layout = QGridLayout()
        AR = Qt.AlignRight

        i_row = 0
        layout.addWidget(QLabel("Hostname: "), i_row, 0, AR)
        layout.addWidget(self._hostname_le, i_row, 1)
        layout.addWidget(QLabel("Port: "), i_row, 2, AR)
        layout.addWidget(self._port_le, i_row, 3)

        i_row += 1
        layout.addWidget(self.start_btn, i_row, 1)
        layout.addWidget(self.stop_btn, i_row, 2)
        layout.addWidget(self.reset_btn, i_row, 3)

        if self._with_dark:
            i_row += 1
            layout.addWidget(self.record_dark_btn, i_row, 0)
            layout.addWidget(self.load_dark_run_btn, i_row, 1)
            layout.addWidget(self.remove_dark_btn, i_row, 2)
            layout.addWidget(self.dark_subtraction_cb, i_row, 3)

        if self._with_levels:
            i_row += 1
            layout.addWidget(self.auto_level_btn, i_row, 3)

        self.setLayout(layout)

    def endpoint(self):
        return f"tcp://{self._hostname_le.text()}:{self._port_le.text()}"

    def updateDefaultPort(self, port: int):
        self._port_le.setText(str(port))

    def onStart(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._hostname_le.setEnabled(False)
        self._port_le.setEnabled(False)
        self.load_dark_run_btn.setEnabled(False)

    def onStop(self):
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self._hostname_le.setEnabled(True)
        self._port_le.setEnabled(True)
        self.load_dark_run_btn.setEnabled(True)


class _BaseAnalysisCtrlWidgetS(QFrame):
    """The base class should be inherited by all concrete ctrl widgets."""
    def __init__(self, topic, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._topic = topic

        # widgets whose values are not allowed to change after the "start"
        # button is clicked
        self._non_reconfigurable_widgets = []

        self.setFrameStyle(QFrame.StyledPanel)

    @abc.abstractmethod
    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    @abc.abstractmethod
    def initConnections(self):
        """Initialization of signal-slot connections."""
        raise NotImplementedError

    def onStart(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(False)

    def onStop(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(True)

    def addRows(self, layout, widgets):
        AR = Qt.AlignRight
        index = 0
        for name, widget in widgets:
            if name:
                layout.addWidget(QLabel(f"{name}: "), index, 0, AR)
            if isinstance(widget, QWidget):
                layout.addWidget(widget, index, 1)
            else:
                layout.addLayout(widget, index, 1)
            index += 1


class _ThreadLogger(QObject):
    """Logging in the thread."""
    # post messages in the main thread
    debug_sgn = pyqtSignal(str)
    info_sgn = pyqtSignal(str)
    warning_sgn = pyqtSignal(str)
    error_sgn = pyqtSignal(str)

    def debug(self, msg):
        """Log debug information in the main GUI."""
        self.debug_sgn.emit(msg)

    def info(self, msg):
        """Log info information in the main GUI."""
        self.info_sgn.emit(msg)

    def warning(self, msg):
        """Log warning information in the main GUI."""
        self.warning_sgn.emit(msg)

    def error(self, msg):
        """Log error information in the main GUI."""
        self.error_sgn.emit(msg)

    def logOnMainThread(self, instance):
        self.debug_sgn.connect(instance.onDebugReceived)
        self.info_sgn.connect(instance.onInfoReceived)
        self.warning_sgn.connect(instance.onWarningReceived)
        self.error_sgn.connect(instance.onErrorReceived)


class QThreadWorker(QObject):
    """Base class of worker running in a thread.

    Attributes:
        _recording_dark (bool): True for recording dark.
        _subtract_dark (bool): True for applying dark subtraction.
    """

    def __init__(self, queue, condition, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._input = queue
        self._cv = condition

        self._output = SimpleQueue(maxsize=1)
        self._running = False

        self._recording_dark = False
        self._subtract_dark = True

        self.log = _ThreadLogger()

    def reset(self):
        """Reset the internal state of process worker."""
        self._input.clear()
        self._output.clear()
        self.onReset()

    def onReset(self):
        """Interface method."""
        pass

    def onRecordDarkToggled(self, state: bool):
        self._recording_dark = state

    def onSubtractDarkToggled(self, state: bool):
        self._subtract_dark = state

    def _loadRunDirectory(self, dirpath):
        """Load a run directory.

        This method should be called inside the onLoadDarkRun
        implementation of the child class.
        """
        try:
            run = RunDirectory(dirpath)
            self.log.info(f"Loaded run from {dirpath}")
            return run
        except Exception as e:
            self.log.error(repr(e))

    def onLoadDarkRun(self, dirpath):
        """Load the dark from a run folder."""
        raise NotImplementedError

    def onRemoveDark(self):
        """Remove the recorded dark data."""
        raise NotImplementedError

    def run(self):
        """Override."""
        self._running = True
        self.reset()
        while self._running:
            try:
                data = self._input.get_nowait()
                try:
                    processed = self.process(data)

                except ProcessingError as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    self.log.debug(repr(traceback.format_tb(exc_traceback))
                                   + repr(e))
                    self.log.error(repr(e))
                    continue

                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    self.log.debug(f"Unexpected Exception!: " +
                                   repr(traceback.format_tb(exc_traceback)) +
                                   repr(e))
                    self.log.error(repr(e))
                    continue

                if processed is not None:
                    # keep the latest processed data in the output
                    self._output.put_pop(processed)

            except Empty:
                with self._cv:
                    self._cv.wait()

                if not self._running:
                    break

    def sources(self):
        """Return a list of (device ID/output channel, property).

        Interface method.

        Child class should implement this method in order to receive data
        from the bridge.
        """
        return []

    @abc.abstractmethod
    def process(self, data):
        """Process data.

        :param dict data: Data received by ZMQ clients have four keys:
            "raw", "processed", "meta", "catalog". data["processed"] is a
             ProcessedData object if the client is a QThreadFoamClient
             instance, and is None if the client is a QThreadKbClient
             instance. It should be noted that the "raw" and "meta" data
             here are different from the data received directly from a
             Karabo bridge. For details, one can check DataTransformer
             class.
        """
        raise NotImplementedError

    def get(self):
        return self._output.get_nowait()

    def terminate(self):
        self._running = False
        with self._cv:
            self._cv.notify()

    def _get_tid(self, meta):
        """Fetch train ID from meta data.

        :param dict meta: meta data.
        """
        try:
            return next(iter(meta.values()))["train_id"]
        except (StopIteration, KeyError) as e:
            raise ProcessingError(f"Train ID not found in meta data: {str(e)}")

    def _get_property_data(self, data, name, ppt):
        """Convenience method to get property data from raw data.

        :param dict data: data.
        :param str name: device ID / output channel.
        :param str ppt: property.

        :returns (value, error str)
        """
        return data[f"{name} {ppt}"]

    def _squeeze_camera_image(self, tid, arr):
        """Return a 2D image data.

        It attempts to squeeze the input array if its dimension is 3D.

        :param int tid: train ID.
        :param numpy.ndarray arr: image data.
        """
        if arr is None:
            return

        if arr.ndim not in (2, 3):
            self.log.error(f"[{tid}] Array dimension must be either 2 or 3! "
                           f"actual {arr.ndim}!")
            return

        if arr.ndim == 3:
            try:
                img = np.squeeze(arr, axis=0)
            except ValueError:
                try:
                    img = np.squeeze(arr, axis=-1)
                except ValueError:
                    self.log.error(
                        "f[{tid}] Failed to squeeze a 3D array to 2D!")
                    return
        else:
            img = arr

        if img.dtype != _IMAGE_DTYPE:
            img = img.astype(_IMAGE_DTYPE)

        return img


class _BaseQThreadClient(QThread):
    def __init__(self, queue, condition, catalog, *args, **kwargs):
        """Initialization.

        :param queue:
        :param Condition condition:
        :param SourceCatalog catalog:
        """
        super().__init__(*args, **kwargs)

        self._output = queue
        self._cv = condition
        self._catalog = catalog
        self._transformer = DataTransformer(catalog)

        self._endpoint = None

        self.log = _ThreadLogger()

    @abc.abstractmethod
    def run(self):
        """Override."""
        raise NotImplementedError

    def get(self):
        return self._cache.get_nowait()

    def stop(self):
        self.requestInterruption()

    def reset(self):
        """Reset the internal state of the client."""
        self._transformer.reset()
        self._output.clear()

    def updateParams(self, params):
        self._endpoint = params["endpoint"]
        ctl = self._catalog

        ctl.clear()
        for name, ppt in params["sources"]:
            ctl.add_item(None, name, None, ppt, None, None)


class QThreadFoamClient(_BaseQThreadClient):
    _client_instance_type = FoamZmqClient

    def run(self):
        """Override."""
        self.reset()

        with self._client_instance_type(
                self._endpoint, timeout=config["BRIDGE_TIMEOUT"]) as client:

            self.log.info(f"Connected to {self._endpoint}")

            while not self.isInterruptionRequested():
                try:
                    data = client.next()
                except TimeoutError:
                    continue

                if data["processed"] is None:
                    self.log.error("Processed data not found! Please check "
                                   "the ZMQ connection!")
                    continue

                # check whether all the requested sources are in the data
                not_found = False
                for src in self._catalog:
                    if src not in data["catalog"]:
                        self.log.error(f"{src} not found in the data!")
                        not_found = True
                        break
                if not_found:
                    continue

                # keep the latest processed data in the output
                self._output.put_pop(data)
                with self._cv:
                    self._cv.notify()

        self.log.info(f"Disconnected with {self._endpoint}")


class QThreadKbClient(_BaseQThreadClient):
    _client_instance_type = KaraboBridgeClient

    def run(self):
        """Override."""
        self.reset()

        correlated = None
        with self._client_instance_type(
                self._endpoint, timeout=config["BRIDGE_TIMEOUT"]) as client:
            self.log.info(f"Connected to {self._endpoint}")
            while not self.isInterruptionRequested():

                try:
                    data = client.next()
                except TimeoutError:
                    continue

                try:
                    correlated, dropped = self._transformer.correlate(data)
                except RuntimeError as e:
                    self.log.error(str(e))

                for tid in dropped:
                    self.log.error(f"Unable to correlate all data sources "
                                   f"for train {tid}")

                if correlated is not None:
                    # keep the latest processed data in the output
                    self._output.put_pop(correlated)
                    with self._cv:
                        self._cv.notify()

        self.log.info(f"Disconnected with {self._endpoint}")


def create_special(ctrl_klass, worker_klass, client_klass):
    """A decorator for the special analysis window."""
    def wrap(instance_type):
        @functools.wraps(instance_type)
        def wrapped_instance_type(*args, **kwargs):
            instance_type._ctrl_instance_type = ctrl_klass
            instance_type._worker_instance_type = worker_klass
            instance_type._client_instance_type = client_klass
            return instance_type(*args, **kwargs)
        return wrapped_instance_type
    return wrap


class _SpecialAnalysisBase(QMainWindow):
    """Base class for special analysis windows."""

    _SPLITTER_HANDLE_WIDTH = 5

    _TOTAL_W, _TOTAL_H = config['GUI_SPECIAL_WINDOW_SIZE']

    started_sgn = pyqtSignal()
    stopped_sgn = pyqtSignal()
    reset_sgn = pyqtSignal()

    def __init__(self, topic, **kwargs):
        """Initialization."""
        super().__init__()

        self._topic = topic

        self.setWindowTitle(f"EXtra-foam {__version__} - {self._title}")

        self._com_ctrl = _SharedCtrlWidgetS(**kwargs)

        cv = Condition()
        catalog = SourceCatalog()
        queue = SimpleQueue(maxsize=1)
        self._client = self._client_instance_type(queue, cv, catalog)
        self._worker = self._worker_instance_type(queue, cv)
        self._worker_thread = QThread()
        self._ctrl_widget = self._ctrl_instance_type(topic)

        if isinstance(self._client, QThreadFoamClient):
            self._com_ctrl.updateDefaultPort(config["EXTENSION_PORT"])

        self._plot_widgets = WeakKeyDictionary()  # book-keeping plot widgets
        self._image_views = WeakKeyDictionary()  # book-keeping ImageView widget

        self._data = None

        self._gui_logger = GuiLogger(parent=self)
        logger.addHandler(self._gui_logger)

        self._cw = QSplitter()
        self._cw.setChildrenCollapsible(False)
        self.setCentralWidget(self._cw)

        self._plot_timer = QTimer()
        self._plot_timer.setInterval(config["GUI_PLOT_UPDATE_TIMER"])
        self._plot_timer.timeout.connect(self.updateWidgetsF)

        # init UI

        self._left_panel = QSplitter(Qt.Vertical)
        self._left_panel.addWidget(self._com_ctrl)
        self._left_panel.addWidget(self._ctrl_widget)
        self._left_panel.addWidget(self._gui_logger.widget)
        self._left_panel.setChildrenCollapsible(False)

        # init Connections

        self._client.log.logOnMainThread(self)
        self._worker.log.logOnMainThread(self)

        # start/stop/reset
        self._com_ctrl.start_btn.clicked.connect(self._onStart)
        self._com_ctrl.stop_btn.clicked.connect(self._onStop)
        self._com_ctrl.reset_btn.clicked.connect(self._onReset)

        # dark operation
        self._com_ctrl.record_dark_btn.toggled.connect(
            self._worker.onRecordDarkToggled)
        self._com_ctrl.record_dark_btn.toggled.emit(
            self._com_ctrl.record_dark_btn.isChecked())

        self._com_ctrl.load_dark_run_btn.clicked.connect(
            self._onSelectDarkRunDirectory)

        self._com_ctrl.remove_dark_btn.clicked.connect(
            self._worker.onRemoveDark)

        self._com_ctrl.dark_subtraction_cb.toggled.connect(
            self._worker.onSubtractDarkToggled)
        self._com_ctrl.dark_subtraction_cb.toggled.emit(
            self._com_ctrl.dark_subtraction_cb.isChecked())

        self._com_ctrl.auto_level_btn.clicked.connect(
            self._onAutoLevel)

    @abc.abstractmethod
    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    @abc.abstractmethod
    def initConnections(self):
        """Initialization of signal-slot connections."""
        raise NotImplementedError

    def startWorker(self):
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker_thread.start()

    def _onStart(self):
        self._client.updateParams({
            "endpoint": self._com_ctrl.endpoint(),
            "sources": self._worker.sources()
        })

        self._com_ctrl.onStart()
        self._ctrl_widget.onStart()

        self._client.start()
        self._plot_timer.start()

        self.started_sgn.emit()
        logger.info("Processing started")

    def _onStop(self):
        self._com_ctrl.onStop()
        self._ctrl_widget.onStop()

        self._client.stop()
        self._plot_timer.stop()

        self.stopped_sgn.emit()
        logger.info("Processing stopped")

    def _onReset(self):
        for widget in self._plot_widgets:
            widget.reset()
        self._worker.reset()
        self._client.reset()

        self.reset_sgn.emit()

    def updateWidgetsF(self):
        """Override."""
        try:
            self._data = self._worker.get()
        except Empty:
            return

        for widget in self._plot_widgets:
            widget.updateF(self._data)

    def registerPlotWidget(self, instance):
        self._plot_widgets[instance] = 1
        if isinstance(instance, ImageViewF):
            self._image_views[instance] = 1

    def unregisterPlotWidget(self, instance):
        del self._plot_widgets[instance]
        if instance in self._image_views:
            del self._image_views[instance]

    @pyqtSlot()
    def _onAutoLevel(self):
        """Override."""
        for view in self._image_views:
            view.updateImageWithAutoLevel()

    @pyqtSlot()
    def _onSelectDarkRunDirectory(self):
        dirpath = QFileDialog.getExistingDirectory(
            options=QFileDialog.ShowDirsOnly)

        if dirpath:
            self._worker.onLoadDarkRun(dirpath)

    @pyqtSlot(str)
    def onDebugReceived(self, msg):
        logger.debug(msg)

    @pyqtSlot(str)
    def onInfoReceived(self, msg):
        logger.info(msg)

    @pyqtSlot(str)
    def onWarningReceived(self, msg):
        logger.warning(msg)

    @pyqtSlot(str)
    def onErrorReceived(self, msg):
        logger.error(msg)

    def closeEvent(self, QCloseEvent):
        # prevent from logging in the GUI when it has been closed
        logger.removeHandler(self._gui_logger)

        self._worker.terminate()
        self._worker_thread.quit()
        self._worker_thread.wait()

        super().closeEvent(QCloseEvent)
