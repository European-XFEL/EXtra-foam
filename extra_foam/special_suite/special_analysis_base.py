"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
import functools
from queue import Empty
import sys
from threading import Condition
import time
import traceback
from weakref import WeakKeyDictionary

import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, Qt, QThread, QTimer
from PyQt5.QtGui import QColor, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QLabel,
    QMainWindow, QPushButton, QSizePolicy, QSplitter
)

from extra_data import RunDirectory
from karabo_bridge import Client as KaraboBridgeClient

from extra_foam.algorithms import intersection
from extra_foam.database import SourceCatalog
from extra_foam.gui.ctrl_widgets import _SingleRoiCtrlWidget, SmartLineEdit
from extra_foam.gui.plot_widgets import ImageViewF
from extra_foam.gui.misc_widgets import GuiLogger, set_button_color
from extra_foam.pipeline.f_queue import SimpleQueue
from extra_foam.pipeline.f_transformer import DataTransformer
from extra_foam.pipeline.f_zmq import FoamZmqClient
from extra_foam.pipeline.exceptions import ProcessingError

from . import __version__


from . import logger
from .config import _IMAGE_DTYPE, config


class _SharedCtrlWidgetS(QFrame):
    """Control widget used in all special analysis window.

    It provides connection setup, start/stop/reset control, dark recording/
    subtraction control as well as other common GUI controls.
    """
    # forward signal with the same name in _SingleRoiCtrlWidget
    roi_geometry_change_sgn = pyqtSignal(object)

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
        self._port_le = SmartLineEdit(str(config["DEFAULT_CLIENT_PORT"]))
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

        self.roi_ctrl = None

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

    def addRoiCtrl(self, roi):
        """Add the ROI ctrl widget.

        :param RectROI roi: Roi item.
        """
        if self.roi_ctrl is not None:
            raise RuntimeError("Only one ImageView with ROI ctrl is allowed!")

        roi.setLocked(False)

        self.roi_ctrl = _SingleRoiCtrlWidget(
            roi, mediator=self, with_lock=False)
        self.roi_ctrl.setLabel("ROI")
        self.roi_ctrl.roi_geometry_change_sgn.connect(
            self.onRoiGeometryChange)
        self.roi_ctrl.notifyRoiParams()
        layout = self.layout()
        self.roi_ctrl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.roi_ctrl, layout.rowCount(), 0, 1, 4)

    def endpoint(self):
        return f"tcp://{self._hostname_le.text()}:{self._port_le.text()}"

    def updateDefaultPort(self, port: int):
        self._port_le.setText(str(port))

    def onStartST(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._hostname_le.setEnabled(False)
        self._port_le.setEnabled(False)
        self.load_dark_run_btn.setEnabled(False)

    def onStopST(self):
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self._hostname_le.setEnabled(True)
        self._port_le.setEnabled(True)
        self.load_dark_run_btn.setEnabled(True)

    def onRoiGeometryChange(self, object):
        self.roi_geometry_change_sgn.emit(object)


class _BaseAnalysisCtrlWidgetS(QFrame):
    """Base class of control widget.

    It should be inherited by all concrete ctrl widgets. Also, each special
    suite app should have only one ctrl widget.
    """
    def __init__(self, topic, *, parent=None):
        """Initialization.

        :param str topic: topic, e.g. SCS, MID, DET, etc.
        """
        super().__init__(parent=parent)

        self._topic_st = topic

        # widgets whose values are not allowed to change after the "start"
        # button is clicked
        self._non_reconfigurable_widgets = []

        self.setFrameStyle(QFrame.StyledPanel)

        # set default layout
        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight)
        self.setLayout(layout)

    ###################################################################
    # Interface start
    ###################################################################

    @property
    def topic(self):
        return self._topic_st

    @abc.abstractmethod
    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    @abc.abstractmethod
    def initConnections(self):
        """Initialization of signal-slot connections."""
        raise NotImplementedError

    ###################################################################
    # Interface end
    ###################################################################

    def onStartST(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(False)

    def onStopST(self):
        for widget in self._non_reconfigurable_widgets:
            widget.setEnabled(True)


def profiler(info):
    def wrap(f):
        @functools.wraps(f)
        def timed_f(*args, **kwargs):
            t0 = time.perf_counter()
            result = f(*args, **kwargs)
            logger.debug(f"Process time spent on {info}: "
                         f"{1000 * (time.perf_counter() - t0):.3f} ms")
            return result
        return timed_f
    return wrap


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
        self.debug_sgn.connect(instance.onDebugReceivedST)
        self.info_sgn.connect(instance.onInfoReceivedST)
        self.warning_sgn.connect(instance.onWarningReceivedST)
        self.error_sgn.connect(instance.onErrorReceivedST)


class QThreadWorker(QObject):
    """Base class of worker running in a thread.

    It should be inherited by all concrete workers.

    Attributes:
        _recording_dark_st (bool): True for recording dark.
        _subtract_dark_st (bool): True for applying dark subtraction.
        _roi_geom_st (tuple): (x, y, w, h) for ROI.
    """

    def __init__(self, queue, condition, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._input_st = queue
        self._cv_st = condition

        self._output_st = SimpleQueue(maxsize=1)
        self._running_st = False

        self._recording_dark_st = False
        self._subtract_dark_st = True

        self._reset_st = True

        self._roi_geom_st = None

        self.log = _ThreadLogger()

    def onResetST(self):
        """Reset the internal state of process worker."""
        self._input_st.clear()
        self._output_st.clear()
        self._reset_st = True

    def onRecordDarkToggledST(self, state: bool):
        self._recording_dark_st = state

    def onSubtractDarkToggledST(self, state: bool):
        self._subtract_dark_st = state

    def _loadRunDirectoryST(self, dirpath):
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

    def onRoiGeometryChange(self, value: tuple):
        """EXtra-foam interface method."""
        idx, activated, locked, x, y, w, h = value
        if activated:
            self._roi_geom_st = (x, y, w, h)
        else:
            # distinguish None from no intersection
            self._roi_geom_st = None

    def getOutputDataST(self):
        """Get data from the output queue."""
        return self._output_st.get_nowait()

    def _processImpST(self, data):
        """Process data."""
        if self._reset_st:
            self.reset()
            self._reset_st = False

        self.preprocess()
        processed = self.process(data)
        self.postprocess()
        return processed

    def runForeverST(self):
        """Run processing in an infinite loop unless interrupted."""
        self._running_st = True
        self.reset()
        while self._running_st:
            try:
                data = self._input_st.get_nowait()
                try:
                    processed = self._processImpST(data)

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
                    self._output_st.put_pop(processed)

            except Empty:
                with self._cv_st:
                    self._cv_st.wait()

                if not self._running_st:
                    break

    def terminateRunST(self):
        """Terminate processing and notify other waiting threads."""
        self._running_st = False
        with self._cv_st:
            self._cv_st.notify()

    ###################################################################
    # Interface start
    ###################################################################

    def subtractDark(self) -> bool:
        """Return whether to apply dark subtraction."""
        return self._subtract_dark_st

    def recordingDark(self) -> bool:
        """Return the state of dark recording."""
        return self._recording_dark_st

    def onLoadDarkRun(self, dirpath):
        """Load the dark from a run folder."""
        raise NotImplementedError

    def onRemoveDark(self):
        """Remove the recorded dark data."""
        raise NotImplementedError

    def str2range(self, s, *, handler=float):
        """Concert a string to a tuple with lower and upper boundary.

        For example: str2range("-inf, inf") = (-math.inf, math.inf)
        """
        splitted = s.split(",")
        return handler(splitted[0]), handler(splitted[1])

    def reset(self):
        """Interface method.

        Concrete child class should re-implement this method to reset
        any internal state when the 'Reset' button is clicked. This method
        will be called once before the next call to 'preprocess'.
        """
        pass

    def sources(self):
        """Interface method.

        Return a list of (device ID/output channel, property).

        Concrete child class should re-implement this method in order to
        receive data from the bridge, transform and correlate them.
        """
        return []

    def preprocess(self):
        """Preprocess before processing data."""
        pass

    def postprocess(self):
        """Postprocess after processing data."""
        pass

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

    def getTrainId(self, meta):
        """Get train ID from meta data.

        :param dict meta: meta data.
        """
        try:
            return next(iter(meta.values()))["train_id"]
        except (StopIteration, KeyError) as e:
            raise ProcessingError(f"Train ID not found in meta data: {str(e)}")

    def getPropertyData(self, data, name, ppt):
        """Convenience method to get property data from raw data.

        :param dict data: data.
        :param str name: device ID / output channel.
        :param str ppt: property.
        """
        return data[f"{name} {ppt}"]

    def squeezeToImage(self, tid, arr):
        """Try to squeeze an array to get a 2D image data.

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

    def getRoiData(self, img, copy=False):
        """Get the ROI(s) of an image or arrays of images.

        :param numpy.ndarray img: image data. Shape = (..., y, x)
        :param bool copy: True for copying the ROI data.
        """
        if self._roi_geom_st is None:
            roi = img
        else:
            img_shape = img.shape[-2:]
            x, y, w, h = intersection(self._roi_geom_st,
                                      (0, 0, img_shape[1], img_shape[0]))

            if w <= 0 or h <= 0:
                w, h = 0, 0
            roi = img[..., y:y+h, x:x+w]

        if copy:
            return roi.copy()
        return roi

    ###################################################################
    # Interface end
    ###################################################################


class _BaseQThreadClient(QThread):
    def __init__(self, queue, condition, catalog, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_st = queue
        self._cv_st = condition
        self._catalog_st = catalog
        self._transformer_st = DataTransformer(catalog)

        self._endpoint_st = None

        self.log = _ThreadLogger()

    def run(self):
        """Override."""
        raise NotImplementedError

    def terminateRunST(self):
        """Terminate running of the thread."""
        self.requestInterruption()

    def onResetST(self):
        """Reset the internal state of the client."""
        self._transformer_st.reset()
        self._output_st.clear()

    def updateParamsST(self, params):
        """Update internal states of the client."""
        self._endpoint_st = params["endpoint"]

        ctl = self._catalog_st
        ctl.clear()
        for name, ppt in params["sources"]:
            ctl.add_item(None, name, None, ppt, None, None)


class QThreadFoamClient(_BaseQThreadClient):
    _client_instance_type = FoamZmqClient

    def run(self):
        """Override."""
        self.onResetST()

        with self._client_instance_type(
                self._endpoint_st, timeout=config["CLIENT_TIME_OUT"]) as client:

            self.log.info(f"Connected to {self._endpoint_st}")

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
                for src in self._catalog_st:
                    if src not in data["catalog"]:
                        self.log.error(f"{src} not found in the data!")
                        not_found = True
                        break
                if not_found:
                    continue

                # keep the latest processed data in the output
                self._output_st.put_pop(data)
                with self._cv_st:
                    self._cv_st.notify()

        self.log.info(f"Disconnected with {self._endpoint_st}")


class QThreadKbClient(_BaseQThreadClient):
    _client_instance_type = KaraboBridgeClient

    def run(self):
        """Override."""
        self.onResetST()

        with self._client_instance_type(
                self._endpoint_st, timeout=config["CLIENT_TIME_OUT"]) as client:
            self.log.info(f"Connected to {self._endpoint_st}")
            correlated = None
            while not self.isInterruptionRequested():

                try:
                    data = client.next()
                except TimeoutError:
                    continue

                try:
                    correlated, dropped = self._transformer_st.correlate(data)
                    for tid, err in dropped:
                        self.log.error(err)
                except Exception as e:
                    # To be on the safe side since any Exception here
                    # will stop the thread
                    self.log.error(str(e))

                if correlated is not None:
                    # keep the latest processed data in the output
                    self._output_st.put_pop(correlated)
                    with self._cv_st:
                        self._cv_st.notify()

        self.log.info(f"Disconnected with {self._endpoint_st}")


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
    """Base class for special analysis windows.

    It should be inherited by all concrete windows.
    """

    _TOTAL_W, _TOTAL_H = config["GUI_SPECIAL_WINDOW_SIZE"]

    started_sgn = pyqtSignal()
    stopped_sgn = pyqtSignal()

    def __init__(self, topic, **kwargs):
        """Initialization.

        :param str topic: topic, e.g. SCS, MID, DET, etc.
        """
        super().__init__()

        self._topic_st = topic

        self.setWindowTitle(f"EXtra-foam {__version__} - " +
                            f"special suite - {self._title}")

        self._com_ctrl_st = _SharedCtrlWidgetS(**kwargs)

        cv = Condition()
        catalog = SourceCatalog()
        queue = SimpleQueue(maxsize=1)
        self._client_st = self._client_instance_type(queue, cv, catalog)
        self._worker_st = self._worker_instance_type(queue, cv)
        self._worker_thread_st = QThread()
        self._ctrl_widget_st = self._ctrl_instance_type(topic)

        if isinstance(self._client_st, QThreadFoamClient):
            self._com_ctrl_st.updateDefaultPort(config["EXTENSION_PORT"])

        # book-keeping plot widgets
        self._plot_widgets_st = WeakKeyDictionary()
        # book-keeping ImageView widget
        self._image_views_st = WeakKeyDictionary()

        self._data_st = None

        self._gui_logger_st = GuiLogger(parent=self)
        logger.addHandler(self._gui_logger_st)

        self._cw_st = QSplitter()
        self._cw_st.setChildrenCollapsible(False)
        self.setCentralWidget(self._cw_st)

        self._plot_timer_st = QTimer()
        self._plot_timer_st.setInterval(config["GUI_PLOT_UPDATE_TIMER"])
        self._plot_timer_st.timeout.connect(self.updateWidgetsST)

        # init UI

        self._left_panel_st = QSplitter(Qt.Vertical)
        self._left_panel_st.addWidget(self._com_ctrl_st)
        self._left_panel_st.addWidget(self._ctrl_widget_st)
        self._left_panel_st.addWidget(self._gui_logger_st.widget)
        self._left_panel_st.setChildrenCollapsible(False)
        self._cw_st.addWidget(self._left_panel_st)

        # init Connections

        self._client_st.log.logOnMainThread(self)
        self._worker_st.log.logOnMainThread(self)

        # start/stop/reset
        self._com_ctrl_st.start_btn.clicked.connect(self._onStartST)
        self._com_ctrl_st.stop_btn.clicked.connect(self._onStopST)
        self._com_ctrl_st.reset_btn.clicked.connect(self._onResetST)

        # dark operation
        self._com_ctrl_st.record_dark_btn.toggled.connect(
            self._worker_st.onRecordDarkToggledST)
        self._com_ctrl_st.record_dark_btn.toggled.emit(
            self._com_ctrl_st.record_dark_btn.isChecked())

        self._com_ctrl_st.load_dark_run_btn.clicked.connect(
            self._onSelectDarkRunDirectoryST)

        self._com_ctrl_st.remove_dark_btn.clicked.connect(
            self._worker_st.onRemoveDark)

        self._com_ctrl_st.dark_subtraction_cb.toggled.connect(
            self._worker_st.onSubtractDarkToggledST)
        self._com_ctrl_st.dark_subtraction_cb.toggled.emit(
            self._com_ctrl_st.dark_subtraction_cb.isChecked())

        self._com_ctrl_st.auto_level_btn.clicked.connect(
            self._onAutoLevelST)

        # ROI ctrl
        self._com_ctrl_st.roi_geometry_change_sgn.connect(
            self._worker_st.onRoiGeometryChange)

    ###################################################################
    # Interface start
    ###################################################################

    @abc.abstractmethod
    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    @abc.abstractmethod
    def initConnections(self):
        """Initialization of signal-slot connections."""
        raise NotImplementedError

    def centralWidget(self):
        """Return the central widget."""
        return self._cw_st

    def startWorker(self):
        """Start worker in thread.

        Concrete child class should call this method at the end of
        initialization.
        """
        self._worker_st.moveToThread(self._worker_thread_st)
        self._worker_thread_st.started.connect(self._worker_st.runForeverST)
        self._worker_thread_st.start()

    ###################################################################
    # Interface end
    ###################################################################

    def _onStartST(self):
        self._client_st.updateParamsST({
            "endpoint": self._com_ctrl_st.endpoint(),
            "sources": self._worker_st.sources()
        })

        self._com_ctrl_st.onStartST()
        self._ctrl_widget_st.onStartST()

        self._client_st.start()
        self._plot_timer_st.start()

        self.started_sgn.emit()
        logger.info("Processing started")

    def _onStopST(self):
        self._com_ctrl_st.onStopST()
        self._ctrl_widget_st.onStopST()

        self._client_st.terminateRunST()
        self._plot_timer_st.stop()

        self.stopped_sgn.emit()
        logger.info("Processing stopped")

    def _onResetST(self):
        for widget in self._plot_widgets_st:
            widget.reset()
        self._worker_st.onResetST()
        self._client_st.onResetST()

    def updateWidgetsST(self):
        try:
            self._data_st = self._worker_st.getOutputDataST()
        except Empty:
            return

        for widget in self._plot_widgets_st:
            try:
                widget.updateF(self._data_st)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.debug(repr(traceback.format_tb(exc_traceback))
                             + repr(e))
                logger.error(f"[Update plots] {repr(e)}")

    @pyqtSlot()
    def _onAutoLevelST(self):
        for view in self._image_views_st:
            view.updateImageWithAutoLevel()

    @pyqtSlot()
    def _onSelectDarkRunDirectoryST(self):
        dirpath = QFileDialog.getExistingDirectory(
            options=QFileDialog.ShowDirsOnly)

        if dirpath:
            self._worker_st.onLoadDarkRun(dirpath)

    @pyqtSlot(str)
    def onDebugReceivedST(self, msg):
        logger.debug(msg)

    @pyqtSlot(str)
    def onInfoReceivedST(self, msg):
        logger.info(msg)

    @pyqtSlot(str)
    def onWarningReceivedST(self, msg):
        logger.warning(msg)

    @pyqtSlot(str)
    def onErrorReceivedST(self, msg):
        logger.error(msg)

    def registerPlotWidget(self, instance):
        """EXtra-foam interface method."""
        self._plot_widgets_st[instance] = 1
        if isinstance(instance, ImageViewF):
            self._image_views_st[instance] = 1
            if instance.rois:
                self._com_ctrl_st.addRoiCtrl(instance.rois[0])

    def unregisterPlotWidget(self, instance):
        """EXtra-foam interface method."""
        del self._plot_widgets_st[instance]
        if instance in self._image_views_st:
            del self._image_views_st[instance]

    def closeEvent(self, QCloseEvent):
        """Override."""
        # prevent from logging in the GUI when it has been closed
        logger.removeHandler(self._gui_logger_st)

        self._worker_st.terminateRunST()
        self._worker_thread_st.quit()
        self._worker_thread_st.wait()

        super().closeEvent(QCloseEvent)
