"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque
import sys
import traceback
import time
import os.path as osp
from queue import Empty
from weakref import WeakKeyDictionary
import functools
import itertools
from threading import Event

from PyQt5.QtCore import (
    pyqtSignal, pyqtSlot, QObject, Qt, QThread, QTimer
)
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction, QFrame, QMainWindow, QScrollArea, QSplitter,
    QTabWidget, QVBoxLayout, QWidget
)
from redis import ConnectionError

from .ctrl_widgets import (
    AnalysisCtrlWidget, ExtensionCtrlWidget, FomFilterCtrlWidget,
    DataSourceWidget
)
from .misc_widgets import AnalysisSetupManager, GuiLogger
from .image_tool import ImageToolWindow
from .windows import (
    BinningWindow, CorrelationWindow, HistogramWindow, PulseOfInterestWindow,
    PumpProbeWindow, FileStreamWindow, AboutWindow
)
from .. import __version__
from ..config import config
from ..logger import logger
from ..utils import profiler
from ..ipc import RedisConnection, RedisPSubscriber
from ..pipeline import MpInQueue
from ..processes import shutdown_all
from ..database import MonProxy


class ThreadLoggerBridge(QObject):
    """QThread which subscribes logs the Redis server.

    This QThread forward the message from the Redis server and send
    it to the MainGUI via signal-slot connection.
    """
    log_msg_sgn = pyqtSignal(str, str)

    _sub = RedisPSubscriber("log:*")

    def __init__(self):
        super().__init__()

        self._running = False

    def recv(self):
        self._running = True
        while self._running:
            try:
                msg = self._sub.get_message(ignore_subscribe_messages=True)
                self.log_msg_sgn.emit(msg['channel'], msg['data'])
            except Exception:
                pass

            time.sleep(0.001)

    def connectToMainThread(self, instance):
        self.log_msg_sgn.connect(instance.onLogMsgReceived)

    def stop(self):
        self._running = False


class MainGUI(QMainWindow):
    """The main GUI for azimuthal integration."""

    _root_dir = osp.dirname(osp.abspath(__file__))

    start_sgn = pyqtSignal()
    stop_sgn = pyqtSignal()
    quit_sgn = pyqtSignal()

    _db = RedisConnection()

    _WIDTH, _HEIGHT = config['GUI_MAIN_GUI_SIZE']

    def __init__(self, pause_ev, close_ev):
        """Initialization."""
        super().__init__()

        self._pause_ev = pause_ev
        self._close_ev = close_ev
        self._input_update_ev = Event()
        self._input = MpInQueue(self._input_update_ev, pause_ev, close_ev)

        self._pulse_resolved = config["PULSE_RESOLVED"]
        self._require_geometry = config["REQUIRE_GEOMETRY"]
        self._queue = deque(maxlen=1)

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.title = f"EXtra-foam {__version__} ({config['DETECTOR']})"
        self.setWindowTitle(self.title + " - main GUI")

        # *************************************************************
        # Central widget
        # *************************************************************

        self._ctrl_widgets = []  # book-keeping control widgets

        self._cw = QSplitter()
        self._cw.setChildrenCollapsible(False)
        self.setCentralWidget(self._cw)

        self._left_cw_container = QScrollArea()
        self._left_cw_container.setFrameShape(QFrame.NoFrame)
        self._left_cw = QTabWidget()
        self._right_cw_container = QScrollArea()
        self._right_cw_container.setFrameShape(QFrame.NoFrame)
        self._right_cw = QSplitter(Qt.Vertical)
        self._right_cw.setChildrenCollapsible(False)

        self._source_cw = self.createCtrlWidget(DataSourceWidget)
        self._extension_cw = self.createCtrlWidget(ExtensionCtrlWidget)

        self._ctrl_panel_cw = QTabWidget()
        self._analysis_cw = QWidget()
        self._statistics_cw = QWidget()

        self._util_panel_container = QWidget()
        self._util_panel_cw = QTabWidget()

        # *************************************************************
        # Tool bar
        # Note: the order of '_addAction` affect the unittest!!!
        # *************************************************************
        self._tool_bar = self.addToolBar("Control")
        # make icon a bit larger
        self._tool_bar.setIconSize(2 * self._tool_bar.iconSize())

        self._start_at = self._addAction("Start bridge", "start.png")
        self._start_at.triggered.connect(self.onStart)

        self._stop_at = self._addAction("Stop bridge", "stop.png")
        self._stop_at.triggered.connect(self.onStop)
        self._stop_at.setEnabled(False)

        self._tool_bar.addSeparator()

        image_tool_at = self._addAction("Image tool", "image_tool.png")
        image_tool_at.triggered.connect(
            lambda: (self._image_tool.show(),
                     self._image_tool.activateWindow()))

        open_poi_window_at = self._addAction("Pulse-of-interest", "poi.png")
        open_poi_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, PulseOfInterestWindow))
        if not self._pulse_resolved:
            open_poi_window_at.setEnabled(False)

        pump_probe_window_at = self._addAction("Pump-probe", "pump-probe.png")
        pump_probe_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, PumpProbeWindow))

        open_correlation_window_at = self._addAction(
            "Correlation", "correlation.png")
        open_correlation_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, CorrelationWindow))

        open_histogram_window_at = self._addAction(
            "Histogram", "histogram.png")
        open_histogram_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, HistogramWindow))

        open_bin2d_window_at = self._addAction("Binning", "binning.png")
        open_bin2d_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, BinningWindow))

        self._tool_bar.addSeparator()

        open_file_stream_window_at = self._addAction(
            "File stream", "file_stream.png")
        open_file_stream_window_at.triggered.connect(
            lambda: self.onOpenSatelliteWindow(FileStreamWindow))

        open_about_at = self._addAction("About EXtra-foam", "about.png")
        open_about_at.triggered.connect(
            lambda: self.onOpenSatelliteWindow(AboutWindow))

        # *************************************************************
        # Miscellaneous
        # *************************************************************

        # book-keeping opened windows
        self._plot_windows = WeakKeyDictionary()
        self._satellite_windows = WeakKeyDictionary()

        self._gui_logger = GuiLogger(parent=self)
        logger.addHandler(self._gui_logger)

        self._analysis_setup_manager = AnalysisSetupManager()

        self._thread_logger = ThreadLoggerBridge()
        self.quit_sgn.connect(self._thread_logger.stop)
        self._thread_logger_t = QThread()
        self._thread_logger.moveToThread(self._thread_logger_t)
        self._thread_logger_t.started.connect(self._thread_logger.recv)
        self._thread_logger.connectToMainThread(self)

        # For real time plot
        self._running = False
        self._plot_timer = QTimer()
        self._plot_timer.timeout.connect(self.updateAll)

        # For checking the connection to the Redis server
        self._redis_timer = QTimer()
        self._redis_timer.timeout.connect(self.pingRedisServer)

        self.__redis_connection_fails = 0

        self._mon_proxy = MonProxy()

        # *************************************************************
        # control widgets
        # *************************************************************

        # analysis control widgets
        self.analysis_ctrl_widget = self.createCtrlWidget(AnalysisCtrlWidget)
        self.fom_filter_ctrl_widget = self.createCtrlWidget(FomFilterCtrlWidget)

        # *************************************************************
        # status bar
        # *************************************************************

        # StatusBar to display topic name
        self.statusBar().showMessage(f"TOPIC: {config['TOPIC']}")
        self.statusBar().setStyleSheet("QStatusBar{font-weight:bold;}")

        # ImageToolWindow is treated differently since it is the second
        # control window.
        self._image_tool = ImageToolWindow(
            queue=self._queue,
            pulse_resolved=self._pulse_resolved,
            require_geometry=self._require_geometry,
            parent=self)

        self.initUI()
        self.initConnections()
        self.updateMetaData()
        self._analysis_setup_manager.onInit()

        self.setMinimumSize(640, 480)
        self.resize(self._WIDTH, self._HEIGHT)

        self.show()

    def createCtrlWidget(self, widget_class):
        widget = widget_class(pulse_resolved=self._pulse_resolved,
                              require_geometry=self._require_geometry,
                              parent=self)
        self._ctrl_widgets.append(widget)
        return widget

    def initUI(self):
        self.initLeftUI()
        self.initRightUI()

        self._cw.addWidget(self._left_cw_container)
        self._cw.addWidget(self._right_cw_container)
        self._cw.setSizes([int(self._WIDTH * 0.6), int(self._WIDTH * 0.4)])

    def initLeftUI(self):
        self._left_cw.setTabPosition(QTabWidget.TabPosition.West)

        self._left_cw.addTab(self._source_cw, "Data source")
        self._left_cw_container.setWidget(self._left_cw)
        self._left_cw_container.setWidgetResizable(True)

        self._left_cw.addTab(self._extension_cw, "Extension")

    def initRightUI(self):
        self.initCtrlUI()
        self.initUtilUI()

        self._right_cw.addWidget(self._ctrl_panel_cw)
        self._right_cw.addWidget(self._util_panel_container)

        self._right_cw_container.setWidget(self._right_cw)
        self._right_cw_container.setWidgetResizable(True)

    def initCtrlUI(self):
        self.initGeneralAnalysisUI()

        self._ctrl_panel_cw.addTab(self._analysis_cw, "General analysis setup")

    def initGeneralAnalysisUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self.analysis_ctrl_widget)
        layout.addWidget(self.fom_filter_ctrl_widget)
        self._analysis_cw.setLayout(layout)

    def initUtilUI(self):
        self._util_panel_cw.addTab(self._gui_logger.widget, "Logger")
        self._util_panel_cw.addTab(self._analysis_setup_manager, "Analysis Setup Manager")
        self._util_panel_cw.setTabPosition(QTabWidget.TabPosition.South)

        layout = QVBoxLayout()
        layout.addWidget(self._util_panel_cw)
        self._util_panel_container.setLayout(layout)

    def initConnections(self):
        self._analysis_setup_manager.load_metadata_sgn.connect(self.loadMetaData)

    def connect_input_to_output(self, output):
        self._input.connect(output)

    @profiler("Update Plots", process_time=True)
    def updateAll(self):
        """Update all the plots in the main and child windows."""
        if not self._running:
            return

        try:
            data = self._input.get()
            self._queue.append(data)
        except Empty:
            return

        # clear the previous plots no matter what comes next
        # for w in self._plot_windows.keys():
        #     w.reset()

        processed = self._queue[0]["processed"]

        self._image_tool.updateWidgetsF()
        for w in itertools.chain(self._plot_windows):
            try:
                w.updateWidgetsF()
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.debug(repr(traceback.format_tb(exc_traceback))
                             + repr(e))
                logger.error(f"[Update plots] {repr(e)}")

        logger.debug(f"Plot train with ID: {processed.tid}")

    def pingRedisServer(self):
        try:
            self._db.ping()
            if self.__redis_connection_fails > 0:
                # Note: Indeed, we do not have mechanism to recover from
                #       a Redis server crash. It is recommended to restart
                #       Extra-foam if you encounter this situation.
                logger.info("Reconnect to the Redis server!")
                self.__redis_connection_fails = 0
        except ConnectionError:
            self.__redis_connection_fails += 1
            rest_attempts = config["REDIS_MAX_PING_ATTEMPTS"] - \
                self.__redis_connection_fails

            if rest_attempts > 0:
                logger.warning(f"No response from the Redis server! Shut "
                               f"down after {rest_attempts} attempts ...")
            else:
                logger.warning(f"No response from the Redis server! "
                               f"Shutting down!")
                self.close()

    def _addAction(self, description, filename):
        icon = QIcon(osp.join(self._root_dir, "icons/" + filename))
        action = QAction(icon, description, self)
        self._tool_bar.addAction(action)
        return action

    def onOpenPlotWindow(self, instance_type):
        """Open a plot window if it does not exist.

        Otherwise bring the opened window to the table top.
        """
        if self.checkWindowExistence(instance_type, self._plot_windows):
            return

        return instance_type(self._queue,
                             pulse_resolved=self._pulse_resolved,
                             require_geometry=self._require_geometry,
                             parent=self)

    def onOpenSatelliteWindow(self, instance_type):
        """Open a satellite window if it does not exist.

        Otherwise bring the opened window to the table top.
        """
        if self.checkWindowExistence(instance_type, self._satellite_windows):
            return
        return instance_type(parent=self)

    def checkWindowExistence(self, instance_type, windows):
        for key in windows:
            if isinstance(key, instance_type):
                key.activateWindow()
                return True
        return False

    def registerWindow(self, instance):
        self._plot_windows[instance] = 1

    def unregisterWindow(self, instance):
        del self._plot_windows[instance]

    def registerSatelliteWindow(self, instance):
        self._satellite_windows[instance] = 1

    def unregisterSatelliteWindow(self, instance):
        del self._satellite_windows[instance]

    @property
    def input(self):
        return self._input

    def start(self):
        """Start running.

        ProcessWorker interface.
        """
        self._thread_logger_t.start()
        self._plot_timer.start(config["GUI_PLOT_UPDATE_TIMER"])
        self._redis_timer.start(config["REDIS_PING_ATTEMPT_INTERVAL"])
        self._input.start()

    def onStart(self):
        if not self.updateMetaData():
            return

        self.start_sgn.emit()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)

        for widget in self._ctrl_widgets:
            widget.onStart()
        for win in self._plot_windows:
            win.onStart()
        self._image_tool.onStart()
        self._analysis_setup_manager.onStart()

        self._running = True  # starting to update plots
        self._input_update_ev.set()  # notify update

    def onStop(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        self.stop_sgn.emit()

        # TODO: wait for some signal

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)

        for widget in self._ctrl_widgets:
            widget.onStop()
        for win in self._plot_windows:
            win.onStop()
        self._image_tool.onStop()
        self._analysis_setup_manager.onStop()

    def updateMetaData(self):
        """Update metadata from all the ctrl widgets.

        :returns bool: True if all metadata successfully parsed
            and emitted, otherwise False.
        """
        for widget in self._ctrl_widgets:
            if not widget.updateMetaData():
                return False

        for win in self._plot_windows:
            if not win.updateMetaData():
                return False

        return self._image_tool.updateMetaData()

    def loadMetaData(self):
        """Load metadata from Redis and set child control widgets."""
        for widget in self._ctrl_widgets:
            widget.loadMetaData()

        for win in self._plot_windows:
            win.loadMetaData()

        self._image_tool.loadMetaData()

    @pyqtSlot(str, str)
    def onLogMsgReceived(self, ch, msg):
        if ch == 'log:debug':
            logger.debug(msg)
        elif ch == 'log:info':
            logger.info(msg)
        elif ch == 'log:warning':
            logger.warning(msg)
        elif ch == 'log:error':
            logger.error(msg)

    def closeEvent(self, QCloseEvent):
        # prevent from logging in the GUI when it has been closed
        logger.removeHandler(self._gui_logger)

        # tell all processes to close
        self._close_ev.set()

        # clean up the logger thread
        self.quit_sgn.emit()
        self._thread_logger_t.quit()
        self._thread_logger_t.wait()

        # shutdown pipeline workers and Redis server
        shutdown_all()

        self._image_tool.close()
        for window in list(itertools.chain(self._plot_windows,
                                           self._satellite_windows)):
            # Close all open child windows to make sure their resources
            # (any running process etc.) are released gracefully. This
            # is especially necessary for the case when file stream was
            # still ongoing when the main GUI was closed.
            window.close()

        super().closeEvent(QCloseEvent)
