"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque
import time
import logging
import os.path as osp
from queue import Empty
from weakref import WeakKeyDictionary
import functools
import itertools
import multiprocessing as mp

from PyQt5.QtCore import (
    pyqtSignal, pyqtSlot, QObject, QSize, Qt, QThread, QTimer
)
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction, QHBoxLayout, QMainWindow, QPushButton, QScrollArea, QSplitter,
    QTabWidget, QVBoxLayout, QWidget
)
from redis import ConnectionError

from .ctrl_widgets import (
    AnalysisCtrlWidget, BinCtrlWidget,
    CorrelationCtrlWidget, PulseFilterCtrlWidget, DataSourceWidget,
    StatisticsCtrlWidget, PumpProbeCtrlWidget, TrXasCtrlWidget,
)
from .misc_widgets import GuiLogger
from .image_tool import ImageToolWindow
from .windows import (
    Bin1dWindow, Bin2dWindow, CorrelationWindow,
    ProcessMonitor, StatisticsWindow, PulseOfInterestWindow,
    PumpProbeWindow, RoiWindow, FileStreamControllerWindow, AboutWindow,
    TrXasWindow
)
from .. import __version__
from ..config import config
from ..logger import logger
from ..utils import profiler
from ..ipc import RedisConnection, RedisPSubscriber
from ..pipeline import MpInQueue
from ..processes import list_fai_processes, shutdown_all
from ..database import MonProxy


class ThreadLoggerBridge(QObject):
    """QThread which subscribes logs the Redis server.

    This QThread forward the message from the Redis server and send
    it to the MainGUI via signal-slot connection.
    """
    log_debug_sgn = pyqtSignal(str)
    log_info_sgn = pyqtSignal(str)
    log_warning_sgn = pyqtSignal(str)
    log_error_sgn = pyqtSignal(str)

    __sub = RedisPSubscriber("log:*")

    def __init__(self):
        super().__init__()

        self._running = False

    def recv(self):
        self._running = True
        while self._running:
            try:
                msg = self.__sub.get_message()

                if msg and isinstance(msg['data'], str):
                    channel = msg['channel']
                    log_msg = msg['data']

                    if channel == 'log:debug':
                        self.log_debug_sgn.emit(log_msg)
                    elif channel == 'log:info':
                        self.log_info_sgn.emit(log_msg)
                    elif channel == 'log:warning':
                        self.log_warning_sgn.emit(log_msg)
                    elif channel == 'log:error':
                        self.log_error_sgn.emit(log_msg)

            except (ConnectionError, RuntimeError, AttributeError, IndexError):
                pass

            # TODO: find a magic number
            time.sleep(0.001)

    def connectToMainThread(self, instance):
        """Connect all log signals to slots in the Main Thread."""
        self.log_debug_sgn.connect(instance.onLogDebugReceived)
        self.log_info_sgn.connect(instance.onLogInfoReceived)
        self.log_warning_sgn.connect(instance.onLogWarningReceived)
        self.log_error_sgn.connect(instance.onLogErrorReceived)

    def stop(self):
        self._running = False


class MainGUI(QMainWindow):
    """The main GUI for azimuthal integration."""

    _root_dir = osp.dirname(osp.abspath(__file__))

    start_sgn = pyqtSignal()
    stop_sgn = pyqtSignal()
    quit_sgn = pyqtSignal()

    process_info_sgn = pyqtSignal(object)

    _db = RedisConnection()

    _SPLITTER_HANDLE_WIDTH = 9
    _SPECIAL_ANALYSIS_ICON_WIDTH = 100

    _WIDTH, _HEIGHT = config['GUI']['MAIN_GUI_SIZE']

    def __init__(self, *, start_thread_logger=False):
        """Initialization.

        :param bool start_thread_logger: True for starting ThreadLogger
            thread which allows processes to log in the MainGUI. For the
            convenience of testing, it is False by default.
        """
        super().__init__()

        self._pulse_resolved = config["PULSE_RESOLVED"]
        self._queue = deque(maxlen=1)

        self._input = MpInQueue("gui:input")
        self._close_ev = mp.Event()
        self._input.run_in_thread(self._close_ev)

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.title = f"karaboFAI {__version__} ({config['DETECTOR']})"
        self.setWindowTitle(self.title + " - main GUI")

        # *************************************************************
        # Central widget
        # *************************************************************

        self._ctrl_widgets = []  # book-keeping control widgets

        self._cw = QSplitter()
        self._cw.setChildrenCollapsible(False)
        self._cw.setHandleWidth(self._SPLITTER_HANDLE_WIDTH)
        self.setCentralWidget(self._cw)

        self._left_cw_container = QScrollArea()
        self._left_cw = QTabWidget()
        self._right_cw_container = QScrollArea()
        self._right_cw = QSplitter(Qt.Vertical)
        self._right_cw.setHandleWidth(self._SPLITTER_HANDLE_WIDTH)
        self._right_cw.setChildrenCollapsible(False)

        self._source_cw = DataSourceWidget(self)

        self._ctrl_panel_cw = QTabWidget()
        self._analysis_cw = QWidget()
        self._special_analysis_cw = QWidget()

        self._util_panel_container = QWidget()
        self._util_panel_cw = QTabWidget()

        # *************************************************************
        # Tool bar
        # Note: the order of 'addAction` affect the unittest!!!
        # *************************************************************
        self._tool_bar = self.addToolBar("Control")

        self._start_at = self._addAction("Start bridge", "start.png")
        self._start_at.triggered.connect(self.onStart)

        self._stop_at = self._addAction("Stop bridge", "stop.png")
        self._stop_at.triggered.connect(self.onStop)
        self._stop_at.setEnabled(False)

        self._tool_bar.addSeparator()

        image_tool_at = self._addAction("Image tool", "image_tool.png")
        image_tool_at.triggered.connect(lambda: (self._image_tool.show(),
                                                 self._image_tool.activateWindow()))

        open_poi_window_at = self._addAction("Pulse-of-interest", "poi.png")
        open_poi_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, PulseOfInterestWindow))
        if not self._pulse_resolved:
            open_poi_window_at.setEnabled(False)

        pump_probe_window_at = self._addAction("Pump-probe", "pump-probe.png")
        pump_probe_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, PumpProbeWindow))

        open_statistics_window_at = self._addAction("Statistics", "statistics.png")
        open_statistics_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, StatisticsWindow))

        open_corr_window_at = self._addAction("Correlation", "scatter.png")
        open_corr_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, CorrelationWindow))

        open_bin1d_window_at = self._addAction("Bin 1D", "binning1d.png")
        open_bin1d_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, Bin1dWindow))

        open_bin2d_window_at = self._addAction("Bin 2D", "heatmap.png")
        open_bin2d_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, Bin2dWindow))

        open_roi_window_at = self._addAction("ROI", "roi_monitor.png")
        open_roi_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, RoiWindow))

        self._tool_bar.addSeparator()

        open_process_monitor_at = self._addAction(
            "Process monitor", "process_monitor.png")
        open_process_monitor_at.triggered.connect(self.openProcessMonitor)

        open_file_stream_window_at = self._addAction("Offline", "offline.png")
        open_file_stream_window_at.triggered.connect(
            self.openFileStreamControllerWindow)

        open_help_at = self._addAction("About karaboFAI", "about.png")
        open_help_at.triggered.connect(self.openAboutWindow)

        # *************************************************************
        # Special analysis
        # *************************************************************

        self._trxas_btn = self._addSpecial("tr_xas.png", TrXasWindow)

        # *************************************************************
        # Miscellaneous
        # *************************************************************

        # book-keeping opened windows
        self._windows = WeakKeyDictionary()
        self._satellite_windows = WeakKeyDictionary()
        self._special_windows = WeakKeyDictionary()

        self._logger = GuiLogger(parent=self)
        logging.getLogger().addHandler(self._logger)

        self._thread_logger = ThreadLoggerBridge()
        self.quit_sgn.connect(self._thread_logger.stop)
        self._thread_logger_t = QThread()
        self._thread_logger.moveToThread(self._thread_logger_t)
        self._thread_logger_t.started.connect(self._thread_logger.recv)
        self._thread_logger.connectToMainThread(self)
        if start_thread_logger:
            self._thread_logger_t.start()

        # For real time plot
        self._running = False
        self._plot_timer = QTimer()
        self._plot_timer.timeout.connect(self.updateAll)
        self._plot_timer.start(config["PLOT_UPDATE_INTERVAL"])

        # For process monitoring
        self._proc_monitor_timer = QTimer()
        self._proc_monitor_timer.timeout.connect(self._update_process_monitoring)
        self._proc_monitor_timer.start(config["PROCESS_MONITOR_HEART_BEAT"])

        self.__redis_connection_fails = 0

        self._mon_proxy = MonProxy()

        # *************************************************************
        # control widgets
        # *************************************************************

        # analysis control widgets
        self.analysis_ctrl_widget = self.createCtrlWidget(AnalysisCtrlWidget)
        self.correlation_ctrl_widget = self.createCtrlWidget(CorrelationCtrlWidget)
        self.pump_probe_ctrl_widget = self.createCtrlWidget(PumpProbeCtrlWidget)
        self.pulse_filter_ctrl_widget = self.createCtrlWidget(PulseFilterCtrlWidget)
        self.bin_ctrl_widget = self.createCtrlWidget(BinCtrlWidget)
        self.statistics_ctrl_widget = self.createCtrlWidget(StatisticsCtrlWidget)

        # special analysis control widgets (do not register them!!!)
        self._trxas_ctrl_widget = TrXasCtrlWidget()

        # *************************************************************
        # status bar
        # *************************************************************

        # StatusBar to display topic name
        self.statusBar().showMessage(f"TOPIC: {config['TOPIC']}")
        self.statusBar().setStyleSheet("QStatusBar{font-weight:bold;}")

        # ImageToolWindow is treated differently since it is the second
        # control window.
        self._image_tool = ImageToolWindow(queue=self._queue,
                                           pulse_resolved=self._pulse_resolved,
                                           parent=self)

        self.initUI()
        self.initConnections()
        self.updateMetaData()

        self.setMinimumSize(640, 480)
        self.resize(self._WIDTH, self._HEIGHT)

        self.show()

    def createCtrlWidget(self, widget_class):
        widget = widget_class(pulse_resolved=self._pulse_resolved)
        self._ctrl_widgets.append(widget)
        return widget

    def initUI(self):
        self.initLeftUI()
        self.initRightUI()

        self._cw.addWidget(self._left_cw_container)
        self._cw.addWidget(self._right_cw_container)
        self._cw.setSizes([self._WIDTH * 0.5, self._WIDTH * 0.5])

    def initLeftUI(self):
        self._left_cw.setTabPosition(QTabWidget.TabPosition.West)

        self._left_cw.addTab(self._source_cw, "Data source")
        self._left_cw_container.setWidget(self._left_cw)
        self._left_cw_container.setWidgetResizable(True)

    def initRightUI(self):
        self.initCtrlUI()
        self.initUtilUI()

        self._right_cw.addWidget(self._ctrl_panel_cw)
        self._right_cw.addWidget(self._util_panel_container)
        self._right_cw_container.setWidget(self._right_cw)
        self._right_cw_container.setWidgetResizable(True)

    def initCtrlUI(self):
        self.initAnalysisUI()
        self.initSpecialAnalysisUI()

        self._ctrl_panel_cw.addTab(self._analysis_cw, "General analysis")
        self._ctrl_panel_cw.addTab(self._special_analysis_cw, "Special analysis")

    def initAnalysisUI(self):
        layout = QVBoxLayout()
        layout.addWidget(self.analysis_ctrl_widget)
        layout.addWidget(self.pump_probe_ctrl_widget)
        layout.addWidget(self.pulse_filter_ctrl_widget)
        layout.addWidget(self.statistics_ctrl_widget)
        layout.addWidget(self.bin_ctrl_widget)
        layout.addWidget(self.correlation_ctrl_widget)
        self._analysis_cw.setLayout(layout)

    def initSpecialAnalysisUI(self):
        ctrl_widget = QTabWidget()
        ctrl_widget.setTabPosition(QTabWidget.TabPosition.East)
        ctrl_widget.addTab(self._trxas_ctrl_widget, "tr-XAS")

        icon_layout = QVBoxLayout()
        icon_layout.addWidget(self._trxas_btn)
        icon_layout.addStretch(1)

        layout = QHBoxLayout()
        layout.addWidget(ctrl_widget)
        layout.addLayout(icon_layout)
        self._special_analysis_cw.setLayout(layout)

    def initUtilUI(self):
        self._util_panel_cw.addTab(self._logger.widget, "Logger")
        self._util_panel_cw.setTabPosition(QTabWidget.TabPosition.South)

        layout = QVBoxLayout()
        layout.addWidget(self._util_panel_cw)
        self._util_panel_container.setLayout(layout)

    def initConnections(self):
        pass

    def connectInputToOutput(self, output):
        self._input.connect(output)

    @profiler("Update Plots", process_time=True)
    def updateAll(self):
        """Update all the plots in the main and child windows."""
        if not self._running:
            return

        # Fetch all data in the queue and update history, then update plots.
        # This prevent costly GUI updating from blocking data acquisition and
        # processing.
        update_plots = False
        while True:
            try:
                processed = self._input.get_nowait()
                processed.update()
                self._queue.append(processed)
                # use this flag to prevent update the same train multiple times
                update_plots = True

                self._mon_proxy.add_tid_with_timestamp(processed.tid)
                logger.info(f"[{processed.tid}] updated")
            except Empty:
                break
            except Exception as e:
                logger.error(f"Unexpected exception: {repr(e)}")
                return

        # clear the previous plots no matter what comes next
        # for w in self._windows.keys():
        #     w.reset()

        if not update_plots:
            return

        data = self._queue[0]
        if data.image is None:
            logger.info(f"Bad train with ID: {data.tid}")
            return

        self._image_tool.updateWidgetsF()
        for w in itertools.chain(self._special_windows, self._windows):
            w.updateWidgetsF()

        logger.debug(f"Plot train with ID: {data.tid}")

    def _update_process_monitoring(self):
        self.process_info_sgn.emit(list_fai_processes())

        try:
            self._db.ping()
            self.__redis_connection_fails = 0
        except ConnectionError:
            self.__redis_connection_fails += 1
            rest_attempts = config["MAX_REDIS_PING_ATTEMPTS"] - \
                self.__redis_connection_fails

            if rest_attempts > 0:
                logger.warning(f"No response from the Redis server! Shutting "
                               f"down karaboFAI after {rest_attempts} "
                               f"attempts ...")
            else:
                logger.warning(f"No response from the Redis server! "
                               f"Shutting down karaboFAI!")
                self.close()

    def _addAction(self, description, filename):
        icon = QIcon(osp.join(self._root_dir, "icons/" + filename))
        action = QAction(icon, description, self)
        self._tool_bar.addAction(action)
        return action

    def _addSpecial(self, filename, instance_type):
        icon = QIcon(osp.join(self._root_dir, "icons/" + filename))
        btn = QPushButton()
        btn.setIcon(icon)
        btn.setIconSize(QSize(self._SPECIAL_ANALYSIS_ICON_WIDTH,
                              self._SPECIAL_ANALYSIS_ICON_WIDTH))
        btn.setFixedSize(btn.minimumSizeHint())
        btn.clicked.connect(
            lambda: self.openSpecialAnalysisWindow(instance_type))
        return btn

    def onOpenPlotWindow(self, instance_type):
        """Open a plot window if it does not exist.

        Otherwise bring the opened window to the table top.
        """
        if self._checkWindowExistence(instance_type, self._windows):
            return

        instance_type(self._queue,
                      pulse_resolved=self._pulse_resolved,
                      parent=self)

    def openSpecialAnalysisWindow(self, instance_type):
        if self._checkWindowExistence(instance_type, self._special_windows):
            return

        instance_type(self._queue, parent=self)

    def openProcessMonitor(self):
        if self._checkWindowExistence(ProcessMonitor, self._satellite_windows):
            return

        w = ProcessMonitor(parent=self)
        self.process_info_sgn.connect(w.onProcessInfoUpdate)
        return w

    def openFileStreamControllerWindow(self):
        if self._checkWindowExistence(FileStreamControllerWindow,
                                      self._satellite_windows):
            return

        return FileStreamControllerWindow(parent=self)

    def openAboutWindow(self):
        if self._checkWindowExistence(AboutWindow, self._satellite_windows):
            return

        return AboutWindow(parent=self)

    def _checkWindowExistence(self, instance_type, windows):
        for key in windows:
            if isinstance(key, instance_type):
                key.activateWindow()
                return True
        return False

    def registerWindow(self, instance):
        self._windows[instance] = 1

    def unregisterWindow(self, instance):
        del self._windows[instance]

    def registerSatelliteWindow(self, instance):
        self._satellite_windows[instance] = 1

    def unregisterSatelliteWindow(self, instance):
        del self._satellite_windows[instance]

    def registerSpecialWindow(self, instance):
        self._special_windows[instance] = 1

    def unregisterSpecialWindow(self, instance):
        del self._special_windows[instance]

    def onStart(self):
        if not self.updateMetaData():
            return
        self.start_sgn.emit()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)

        for widget in self._ctrl_widgets:
            widget.onStart()
        self._image_tool.onStart()

        self._running = True  # starting to update plots

    def onStop(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        self.stop_sgn.emit()

        # TODO: wait for some signal

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)

        for widget in self._ctrl_widgets:
            widget.onStop()
        self._image_tool.onStop()

    def updateMetaData(self):
        """Update metadata from all the ctrl widgets.

        :returns bool: True if all metadata successfully parsed
            and emitted, otherwise False.
        """
        for widget in self._ctrl_widgets:
            succeeded = widget.updateMetaData()
            if not succeeded:
                return False
        return self._image_tool.updateMetaData()

    @pyqtSlot(str)
    def onLogDebugReceived(self, msg):
        logger.debug(msg)

    @pyqtSlot(str)
    def onLogInfoReceived(self, msg):
        logger.info(msg)

    @pyqtSlot(str)
    def onLogWarningReceived(self, msg):
        logger.warning(msg)

    @pyqtSlot(str)
    def onLogErrorReceived(self, msg):
        logger.error(msg)

    def closeEvent(self, QCloseEvent):
        # prevent from logging in the GUI when it has been closed
        logging.getLogger().removeHandler(self._logger)

        # clean up the input queue
        self._close_ev.set()

        # clean up the logger thread
        self.quit_sgn.emit()
        self._thread_logger_t.quit()
        self._thread_logger_t.wait()

        # shutdown pipeline workers and Redis server
        shutdown_all()

        self._image_tool.close()
        for window in itertools.chain(self._windows,
                                      self._satellite_windows,
                                      self._special_windows):
            # Close all open child windows to make sure their resources
            # (any running process etc.) are released gracefully
            window.close()

        super().closeEvent(QCloseEvent)
