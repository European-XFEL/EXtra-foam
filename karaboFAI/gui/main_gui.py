"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Main karaboFAI GUI.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
import logging
import os.path as osp
from queue import Empty
from weakref import WeakKeyDictionary
import functools
import multiprocessing as mp

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject

from redis import ConnectionError

from .ctrl_widgets import (
    AzimuthalIntegCtrlWidget, AnalysisCtrlWidget, BinCtrlWidget,
    CorrelationCtrlWidget, PulseFilterCtrlWidget, DataSourceWidget,
    StatisticsCtrlWidget, GeometryCtrlWidget,
    PumpProbeCtrlWidget, RoiCtrlWidget
)
from .misc_widgets import GuiLogger
from .windows import (
    Bin1dWindow, Bin2dWindow, CorrelationWindow, DarkRunWindow,
    ImageToolWindow, OverviewWindow, ProcessMonitor,
    AzimuthalIntegrationWindow, StatisticsWindow, PulseOfInterestWindow,
    PumpProbeWindow, RoiWindow, FileStreamControllerWindow, AboutWindow
)
from .. import __version__
from ..config import config
from ..logger import logger
from ..utils import profiler
from ..ipc import RedisConnection, RedisPSubscriber
from ..pipeline import MpInQueue
from ..processes import list_fai_processes, shutdown_all
from ..database import MonProxy


class Data4Visualization:
    """Data shared between all the windows and widgets.

    The internal data is only modified in MainGUI.updateAll()
    """
    def __init__(self):
        self.__value = None

    def get(self):
        return self.__value

    def set(self, value):
        self.__value = value


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

    def recv_messages(self):
        while True:
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


class MainGUI(QtGui.QMainWindow):
    """The main GUI for azimuthal integration."""

    _root_dir = osp.dirname(osp.abspath(__file__))

    start_sgn = pyqtSignal()
    stop_sgn = pyqtSignal()

    process_info_sgn = pyqtSignal(object)

    available_sources_sgn = pyqtSignal(object)

    _db = RedisConnection()

    SPLITTER_HANDLE_WIDTH = 9

    def __init__(self, *, start_thread_logger=False):
        """Initialization.

        :param bool start_thread_logger: True for starting ThreadLogger
            thread which allows processes to log in the MainGUI. For the
            convenience of testing, it is False by default.
        """
        super().__init__()

        self._pulse_resolved = config["PULSE_RESOLVED"]
        self._input = MpInQueue("gui:input")
        self._close_ev = mp.Event()
        self._input.run_in_thread(self._close_ev)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.title = f"karaboFAI {__version__} ({config['DETECTOR']})"
        self.setWindowTitle(self.title + " - main GUI")

        # *************************************************************
        # Central widget
        # *************************************************************

        self._cw = QtWidgets.QSplitter()
        self._cw.setChildrenCollapsible(False)
        self._cw.setHandleWidth(self.SPLITTER_HANDLE_WIDTH)
        self.setCentralWidget(self._cw)

        self._left_cw = QtWidgets.QTabWidget()
        self._middle_cw = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._middle_cw.setHandleWidth(self.SPLITTER_HANDLE_WIDTH)
        self._middle_cw.setChildrenCollapsible(False)

        self._source_cw = DataSourceWidget()

        self._ctrl_panel_cw = QtWidgets.QTabWidget()
        self._analysis_cw = QtWidgets.QScrollArea()
        self._analysis_cw.setWidgetResizable(True)
        self._geometry_cw = QtWidgets.QScrollArea()
        self._geometry_cw.setWidgetResizable(True)

        self._util_panel_cw = QtWidgets.QTabWidget()

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

        image_tool_at = self._addAction("Image tool", "image_tool.png")
        image_tool_at.triggered.connect(lambda: ImageToolWindow(
            self._data, parent=self))

        open_overview_window_at = self._addAction("Overview", "overview.png")
        open_overview_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, OverviewWindow))

        pump_probe_window_at = self._addAction("Pump-probe", "pump-probe.png")
        pump_probe_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, PumpProbeWindow))

        open_statistics_window_at = self._addAction("Statistics", "statistics.png")
        open_statistics_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, StatisticsWindow))

        open_corr_window_at = self._addAction("Correlations", "scatter.png")
        open_corr_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, CorrelationWindow))

        open_bin1d_window_at = self._addAction("Bin 1D", "binning1d.png")
        open_bin1d_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, Bin1dWindow))

        open_bin2d_window_at = self._addAction("Bin 2D", "heatmap.png")
        open_bin2d_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, Bin2dWindow))

        open_poi_window_at = self._addAction("Pulse-of-interest", "poi.png")
        open_poi_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, PulseOfInterestWindow))
        if not self._pulse_resolved:
            open_poi_window_at.setEnabled(False)

        open_ai_window_at = self._addAction(
            "Azimuthal Integration", "azimuthal_integration.png")
        open_ai_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, AzimuthalIntegrationWindow))

        open_roi_window_at = self._addAction("ROI", "roi_monitor.png")
        open_roi_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, RoiWindow))

        open_dark_run_window_at = self._addAction("Dark run", "dark_run.png")
        open_dark_run_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, DarkRunWindow))

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
        # Miscellaneous
        # *************************************************************

        self._data = Data4Visualization()

        # book-keeping opened windows
        self._windows = WeakKeyDictionary()

        # book-keeping control widgets
        self._ctrl_widgets = []

        self._mask_image = None

        self._logger = GuiLogger(parent=self)
        logging.getLogger().addHandler(self._logger)

        self._thread_logger = ThreadLoggerBridge()
        self._thread_logger_t = QtCore.QThread()
        self._thread_logger.moveToThread(self._thread_logger_t)
        self._thread_logger_t.started.connect(
            self._thread_logger.recv_messages)
        self._thread_logger.connectToMainThread(self)
        if start_thread_logger:
            self._thread_logger_t.start()

        # For real time plot
        self._running = False
        self._plot_timer = QtCore.QTimer()
        self._plot_timer.timeout.connect(self.updateAll)
        self._plot_timer.start(config["PLOT_UPDATE_INTERVAL"])

        # For process monitoring
        self._proc_monitor_timer = QtCore.QTimer()
        self._proc_monitor_timer.timeout.connect(self._update_process_monitoring)
        self._proc_monitor_timer.start(config["PROCESS_MONITOR_HEART_BEAT"])

        self.__redis_connection_fails = 0

        self._mon_proxy = MonProxy()

        # *************************************************************
        # control widgets
        # *************************************************************

        self.registerCtrlWidget(self._source_cw.connection_ctrl_widget)

        # analysis control widgets
        self.azimuthal_integ_ctrl_widget = self.createCtrlWidget(
            AzimuthalIntegCtrlWidget)
        self.analysis_ctrl_widget = self.createCtrlWidget(AnalysisCtrlWidget)
        self.roi_ctrl_widget = self.createCtrlWidget(RoiCtrlWidget)
        self.correlation_ctrl_widget = self.createCtrlWidget(CorrelationCtrlWidget)
        self.pump_probe_ctrl_widget = self.createCtrlWidget(PumpProbeCtrlWidget)
        self.pulse_filter_ctrl_widget = self.createCtrlWidget(PulseFilterCtrlWidget)
        self.bin_ctrl_widget = self.createCtrlWidget(BinCtrlWidget)
        self.statistics_ctrl_widget = self.createCtrlWidget(StatisticsCtrlWidget)

        # geometry control widgets
        self.geometry_ctrl_widget = self.createCtrlWidget(GeometryCtrlWidget)

        # *************************************************************
        # status bar
        # *************************************************************

        # StatusBar to display topic name
        self.statusBar().showMessage(f"TOPIC: {config['TOPIC']}")
        self.statusBar().setStyleSheet("QStatusBar{font-weight:bold;}")

        self.initUI()
        self.updateMetaData()

        image_tool_at.trigger()

        self.setMinimumSize(640, 480)
        # FIXME:
        self.resize(1650, 1000)

        self.show()

    def createCtrlWidget(self, widget_class):
        widget = widget_class(pulse_resolved=self._pulse_resolved)
        self.registerCtrlWidget(widget)
        return widget

    def initUI(self):
        self._cw.addWidget(self._left_cw)
        self._cw.addWidget(self._middle_cw)

        self._middle_cw.addWidget(self._ctrl_panel_cw)
        self._middle_cw.addWidget(self._util_panel_cw)

        self.initLeftUI()
        self.initMiddleUI()

    def initLeftUI(self):
        self._left_cw.setTabPosition(
            QtWidgets.QTabWidget.TabPosition.West)

        self._left_cw.addTab(self._source_cw, "Data source")

        self.available_sources_sgn.connect(self._source_cw.updateDeviceList)

    def initMiddleUI(self):
        self.initCtrlUI()
        self.initUtilUI()

    def initCtrlUI(self):
        self._ctrl_panel_cw.addTab(self._analysis_cw, "Analysis setup")
        geom_tab = self._ctrl_panel_cw.addTab(
            self._geometry_cw, "Geometry setup")
        if not config['REQUIRE_GEOMETRY']:
            self._ctrl_panel_cw.setTabEnabled(geom_tab, False)

        self.initAnalysisUI()
        self.initGeometryUI()

    def initAnalysisUI(self):
        left_ctrl_widget = QtWidgets.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.analysis_ctrl_widget)
        layout.addWidget(self.pump_probe_ctrl_widget)
        layout.addWidget(self.azimuthal_integ_ctrl_widget)
        layout.addWidget(self.roi_ctrl_widget)
        left_ctrl_widget.setLayout(layout)

        right_ctrl_widget = QtWidgets.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.pulse_filter_ctrl_widget)
        layout.addWidget(self.statistics_ctrl_widget)
        layout.addWidget(self.bin_ctrl_widget)
        layout.addWidget(self.correlation_ctrl_widget)
        right_ctrl_widget.setLayout(layout)

        container = QtWidgets.QWidget()
        layout = QtGui.QHBoxLayout()
        layout.addWidget(left_ctrl_widget)
        layout.addWidget(right_ctrl_widget)
        container.setLayout(layout)
        self._analysis_cw.setWidget(container)

    def initGeometryUI(self):
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.geometry_ctrl_widget)
        layout.addStretch(1)
        self._geometry_cw.setLayout(layout)

    def initUtilUI(self):
        self._util_panel_cw.addTab(self._logger.widget, "Logger")

        self._util_panel_cw.setTabPosition(
            QtWidgets.QTabWidget.TabPosition.South)

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
        tid = -1
        plot_data = False
        while True:
            try:
                processed = self._input.get_nowait()
                plot_data = True
                processed.update()
                tid = processed.tid

                self._mon_proxy.add_tid_with_timestamp(tid)

                self._data.set(processed)

                self.available_sources_sgn.emit(processed.sources)

                logger.info(f"[{tid}] updated")
            except Empty:
                if plot_data:
                    break
                # no data
                return
            except Exception as e:
                logger.error(f"Unexpected exception: {repr(e)}")
                return

        # clear the previous plots no matter what comes next
        # for w in self._windows.keys():
        #     w.reset()

        if self._data.get().image is None:
            # FIXME: will this happen? I guess this is some very
            #        old code.
            logger.info(f"Bad train with ID: {tid}")
            return

        for w in self._windows.keys():
            w.update()
        logger.debug(f"Plot train with ID: {tid}")

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
        icon = QtGui.QIcon(osp.join(self._root_dir, "icons/" + filename))
        action = QtGui.QAction(icon, description, self)
        self._tool_bar.addAction(action)
        return action

    def onOpenPlotWindow(self, instance_type):
        """Open a plot window if it does not exist.

        Otherwise bring the opened window to the table top.
        """
        if self._checkWindowExistence(instance_type):
            return

        instance_type(self._data,
                      pulse_resolved=self._pulse_resolved, parent=self)

    def openProcessMonitor(self):
        if self._checkWindowExistence(ProcessMonitor):
            return

        w = ProcessMonitor(parent=self)
        self.process_info_sgn.connect(w.onProcessInfoUpdate)
        return w

    def openFileStreamControllerWindow(self):
        if self._checkWindowExistence(FileStreamControllerWindow):
            return

        w = FileStreamControllerWindow(parent=self)
        return w

    def openAboutWindow(self):
        if self._checkWindowExistence(AboutWindow):
            return

        return AboutWindow(parent=self)

    def _checkWindowExistence(self, instance_type):
        for key in self._windows:
            if isinstance(key, instance_type):
                key.activateWindow()
                return True
        return False

    def registerWindow(self, instance):
        self._windows[instance] = 1

    def unregisterWindow(self, instance):
        del self._windows[instance]

    def registerCtrlWidget(self, instance):
        self._ctrl_widgets.append(instance)

    def onStart(self):
        if not self.updateMetaData():
            return
        self.start_sgn.emit()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)

        for widget in self._ctrl_widgets:
            widget.onStart()

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

    def updateMetaData(self):
        """Update metadata from all the ctrl widgets.

        :returns bool: True if all metadata successfully parsed
            and emitted, otherwise False.
        """
        for widget in self._ctrl_widgets:
            succeeded = widget.updateMetaData()
            if not succeeded:
                return False

        return True

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

        self._close_ev.set()

        self._thread_logger_t.quit()

        shutdown_all()
        for window in list(self._windows):
            # Close all open child windows to make sure their resources
            # (any running process etc.) are released gracefully
            window.close()

        super().closeEvent(QCloseEvent)
