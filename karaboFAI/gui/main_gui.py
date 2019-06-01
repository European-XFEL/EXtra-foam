"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Main karaboFAI GUI.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import logging
import os.path as osp
from queue import Empty
from weakref import WeakKeyDictionary
import functools

from PyQt5 import QtCore, QtGui

from .ctrl_widgets import (
    AzimuthalIntegCtrlWidget, AnalysisCtrlWidget, BinCtrlWidget,
    CorrelationCtrlWidget, DataCtrlWidget, GeometryCtrlWidget,
    PumpProbeCtrlWidget, RoiCtrlWidget, XasCtrlWidget
)
from .misc_widgets import GuiLogger
from .windows import (
    Bin1DWindow, CorrelationWindow, ImageToolWindow, OverviewWindow,
    PulsedAzimuthalIntegrationWindow, PumpProbeWindow, RoiWindow,
    XasWindow
)
from .windows import (
    ProcessMonitor,
)
from .. import __version__
from ..config import config
from ..logger import logger
from ..utils import profiler
from ..pipeline import Data4Visualization
from ..pipeline.worker import ProcessManager, list_fai_processes
from .thread_worker import ThreadLoggerBridge
from ..offline import FileServerManager


class MainGUI(QtGui.QMainWindow):
    """The main GUI for azimuthal integration."""

    _root_dir = osp.dirname(osp.abspath(__file__))

    start_sgn = QtCore.pyqtSignal()
    stop_sgn = QtCore.pyqtSignal()

    process_info_sgn = QtCore.pyqtSignal(object)

    def __init__(self, *, start_thread_logger=False):
        """Initialization.

        :param bool start_thread_logger: True for starting ThreadLogger
            thread which allows processes to log in the MainGUI. For the
            convenience of testing, it is False by default.
        """
        super().__init__()

        self._pulse_resolved = config["PULSE_RESOLVED"]

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.title = f"karaboFAI {__version__} ({config['DETECTOR']})"
        self.setWindowTitle(self.title + " - main GUI")

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

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

        open_corr_window_at = self._addAction("Correlations", "scatter.png")
        open_corr_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, CorrelationWindow))

        open_xas_window_at = self._addAction("XAS", "xas.png")
        open_xas_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, XasWindow))

        open_pulsed_ai_window_at = self._addAction("Pulsed A.I", "pulsed_ai.png")
        open_pulsed_ai_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, PulsedAzimuthalIntegrationWindow))

        open_bin_window_at = self._addAction("Bin", "binning.png")
        open_bin_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, Bin1DWindow))

        open_roi_window_at = self._addAction("ROI", "roi_monitor.png")
        open_roi_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, RoiWindow))

        self._tool_bar.addSeparator()

        open_process_monitor_at = self._addAction(
            "Process monitor", "process_monitor.png")
        open_process_monitor_at.triggered.connect(self.openProcessMonitor)

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
        self._proc_monitor_timer.timeout.connect(self._publish_process_info)
        self._proc_monitor_timer.start(config["PROCESS_MONITOR_HEART_BEAT"])

        # a file server which streams data from files
        self._file_server = FileServerManager()

        # *************************************************************
        # control widgets
        # *************************************************************

        self.azimuthal_integ_ctrl_widget = AzimuthalIntegCtrlWidget(parent=self)
        if config['REQUIRE_GEOMETRY']:
            self.geometry_ctrl_widget = GeometryCtrlWidget(parent=self)

        self.analysis_ctrl_widget = AnalysisCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.roi_ctrl_widget = RoiCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.correlation_ctrl_widget = CorrelationCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.pump_probe_ctrl_widget = PumpProbeCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.xas_ctrl_widget = XasCtrlWidget(
            pulse_resolved=self._pulse_resolved, parent=self)

        self.data_ctrl_widget = DataCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.bin_ctrl_widget = BinCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.initUI()
        self.updateSharedParameters()

        image_tool_at.trigger()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.show()

    def initUI(self):
        analysis_layout = QtGui.QVBoxLayout()
        analysis_layout.addWidget(self.analysis_ctrl_widget)
        analysis_layout.addWidget(self.azimuthal_integ_ctrl_widget)
        analysis_layout.addWidget(self.roi_ctrl_widget)
        analysis_layout.addWidget(self.pump_probe_ctrl_widget)
        analysis_layout.addWidget(self.xas_ctrl_widget)

        misc_layout = QtGui.QVBoxLayout()
        misc_layout.addWidget(self.data_ctrl_widget)
        misc_layout.addWidget(self.bin_ctrl_widget)
        misc_layout.addWidget(self.correlation_ctrl_widget)
        if config['REQUIRE_GEOMETRY']:
            misc_layout.addWidget(self.geometry_ctrl_widget)
        misc_layout.addWidget(self._logger.widget)

        layout = QtGui.QHBoxLayout()
        layout.addLayout(analysis_layout, 1)
        layout.addLayout(misc_layout, 3)
        self._cw.setLayout(layout)

    def connectInput(self, worker):
        self._input = worker.output

    def updateAll(self):
        """Update all the plots in the main and child windows."""
        if not self._running:
            return

        try:
            processed_data = self._input.get_nowait()
            processed_data.update_hist()
            self._data.set(processed_data)
        except Empty:
            return

        # clear the previous plots no matter what comes next
        # for w in self._windows.keys():
        #     w.reset()

        if self._data.get().image is None:
            logger.info("Bad train with ID: {}".format(self._data.get().tid))
            return

        self._updateAllPlots()

        logger.info("Updated train with ID: {}".format(self._data.get().tid))

    @profiler("Update Plots")
    def _updateAllPlots(self):
        for w in self._windows.keys():
            w.update()

    def _publish_process_info(self):
        self.process_info_sgn.emit(list_fai_processes())

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
        if not self.updateSharedParameters():
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

    def updateSharedParameters(self):
        """Update shared parameters for all child windows.

        :returns bool: True if all shared parameters successfully parsed
            and emitted, otherwise False.
        """
        for widget in self._ctrl_widgets:
            succeeded = widget.updateSharedParameters()
            if not succeeded:
                return False

        return True

    @QtCore.pyqtSlot(str)
    def onLogDebugReceived(self, msg):
        logger.debug(msg)

    @QtCore.pyqtSlot(str)
    def onLogInfoReceived(self, msg):
        logger.info(msg)

    @QtCore.pyqtSlot(str)
    def onLogWarningReceived(self, msg):
        logger.warning(msg)

    @QtCore.pyqtSlot(str)
    def onLogErrorReceived(self, msg):
        logger.error(msg)

    def closeEvent(self, QCloseEvent):
        # prevent from logging in the GUI when it has been closed
        logging.getLogger().removeHandler(self._logger)

        self._file_server.shutdown()

        self._thread_logger_t.quit()

        ProcessManager.shutdown_pipeline()
        ProcessManager.shutdown_redis()

        super().closeEvent(QCloseEvent)
