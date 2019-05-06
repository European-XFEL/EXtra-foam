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
import time

import zmq

from .pyqtgraph import QtCore, QtGui

from .mediator import Mediator
from .ctrl_widgets import (
    AiCtrlWidget, AnalysisCtrlWidget, CorrelationCtrlWidget, DataCtrlWidget,
    GeometryCtrlWidget, PumpProbeCtrlWidget, XasCtrlWidget
)
from .misc_widgets import GuiLogger
from .windows import (
    CorrelationWindow, ImageToolWindow, OverviewWindow,
    PulsedAzimuthalIntegrationWindow, PumpProbeWindow, RoiWindow,
    SingletonWindow, XasWindow
)
from .. import __version__
from ..config import config
from ..logger import logger
from ..helpers import profiler
from ..offline import FileServer
from ..pipeline import Data4Visualization


class MainGUI(QtGui.QMainWindow):
    """The main GUI for azimuthal integration."""

    _root_dir = osp.dirname(osp.abspath(__file__))

    image_mask_sgn = QtCore.pyqtSignal(str)  # filename

    bridge_started_sgn = QtCore.pyqtSignal()
    bridge_stopped_sgn = QtCore.pyqtSignal()
    file_server_started_sgn = QtCore.pyqtSignal()
    file_server_stopped_sgn = QtCore.pyqtSignal()

    def __init__(self, bridge=None, scheduler=None):
        """Initialization."""
        super().__init__()

        self._bridge = bridge
        self._scheduler = scheduler

        mediator = Mediator()
        mediator.connect_bridge(bridge)
        mediator.connect_scheduler(scheduler)

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
        self._start_at.triggered.connect(self.onStartBridge)

        self._stop_at = self._addAction("Stop bridge", "stop.png")
        self._stop_at.triggered.connect(self.onStopBridge)
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

        open_roi_window_at = self._addAction("ROI", "roi_monitor.png")
        open_roi_window_at.triggered.connect(
            functools.partial(self.onOpenPlotWindow, RoiWindow))

        # *************************************************************
        # Miscellaneous
        # *************************************************************

        self._data = Data4Visualization()

        # book-keeping opened windows
        self._windows = WeakKeyDictionary()

        # book-keeping control widgets
        self._ctrl_widgets = []

        self._mask_image = None

        self._logger = GuiLogger(self)
        logging.getLogger().addHandler(self._logger)

        self._file_server = None

        # For real time plot
        self._running = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateAll)
        self.timer.start(config["TIMER_INTERVAL"])

        # *************************************************************
        # control widgets
        # *************************************************************

        self.ai_ctrl_widget = AiCtrlWidget(parent=self)
        if config['REQUIRE_GEOMETRY']:
            self.geometry_ctrl_widget = GeometryCtrlWidget(parent=self)

        self.analysis_ctrl_widget = AnalysisCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.correlation_ctrl_widget = CorrelationCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.pump_probe_ctrl_widget = PumpProbeCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.xas_ctrl_widget = XasCtrlWidget(
            pulse_resolved=self._pulse_resolved, parent=self)

        self.data_ctrl_widget = DataCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self.initUI()
        self.initConnection()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.show()

    def initConnection(self):
        """Set up all signal and slot connections."""
        if self._bridge is not None:
            self._bridge.message.connect(self.onMessageReceived)

        if self._scheduler is None:
            return

        self._scheduler.message.connect(self.onMessageReceived)

        if config['REQUIRE_GEOMETRY']:
            self.geometry_ctrl_widget.geometry_sgn.connect(
                self._scheduler.onGeometryChange)

        self.ai_ctrl_widget.photon_energy_sgn.connect(
            self._scheduler.onPhotonEnergyChange)
        self.ai_ctrl_widget.sample_distance_sgn.connect(
            self._scheduler.onSampleDistanceChange)
        self.ai_ctrl_widget.integration_center_sgn.connect(
            self._scheduler.onIntegrationCenterChange)
        self.ai_ctrl_widget.integration_method_sgn.connect(
            self._scheduler.onIntegrationMethodChange)
        self.ai_ctrl_widget.integration_range_sgn.connect(
            self._scheduler.onIntegrationRangeChange)
        self.ai_ctrl_widget.integration_points_sgn.connect(
            self._scheduler.onIntegrationPointsChange)
        self.ai_ctrl_widget.ai_normalizer_sgn.connect(
            self._scheduler.onAiNormalizeChange)
        self.ai_ctrl_widget.auc_x_range_sgn.connect(
            self._scheduler.onAucXRangeChange)
        self.ai_ctrl_widget.fom_integration_range_sgn.connect(
            self._scheduler.onFomIntegrationRangeChange)
        self.ai_ctrl_widget.pulsed_ai_cb.stateChanged.connect(
            self._scheduler.onPulsedAiStateChange)

        self.analysis_ctrl_widget.pulse_id_range_sgn.connect(
            self._scheduler.onPulseIdRangeChange)

        self.pump_probe_ctrl_widget.pp_pulse_ids_sgn.connect(
            self._scheduler.onPpPulseStateChange)
        self.pump_probe_ctrl_widget.pp_analysis_type_sgn.connect(
            self._scheduler.onPpAnalysisTypeChange)
        self.pump_probe_ctrl_widget.abs_difference_sgn.connect(
            self._scheduler.onPpDifferenceTypeChange)
        self.pump_probe_ctrl_widget.reset_btn.clicked.connect(
            self._scheduler.onPumpProbeReset)

        self.correlation_ctrl_widget.correlation_fom_change_sgn.connect(
            self._scheduler.onCorrelationFomChange)
        self.correlation_ctrl_widget.correlation_param_change_sgn.connect(
            self._scheduler.onCorrelationParamChange)
        self.correlation_ctrl_widget.clear_btn.clicked.connect(
            self._scheduler.onCorrelationReset)

    def initUI(self):
        analysis_layout = QtGui.QVBoxLayout()
        analysis_layout.addWidget(self.analysis_ctrl_widget)
        analysis_layout.addWidget(self.ai_ctrl_widget)
        analysis_layout.addWidget(self.pump_probe_ctrl_widget)
        analysis_layout.addWidget(self.xas_ctrl_widget)

        misc_layout = QtGui.QVBoxLayout()
        misc_layout.addWidget(self.data_ctrl_widget)
        misc_layout.addWidget(self.correlation_ctrl_widget)
        if config['REQUIRE_GEOMETRY']:
            misc_layout.addWidget(self.geometry_ctrl_widget)
        misc_layout.addWidget(self._logger.widget)

        layout = QtGui.QHBoxLayout()
        layout.addLayout(analysis_layout, 1)
        layout.addLayout(misc_layout, 3)
        self._cw.setLayout(layout)

    def updateAll(self):
        """Update all the plots in the main and child windows."""
        if not self._running:
            return

        # TODO: improve plot updating
        # Use multithreading for plot updating. However, this is not the
        # bottleneck for the performance.

        try:
            self._data.set(self._scheduler._output.get_nowait())
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

    def _addAction(self, description, filename):
        icon = QtGui.QIcon(osp.join(self._root_dir, "icons/" + filename))
        action = QtGui.QAction(icon, description, self)
        self._tool_bar.addAction(action)
        return action

    def onOpenPlotWindow(self, instance_type):
        """Open a plot window if it does not exist.

        Otherwise bring the opened window to the table top.
        """
        for key in self._windows:
            if isinstance(key, instance_type):
                key.activateWindow()
                return

        instance_type(self._data,
                      pulse_resolved=self._pulse_resolved, parent=self)

    def registerWindow(self, instance):
        self._windows[instance] = 1

    def unregisterWindow(self, instance):
        del self._windows[instance]

    def registerCtrlWidget(self, instance):
        self._ctrl_widgets.append(instance)
        self.bridge_started_sgn.connect(instance.onBridgeStarted)
        self.bridge_stopped_sgn.connect(instance.onBridgeStopped)

    def onStartBridge(self):
        """Actions taken before the start of a 'run'."""
        self.clearQueues()
        self._running = True  # starting to update plots

        if not self.updateSharedParameters():
            return
        self._scheduler.start()
        self._bridge.start()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)

        self.bridge_started_sgn.emit()

    def onStopBridge(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        time.sleep(0.2)

        self.clearWorkers()
        self.clearQueues()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)

        self.bridge_stopped_sgn.emit()

    def clearWorkers(self):
        self._scheduler.terminate()
        self._bridge.terminate()
        self._scheduler.wait()
        self._bridge.wait()

    def clearQueues(self):
        self._bridge.clear_queue()
        self._scheduler.clear_queue()

    def onStartServeFile(self):
        """Actions taken before the start of file serving."""
        # process can only be start once
        folder, port = self.data_ctrl_widget.file_server
        self._file_server = FileServer(folder, port)
        try:
            # TODO: signal the end of file serving
            self._file_server.start()
            logger.info("Start serving file in the folder {} through port {}"
                        .format(folder, port))
        except FileNotFoundError:
            logger.info("{} does not exist!".format(folder))
            return
        except zmq.error.ZMQError:
            logger.info("Port {} is already in use!".format(port))
            return

        self.file_server_started_sgn.emit()

    def onStopServeFile(self):
        """Actions taken before the end of file serving."""
        self._file_server.terminate()

        self.file_server_stopped_sgn.emit()

    def updateSharedParameters(self):
        """Update shared parameters for all child windows.

        :returns bool: True if all shared parameters successfully parsed
            and emitted, otherwise False.
        """
        for widget in self._ctrl_widgets:
            info = widget.updateSharedParameters()
            if not info:
                return False
        return True

    @QtCore.pyqtSlot(str)
    def onMessageReceived(self, msg):
        logger.info(msg)

    def closeEvent(self, QCloseEvent):
        self.clearWorkers()

        if self._file_server is not None and self._file_server.is_alive():
            self._file_server.terminate()

        # useful in unittests
        SingletonWindow._instances.clear()

        super().closeEvent(QCloseEvent)
