"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Main karaboFAI GUI.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse
import logging
import os.path as osp
from queue import Queue, Empty
import sys
from weakref import WeakKeyDictionary

import zmq

from .pyqtgraph import QtCore, QtGui

from .mediator import Mediator
from .ctrl_widgets import (
    AiCtrlWidget, AnalysisCtrlWidget, CorrelationCtrlWidget, DataCtrlWidget,
    GeometryCtrlWidget, PumpProbeCtrlWidget, XasCtrlWidget
)
from .misc_widgets import GuiLogger
from .windows import (
    CorrelationWindow, ImageToolWindow, OverviewWindow, PumpProbeWindow,
    XasWindow
)
from .. import __version__
from ..config import config
from ..logger import logger
from ..helpers import profiler
from ..offline import FileServer
from ..pipeline import DataAcquisition, PipelineLauncher, Data4Visualization


class MainGUI(QtGui.QMainWindow):
    """The main GUI for azimuthal integration."""

    _root_dir = osp.dirname(osp.abspath(__file__))

    image_mask_sgn = QtCore.pyqtSignal(str)  # filename

    daq_started_sgn = QtCore.pyqtSignal()
    daq_stopped_sgn = QtCore.pyqtSignal()
    file_server_started_sgn = QtCore.pyqtSignal()
    file_server_stopped_sgn = QtCore.pyqtSignal()

    def __init__(self, detector):
        """Initialization."""
        super().__init__()

        # update global configuration
        config.load(detector)

        self._pulse_resolved = config["PULSE_RESOLVED"]

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.title = f"karaboFAI {__version__} ({detector})"
        self.setWindowTitle(self.title + " - main GUI")

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        # *************************************************************
        # Tool bar
        # Note: the order of 'addAction` affect the unittest!!!
        # *************************************************************
        self._tool_bar = self.addToolBar("Control")

        self._start_at = self._addAction("Start DAQ", "start.png")
        self._start_at.triggered.connect(self.onStartDAQ)

        self._stop_at = self._addAction("Stop DAQ", "stop.png")
        self._stop_at.triggered.connect(self.onStopDAQ)
        self._stop_at.setEnabled(False)

        image_tool_at = self._addAction("Image tool", "image_tool.png")
        image_tool_at.triggered.connect(lambda: ImageToolWindow(
            self._data, parent=self))

        open_overview_window_at = self._addAction("Overview", "overview.png")
        open_overview_window_at.triggered.connect(
            lambda: OverviewWindow(self._data,
                                   pulse_resolved=self._pulse_resolved,
                                   parent=self))

        pump_probe_window_at = self._addAction("Pump-probe", "pump-probe.png")
        pump_probe_window_at.triggered.connect(
            lambda: PumpProbeWindow(self._data,
                                    pulse_resolved=self._pulse_resolved,
                                    parent=self))

        open_corr_window_at = self._addAction("Correlations", "scatter.png")
        open_corr_window_at.triggered.connect(
            lambda: CorrelationWindow(self._data,
                                      pulse_resolved=self._pulse_resolved,
                                      parent=self))

        open_xas_window_at = self._addAction("XAS", "xas.png")
        open_xas_window_at.triggered.connect(
            lambda: XasWindow(self._data,
                              pulse_resolved=self._pulse_resolved,
                              parent=self))

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

        self._daq_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._proc_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])

        # a DAQ worker which acquires the data in another thread
        self._daq_worker = DataAcquisition(self._daq_queue)
        # a data processing worker which processes the data in another thread
        self._pipe_worker = PipelineLauncher(self._daq_queue, self._proc_queue)

        # initializing mediator
        mediator = Mediator()
        mediator.setPipeline(self._pipe_worker)
        mediator.setDaq(self._daq_worker)

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
        self._daq_worker.message.connect(self.onMessageReceived)

        self._pipe_worker.message.connect(self.onMessageReceived)

        if config['REQUIRE_GEOMETRY']:
            self.geometry_ctrl_widget.geometry_sgn.connect(
                self._pipe_worker.onGeometryChange)

        self.ai_ctrl_widget.photon_energy_sgn.connect(
            self._pipe_worker.onPhotonEnergyChange)
        self.ai_ctrl_widget.sample_distance_sgn.connect(
            self._pipe_worker.onSampleDistanceChange)
        self.ai_ctrl_widget.integration_center_sgn.connect(
            self._pipe_worker.onIntegrationCenterChange)
        self.ai_ctrl_widget.integration_method_sgn.connect(
            self._pipe_worker.onIntegrationMethodChange)
        self.ai_ctrl_widget.integration_range_sgn.connect(
            self._pipe_worker.onIntegrationRangeChange)
        self.ai_ctrl_widget.integration_points_sgn.connect(
            self._pipe_worker.onIntegrationPointsChange)
        self.ai_ctrl_widget.ai_normalizer_sgn.connect(
            self._pipe_worker.onAiNormalizeChange)
        self.ai_ctrl_widget.auc_x_range_sgn.connect(
            self._pipe_worker.onAucXRangeChange)
        self.ai_ctrl_widget.fom_integration_range_sgn.connect(
            self._pipe_worker.onFomIntegrationRangeChange)

        self.analysis_ctrl_widget.enable_ai_cb.stateChanged.connect(
            self._pipe_worker.onEnableAiStateChange)
        self.analysis_ctrl_widget.pulse_id_range_sgn.connect(
            self._pipe_worker.onPulseIdRangeChange)

        self.pump_probe_ctrl_widget.pp_pulse_ids_sgn.connect(
            self._pipe_worker.onPpPulseStateChange)
        self.pump_probe_ctrl_widget.pp_analysis_type_sgn.connect(
            self._pipe_worker.onPpAnalysisTypeChange)
        self.pump_probe_ctrl_widget.abs_difference_sgn.connect(
            self._pipe_worker.onPpDifferenceTypeChange)
        self.pump_probe_ctrl_widget.reset_btn.clicked.connect(
            self._pipe_worker.onLaserOnOffClear)

        self.correlation_ctrl_widget.correlation_fom_change_sgn.connect(
            self._pipe_worker.onCorrelationFomChange)
        self.correlation_ctrl_widget.correlation_param_change_sgn.connect(
            self._pipe_worker.onCorrelationParamChange)
        self.correlation_ctrl_widget.clear_btn.clicked.connect(
            self._pipe_worker.onCorrelationClear)

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
            self._data.set(self._proc_queue.get_nowait())
        except Empty:
            return

        # clear the previous plots no matter what comes next
        for w in self._windows.keys():
            w.reset()

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

    def registerWindow(self, instance):
        self._windows[instance] = 1

    def unregisterWindow(self, instance):
        del self._windows[instance]

    def registerCtrlWidget(self, instance):
        self._ctrl_widgets.append(instance)

    def onStartDAQ(self):
        """Actions taken before the start of a 'run'."""
        self.clearQueues()
        self._running = True  # starting to update plots

        if not self.updateSharedParameters():
            return
        self._pipe_worker.start()
        self._daq_worker.start()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)

        self.daq_started_sgn.emit()

    def onStopDAQ(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        self.clearWorkers()
        self.clearQueues()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)

        self.daq_stopped_sgn.emit()

    def clearWorkers(self):
        self._pipe_worker.terminate()
        self._daq_worker.terminate()
        self._pipe_worker.wait()
        self._daq_worker.wait()

    def clearQueues(self):
        with self._daq_queue.mutex:
            self._daq_queue.queue.clear()
        with self._proc_queue.mutex:
            self._proc_queue.queue.clear()

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

        super().closeEvent(QCloseEvent)


def start():
    parser = argparse.ArgumentParser(prog="karaboFAI")
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("detector", help="detector name (case insensitive)",
                        choices=['AGIPD', 'LPD', 'JUNGFRAU', 'FASTCCD'],
                        type=lambda s: s.upper())

    args = parser.parse_args()

    detector = args.detector
    if detector == 'JUNGFRAU':
        detector = 'JungFrau'
    elif detector == 'FASTCCD':
        detector = 'FastCCD'
    else:
        detector = detector.upper()

    app = QtGui.QApplication(sys.argv)
    ex = MainGUI(detector)
    app.exec_()


if __name__ == "__main__":
    start()
