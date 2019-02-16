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
import os
import sys
import time
from queue import Queue, Empty
from weakref import WeakKeyDictionary
import logging

import zmq

from .data_acquisition import DataAcquisition
from .data_processing import DataProcessor, Data4Visualization
from .widgets.pyqtgraph import QtCore, QtGui
from .widgets import (
    AiCtrlWidget, AnalysisCtrlWidget, CorrelationCtrlWidget, DataCtrlWidget,
    GeometryCtrlWidget, GuiLogger, PumpProbeCtrlWidget
)
from .windows import (
    CorrelationWindow, DrawMaskWindow, ImageToolWindow,
    OverviewWindow, OverviewWindowTrainResolved
)
from .file_server import FileServer
from .config import config
from .logger import logger

from . import __version__


class Mediator(QtCore.QObject):
    roi_displayed_range_sgn = QtCore.pyqtSignal(int)
    roi_hist_clear_sgn = QtCore.pyqtSignal()
    roi_value_tyoe_change_sgn = QtCore.pyqtSignal(object)
    roi1_region_change_sgn = QtCore.pyqtSignal(bool, int, int, int, int)
    roi2_region_change_sgn = QtCore.pyqtSignal(bool, int, int, int, int)
    bkg_change_sgn = QtCore.pyqtSignal(float)

    crop_area_change_sgn = QtCore.pyqtSignal(bool, int, int, int, int)

    threshold_mask_change_sgn = QtCore.pyqtSignal(float, float)

    @QtCore.pyqtSlot()
    def onRoiDisplayedRangeChange(self):
        v = int(self.sender().text())
        self.roi_displayed_range_sgn.emit(v)

    @QtCore.pyqtSlot()
    def onRoiHistClear(self):
        self.roi_hist_clear_sgn.emit()

    @QtCore.pyqtSlot(object)
    def onRoiValueTypeChange(self, state):
        self.roi_value_tyoe_change_sgn.emit(state)

    @QtCore.pyqtSlot(bool, int, int, int, int)
    def onRoi1Change(self, activated, w, h, px, py):
        self.roi1_region_change_sgn.emit(activated, w, h, px, py)

    @QtCore.pyqtSlot(bool, int, int, int, int)
    def onRoi2Change(self, activated, w, h, px, py):
        self.roi2_region_change_sgn.emit(activated, w, h, px, py)

    @QtCore.pyqtSlot()
    def onBkgChange(self):
        self.bkg_change_sgn.emit(float(self.sender().text()))

    @QtCore.pyqtSlot(bool, int, int, int, int)
    def onCropAreaChange(self, restore, w, h, px, py):
        self.crop_area_change_sgn.emit(restore, w, h, px, py)

    @QtCore.pyqtSlot(float, float)
    def onThresholdMaskChange(self, lb, ub):
        self.threshold_mask_change_sgn.emit(lb, ub)


class MainGUI(QtGui.QMainWindow):
    """The main GUI for azimuthal integration."""

    _root_dir = os.path.dirname(os.path.abspath(__file__))

    image_mask_sgn = QtCore.pyqtSignal(str)  # filename

    daq_started_sgn = QtCore.pyqtSignal()
    daq_stopped_sgn = QtCore.pyqtSignal()
    file_server_started_sgn = QtCore.pyqtSignal()
    file_server_stopped_sgn = QtCore.pyqtSignal()

    def __init__(self, detector):
        """Initialization."""
        super().__init__()

        self._mediator = Mediator()

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

        #
        self._start_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/start.png")),
            "Start DAQ",
            self)
        self._tool_bar.addAction(self._start_at)
        self._start_at.triggered.connect(self.onStartDAQ)

        #
        self._stop_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/stop.png")),
            "Stop DAQ",
            self)
        self._tool_bar.addAction(self._stop_at)
        self._stop_at.triggered.connect(self.onStopDAQ)
        self._stop_at.setEnabled(False)

        #
        self._draw_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/draw_mask.png")),
            "Draw mask",
            self)
        self._draw_mask_at.triggered.connect(
            lambda: DrawMaskWindow(self._data, parent=self))
        self._tool_bar.addAction(self._draw_mask_at)

        #
        load_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/load_mask.png")),
            "Load mask",
            self)
        load_mask_at.triggered.connect(self.loadMaskImage)
        self._tool_bar.addAction(load_mask_at)

        #
        image_tool_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/image_tool.png")),
            "Image tool",
            self)
        image_tool_at.triggered.connect(
            lambda: ImageToolWindow(
                self._data, mediator=self._mediator, parent=self))
        self._tool_bar.addAction(image_tool_at)

        open_overview_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/overview.png")),
            "Overview",
            self)
        if self._pulse_resolved:
            open_overview_window_at.triggered.connect(
                lambda: OverviewWindow(self._data, parent=self))
        else:
            open_overview_window_at.triggered.connect(
                lambda: OverviewWindowTrainResolved(
                    self._data, mediator=self._mediator, parent=self))

        self._tool_bar.addAction(open_overview_window_at)

        #
        open_correlation_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(self._root_dir, "icons/scatter.png")),
            "Correlations",
            self)
        open_correlation_window_at.triggered.connect(
            lambda: CorrelationWindow(self._data,
                                      parent=self,
                                      pulse_resolved=self._pulse_resolved))
        self._tool_bar.addAction(open_correlation_window_at)

        # *************************************************************
        # Miscellaneous
        # *************************************************************

        self._data = Data4Visualization()

        # book-keeping opened windows
        self._windows = WeakKeyDictionary()

        # book-keeping control widgets
        self._ctrl_widgets = []

        self._mask_image = None

        self._disabled_widgets_during_daq = [
            load_mask_at,
        ]

        self._logger = GuiLogger(self)
        logging.getLogger().addHandler(self._logger)

        self._file_server = None

        self._daq_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._proc_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])

        # a DAQ worker which acquires the data in another thread
        self._daq_worker = DataAcquisition(self._daq_queue)
        # a data processing worker which processes the data in another thread
        self._proc_worker = None

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

        self.data_ctrl_widget = DataCtrlWidget(
            parent=self, pulse_resolved=self._pulse_resolved)

        self._proc_worker = DataProcessor(self._daq_queue, self._proc_queue)

        self.initUI()
        self.initConnection()

        self.setFixedSize(self.minimumSizeHint())

        self.show()

    def initConnection(self):
        """Set up all signal and slot connections."""
        self._daq_worker.message.connect(self.onMessageReceived)

        self.data_ctrl_widget.source_type_sgn.connect(
            self._proc_worker.onSourceTypeChange)
        self.data_ctrl_widget.server_tcp_sgn.connect(
            self._daq_worker.onServerTcpChanged)
        self.data_ctrl_widget.source_name_sgn.connect(
            self._proc_worker.onSourceNameChange)

        self._proc_worker.message.connect(self.onMessageReceived)

        self.image_mask_sgn.connect(self._proc_worker.onImageMaskChanged)

        if config['REQUIRE_GEOMETRY']:
            self.geometry_ctrl_widget.geometry_sgn.connect(
                self._proc_worker.onGeometryChanged)

        self.ai_ctrl_widget.photon_energy_sgn.connect(
            self._proc_worker.onPhotonEnergyChanged)
        self.ai_ctrl_widget.sample_distance_sgn.connect(
            self._proc_worker.onSampleDistanceChanged)
        self.ai_ctrl_widget.poni_sgn.connect(
            self._proc_worker.onPoniChange)
        self.ai_ctrl_widget.integration_method_sgn.connect(
            self._proc_worker.onIntegrationMethodChanged)
        self.ai_ctrl_widget.integration_range_sgn.connect(
            self._proc_worker.onIntegrationRangeChanged)
        self.ai_ctrl_widget.integration_points_sgn.connect(
            self._proc_worker.onIntegrationPointsChanged)

        self.analysis_ctrl_widget.pulse_id_range_sgn.connect(
            self._proc_worker.onPulseRangeChanged)
        self.analysis_ctrl_widget.enable_ai_cb.stateChanged.connect(
            self._proc_worker.onEnableAiStateChange)
        self.analysis_ctrl_widget.enable_ai_cb.setChecked(True)
        self.analysis_ctrl_widget.ai_normalizer_sgn.connect(
            self._proc_worker.onAiNormalizeChange)
        self.analysis_ctrl_widget.normalization_range_sgn.connect(
            self._proc_worker.onNormalizationRangeChange)
        self.analysis_ctrl_widget.integration_range_sgn.connect(
            self._proc_worker.onFomIntegrationRangeChange)

        self._mediator.roi_hist_clear_sgn.connect(
            self._proc_worker.onRoiHistClear)
        self._mediator.roi1_region_change_sgn.connect(
            self._proc_worker.onRoi1Change)
        self._mediator.roi_value_tyoe_change_sgn.connect(
            self._proc_worker.onRoiValueTypeChange)
        self._mediator.roi2_region_change_sgn.connect(
            self._proc_worker.onRoi2Change)
        self._mediator.bkg_change_sgn.connect(
            self._proc_worker.onBkgChange)
        self._mediator.threshold_mask_change_sgn.connect(
            self._proc_worker.onThresholdMaskChange)
        self._mediator.crop_area_change_sgn.connect(
            self._proc_worker.onCropAreaChange)

        self.pump_probe_ctrl_widget.on_off_pulse_ids_sgn.connect(
            self._proc_worker.onOffPulseStateChange)
        self.pump_probe_ctrl_widget.abs_difference_sgn.connect(
            self._proc_worker.onAbsDifferenceStateChange)
        self.pump_probe_ctrl_widget.moving_avg_window_sgn.connect(
            self._proc_worker.onMovingAverageWindowChange)
        self.pump_probe_ctrl_widget.reset_btn.clicked.connect(
            self._proc_worker.onLaserOnOffClear)

        self.correlation_ctrl_widget.correlation_param_sgn.connect(
            self._proc_worker.onCorrelationParamChange)
        self.correlation_ctrl_widget.correlation_fom_sgn.connect(
            self._proc_worker.onCorrelationFomChange)

    def initUI(self):
        misc_layout = QtGui.QHBoxLayout()
        misc_layout.addWidget(self.ai_ctrl_widget)
        if config['REQUIRE_GEOMETRY']:
            misc_layout.addWidget(self.geometry_ctrl_widget)
        misc_layout.addWidget(self.data_ctrl_widget)

        right_layout = QtGui.QVBoxLayout()
        right_layout.addLayout(misc_layout)
        right_layout.addWidget(self._logger.widget)

        analysis_layout = QtGui.QVBoxLayout()
        analysis_layout.addWidget(self.analysis_ctrl_widget)
        analysis_layout.addWidget(self.correlation_ctrl_widget)
        analysis_layout.addWidget(self.pump_probe_ctrl_widget)

        layout = QtGui.QHBoxLayout()
        layout.addLayout(analysis_layout)
        layout.addLayout(right_layout)
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
            w.clear()

        if self._data.get().image is None:
            logger.info("Bad train with ID: {}".format(self._data.get().tid))
            return

        t0 = time.perf_counter()

        # update the all the plots
        for w in self._windows.keys():
            w.update()

        logger.debug("Time for updating the plots: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        logger.info("Updated train with ID: {}".format(self._data.get().tid))

    def registerWindow(self, instance):
        self._windows[instance] = 1

    def unregisterWindow(self, instance):
        del self._windows[instance]

    def registerCtrlWidget(self, instance):
        self._ctrl_widgets.append(instance)

    def loadMaskImage(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if not filename:
            logger.error("Please specify the image mask file!")
        self.image_mask_sgn.emit(filename)

    def onStartDAQ(self):
        """Actions taken before the start of a 'run'."""
        self.clearQueues()
        self._running = True  # starting to update plots

        if not self.updateSharedParameters():
            return
        self._proc_worker.start()
        self._daq_worker.start()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(False)
        self.daq_started_sgn.emit()

    def onStopDAQ(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        self.clearWorkers()
        self.clearQueues()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(True)
        self.daq_stopped_sgn.emit()

    def clearWorkers(self):
        self._proc_worker.terminate()
        self._daq_worker.terminate()
        self._proc_worker.wait()
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
        total_info = ""
        for widget in self._ctrl_widgets:
            info = widget.updateSharedParameters()
            if info is None:
                return False
            total_info += info

        logger.info(total_info)
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
