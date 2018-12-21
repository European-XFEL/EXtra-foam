"""
Offline and online data analysis and visualization tool for Centre  of
mass analysis from different data acquired with various detectors at
European XFEL.

Main Bragg GUI.

Author: Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import time
import logging
from queue import Queue, Empty
from weakref import WeakKeyDictionary

import zmq

from .logger import logger
from .widgets.pyqtgraph import QtCore, QtGui
from .widgets import (
    CustomGroupBox, FixedWidthLineEdit, GuiLogger, InputDialogWithCheckBox,
    AiSetUpWidget, GmtSetUpWidget, ExpSetUpWidget, DataSrcWidget
)
from .windows import (
    BraggSpotsWindow, DrawMaskWindow, LaserOnOffWindow, OverviewWindow
)
from .data_acquisition import DataAcquisition
from .data_processing import DataSource, DataProcessor, ProcessedData
from .file_server import FileServer
from .config import config
from .helpers import parse_ids, parse_boundary, parse_table_widget


class MainBraggGUI(QtGui.QMainWindow):
    """The main GUI for azimuthal integration."""

    class Data4Visualization:
        """Data shared between all the windows and widgets.

        The internal data is only modified in MainGUI.updateAll()
        """
        def __init__(self):
            self.__value = ProcessedData(-1)

        def get(self):
            return self.__value

        def set(self, value):
            self.__value = value

    # *************************************************************
    # signals related to shared parameters
    # *************************************************************

    server_tcp_sgn = QtCore.pyqtSignal(str, str)
    data_source_sgn = QtCore.pyqtSignal(object)

    # (geometry file, quadrant positions)
    geometry_sgn = QtCore.pyqtSignal(str, list)

    sample_distance_sgn = QtCore.pyqtSignal(float)
    center_coordinate_sgn = QtCore.pyqtSignal(int, int)  # (cx, cy)
    integration_method_sgn = QtCore.pyqtSignal(str)
    integration_range_sgn = QtCore.pyqtSignal(float, float)
    integration_points_sgn = QtCore.pyqtSignal(int)

    mask_range_sgn = QtCore.pyqtSignal(float, float)
    diff_integration_range_sgn = QtCore.pyqtSignal(float, float)
    normalization_range_sgn = QtCore.pyqtSignal(float, float)
    ma_window_size_sgn = QtCore.pyqtSignal(int)
    # (mode, on-pulse ids, off-pulse ids)
    on_off_pulse_ids_sgn = QtCore.pyqtSignal(str, list, list)
    photon_energy_sgn = QtCore.pyqtSignal(float)

    pulse_range_sgn = QtCore.pyqtSignal(int, int)

    # *************************************************************
    # other signals
    # *************************************************************

    image_mask_sgn = QtCore.pyqtSignal(str)  # filename

    _height = 600  # window height, in pixel
    _width = 1380  # window width, in pixel

    def __init__(self, topic, screen_size=None):
        """Initialization.

        :param str topic: detector topic, allowed options "SPB", "FXE"
        """
        super().__init__()

        # update global configuration
        config.load(topic)

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(self._width, self._height)

        self.title = topic + " Centre of Mass Analysis"
        self.setWindowTitle(self.title + " - main GUI")

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        # *************************************************************
        # Tool bar
        # *************************************************************
        tool_bar = self.addToolBar("Control")

        root_dir = os.path.dirname(os.path.abspath(__file__))

        #
        self._start_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/start.png")),
            "Start DAQ",
            self)
        tool_bar.addAction(self._start_at)
        self._start_at.triggered.connect(self._onStartDAQ)

        #
        self._stop_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/stop.png")),
            "Stop DAQ",
            self)
        tool_bar.addAction(self._stop_at)
        self._stop_at.triggered.connect(self._onStopDAQ)
        self._stop_at.setEnabled(False)

        #
        open_overview_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/overview.png")),
            "Overview",
            self)
        open_overview_window_at.triggered.connect(
            lambda: OverviewWindow(self._data, parent=self))
        tool_bar.addAction(open_overview_window_at)

        #
        open_laseronoff_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/on_off_pulses.png")),
            "On- and off- pulses",
            self)
        open_laseronoff_window_at.triggered.connect(
            lambda: LaserOnOffWindow(self._data, parent=self))
        tool_bar.addAction(open_laseronoff_window_at)

        #
        open_bragg_spots_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/bragg_spots.png")),
            "Bragg spots",
            self)
        open_bragg_spots_window_at.triggered.connect(
            lambda: BraggSpotsWindow(self._data, parent=self))
        tool_bar.addAction(open_bragg_spots_window_at)

        #
        self._draw_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/draw_mask.png")),
            "Draw mask",
            self)
        self._draw_mask_at.triggered.connect(
            lambda: DrawMaskWindow(self._data, parent=self))
        tool_bar.addAction(self._draw_mask_at)

        #
        self._load_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/load_mask.png")),
            "Load mask",
            self)
        self._load_mask_at.triggered.connect(self._loadMaskImage)
        tool_bar.addAction(self._load_mask_at)

        #
        self._load_geometry_file_at = QtGui.QAction(
            QtGui.QIcon(
                self.style().standardIcon(QtGui.QStyle.SP_DriveCDIcon)),
            "geometry file",
            self)
        self._load_geometry_file_at.triggered.connect(
            self._loadGeometryFile)
        tool_bar.addAction(self._load_geometry_file_at)

        # *************************************************************
        # Miscellaneous
        # *************************************************************
        self._data = self.Data4Visualization()

        # book-keeping opened widgets and windows
        self._plot_windows = WeakKeyDictionary()

        # book-keeping control widgets
        self._control_widgets = WeakKeyDictionary()

        self._mask_image = None

        self._ai_setup_gp = AiSetUpWidget(parent=self)
        self._gmt_setup_gp = GmtSetUpWidget(parent=self)
        self._exp_setup_gp = ExpSetUpWidget(parent=self)
        self._data_src_gp = DataSrcWidget(parent=self)

        self._ai_children = self._ai_setup_gp.children
        self._gmt_children = self._gmt_setup_gp.children
        self._exp_children = self._exp_setup_gp.children
        self._datasrc_children = self._data_src_gp.children

        # *************************************************************
        # log window
        # *************************************************************

        self._logger = GuiLogger(self)
        logging.getLogger().addHandler(self._logger)

        # *************************************************************
        # file server
        # *************************************************************
        self._file_server = None

        self._file_server_widget = CustomGroupBox("Data stream server")
        self._server_start_btn = QtGui.QPushButton("Serve")
        self._server_start_btn.clicked.connect(self._onStartServeFile)
        self._server_terminate_btn = QtGui.QPushButton("Terminate")
        self._server_terminate_btn.setEnabled(False)
        self._server_terminate_btn.clicked.connect(
            self._onStopServeFile)

        self._disabled_widgets_during_file_serving = [
            self._datasrc_children.source_name_le,
        ]

        self._initFileServerUI()
        self._initUI()

        if screen_size is None:
            self.move(0, 0)
        else:
            self.move(screen_size.width()/2 - self._width/2,
                      screen_size.height()/20)

        # TODO: implement
        self._datasrc_children.pulse_range0_le.setEnabled(False)

        self._disabled_widgets_during_daq = [
            self._load_mask_at,
            self._load_geometry_file_at,
            self._datasrc_children.hostname_le,
            self._datasrc_children.port_le,
            self._datasrc_children.pulse_range1_le,
            self._ai_children.sample_dist_le,
            self._ai_children.cx_le,
            self._ai_children.cy_le,
            self._ai_children.itgt_method_cb,
            self._ai_children.itgt_range_le,
            self._ai_children.itgt_points_le,
            self._ai_children.mask_range_le,
            self._gmt_children.geom_file_le,
            self._gmt_children.quad_positions_tb,
            self._exp_children.photon_energy_le,
            self._exp_children.laser_mode_cb,
            self._exp_children.on_pulse_le,
            self._exp_children.off_pulse_le,
            self._exp_children.normalization_range_le,
            self._exp_children.diff_integration_range_le,
            self._exp_children.ma_window_le
        ]

        self._disabled_widgets_during_daq.extend(self._datasrc_children.data_src_rbts)

        self._daq_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._proc_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])

        # a DAQ worker which acquires the data in another thread
        self._daq_worker = DataAcquisition(self._daq_queue)
        # a data processing worker which processes the data in another thread
        self._proc_worker = DataProcessor(self._daq_queue, self._proc_queue)

        self._initPipeline()
        # For real time plot
        self._running = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._updateAll)
        self.timer.start(config["TIMER_INTERVAL"])
        self.show()

    def _initPipeline(self):
        """Set up all signal and slot connections for pipeline."""
        # *************************************************************
        # DataProcessor
        # *************************************************************

        self._daq_worker.message.connect(self.onMessageReceived)

        self.server_tcp_sgn.connect(self._daq_worker.onServerTcpChanged)

        # *************************************************************
        # DataProcessor
        # *************************************************************

        self._proc_worker.message.connect(self.onMessageReceived)

        self.data_source_sgn.connect(self._proc_worker.onSourceChanged)
        self.geometry_sgn.connect(self._proc_worker.onGeometryChanged)
        self.sample_distance_sgn.connect(
            self._proc_worker.onSampleDistanceChanged)
        self.center_coordinate_sgn.connect(
            self._proc_worker.onCenterCoordinateChanged)
        self.integration_method_sgn.connect(
            self._proc_worker.onIntegrationMethodChanged)
        self.integration_range_sgn.connect(
            self._proc_worker.onIntegrationRangeChanged)
        self.integration_points_sgn.connect(
            self._proc_worker.onIntegrationPointsChanged)
        self.mask_range_sgn.connect(self._proc_worker.onMaskRangeChanged)
        self.photon_energy_sgn.connect(self._proc_worker.onPhotonEnergyChanged)
        self.pulse_range_sgn.connect(self._proc_worker.onPulseRangeChanged)

        self.image_mask_sgn.connect(self._proc_worker.onImageMaskChanged)

    def _initUI(self):
        layout = QtGui.QGridLayout()

        layout.addWidget(self._ai_setup_gp, 0, 0, 4, 1)
        layout.addWidget(self._gmt_setup_gp, 0, 1, 4, 1)
        layout.addWidget(self._exp_setup_gp, 0, 2, 4, 1)
        layout.addWidget(self._data_src_gp, 0, 3, 4, 1)
        layout.addWidget(self._logger.widget, 4, 0, 2, 3)
        layout.addWidget(self._file_server_widget, 4, 3, 2, 1)
        self._cw.setLayout(layout)

    def registerControlWidget(self, instance):
        self._control_widgets[instance] = 1

    def unregisterControlWidget(self, instance):
        del self._control_widgets[instance]

    def _initFileServerUI(self):
        layout = QtGui.QGridLayout()

        layout.addWidget(self._server_start_btn, 0, 0, 1, 1)
        layout.addWidget(self._server_terminate_btn, 0, 1, 1, 1)

        self._file_server_widget.setLayout(layout)


    def _updateAll(self):
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
        for w in self._plot_windows.keys():
            w.clear()

        if self._data.get().empty():
            logger.info("Bad train with ID: {}".format(self._data.get().tid))
            return

        t0 = time.perf_counter()

        # update the all the plots
        for w in self._plot_windows.keys():
            w.update()

        logger.debug("Time for updating the plots: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        logger.info("Updated train with ID: {}".format(self._data.get().tid))

    def registerPlotWindow(self, instance):
        self._plot_windows[instance] = 1

    def unregisterPlotWindow(self, instance):
        del self._plot_windows[instance]

    def _loadGeometryFile(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if filename:
            self._gmt_children.geom_file_le.setText(filename)

    def _loadMaskImage(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if not filename:
            logger.error("Please specify the image mask file!")
        self.image_mask_sgn.emit(filename)

    def _onStartDAQ(self):
        """Actions taken before the start of a 'run'."""
        self._clearQueues()
        self._running = True  # starting to update plots

        for widget in self._control_widgets:
            if not widget.updateSharedParameters(True):
                return

        # if not self.updateSharedParameters(True):
        #     return
        self._proc_worker.start()
        self._daq_worker.start()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(False)

    def _onStopDAQ(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        self._clearWorkers()
        self._clearQueues()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(True)

    def _clearWorkers(self):
        self._proc_worker.terminate()
        self._daq_worker.terminate()
        self._proc_worker.wait()
        self._daq_worker.wait()

    def _clearQueues(self):
        with self._daq_queue.mutex:
            self._daq_queue.queue.clear()
        with self._proc_queue.mutex:
            self._proc_queue.queue.clear()

    def _onStartServeFile(self):
        """Actions taken before the start of file serving."""
        folder = self._datasrc_children.source_name_le.text().strip()
        port = int(self._datasrc_children.port_le.text().strip())

        # process can only be start once
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

        self._server_terminate_btn.setEnabled(True)
        self._server_start_btn.setEnabled(False)
        for widget in self._disabled_widgets_during_file_serving:
            widget.setEnabled(False)

    def _onStopServeFile(self):
        """Actions taken before the end of file serving."""
        self._file_server.terminate()
        self._server_terminate_btn.setEnabled(False)
        self._server_start_btn.setEnabled(True)
        for widget in self._disabled_widgets_during_file_serving:
            widget.setEnabled(True)


    @QtCore.pyqtSlot(str)
    def onMessageReceived(self, msg):
        logger.info(msg)

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)

        self._clearWorkers()

        if self._file_server is not None and self._file_server.is_alive():
            self._file_server.terminate()

