"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Main GUI.

Author: Jun Zhu <jun.zhu@xfel.eu>
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
from .widgets import CustomGroupBox, FixedWidthLineEdit, GuiLogger
from .windows import (
    BraggSpotsWindow, DrawMaskWindow, LaserOnOffWindow, OverviewWindow
)
from .data_acquisition import DataAcquisition
from .data_processing import DataSource, DataProcessor, ProcessedData
from .file_server import FileServer
from .config import config
from .helpers import parse_ids, parse_boundary, parse_table_widget


class MainGUI(QtGui.QMainWindow):
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

        self.title = topic + " Azimuthal Integration"
        self.setWindowTitle(self.title + " - main GUI")

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        # *************************************************************
        # Tool bar
        # Note: the order of 'addAction` affect the unittest!!!
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

        self._tool_bar = tool_bar

        # *************************************************************
        # Miscellaneous
        # *************************************************************
        self._data = self.Data4Visualization()

        # book-keeping opened windows
        self._plot_windows = WeakKeyDictionary()

        self._mask_image = None
        self._ctrl_pannel = QtGui.QWidget()

        # *************************************************************
        # Azimuthal integration setup
        # *************************************************************
        self._ai_setup_gp = CustomGroupBox("Azimuthal integration setup")

        w = 100
        self._sample_dist_le = FixedWidthLineEdit(w, str(config["DISTANCE"]))
        self._cx_le = FixedWidthLineEdit(w, str(config["CENTER_X"]))
        self._cy_le = FixedWidthLineEdit(w, str(config["CENTER_Y"]))
        self._itgt_method_cb = QtGui.QComboBox()
        self._itgt_method_cb.setFixedWidth(w)
        for method in config["INTEGRATION_METHODS"]:
            self._itgt_method_cb.addItem(method)
        self._itgt_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._itgt_points_le = FixedWidthLineEdit(
            w, str(config["INTEGRATION_POINTS"]))
        self._mask_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in config["MASK_RANGE"]]))

        # *************************************************************
        # Geometry setup
        # *************************************************************
        self._gmt_setup_gp = CustomGroupBox("Geometry setup")
        self._quad_positions_tb = QtGui.QTableWidget()
        self._geom_file_le = FixedWidthLineEdit(285, config["GEOMETRY_FILE"])

        # *************************************************************
        # Experiment setup
        # *************************************************************
        self._ep_setup_gp = CustomGroupBox("Experiment setup")

        w = 100
        self._photon_energy_le = FixedWidthLineEdit(
            w, str(config["PHOTON_ENERGY"]))
        self._laser_mode_cb = QtGui.QComboBox()
        self._laser_mode_cb.setFixedWidth(w)
        self._laser_mode_cb.addItems(LaserOnOffWindow.available_modes.keys())
        self._on_pulse_le = FixedWidthLineEdit(w, "0:8:2")
        self._off_pulse_le = FixedWidthLineEdit(w, "1:8:2")
        self._normalization_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._diff_integration_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._ma_window_le = FixedWidthLineEdit(w, "9999")

        # *************************************************************
        # data source options
        # *************************************************************
        self._data_src_gp = CustomGroupBox("Data source")

        self._hostname_le = FixedWidthLineEdit(165, config["SERVER_ADDR"])
        self._port_le = FixedWidthLineEdit(70, str(config["SERVER_PORT"]))
        self._source_name_le = FixedWidthLineEdit(280, config["SOURCE_NAME"])
        self._pulse_range0_le = FixedWidthLineEdit(60, str(0))
        self._pulse_range1_le = FixedWidthLineEdit(60, str(2699))

        self._data_src_rbts = []
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@files"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@ZMQ bridge"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Assembled data@ZMQ bridge"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Processed data@ZMQ bridge"))
        self._data_src_rbts[int(config["SOURCE_TYPE"])].setChecked(True)

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
            self._source_name_le,
        ]

        # *************************************************************
        # Initialize UI
        # *************************************************************
        self._initCtrlUI()
        self._initFileServerUI()
        self._initUI()

        if screen_size is None:
            self.move(0, 0)
        else:
            self.move(screen_size.width()/2 - self._width/2,
                      screen_size.height()/20)

        # TODO: implement
        self._pulse_range0_le.setEnabled(False)

        self._disabled_widgets_during_daq = [
            self._load_mask_at,
            self._load_geometry_file_at,
            self._hostname_le,
            self._port_le,
            self._pulse_range1_le,
            self._sample_dist_le,
            self._cx_le,
            self._cy_le,
            self._itgt_method_cb,
            self._itgt_range_le,
            self._itgt_points_le,
            self._mask_range_le,
            self._geom_file_le,
            self._quad_positions_tb,
            self._photon_energy_le,
            self._laser_mode_cb,
            self._on_pulse_le,
            self._off_pulse_le,
            self._normalization_range_le,
            self._diff_integration_range_le,
            self._ma_window_le
        ]
        self._disabled_widgets_during_daq.extend(self._data_src_rbts)

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

        layout.addWidget(self._ctrl_pannel, 0, 0, 4, 6)
        layout.addWidget(self._logger.widget, 4, 0, 2, 4)
        layout.addWidget(self._file_server_widget, 4, 4, 2, 2)

        self._cw.setLayout(layout)

    def _initCtrlUI(self):
        # *************************************************************
        # Azimuthal integration setup panel
        # *************************************************************
        sample_dist_lb = QtGui.QLabel("Sample distance (m): ")
        cx = QtGui.QLabel("Cx (pixel): ")
        cy = QtGui.QLabel("Cy (pixel): ")
        itgt_method_lb = QtGui.QLabel("Integration method: ")
        itgt_points_lb = QtGui.QLabel("Integration points: ")
        itgt_range_lb = QtGui.QLabel("Integration range (1/A): ")
        mask_range_lb = QtGui.QLabel("Mask range: ")

        self._initQuadTable()

        # first column
        layout = QtGui.QGridLayout()
        layout.addWidget(sample_dist_lb, 0, 0, 1, 1)
        layout.addWidget(self._sample_dist_le, 0, 1, 1, 1)
        layout.addWidget(cx, 1, 0, 1, 1)
        layout.addWidget(self._cx_le, 1, 1, 1, 1)
        layout.addWidget(cy, 2, 0, 1, 1)
        layout.addWidget(self._cy_le, 2, 1, 1, 1)
        layout.addWidget(itgt_method_lb, 4, 0, 1, 1)
        layout.addWidget(self._itgt_method_cb, 4, 1, 1, 1)
        layout.addWidget(itgt_points_lb, 5, 0, 1, 1)
        layout.addWidget(self._itgt_points_le, 5, 1, 1, 1)
        layout.addWidget(itgt_range_lb, 6, 0, 1, 1)
        layout.addWidget(self._itgt_range_le, 6, 1, 1, 1)
        layout.addWidget(mask_range_lb, 7, 0, 1, 1)
        layout.addWidget(self._mask_range_le, 7, 1, 1, 1)

        self._ai_setup_gp.setLayout(layout)

        # *************************************************************
        # Geometry setup
        # *************************************************************
        geom_file_lb = QtGui.QLabel("Geometry file:")
        quad_positions_lb = QtGui.QLabel("Quadrant positions:")

        layout = QtGui.QGridLayout()
        layout.addWidget(geom_file_lb, 0, 0, 1, 3)
        layout.addWidget(self._geom_file_le, 1, 0, 1, 3)
        layout.addWidget(quad_positions_lb, 2, 0, 1, 2)
        layout.addWidget(self._quad_positions_tb, 3, 0, 1, 2)

        self._gmt_setup_gp.setLayout(layout)

        # *************************************************************
        # Experiment setup panel
        # *************************************************************
        photon_energy_lb = QtGui.QLabel("Photon energy (keV): ")
        laser_mode_lb = QtGui.QLabel("Laser on/off mode: ")
        on_pulse_lb = QtGui.QLabel("On-pulse IDs: ")
        off_pulse_lb = QtGui.QLabel("Off-pulse IDs: ")
        normalization_range_lb = QtGui.QLabel("Normalization range (1/A): ")
        diff_integration_range_lb = QtGui.QLabel(
            "Diff integration range (1/A): ")
        ma_window_lb = QtGui.QLabel("M.A. window size: ")

        layout = QtGui.QGridLayout()
        layout.addWidget(photon_energy_lb, 0, 0, 1, 1)
        layout.addWidget(self._photon_energy_le, 0, 1, 1, 1)
        layout.addWidget(laser_mode_lb, 1, 0, 1, 1)
        layout.addWidget(self._laser_mode_cb, 1, 1, 1, 1)
        layout.addWidget(on_pulse_lb, 2, 0, 1, 1)
        layout.addWidget(self._on_pulse_le, 2, 1, 1, 1)
        layout.addWidget(off_pulse_lb, 3, 0, 1, 1)
        layout.addWidget(self._off_pulse_le, 3, 1, 1, 1)
        layout.addWidget(normalization_range_lb, 4, 0, 1, 1)
        layout.addWidget(self._normalization_range_le, 4, 1, 1, 1)
        layout.addWidget(diff_integration_range_lb, 5, 0, 1, 1)
        layout.addWidget(self._diff_integration_range_le, 5, 1, 1, 1)
        layout.addWidget(ma_window_lb, 6, 0, 1, 1)
        layout.addWidget(self._ma_window_le, 6, 1, 1, 1)

        self._ep_setup_gp.setLayout(layout)

        # *************************************************************
        # data source panel
        # *************************************************************
        hostname_lb = QtGui.QLabel("Hostname: ")
        self._hostname_le.setAlignment(QtCore.Qt.AlignCenter)
        port_lb = QtGui.QLabel("Port: ")
        self._port_le.setAlignment(QtCore.Qt.AlignCenter)
        source_name_lb = QtGui.QLabel("Source: ")
        self._source_name_le.setAlignment(QtCore.Qt.AlignCenter)
        pulse_range_lb = QtGui.QLabel("Pulse ID range: ")
        self._pulse_range0_le.setAlignment(QtCore.Qt.AlignCenter)
        self._pulse_range1_le.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        sub_layout1 = QtGui.QHBoxLayout()
        sub_layout1.addWidget(hostname_lb)
        sub_layout1.addWidget(self._hostname_le)
        sub_layout1.addWidget(port_lb)
        sub_layout1.addWidget(self._port_le)
        sub_layout2 = QtGui.QHBoxLayout()
        sub_layout2.addWidget(pulse_range_lb)
        sub_layout2.addWidget(self._pulse_range0_le)
        sub_layout2.addWidget(QtGui.QLabel(" to "))
        sub_layout2.addWidget(self._pulse_range1_le)
        sub_layout2.addStretch(2)
        sub_layout3 = QtGui.QHBoxLayout()
        sub_layout3.addWidget(source_name_lb)
        sub_layout3.addWidget(self._source_name_le)
        layout.addLayout(sub_layout1)
        layout.addLayout(sub_layout3)
        for btn in self._data_src_rbts:
            layout.addWidget(btn)
        layout.addLayout(sub_layout2)
        self._data_src_gp.setLayout(layout)

        # ------------------------------------------------------------
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._ai_setup_gp, 3)
        layout.addWidget(self._gmt_setup_gp, 2)
        layout.addWidget(self._ep_setup_gp, 3)
        layout.addWidget(self._data_src_gp, 3)

        self._ctrl_pannel.setLayout(layout)

    def _initQuadTable(self):
        n_row = 4
        n_col = 2
        widget = self._quad_positions_tb
        widget.setRowCount(n_row)
        widget.setColumnCount(n_col)
        try:
            for i in range(n_row):
                for j in range(n_col):
                    widget.setItem(i, j, QtGui.QTableWidgetItem(
                        str(config["QUAD_POSITIONS"][i][j])))
        except IndexError:
            pass

        widget.move(0, 0)
        widget.setHorizontalHeaderLabels(['x', 'y'])
        widget.setVerticalHeaderLabels(['1', '2', '3', '4'])
        widget.setColumnWidth(0, 80)
        widget.setColumnWidth(1, 80)

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
            self._geom_file_le.setText(filename)

    def _loadMaskImage(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if not filename:
            logger.error("Please specify the image mask file!")
        self.image_mask_sgn.emit(filename)

    def _onStartDAQ(self):
        """Actions taken before the start of a 'run'."""
        self._clearQueues()
        self._running = True  # starting to update plots

        if not self.updateSharedParameters(True):
            return
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
        folder = self._source_name_le.text().strip()
        port = int(self._port_le.text().strip())

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

    def updateSharedParameters(self, log=False):
        """Update shared parameters for all child windows.

        :params bool log: True for logging shared parameters and False
            for not.

        Returns bool: True if all shared parameters successfully parsed
            and emitted, otherwise False.
        """
        if self._data_src_rbts[DataSource.CALIBRATED_FILE].isChecked() is True:
            data_source = DataSource.CALIBRATED_FILE
        elif self._data_src_rbts[DataSource.CALIBRATED].isChecked() is True:
            data_source = DataSource.CALIBRATED
        elif self._data_src_rbts[DataSource.ASSEMBLED].isChecked() is True:
            data_source = DataSource.ASSEMBLED
        else:
            data_source = DataSource.PROCESSED
        self.data_source_sgn.emit(data_source)

        try:
            geom_file = self._geom_file_le.text()
            quad_positions = parse_table_widget(self._quad_positions_tb)
            self.geometry_sgn.emit(geom_file, quad_positions)
        except ValueError as e:
            logger.error("<Quadrant positions>: " + str(e))
            return False

        sample_distance = float(self._sample_dist_le.text().strip())
        if sample_distance <= 0:
            logger.error("<Sample distance>: Invalid input! Must be positive!")
            return False
        else:
            self.sample_distance_sgn.emit(sample_distance)

        center_x = int(self._cx_le.text().strip())
        center_y = int(self._cy_le.text().strip())
        self.center_coordinate_sgn.emit(center_x, center_y)

        integration_method = self._itgt_method_cb.currentText()
        self.integration_method_sgn.emit(integration_method)

        integration_points = int(self._itgt_points_le.text().strip())
        if integration_points <= 0:
            logger.error(
                "<Integration points>: Invalid input! Must be positive!")
            return False
        else:
            self.integration_points_sgn.emit(integration_points)

        try:
            integration_range = parse_boundary(self._itgt_range_le.text())
            self.integration_range_sgn.emit(*integration_range)
        except ValueError as e:
            logger.error("<Integration range>: " + str(e))
            return False

        try:
            mask_range = parse_boundary(self._mask_range_le.text())
            self.mask_range_sgn.emit(*mask_range)
        except ValueError as e:
            logger.error("<Mask range>: " + str(e))
            return False

        try:
            normalization_range = parse_boundary(
                self._normalization_range_le.text())
            self.normalization_range_sgn.emit(*normalization_range)
        except ValueError as e:
            logger.error("<Normalization range>: " + str(e))
            return False

        try:
            diff_integration_range = parse_boundary(
                self._diff_integration_range_le.text())
            self.diff_integration_range_sgn.emit(*diff_integration_range)
        except ValueError as e:
            logger.error("<Diff integration range>: " + str(e))
            return False

        try:
            # check pulse ID only when laser on/off pulses are in the same
            # train (the "normal" mode)
            mode = self._laser_mode_cb.currentText()
            on_pulse_ids = parse_ids(self._on_pulse_le.text())
            off_pulse_ids = parse_ids(self._off_pulse_le.text())
            if mode == list(LaserOnOffWindow.available_modes.keys())[0]:
                common = set(on_pulse_ids).intersection(off_pulse_ids)
                if common:
                    logger.error(
                        "Pulse IDs {} are found in both on- and off- pulses.".
                        format(','.join([str(v) for v in common])))
                    return False

            self.on_off_pulse_ids_sgn.emit(mode, on_pulse_ids, off_pulse_ids)
        except ValueError:
            logger.error("Invalid input! Enter on/off pulse IDs separated "
                         "by ',' and/or use the range operator ':'!")
            return False

        try:
            window_size = int(self._ma_window_le.text())
            if window_size < 1:
                logger.error("Moving average window width < 1!")
                return False
            self.ma_window_size_sgn.emit(window_size)
        except ValueError as e:
            logger.error("<Moving average window size>: " + str(e))
            return False

        photon_energy = float(self._photon_energy_le.text().strip())
        if photon_energy <= 0:
            logger.error("<Photon energy>: Invalid input! Must be positive!")
            return False
        else:
            self.photon_energy_sgn.emit(photon_energy)

        pulse_range = (int(self._pulse_range0_le.text()),
                       int(self._pulse_range1_le.text()))
        if pulse_range[1] <= 0:
            logger.error("<Pulse range>: Invalid input!")
            return False
        else:
            self.pulse_range_sgn.emit(*pulse_range)

        server_hostname = self._hostname_le.text().strip()
        server_port = self._port_le.text().strip()
        self.server_tcp_sgn.emit(server_hostname, server_port)

        if log:
            logger.info("--- Shared parameters ---")
            logger.info("<Host name>, <Port>: {}, {}".
                        format(server_hostname, server_port))
            logger.info("<Data source>: {}".format(data_source))
            logger.info("<Geometry file>: {}".format(geom_file))
            logger.info("<Quadrant positions>: [{}]".format(
                ", ".join(["[{}, {}]".format(p[0], p[1])
                           for p in quad_positions])))
            logger.info("<Sample distance (m)>: {}".format(sample_distance))
            logger.info("<Cx (pixel), Cy (pixel>: ({:d}, {:d})".
                        format(center_x, center_y))
            logger.info("<Cy (pixel)>: {:d}".format(center_y))
            logger.info("<Integration method>: '{}'".format(integration_method))
            logger.info("<Integration range (1/A)>: ({}, {})".
                        format(*integration_range))
            logger.info("<Number of integration points>: {}".
                        format(integration_points))
            logger.info("<Mask range>: ({}, {})".format(*mask_range))
            logger.info("<Normalization range>: ({}, {})".
                        format(*normalization_range))
            logger.info("<Diff integration range>: ({}, {})".
                        format(*diff_integration_range))
            logger.info("<Optical laser mode>: {}".format(mode))
            logger.info("<On-pulse IDs>: {}".format(on_pulse_ids))
            logger.info("<Off-pulse IDs>: {}".format(off_pulse_ids))
            logger.info("<Moving average window size>: {}".
                        format(window_size))
            logger.info("<Photon energy (keV)>: {}".format(photon_energy))
            logger.info("<Pulse range>: ({}, {})".format(*pulse_range))

        return True

    @QtCore.pyqtSlot(str)
    def onMessageReceived(self, msg):
        logger.info(msg)

    def closeEvent(self, QCloseEvent):
        self._clearWorkers()

        if self._file_server is not None and self._file_server.is_alive():
            self._file_server.terminate()

        super().closeEvent(QCloseEvent)
