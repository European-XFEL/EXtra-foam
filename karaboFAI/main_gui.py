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

import fabio
import zmq

from .logger import GuiLogger, logger
from .widgets.pyqtgraph import QtCore, QtGui
from .widgets import (
    BraggSpotsWindow, CustomGroupBox, DrawMaskWindow, FixedWidthLineEdit,
    IndividualPulseWindow, InputDialogWithCheckBox, LaserOnOffWindow,
    MainGuiImageViewWidget, MainGuiLinePlotWidget, SampleDegradationMonitor
)
from .data_acquisition import DaqWorker
from .data_processing import DataSource, DataProcessor, ProcessedData
from .file_server import FileServer
from .config import config
from .helpers import parse_ids, parse_boundary, parse_quadrant_table


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

    # Shared parameters are pyqtSignals
    # Note: shared parameters should end with '_sp'
    mask_range_sp = QtCore.pyqtSignal(float, float)
    fom_range_sp = QtCore.pyqtSignal(float, float)
    normalization_range_sp = QtCore.pyqtSignal(float, float)
    ma_window_size_sp = QtCore.pyqtSignal(int)
    # (mode, on-pulse ids, off-pulse ids)
    on_off_pulse_ids_sp = QtCore.pyqtSignal(str, list, list)

    _height = 1000  # window height, in pixel
    _width = 1380  # window width, in pixel
    _plot_height = 480  # height of the plot widgets, in pixel

    _logger_fontsize = 12  # fontsize in logger window

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
        self.setWindowTitle(self.title)

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        self._daq_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._proc_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])

        # a DAQ worker which acquires the data in another thread
        self._daq_worker = None
        # a data processing worker which processes the data in another thread
        self._proc_worker = None

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
        open_individual_pulse_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/individual_pulse.png")),
            "Individual pulse",
            self)
        open_individual_pulse_at.triggered.connect(
            self._openIndividualPulseWindowDialog)
        tool_bar.addAction(open_individual_pulse_at)

        #
        open_laseronoff_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/fom_evolution.png")),
            "On- and off- pulses",
            self)
        open_laseronoff_window_at.triggered.connect(
            self._openLaserOnOffWindow)
        tool_bar.addAction(open_laseronoff_window_at)

        #
        open_bragg_spots_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/bragg_spots.png")),
            "Bragg spots",
            self)
        open_bragg_spots_window_at.triggered.connect(
            self._openBraggSpotsWindow)
        tool_bar.addAction(open_bragg_spots_window_at)

        #
        open_sample_degradation_monitor_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/sample_degradation.png")),
            "Sample degradation monitor",
            self)
        open_sample_degradation_monitor_at.triggered.connect(
            self._openSampleDegradationMonitor)
        tool_bar.addAction(open_sample_degradation_monitor_at)

        #
        self._draw_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/draw_mask.png")),
            "Draw mask",
            self)
        self._draw_mask_at.triggered.connect(self._openDrawMaskWindow)
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
        # Plots
        # *************************************************************
        self._data = self.Data4Visualization()

        self._lineplot_widget = MainGuiLinePlotWidget(self._data)
        self._lineplot_widget.setFixedSize(
            self._width - self._plot_height - 25, self._plot_height)

        self._image_widget = MainGuiImageViewWidget(self._data)
        self._image_widget.setFixedSize(self._plot_height, self._plot_height)

        # book-keeping opened widgets and windows
        self._opened_windows = dict()
        self._opened_windows[self._lineplot_widget] = 1
        self._opened_windows[self._image_widget] = 1

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
        self._energy_le = FixedWidthLineEdit(w, str(config["PHOTON_ENERGY"]))
        self._laser_mode_cb = QtGui.QComboBox()
        self._laser_mode_cb.setFixedWidth(w)
        self._laser_mode_cb.addItems(LaserOnOffWindow.available_modes.keys())
        self._on_pulse_le = FixedWidthLineEdit(w, "0, 3:16:2")
        self._off_pulse_le = FixedWidthLineEdit(w, "1, 2:16:2")
        self._normalization_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._fom_range_le = FixedWidthLineEdit(
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
        self._log_window = QtGui.QPlainTextEdit()
        self._log_window.setReadOnly(True)
        self._log_window.setMaximumBlockCount(config["MAX_LOGGING"])
        logger_font = QtGui.QFont()
        logger_font.setPointSize(self._logger_fontsize)
        self._log_window.setFont(logger_font)
        self._logger = GuiLogger(self._log_window)
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
            self._energy_le,
            self._laser_mode_cb,
            self._on_pulse_le,
            self._off_pulse_le,
            self._normalization_range_le,
            self._fom_range_le,
            self._ma_window_le
        ]
        self._disabled_widgets_during_daq.extend(self._data_src_rbts)

        # For real time plot
        self._is_running = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._updateAll)
        self.timer.start(10)

        self.show()

    def _initUI(self):
        layout = QtGui.QGridLayout()

        layout.addWidget(self._ctrl_pannel, 0, 0, 4, 6)
        layout.addWidget(self._image_widget, 4, 0, 5, 1)
        layout.addWidget(self._lineplot_widget, 4, 1, 5, 5)
        layout.addWidget(self._log_window, 9, 0, 2, 4)
        layout.addWidget(self._file_server_widget, 9, 4, 2, 2)

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
        energy_lb = QtGui.QLabel("Photon energy (keV): ")
        laser_mode_lb = QtGui.QLabel("Laser on/off mode: ")
        on_pulse_lb = QtGui.QLabel("On-pulse IDs: ")
        off_pulse_lb = QtGui.QLabel("Off-pulse IDs: ")
        normalization_range_lb = QtGui.QLabel("Normalization range (1/A): ")
        fom_range_lb = QtGui.QLabel("FOM range (1/A): ")
        ma_window_lb = QtGui.QLabel("M.A. window size: ")

        layout = QtGui.QGridLayout()
        layout.addWidget(energy_lb, 0, 0, 1, 1)
        layout.addWidget(self._energy_le, 0, 1, 1, 1)
        layout.addWidget(laser_mode_lb, 1, 0, 1, 1)
        layout.addWidget(self._laser_mode_cb, 1, 1, 1, 1)
        layout.addWidget(on_pulse_lb, 2, 0, 1, 1)
        layout.addWidget(self._on_pulse_le, 2, 1, 1, 1)
        layout.addWidget(off_pulse_lb, 3, 0, 1, 1)
        layout.addWidget(self._off_pulse_le, 3, 1, 1, 1)
        layout.addWidget(normalization_range_lb, 4, 0, 1, 1)
        layout.addWidget(self._normalization_range_le, 4, 1, 1, 1)
        layout.addWidget(fom_range_lb, 5, 0, 1, 1)
        layout.addWidget(self._fom_range_le, 5, 1, 1, 1)
        layout.addWidget(ma_window_lb, 6, 0, 1, 1)
        layout.addWidget(self._ma_window_le, 6, 1, 1, 1)

        self._ep_setup_gp.setLayout(layout)

        # *************************************************************
        # data source panel
        # *************************************************************
        hostname_lb = QtGui.QLabel("Hostname: ")
        self._hostname_le.setAlignment(QtCore.Qt.AlignCenter)
        self._hostname_le.setFixedHeight(28)
        port_lb = QtGui.QLabel("Port: ")
        self._port_le.setAlignment(QtCore.Qt.AlignCenter)
        self._port_le.setFixedHeight(28)
        source_name_lb = QtGui.QLabel("Source: ")
        self._source_name_le.setAlignment(QtCore.Qt.AlignCenter)
        self._source_name_le.setFixedHeight(28)
        pulse_range_lb = QtGui.QLabel("Pulse ID range: ")
        self._pulse_range0_le.setAlignment(QtCore.Qt.AlignCenter)
        self._pulse_range0_le.setFixedHeight(28)
        self._pulse_range1_le.setAlignment(QtCore.Qt.AlignCenter)
        self._pulse_range1_le.setFixedHeight(28)

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
        if self._is_running is False:
            return

        # TODO: improve plot updating
        # Use multithreading for plot updating. However, this is not the
        # bottleneck for the performance.

        try:
            self._data.set(self._proc_queue.get_nowait())
        except Empty:
            return

        # clear the previous plots no matter what comes next
        for w in self._opened_windows.keys():
            w.clearPlots()

        if self._data.get().empty():
            logger.info("Bad train with ID: {}".format(self._data.get().tid))
            return

        t0 = time.perf_counter()

        # update the all the plots
        for w in self._opened_windows.keys():
            w.updatePlots()

        logger.debug("Time for updating the plots: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        logger.info("Updated train with ID: {}".format(self._data.get().tid))

    def _openIndividualPulseWindowDialog(self):
        """A dialog for opening an IndividualPulseWindow."""
        ret, ok = InputDialogWithCheckBox.getResult(
            self,
            'Input Dialog',
            'Enter pulse IDs (separated by comma):',
            "Include detector image")

        err_msg = "Invalid input! " \
                  "Enter pulse IDs within the pulse range separated by ','!"

        try:
            pulse_ids = parse_ids(ret[0])
        except ValueError:
            logger.error(err_msg)
            return

        if not pulse_ids:
            logger.error(err_msg)
            return

        if ok:
            self._openIndividualPulseWindow(pulse_ids, ret[1])

    def _openIndividualPulseWindow(self, pulse_ids, show_image):
        w = IndividualPulseWindow(self._data,
                                  pulse_ids,
                                  parent=self,
                                  show_image=show_image)
        self._opened_windows[w] = 1
        w.show()

    def _openLaserOnOffWindow(self):
        w = LaserOnOffWindow(self._data, parent=self)
        self._opened_windows[w] = 1
        w.show()

    def _openBraggSpotsWindow(self):
        w = BraggSpotsWindow(self._data, parent=self)
        self._opened_windows[w] = 1
        w.show()

    def _openSampleDegradationMonitor(self):
        w = SampleDegradationMonitor(self._data, parent=self)
        self._opened_windows[w] = 1
        w.show()

    def _openDrawMaskWindow(self):
        w = DrawMaskWindow(self._data, parent=self)
        self._opened_windows[w] = 1
        w.show()

    def removeWindow(self, instance):
        del self._opened_windows[instance]

    def _loadGeometryFile(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if filename:
            self._geom_file_le.setText(filename)

    def _loadMaskImage(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        self._mask_image = None
        if filename:
            try:
                self._mask_image = fabio.open(filename).data
                logger.info("Load mask image at {}".format(filename))
            except IOError as e:
                logger.error(e)
            except Exception:
                raise
        else:
            logger.error("Please specify the mask image file!")

    def _onStartDAQ(self):
        """Actions taken before the start of a 'run'."""
        self._is_running = True

        if self._data_src_rbts[DataSource.CALIBRATED_FILE].isChecked() is True:
            data_source = DataSource.CALIBRATED_FILE
        elif self._data_src_rbts[DataSource.CALIBRATED].isChecked() is True:
            data_source = DataSource.CALIBRATED
        elif self._data_src_rbts[DataSource.ASSEMBLED].isChecked() is True:
            data_source = DataSource.ASSEMBLED
        else:
            data_source = DataSource.PROCESSED

        pulse_range = (int(self._pulse_range0_le.text()),
                       int(self._pulse_range1_le.text()))

        geom_file = self._geom_file_le.text()
        quad_positions = parse_quadrant_table(self._quad_positions_tb)
        energy = float(self._energy_le.text().strip())
        sample_distance = float(self._sample_dist_le.text().strip())
        center_x = float(self._cx_le.text().strip())
        center_y = float(self._cy_le.text().strip())
        integration_method = self._itgt_method_cb.currentText()

        try:
            integration_range = parse_boundary(self._itgt_range_le.text())
        except ValueError as e:
            logger.error("<Integration range>: " + str(e))
            return
        try:
            mask_range = parse_boundary(self._mask_range_le.text())
        except ValueError as e:
            logger.error("<Mask range>: " + str(e))
            return

        integration_points = int(self._itgt_points_le.text().strip())

        if not self.updateSharedParameters(True):
            return

        client_addr = "tcp://" \
                      + self._hostname_le.text().strip() \
                      + ":" \
                      + self._port_le.text().strip()

        try:
            self._proc_worker = DataProcessor(
                self._daq_queue, self._proc_queue,
                source=data_source,
                pulse_range=pulse_range,
                geom_file=geom_file,
                quad_positions=quad_positions,
                photon_energy=energy,
                sample_dist=sample_distance,
                cx=center_x,
                cy=center_y,
                integration_method=integration_method,
                integration_range=self._getIntegrationRange(),
                integration_points=integration_points,
                mask_range=self._getMaskRange(),
                mask=self._mask_image
            )

            self._daq_worker = DaqWorker(client_addr, self._daq_queue)
        except Exception as e:
            logger.error(e)
            return

        # remove when Client.next() has timeout option
        with self._daq_queue.mutex:
            self._daq_queue.queue.clear()

        logger.debug("Size of in and out queues: {}, {}".
                     format(self._daq_queue.qsize(), self._proc_queue.qsize()))

        self._daq_worker.start()
        self._proc_worker.start()

        logger.info("DAQ started!")
        # logger.info("Azimuthal integration parameters:\n"
        #             " - pulse range: {}\n"
        #             " - photon energy (keV): {}\n"
        #             " - sample distance (m): {}\n"
        #             " - cx (pixel): {}\n"
        #             " - cy (pixel): {}\n"
        #             " - integration method: '{}'\n"
        #             " - integration range (1/A): ({}, {})\n"
        #             " - number of integration points: {}\n"
        #             " - mask range: ({:d}, {:d})\n"
        #             " - quadrant positions: {}".
        #             format(pulse_range,
        #                    energy,
        #                    sample_distance,
        #                    center_x, center_y,
        #                    integration_method,
        #                    integration_range[0], integration_range[1],
        #                    integration_points,
        #                    mask_range[0], mask_range[1],
        #                    ", ".join(["({}, {})".format(p[0], p[1])
        #                               for p in quad_positions]))
        #             )

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(False)

    def _onStopDAQ(self):
        """Actions taken before the end of a 'run'."""
        self._is_running = False

        self._daq_worker.terminate()
        self._proc_worker.terminate()

        # TODO: self._daq_worker.join()
        self._proc_worker.join()

        with self._daq_queue.mutex:
            self._daq_queue.queue.clear()
        with self._proc_queue.mutex:
            self._proc_queue.queue.clear()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(True)

        logger.info("DAQ stopped!")

    def _onStartServeFile(self):
        """Actions taken before the start of file serving."""
        folder = self._source_name_le.text().strip()
        port = int(self._port_le.text().strip())

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

    def _getIntegrationRange(self):
        try:
            return parse_boundary(self._itgt_range_le.text())
        except ValueError as e:
            logger.error("<Integration range>: " + str(e))

    def _getMaskRange(self):
        try:
            return parse_boundary(self._mask_range_le.text())
        except ValueError as e:
            logger.error("<Mask range>: " + str(e))

    def updateSharedParameters(self, log=False):
        """Update shared parameters for all child windows.

        :params bool log: True for logging shared parameters and False
            for not.

        Returns bool: True if all shared parameters successfully parsed
            and emitted, otherwise False.
        """
        try:
            lb, ub = parse_boundary(self._mask_range_le.text())
            self.mask_range_sp.emit(lb, ub)
            if log:
                logger.info("<Mask range>: ({}, {})".format(lb, ub))
        except ValueError as e:
            logger.error("<Mask range>: " + str(e))
            return False

        try:
            lb, ub = parse_boundary(self._normalization_range_le.text())
            self.normalization_range_sp.emit(lb, ub)
            if log:
                logger.info("<Normalization range>: ({}, {})".format(lb, ub))
        except ValueError as e:
            logger.error("<Normalization range>: " + str(e))
            return False

        try:
            lb, ub = parse_boundary(self._fom_range_le.text())
            self.fom_range_sp.emit(lb, ub)
            if log:
                logger.info("<FOM range>: ({}, {})".format(lb, ub))
        except ValueError as e:
            logger.error("<FOM range>: " + str(e))
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

            self.on_off_pulse_ids_sp.emit(mode, on_pulse_ids, off_pulse_ids)
            if log:
                logger.info("<Optical laser mode>: {}".format(mode))
                logger.info("<On-pulse IDs>: {}".format(on_pulse_ids))
                logger.info("<Off-pulse IDs>: {}".format(off_pulse_ids))
        except ValueError:
            logger.error("Invalid input! Enter on/off pulse IDs separated "
                         "by ',' and/or use the range operator ':'!")
            return False

        try:
            window_size = int(self._ma_window_le.text())
            if window_size < 1:
                logger.error("Moving average window width < 1!")
                return False
            self.ma_window_size_sp.emit(window_size)
            if log:
                logger.info("<Moving average window size>: {}".
                            format(window_size))
        except ValueError as e:
            logger.error("<Moving average window size>: " + str(e))
            return False

        return True
