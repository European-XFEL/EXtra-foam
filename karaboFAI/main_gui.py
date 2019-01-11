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
from .widgets import (
    AiSetUpWidget, DataSrcWidget, ExpSetUpWidget, FileServerWidget,
    GmtSetUpWidget, GuiLogger
)
from .windows import (
    DrawMaskWindow, LaserOnOffWindow, OverviewWindow
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
        # *************************************************************
        tool_bar = self.addToolBar("Control")

        root_dir = os.path.dirname(os.path.abspath(__file__))

        #
        self._start_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/start.png")),
            "Start DAQ",
            self)
        tool_bar.addAction(self._start_at)
        self._start_at.triggered.connect(self.onStartDAQ)

        #
        self._stop_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/stop.png")),
            "Stop DAQ",
            self)
        tool_bar.addAction(self._stop_at)
        self._stop_at.triggered.connect(self.onStopDAQ)
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
        self._load_mask_at.triggered.connect(self.loadMaskImage)
        tool_bar.addAction(self._load_mask_at)

        #
        self._load_geometry_file_at = QtGui.QAction(
            QtGui.QIcon(
                self.style().standardIcon(QtGui.QStyle.SP_DriveCDIcon)),
            "geometry file",
            self)
        self._load_geometry_file_at.triggered.connect(
            self.loadGeometryFile)
        tool_bar.addAction(self._load_geometry_file_at)

        # *************************************************************
        # Miscellaneous
        # *************************************************************
        self._data = self.Data4Visualization()

        # book-keeping opened windows
        self._plot_windows = WeakKeyDictionary()

        self._mask_image = None

        self._disabled_widgets_during_daq = [
            self._load_mask_at,
            self._load_geometry_file_at,
        ]

        self.ai_setup_widget = AiSetUpWidget(parent=self)
        self.gmt_setup_widget = GmtSetUpWidget(parent=self)
        self.exp_setup_widget = ExpSetUpWidget(parent=self)
        self.data_src_widget = DataSrcWidget(parent=self)
        self.file_server_widget = FileServerWidget(parent=self)

        self._logger = GuiLogger(self) 
        logging.getLogger().addHandler(self._logger)

        self.initUI()

        if screen_size is None:
            self.move(0, 0)
        else:
            self.move(screen_size.width()/2 - self._width/2,
                      screen_size.height()/20)

        # TODO: implement
        # self._pulse_range0_le.setEnabled(False)

        # self._disabled_widgets_during_daq.extend(self._data_src_rbts)

        self._daq_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])
        self._proc_queue = Queue(maxsize=config["MAX_QUEUE_SIZE"])

        # a DAQ worker which acquires the data in another thread
        self._daq_worker = DataAcquisition(self._daq_queue)
        # a data processing worker which processes the data in another thread
        self._proc_worker = DataProcessor(self._daq_queue, self._proc_queue)

        self.initPipeline()

        # For real time plot
        self._running = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateAll)
        self.timer.start(config["TIMER_INTERVAL"])

        self.show()

    def initPipeline(self):
        """Set up all signal and slot connections for pipeline."""
        # *************************************************************
        # DataProcessor
        # *************************************************************

        # self._daq_worker.message.connect(self.onMessageReceived)
        #
        # self.server_tcp_sgn.connect(self._daq_worker.onServerTcpChanged)
        #
        # # *************************************************************
        # # DataProcessor
        # # *************************************************************
        #
        # self._proc_worker.message.connect(self.onMessageReceived)
        #
        # self.data_source_sgn.connect(self._proc_worker.onSourceChanged)
        # self.geometry_sgn.connect(self._proc_worker.onGeometryChanged)
        # self.sample_distance_sgn.connect(
        #     self._proc_worker.onSampleDistanceChanged)
        # self.center_coordinate_sgn.connect(
        #     self._proc_worker.onCenterCoordinateChanged)
        # self.integration_method_sgn.connect(
        #     self._proc_worker.onIntegrationMethodChanged)
        # self.integration_range_sgn.connect(
        #     self._proc_worker.onIntegrationRangeChanged)
        # self.integration_points_sgn.connect(
        #     self._proc_worker.onIntegrationPointsChanged)
        # self.mask_range_sgn.connect(self._proc_worker.onMaskRangeChanged)
        # self.photon_energy_sgn.connect(self._proc_worker.onPhotonEnergyChanged)
        # self.pulse_range_sgn.connect(self._proc_worker.onPulseRangeChanged)
        #
        # self.image_mask_sgn.connect(self._proc_worker.onImageMaskChanged)

    def initUI(self):
        layout = QtGui.QGridLayout()

        layout.addWidget(self.ai_setup_widget, 0, 0, 3, 1)
        layout.addWidget(self.exp_setup_widget, 0, 1, 3, 1)
        layout.addWidget(self.data_src_widget, 0, 2, 2, 1)
        layout.addWidget(self.file_server_widget, 2, 2, 1, 1)

        layout.addWidget(self._logger.widget, 3, 0, 1, 1)
        layout.addWidget(self.gmt_setup_widget, 3, 1, 1, 2)

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

    def loadGeometryFile(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if filename:
            self._geom_file_le.setText(filename)

    def loadMaskImage(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if not filename:
            logger.error("Please specify the image mask file!")
        self.image_mask_sgn.emit(filename)

    def onStartDAQ(self):
        """Actions taken before the start of a 'run'."""
        self.clearQueues()
        self._running = True  # starting to update plots

        if not self.updateSharedParameters(True):
            return
        self._proc_worker.start()
        self._daq_worker.start()

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(False)

    def onStopDAQ(self):
        """Actions taken before the end of a 'run'."""
        self._running = False

        self.clearWorkers()
        self.clearQueues()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(True)

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

    def onStopServeFile(self):
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
        super().closeEvent(QCloseEvent)

        self.clearWorkers()

        if self._file_server is not None and self._file_server.is_alive():
            self._file_server.terminate()
