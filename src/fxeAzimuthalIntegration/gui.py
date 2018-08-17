"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

Main GUI module.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European XFEL GmbH Hamburg. All rights reserved.
"""
import sys
import logging
import time
from collections import deque

import numpy as np
import zmq

from karabo_bridge import Client

from .pyqtgraph.Qt import QtCore, QtGui
from .pyqtgraph import mkPen, intColor
from .logging import logger, GuiLogger
from .plot_widgets import (
    MainLinePlotWidget, ImageViewWidget, IndividualPulseWindow
)

from .data_acquisition import DaqWorker
from .file_server import FileServer
from .config import Config as cfg
from .config import DataSource


GROUP_BOX_STYLE_SHEET = 'QGroupBox:title {'\
                        'border: 1px;'\
                        'subcontrol-origin: margin;'\
                        'subcontrol-position: top left;'\
                        'padding-left: 10px;'\
                        'padding-top: 10px;'\
                        'margin-top: 0.0em;}'


class InputDialogWithCheckBox(QtGui.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

    @classmethod
    def getResult(cls, parent, window_title, input_label, checkbox_label):
        dialog = cls(parent)

        dialog.setWindowTitle(window_title)

        label = QtGui.QLabel(input_label)
        text_le = QtGui.QLineEdit()

        ok_cb = QtGui.QCheckBox(checkbox_label)
        ok_cb.setChecked(True)

        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(text_le)
        layout.addWidget(ok_cb)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        result = dialog.exec_()

        return (text_le.text(), ok_cb.isChecked()), \
               result == QtGui.QDialog.Accepted


class MainGUI(QtGui.QMainWindow):
    """The main GUI for FXE azimuthal integration."""
    def __init__(self, screen_size=None):
        """Initialization."""
        super().__init__()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(cfg.MAIN_WINDOW_WIDTH, cfg.MAIN_WINDOW_HEIGHT)
        self.setWindowTitle('FXE Azimuthal Integration')

        self._opened_windows_count = 0
        self._opened_windows = dict()  # book keeping opened windows

        self._cw = QtGui.QWidget()  # the central widget
        self.setCentralWidget(self._cw)

        # drop the oldest element when the queue is full
        self._daq_queue = deque(maxlen=cfg.MAX_QUEUE_SIZE)
        # a DAQ worker which process the data in another thread
        self._daq_worker = None
        self._client = None

        # *************************************************************
        # Tool bar
        # *************************************************************
        tool_bar = self.addToolBar("Control")

        self._start_at = QtGui.QAction(
            QtGui.QIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay)),
            "Start",
            self)
        tool_bar.addAction(self._start_at)
        self._start_at.triggered.connect(self.on_enter_running)

        self._stop_at = QtGui.QAction(
            QtGui.QIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop)),
            "Stop",
            self)
        tool_bar.addAction(self._stop_at)
        self._stop_at.triggered.connect(self.on_exit_running)
        self._stop_at.setEnabled(False)

        self._insert_image_at = QtGui.QAction(
            QtGui.QIcon(self.style().standardIcon(QtGui.QStyle.SP_FileDialogListView)),
            "Plot individual pulse",
            self)
        self._insert_image_at.triggered.connect(
            self._show_individual_pulse_dialog)
        tool_bar.addAction(self._insert_image_at)

        # open an input dialog for opening a moving average window
        self._open_ma_window_at = QtGui.QAction(
            QtGui.QIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaSeekForward)),
            "Plot moving average for 'on' and 'off' pulses",
            self)
        tool_bar.addAction(self._open_ma_window_at)

        self._open_geometry_file_at = QtGui.QAction(
            QtGui.QIcon(self.style().standardIcon(QtGui.QStyle.SP_DriveCDIcon)),
            "Specify geometry file",
            self)
        self._open_geometry_file_at.triggered.connect(
            self._choose_geometry_file)
        tool_bar.addAction(self._open_geometry_file_at)

        # *************************************************************
        # Plots
        # *************************************************************
        self._plot = MainLinePlotWidget()
        self._image = ImageViewWidget()

        self._ctrl_pannel = QtGui.QWidget()

        # *************************************************************
        # hostname and port
        # *************************************************************
        self._hostname_le = QtGui.QLineEdit(cfg.DEFAULT_SERVER_ADDR)
        self._port_le = QtGui.QLineEdit(cfg.DEFAULT_SERVER_PORT)

        # *************************************************************
        # Azimuthal integration setup
        # *************************************************************
        self._geom_file = cfg.DEFAULT_GEOMETRY_FILE

        self._sample_dist_le = QtGui.QLineEdit(str(cfg.DIST))
        self._cx_le = QtGui.QLineEdit(str(cfg.CENTER_X))
        self._cy_le = QtGui.QLineEdit(str(cfg.CENTER_Y))
        self._energy_le = QtGui.QLineEdit(str(cfg.PHOTON_ENERGY))
        self._itgt_method_cb = QtGui.QComboBox()
        for method in cfg.INTEGRATION_METHODS:
            self._itgt_method_cb.addItem(method)
        self._itgt_range_lb_le = \
            QtGui.QLineEdit(str(cfg.INTEGRATION_RANGE[0]))
        self._itgt_range_ub_le = \
            QtGui.QLineEdit(str(cfg.INTEGRATION_RANGE[1]))
        self._itgt_points_le = QtGui.QLineEdit(str(cfg.INTEGRATION_POINTS))

        # *************************************************************
        # Experiment setup
        # *************************************************************
        self._on_pulses_le = QtGui.QLineEdit()
        self._normalization_range_lb_le = \
            QtGui.QLineEdit(str(cfg.INTEGRATION_RANGE[0]))
        self._normalization_range_ub_le = \
            QtGui.QLineEdit(str(cfg.INTEGRATION_RANGE[1]))
        self._FOM_range_lb_le = \
            QtGui.QLineEdit(str(cfg.INTEGRATION_RANGE[0]))
        self._FOM_range_ub_le = \
            QtGui.QLineEdit(str(cfg.INTEGRATION_RANGE[1]))


        # *************************************************************
        # data source options
        # *************************************************************
        self._data_src_rbts = []
        self._data_src_rbts.append(QtGui.QRadioButton("Calibrated (file)"))
        self._data_src_rbts.append(QtGui.QRadioButton("Calibrated (bridge)"))
        self._data_src_rbts.append(QtGui.QRadioButton("Assembled (bridge)"))
        self._data_src_rbts.append(QtGui.QRadioButton("Processed (bridge)"))
        self._data_src_rbts[int(cfg.DEFAULT_SERVER_SRC)].setChecked(True)

        # *************************************************************
        # plot options
        # *************************************************************
        self._is_normalized_cb = QtGui.QCheckBox("Normalized")
        self._is_normalized_cb.setChecked(False)

        # *************************************************************
        # log window
        # *************************************************************
        self._log_window = QtGui.QPlainTextEdit()
        self._log_window.setReadOnly(True)
        self._log_window.setMaximumBlockCount(cfg.MAX_LOGGING)
        logger_font = QtGui.QFont()
        logger_font.setPointSize(cfg.LOGGER_FONT_SIZE)
        self._log_window.setFont(logger_font)
        self._logger = GuiLogger(self._log_window)
        logging.getLogger().addHandler(self._logger)

        # *************************************************************
        # file server
        # *************************************************************
        self._file_server = None

        self._file_server_widget = QtGui.QGroupBox("Data stream server")
        self._file_server_widget.setStyleSheet(GROUP_BOX_STYLE_SHEET)
        self._server_start_btn = QtGui.QPushButton("Serve")
        self._server_start_btn.clicked.connect(self._on_start_serve_file)
        self._server_terminate_btn = QtGui.QPushButton("Terminate")
        self._server_terminate_btn.setEnabled(False)
        self._server_terminate_btn.clicked.connect(self._on_terminate_serve_file)
        self._select_btn = QtGui.QPushButton("Select")
        self._file_server_port_le = QtGui.QLineEdit(
            cfg.DEFAULT_FILE_SERVER_PORT)
        self._file_server_data_folder_le = QtGui.QLineEdit(
            cfg.DEFAULT_FILE_SERVER_FOLDER)

        self._disabled_widgets_during_file_serving = [
            self._file_server_port_le,
            self._file_server_data_folder_le,
            self._select_btn
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
            self.move(screen_size.width()/2 - cfg.MAIN_WINDOW_WIDTH/2,
                      screen_size.height()/20)

        self._disabled_widgets_during_daq = [
            self._open_geometry_file_at,
            self._hostname_le,
            self._port_le,
            self._sample_dist_le,
            self._cx_le,
            self._cy_le,
            self._energy_le,
            self._itgt_method_cb,
            self._itgt_range_lb_le,
            self._itgt_range_ub_le,
            self._itgt_points_le
        ]
        self._disabled_widgets_during_daq.extend(self._data_src_rbts)

        # For real time plot
        self._is_running = False
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(100)

        self.show()

    def _initUI(self):
        layout = QtGui.QGridLayout()

        layout.addWidget(self._ctrl_pannel, 0, 0, 4, 6)
        layout.addWidget(self._image, 4, 0, 5, 1)
        layout.addWidget(self._plot, 4, 1, 5, 5)
        layout.addWidget(self._log_window, 9, 0, 2, 4)
        layout.addWidget(self._file_server_widget, 9, 4, 2, 2)

        self._cw.setLayout(layout)

    def _initCtrlUI(self):
        # *************************************************************
        # Azimuthal integration setup panel
        # *************************************************************
        ai_setup_gp = QtGui.QGroupBox("Azimuthal integration setup")
        ai_setup_gp.setStyleSheet(GROUP_BOX_STYLE_SHEET)

        energy_lb = QtGui.QLabel("Photon energy (keV): ")
        sample_dist_lb = QtGui.QLabel("Sample distance (m): ")
        cx = QtGui.QLabel("Cx (pixel): ")
        cy = QtGui.QLabel("Cy (pixel): ")
        itgt_method_lb = QtGui.QLabel("Integration method: ")
        itgt_range_lb1 = QtGui.QLabel("Integration range (1/A): ")
        itgt_range_lb2 = QtGui.QLabel(" to ")
        itgt_points_lb = QtGui.QLabel("Integration points: ")

        itgt_range_layout = QtGui.QHBoxLayout()
        itgt_range_layout.addWidget(itgt_range_lb1)
        itgt_range_layout.addWidget(self._itgt_range_lb_le)
        itgt_range_layout.addWidget(itgt_range_lb2)
        itgt_range_layout.addWidget(self._itgt_range_ub_le)

        layout = QtGui.QGridLayout()
        layout.addWidget(sample_dist_lb, 0, 0, 1, 1)
        layout.addWidget(self._sample_dist_le, 0, 1, 1, 1)
        layout.addWidget(cx, 1, 0, 1, 1)
        layout.addWidget(self._cx_le, 1, 1, 1, 1)
        layout.addWidget(cy, 2, 0, 1, 1)
        layout.addWidget(self._cy_le, 2, 1, 1, 1)
        layout.addWidget(energy_lb, 3, 0, 1, 1)
        layout.addWidget(self._energy_le, 3, 1, 1, 1)
        layout.addWidget(itgt_method_lb, 0, 2, 1, 1)
        layout.addWidget(self._itgt_method_cb, 0, 3, 1, 1)
        layout.addLayout(itgt_range_layout, 1, 2, 1, 2)
        layout.addWidget(itgt_points_lb, 2, 2, 1, 1)
        layout.addWidget(self._itgt_points_le, 2, 3, 1, 1)

        ai_setup_gp.setLayout(layout)

        # *************************************************************
        # Experiment setup panel
        # *************************************************************
        ep_setup_gp = QtGui.QGroupBox("Experiment setup")
        ep_setup_gp.setStyleSheet(GROUP_BOX_STYLE_SHEET)

        on_pulses_lb = QtGui.QLabel("On-pulse IDs: ")
        normalization_range_lb1 = QtGui.QLabel("Normalization range (1/A): ")
        normalization_range_lb2 = QtGui.QLabel(" to ")
        FOM_range_lb1 = QtGui.QLabel("FOM range (1/A): ")
        FOM_range_lb2 = QtGui.QLabel(" to ")

        layout = QtGui.QGridLayout()
        layout.addWidget(on_pulses_lb, 0, 0, 1, 1)
        layout.addWidget(self._on_pulses_le, 0, 1, 1, 3)
        layout.addWidget(normalization_range_lb1, 2, 0, 1, 1)
        layout.addWidget(self._normalization_range_lb_le, 2, 1, 1, 1)
        layout.addWidget(normalization_range_lb2, 2, 2, 1, 1)
        layout.addWidget(self._normalization_range_ub_le, 2, 3, 1, 1)
        layout.addWidget(FOM_range_lb1, 3, 0, 1, 1)
        layout.addWidget(self._FOM_range_lb_le, 3, 1, 1, 1)
        layout.addWidget(FOM_range_lb2, 3, 2, 1, 1)
        layout.addWidget(self._FOM_range_ub_le, 3, 3, 1, 1)

        ep_setup_gp.setLayout(layout)

        # *************************************************************
        # TCP connection panel
        # *************************************************************
        tcp_connection_gp = QtGui.QGroupBox("TCP connection")
        tcp_connection_gp.setStyleSheet(GROUP_BOX_STYLE_SHEET
                                        )
        hostname_lb = QtGui.QLabel("Hostname: ")
        self._hostname_le.setAlignment(QtCore.Qt.AlignCenter)
        self._hostname_le.setFixedHeight(28)
        port_lb = QtGui.QLabel("Port: ")
        self._port_le.setAlignment(QtCore.Qt.AlignCenter)
        self._port_le.setFixedHeight(28)

        layout = QtGui.QHBoxLayout()
        layout.addWidget(hostname_lb, 2)
        layout.addWidget(self._hostname_le, 3)
        layout.addWidget(port_lb, 1)
        layout.addWidget(self._port_le, 2)
        tcp_connection_gp.setLayout(layout)

        # *************************************************************
        # data source panel
        # *************************************************************
        data_src_gp = QtGui.QGroupBox("Data source")
        data_src_gp.setStyleSheet(GROUP_BOX_STYLE_SHEET)
        layout = QtGui.QVBoxLayout()
        for btn in self._data_src_rbts:
            layout.addWidget(btn)
        data_src_gp.setLayout(layout)

        # *************************************************************
        # plot option panel
        # *************************************************************
        plot_option_gp = QtGui.QGroupBox('Plot options')
        plot_option_gp.setStyleSheet(GROUP_BOX_STYLE_SHEET)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._is_normalized_cb)
        plot_option_gp.setLayout(layout)

        layout = QtGui.QGridLayout()
        layout.addWidget(ai_setup_gp, 0, 0, 4, 3)
        layout.addWidget(ep_setup_gp, 0, 3, 4, 2)
        layout.addWidget(tcp_connection_gp, 0, 5, 1, 2)
        layout.addWidget(plot_option_gp, 1, 5, 3, 1)
        layout.addWidget(data_src_gp, 1, 6, 3, 1)

        self._ctrl_pannel.setLayout(layout)

    def _initFileServerUI(self):
        layout = QtGui.QGridLayout()

        port_lb = QtGui.QLabel("Port: ")
        self._file_server_port_le.setFixedHeight(28)
        self._select_btn.clicked.connect(self._set_data_folder)
        self._file_server_data_folder_le.setFixedHeight(28)
        self._select_btn.setToolTip("Select data folder")

        layout.addWidget(self._server_start_btn, 0, 0, 1, 1)
        layout.addWidget(self._server_terminate_btn, 0, 1, 1, 1)
        layout.addWidget(port_lb, 0, 4, 1, 1)
        layout.addWidget(self._file_server_port_le, 0, 5, 1, 1)
        layout.addWidget(self._select_btn, 1, 0, 1, 1)
        layout.addWidget(self._file_server_data_folder_le, 1, 1, 1, 5)
        self._file_server_widget.setLayout(layout)

    def _update(self):
        """Update plots in the main window and child windows."""
        if self._is_running is False:
            return

        # TODO: improve plot updating
        # Use multithreading for plot updating. However, this is not the
        # bottleneck for the performance.

        try:
            data = self._daq_queue.pop()
        except IndexError:
            return

        self._plot.clear_()
        self._image.clear_()
        for w in self._opened_windows.values():
            w.clear()

        # update the plots in the main GUI
        t0 = time.perf_counter()
        if data is not None:
            max_intensity = data["intensity"].max()
            for i, intensity in enumerate(data["intensity"]):
                if self._is_normalized_cb.isChecked() is True:
                    # data["intensity"] is also changed, so the plots in
                    # the other windows also become normalized.
                    intensity /= max_intensity

                self._plot.update(data["momentum"], intensity,
                                  pen=mkPen(intColor(i, hues=9, values=5),
                                            width=2))
            self._plot.set_title("Train ID: {}, No. pulses: {}".
                                 format(data["tid"], i+1))
            self._image.update(np.mean(data["image"], axis=0))

        # update the plots in child windows
        for w in self._opened_windows.values():
            w.update(data)

        logger.debug("Time for updating the plots: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

    def _show_individual_pulse_dialog(self):
        """A pop-up window."""
        ret, ok = InputDialogWithCheckBox.getResult(
            self,
            'Input Dialog',
            'Enter pulse IDs (separated by comma):',
            "Include detector image")

        if ok is True:
            self._open_individual_pulse_window(ret)

    def _open_individual_pulse_window(self, ret):
        """Open another window."""
        text = ret[0]
        show_image = ret[1]
        if not text:
            logger.info("Invalid input! Please specify pulse IDs!")
            return

        try:
            pulse_ids = text.split(",")
            pulse_ids = [int(i.strip()) for i in pulse_ids if i.strip()]
        except ValueError:
            logger.info("Invalid input! Enter pulse IDs separated by ','!")
            return

        window_id = "{:06d}".format(self._opened_windows_count)
        w = IndividualPulseWindow(window_id, pulse_ids,
                                  parent=self,
                                  show_image=show_image)
        self._opened_windows_count += 1
        self._opened_windows[window_id] = w
        logger.info("Open new window for pulse(s): {}".
                    format(", ".join(str(i) for i in pulse_ids)))
        w.show()

    def _choose_geometry_file(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if filename:
            self._geom_file = filename

    def remove_window(self, w_id):
        del self._opened_windows[w_id]

    def on_exit_running(self):
        """Actions taken at the beginning of 'run' state."""
        self._is_running = False
        logger.info("DAQ stopped!")

        self._daq_worker.terminate()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(True)

    def on_enter_running(self):
        """Actions taken at the end of 'run' state."""
        self._is_running = True

        client_addr = "tcp://" \
                      + self._hostname_le.text().strip() \
                      + ":" \
                      + self._port_le.text().strip()
        self._client = Client(client_addr)
        logger.info("Bind to {}".format(client_addr))

        if self._data_src_rbts[DataSource.CALIBRATED_FILE].isChecked() is True:
            data_source = DataSource.CALIBRATED_FILE
        elif self._data_src_rbts[DataSource.CALIBRATED].isChecked() is True:
            data_source = DataSource.CALIBRATED
        elif self._data_src_rbts[DataSource.ASSEMBLED].isChecked() is True:
            data_source = DataSource.ASSEMBLED
        else:
            data_source = DataSource.PROCESSED

        energy = float(self._energy_le.text().strip())
        sample_distance = float(self._sample_dist_le.text().strip())
        center_x = float(self._cx_le.text().strip())
        center_y = float(self._cy_le.text().strip())
        integration_method = self._itgt_method_cb.currentText()
        integration_range = float(self._itgt_range_lb_le.text().strip()), \
                            float(self._itgt_range_ub_le.text().strip()),
        integration_points = int(self._itgt_points_le.text().strip())
        try:
            self._daq_worker = DaqWorker(
                self._client,
                self._daq_queue,
                data_source,
                geom_file=self._geom_file,
                photon_energy=energy,
                sample_dist=sample_distance,
                cx=center_x,
                cy=center_y,
                integration_method=integration_method,
                integration_range=integration_range,
                integration_points=integration_points
            )
        except OSError as e:
            logger.info(e)
            return

        self._daq_worker.start()

        logger.info("DAQ started!")
        logger.info("Azimuthal integration parameters:\n"
                    " - photon energy (keV): {}\n"
                    " - sample distance (m): {}\n"
                    " - cx (pixel): {}\n"
                    " - cy (pixel): {}\n"
                    " - integration method: '{}'\n"
                    " - integration range (1/A): ({}, {})\n"
                    " - number of integration points: {}".
                    format(energy, sample_distance, center_x, center_y,
                           integration_method, integration_range[0],
                           integration_range[1], integration_points))

        self._start_at.setEnabled(False)
        self._stop_at.setEnabled(True)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(False)

    def _set_data_folder(self):
        folder = QtGui.QFileDialog.getExistingDirectory(
            self, 'Select directory', '/home')
        self._file_server_data_folder_le.setText(folder)

    def _on_start_serve_file(self):
        """Actions taken at the beginning of file serving."""
        folder = self._file_server_data_folder_le.text().strip()
        port = int(self._file_server_port_le.text().strip())

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

    def _on_terminate_serve_file(self):
        """Actions taken at the termination of file serving."""
        self._file_server.terminate()
        self._server_terminate_btn.setEnabled(False)
        self._server_start_btn.setEnabled(True)
        for widget in self._disabled_widgets_during_file_serving:
            widget.setEnabled(True)


def fxe_ai():
    app = QtGui.QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainGUI(screen_size=screen_size)
    app.exec_()
