"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

Main GUI module.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import sys
import os
import time
import logging
import ast
from queue import Queue, Empty

import numpy as np
from imageio import imread, imsave
import zmq

from .pyqtgraph.Qt import QtCore, QtGui
from .logging import GuiLogger, logger
from .plot_widgets import (
    IndividualPulseWindow, LaserOnOffWindow, MainGuiImageViewWidget,
    MainGuiLinePlotWidget, SanityCheckWindow
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


class FixedWidthLineEdit(QtGui.QLineEdit):
    def __init__(self, width, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedWidth(width)


class CustomGroupBox(QtGui.QGroupBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(GROUP_BOX_STYLE_SHEET)


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


class InputDialogForMA(QtGui.QDialog):
    """Input dialog for moving average on-off pulses."""
    def __init__(self, parent=None):
        super().__init__(parent)

    @classmethod
    def getResult(cls, parent, *, on_pulse_ids=None, off_pulse_ids=None):
        dialog = cls(parent)

        dialog.setWindowTitle("Input dialog")

        on_pulse_lb = QtGui.QLabel("On-pulse IDs")
        on_pulse_le = QtGui.QLineEdit(on_pulse_ids)
        off_pulse_lb = QtGui.QLabel("Off-pulse IDs")
        off_pulse_le = QtGui.QLineEdit(off_pulse_ids)
        buttons = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(on_pulse_lb)
        layout.addWidget(on_pulse_le)
        layout.addWidget(off_pulse_lb)
        layout.addWidget(off_pulse_le)
        layout.addWidget(buttons)
        dialog.setLayout(layout)

        result = dialog.exec_()

        return (on_pulse_le.text(), off_pulse_le.text()), \
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

        self._daq_queue = Queue(maxsize=cfg.MAX_QUEUE_SIZE)
        # a DAQ worker which process the data in another thread
        self._daq_worker = None

        # *************************************************************
        # Tool bar
        # *************************************************************
        tool_bar = self.addToolBar("Control")

        root_dir = os.path.dirname(os.path.abspath(__file__))

        self._start_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/start.png")),
            "Start DAQ",
            self)
        tool_bar.addAction(self._start_at)
        self._start_at.triggered.connect(self.on_enter_running)

        self._stop_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/stop.png")),
            "Stop DAQ",
            self)
        tool_bar.addAction(self._stop_at)
        self._stop_at.triggered.connect(self.on_exit_running)
        self._stop_at.setEnabled(False)

        # open an input dialog for opening plots for individual pulses
        open_ip_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/individual_pulse.png")),
            "Individual pulse",
            self)
        open_ip_window_at.triggered.connect(self._show_ip_window_dialog)
        tool_bar.addAction(open_ip_window_at)

        # open an input dialog for opening a moving average window
        open_laseronoff_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/fom_evolution.png")),
            "On- and off- pulses",
            self)
        open_laseronoff_window_at.triggered.connect(
            self._show_laseronoff_window_dialog)
        tool_bar.addAction(open_laseronoff_window_at)

        # open the in-train pulse comparison window
        open_sanitycheck_window_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/sanity_check.png")),
            "In-train pulses comparison",
            self)
        open_sanitycheck_window_at.triggered.connect(
            self._open_sanitycheck_window)
        tool_bar.addAction(open_sanitycheck_window_at)

        self._open_geometry_file_at = QtGui.QAction(
            QtGui.QIcon(
                self.style().standardIcon(QtGui.QStyle.SP_DriveCDIcon)),
            "geometry file",
            self)
        self._open_geometry_file_at.triggered.connect(
            self._choose_geometry_file)
        tool_bar.addAction(self._open_geometry_file_at)

        self._save_image_at = QtGui.QAction(
            QtGui.QIcon(
                self.style().standardIcon(QtGui.QStyle.SP_DialogSaveButton)),
            "Save image",
            self)
        self._save_image_at.triggered.connect(self._save_image)
        tool_bar.addAction(self._save_image_at)

        self._load_mask_at = QtGui.QAction(
            QtGui.QIcon(os.path.join(root_dir, "icons/load_mask.png")),
            "Load mask",
            self)
        self._load_mask_at.triggered.connect(self._choose_mask_image)
        tool_bar.addAction(self._load_mask_at)

        # *************************************************************
        # Plots
        # *************************************************************
        self._lineplot_widget = MainGuiLinePlotWidget()
        self._image_widget = MainGuiImageViewWidget()
        self._data = None
        self._mask_image = None

        self._ctrl_pannel = QtGui.QWidget()

        # *************************************************************
        # Azimuthal integration setup
        # *************************************************************
        self._ai_setup_gp = CustomGroupBox("Azimuthal integration setup")

        w = 100
        self._sample_dist_le = FixedWidthLineEdit(w, str(cfg.DIST))
        self._cx_le = FixedWidthLineEdit(w, str(cfg.CENTER_X))
        self._cy_le = FixedWidthLineEdit(w, str(cfg.CENTER_Y))
        self._itgt_method_cb = QtGui.QComboBox()
        self._itgt_method_cb.setFixedWidth(w)
        for method in cfg.INTEGRATION_METHODS:
            self._itgt_method_cb.addItem(method)
        self._itgt_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in cfg.INTEGRATION_RANGE]))
        self._itgt_points_le = FixedWidthLineEdit(
            w, str(cfg.INTEGRATION_POINTS))
        self._mask_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in cfg.MASK_RANGE]))

        # *************************************************************
        # Geometry setup
        # *************************************************************
        self._gmt_setup_gp = CustomGroupBox("Geometry setup")
        self._quad_positions_tb = QtGui.QTableWidget()
        self._geom_file_le = FixedWidthLineEdit(285, cfg.DEFAULT_GEOMETRY_FILE)

        # *************************************************************
        # Experiment setup
        # *************************************************************
        self._ep_setup_gp = CustomGroupBox("Experiment setup")

        w = 90
        self._energy_le = FixedWidthLineEdit(w, str(cfg.PHOTON_ENERGY))
        self._on_pulse_le = FixedWidthLineEdit(w, "0, 3:16:2")
        self._off_pulse_le = FixedWidthLineEdit(w, "1, 2:16:2")
        self._normalization_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in cfg.INTEGRATION_RANGE]))
        self._fom_range_le = FixedWidthLineEdit(
            w, ', '.join([str(v) for v in cfg.INTEGRATION_RANGE]))
        self._ma_window_le = FixedWidthLineEdit(w, "10")

        # *************************************************************
        # data source options
        # *************************************************************
        self._data_src_gp = CustomGroupBox("Data source")

        self._hostname_le = FixedWidthLineEdit(130, cfg.DEFAULT_SERVER_ADDR)
        self._port_le = FixedWidthLineEdit(60, cfg.DEFAULT_SERVER_PORT)
        self._pulse_range0_le = FixedWidthLineEdit(60, str(0))
        self._pulse_range1_le = FixedWidthLineEdit(60, str(2999))

        self._data_src_rbts = []
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@files"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@ZMQ bridge"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Assembled data@ZMQ bridge"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Processed data@ZMQ bridge"))
        self._data_src_rbts[int(cfg.DEFAULT_SERVER_SRC)].setChecked(True)

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
        self._server_terminate_btn.clicked.connect(
            self._on_terminate_serve_file)
        self._select_btn = QtGui.QPushButton("Select")
        self._file_server_data_folder_le = QtGui.QLineEdit(
            cfg.DEFAULT_FILE_SERVER_FOLDER)

        self._disabled_widgets_during_file_serving = [
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

        # TODO: implement
        self._pulse_range0_le.setEnabled(False)

        self._disabled_widgets_during_daq = [
            self._open_geometry_file_at,
            self._save_image_at,
            self._load_mask_at,
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
        self.timer.timeout.connect(self._update)
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
        on_pulse_lb = QtGui.QLabel("On-pulse IDs: ")
        off_pulse_lb = QtGui.QLabel("Off-pulse IDs: ")
        normalization_range_lb = QtGui.QLabel("Normalization range (1/A): ")
        fom_range_lb = QtGui.QLabel("FOM range (1/A): ")
        ma_window_lb = QtGui.QLabel("M.A. window: ")

        layout = QtGui.QGridLayout()
        layout.addWidget(energy_lb, 0, 0, 1, 1)
        layout.addWidget(self._energy_le, 0, 1, 1, 1)
        layout.addWidget(on_pulse_lb, 1, 0, 1, 1)
        layout.addWidget(self._on_pulse_le, 1, 1, 1, 1)
        layout.addWidget(off_pulse_lb, 2, 0, 1, 1)
        layout.addWidget(self._off_pulse_le, 2, 1, 1, 1)
        layout.addWidget(normalization_range_lb, 3, 0, 1, 1)
        layout.addWidget(self._normalization_range_le, 3, 1, 1, 1)
        layout.addWidget(fom_range_lb, 4, 0, 1, 1)
        layout.addWidget(self._fom_range_le, 4, 1, 1, 1)
        layout.addWidget(ma_window_lb, 5, 0, 1, 1)
        layout.addWidget(self._ma_window_le, 5, 1, 1, 1)

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
        layout.addLayout(sub_layout1)
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
        for i in range(n_row):
            for j in range(n_col):
                widget.setItem(
                    i, j,
                    QtGui.QTableWidgetItem(str(cfg.QUAD_POSITIONS[i][j])))

        widget.move(0, 0)
        widget.setHorizontalHeaderLabels(['x', 'y'])
        widget.setVerticalHeaderLabels(['1', '2', '3', '4'])
        widget.setColumnWidth(0, 80)
        widget.setColumnWidth(1, 80)

    def _initFileServerUI(self):
        layout = QtGui.QGridLayout()

        self._select_btn.clicked.connect(self._set_data_folder)
        self._file_server_data_folder_le.setFixedHeight(28)
        self._select_btn.setToolTip("Select data folder")

        layout.addWidget(self._server_start_btn, 0, 0, 1, 1)
        layout.addWidget(self._server_terminate_btn, 0, 1, 1, 1)
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
            self._data = self._daq_queue.get_nowait()
        except Empty:
            time.sleep(0)
            return

        # clear the previous plots no matter what comes next
        self._lineplot_widget.clear_()
        self._image_widget.clear_()
        for w in self._opened_windows.values():
            w.clear()

        if self._data.empty():
            logger.info("Bad train with ID: {}".format(self._data.tid))
            return

        # update the plots in the main GUI
        self._lineplot_widget.update(self._data)
        self._image_widget.update(self._data)

        # update the plots in child windows
        for w in self._opened_windows.values():
            w.update(self._data)

        logger.info("Updated train with ID: {}".format(self._data.tid))

    def _show_ip_window_dialog(self):
        """A dialog for individual pulse plot."""
        ret, ok = InputDialogWithCheckBox.getResult(
            self,
            'Input Dialog',
            'Enter pulse IDs (separated by comma):',
            "Include detector image")

        err_msg = "Invalid input! " \
                  "Enter pulse IDs within the pulse range separated by ','!"

        try:
            pulse_ids = self._parse_ids(ret[0])
        except ValueError:
            logger.error(err_msg)
            return

        if not pulse_ids:
            logger.error(err_msg)
            return

        if ok:
            self._open_ip_window(pulse_ids, ret[1])

    def _open_ip_window(self, pulse_ids, show_image):
        """Open individual pulse plot window."""
        window_id = "{:06d}".format(self._opened_windows_count)
        w = IndividualPulseWindow(window_id, pulse_ids,
                                  parent=self,
                                  show_image=show_image)
        self._opened_windows_count += 1
        self._opened_windows[window_id] = w
        logger.info("Open new window for pulse(s): {}".
                    format(", ".join(str(i) for i in pulse_ids)))
        w.show()

    def _show_laseronoff_window_dialog(self):
        """A dialog for moving average on-off pulses plot."""
        ret, ok = InputDialogForMA.getResult(
            self,
            on_pulse_ids=self._on_pulse_le.text(),
            off_pulse_ids=self._off_pulse_le.text()
        )

        err_msg = "Invalid input! Enter on/off pulse IDs separated by ',' " \
                  "and/or use the range operator ':'!"
        try:
            on_pulse_ids = self._parse_ids(ret[0])
            off_pulse_ids = self._parse_ids(ret[1])
        except ValueError:
            logger.error(err_msg)
            return

        if not on_pulse_ids or not off_pulse_ids:
            logger.error(err_msg)
            return

        common = set(on_pulse_ids).intersection(off_pulse_ids)
        if common:
            logger.error("Pulse IDs {} are found in both on- and off- pulses.".
                         format(','.join([str(v) for v in common])))
            return

        if ok:
            self._open_laseronoff_window(on_pulse_ids, off_pulse_ids)

    def _open_laseronoff_window(self, on_pulse_ids, off_pulse_ids):
        """Open moving average on-off pulses window."""
        window_id = "{:06d}".format(self._opened_windows_count)

        try:
            normalization_range = \
                self._parse_boundary(self._normalization_range_le.text())
        except ValueError as e:
            logger.error("<Normalization range>: " + str(e))
            return
        try:
            fom_range = self._parse_boundary(self._fom_range_le.text())
        except ValueError as e:
            logger.error("<FOM range>: " + str(e))
            return

        ma_window_width = int(self._ma_window_le.text())
        if ma_window_width < 1:
            logger.error("Moving average window width < 1!")
            return

        w = LaserOnOffWindow(
            window_id,
            on_pulse_ids,
            off_pulse_ids,
            normalization_range,
            fom_range,
            ma_window_width=ma_window_width,
            parent=self)

        self._opened_windows_count += 1
        self._opened_windows[window_id] = w
        logger.info("Open new window for on-pulse(s): {} and off-pulse(s): {}".
                    format(", ".join(str(i) for i in on_pulse_ids),
                           ", ".join(str(i) for i in off_pulse_ids)))
        w.show()

    def _open_sanitycheck_window(self):
        window_id = "{:06d}".format(self._opened_windows_count)

        try:
            normalization_range = \
                self._parse_boundary(self._normalization_range_le.text())
        except ValueError as e:
            logger.error("<Normalization range>: " + str(e))
            return
        try:
            fom_range = self._parse_boundary(self._fom_range_le.text())
        except ValueError as e:
            logger.error("<FOM range>: " + str(e))
            return

        w = SanityCheckWindow(window_id, normalization_range, fom_range,
                              parent=self)

        self._opened_windows_count += 1
        self._opened_windows[window_id] = w
        logger.info("Open new window for sanity check")
        w.show()

    def _choose_geometry_file(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if filename:
            self._geom_file_le.setText(filename)

    def remove_window(self, w_id):
        del self._opened_windows[w_id]

    def on_exit_running(self):
        """Actions taken at the beginning of 'run' state."""
        self._is_running = False

        self._daq_worker.terminate()

        self._start_at.setEnabled(True)
        self._stop_at.setEnabled(False)
        for widget in self._disabled_widgets_during_daq:
            widget.setEnabled(True)

        logger.info("DAQ stopped!")

    def on_enter_running(self):
        """Actions taken at the end of 'run' state."""
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
        quad_positions = self._parse_quadrant_table(self._quad_positions_tb)
        energy = float(self._energy_le.text().strip())
        sample_distance = float(self._sample_dist_le.text().strip())
        center_x = float(self._cx_le.text().strip())
        center_y = float(self._cy_le.text().strip())
        integration_method = self._itgt_method_cb.currentText()
        try:
            integration_range = self._parse_boundary(
                self._itgt_range_le.text())
        except ValueError as e:
            logger.error("<Integration range>: " + str(e))
            return
        try:
            mask_range = self._parse_boundary(self._mask_range_le.text())
        except ValueError as e:
            logger.error("<Mask range>: " + str(e))
            return

        integration_points = int(self._itgt_points_le.text().strip())

        client_addr = "tcp://" \
                      + self._hostname_le.text().strip() \
                      + ":" \
                      + self._port_le.text().strip()

        try:

            self._daq_worker = DaqWorker(
                client_addr,
                self._daq_queue,
                data_source,
                pulse_range=pulse_range,
                geom_file=geom_file,
                quad_positions=quad_positions,
                photon_energy=energy,
                sample_dist=sample_distance,
                cx=center_x,
                cy=center_y,
                integration_method=integration_method,
                integration_range=integration_range,
                integration_points=integration_points,
                mask_range=mask_range,
                mask=self._mask_image
            )
        except Exception as e:
            logger.error(e)
            return

        self._daq_worker.start()

        logger.info("DAQ started!")
        logger.info("Azimuthal integration parameters:\n"
                    " - pulse range: {}\n"
                    " - photon energy (keV): {}\n"
                    " - sample distance (m): {}\n"
                    " - cx (pixel): {}\n"
                    " - cy (pixel): {}\n"
                    " - integration method: '{}'\n"
                    " - integration range (1/A): ({}, {})\n"
                    " - number of integration points: {}\n"
                    " - mask range: ({:d}, {:d})\n"
                    " - quadrant positions: {}".
                    format(pulse_range,
                           energy,
                           sample_distance,
                           center_x, center_y,
                           integration_method,
                           integration_range[0], integration_range[1],
                           integration_points,
                           mask_range[0], mask_range[1],
                           ", ".join(["({}, {})".format(p[0], p[1])
                                      for p in quad_positions]))
                    )

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

    def _on_terminate_serve_file(self):
        """Actions taken at the termination of file serving."""
        self._file_server.terminate()
        self._server_terminate_btn.setEnabled(False)
        self._server_start_btn.setEnabled(True)
        for widget in self._disabled_widgets_during_file_serving:
            widget.setEnabled(True)

    def _choose_mask_image(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]

        if filename:
            self._mask_image = imread(filename)
            logger.info("Load mask image at {}".format(filename))
        else:
            self._mask_image = None

    def _save_image(self):
        if self._data.empty():
            return

        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save File')[0]
        if filename:
            imsave(filename, self._data.image_avg)
            logger.info("Image saved at {}".format(filename))

    @staticmethod
    def _parse_boundary(text):
        try:
            lb, ub = [ast.literal_eval(x.strip()) for x in text.split(",")]
        except Exception:
            raise ValueError("Invalid input!")

        if lb > ub:
            raise ValueError("lower boundary > upper boundary!")

        return lb, ub

    @staticmethod
    def _parse_ids(text):
        def parse_item(v):
            if not v:
                return []

            if ':' in v:
                x = v.split(':')
                if len(x) < 2 or len(x) > 3:
                    raise ValueError
                try:
                    start = int(x[0].strip())
                    if start < 0:
                        raise ValueError("Pulse ID cannot be negative!")
                    end = int(x[1].strip())
                    if end <= start:
                        raise ValueError

                    if len(x) == 3:
                        inc = int(x[2].strip())
                    else:
                        inc = 1
                except ValueError:
                    raise

                return list(range(start, end, inc))

            try:
                v = int(v)
                if v < 0:
                    raise ValueError
            except ValueError:
                raise ValueError

            return v

        ret = set()
        for item in text.split(","):
            item = parse_item(item.strip())
            if isinstance(item, int):
                ret.add(item)
            else:
                ret.update(item)

        return list(ret)

    @staticmethod
    def _parse_quadrant_table(widget):
        n_row, n_col = widget.rowCount(), widget.columnCount()
        ret = np.zeros((n_row, n_col))
        for i in range(n_row):
            for j in range(n_col):
                ret[i, j] = float(widget.item(i, j).text())
        return ret


def fxe_ai():
    app = QtGui.QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainGUI(screen_size=screen_size)
    app.exec_()
