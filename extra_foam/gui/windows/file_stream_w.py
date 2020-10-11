"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Ebad Kamil <ebad.kamil@xfel.eu> and Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
from multiprocessing import Process, Value
from contextlib import closing
import socket

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFontMetrics, QIntValidator, QValidator
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
    QHeaderView, QGroupBox, QLabel, QLineEdit, QLCDNumber, QProgressBar,
    QPushButton, QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget
)

from zmq.error import ZMQError

from .base_window import _AbstractSatelliteWindow
from ..ctrl_widgets.smart_widgets import SmartLineEdit
from ...gui import QApplication
from ...gui.gui_helpers import create_icon_button
from ...gui.misc_widgets import GuiLogger
from ...logger import logger_stream as logger
from ...offline import (
    gather_sources, load_runs, run_info, serve_files, StreamMode
)


class _FileStreamCtrlWidget(QWidget):

    GROUP_BOX_STYLE_SHEET = 'QGroupBox:title {' \
                            'color: #8B008B;' \
                            'border: 1px;' \
                            'subcontrol-origin: margin;' \
                            'subcontrol-position: top left;' \
                            'padding-left: 10px;' \
                            'padding-top: 10px;' \
                            'margin-top: 0.0em;}'

    _available_modes = OrderedDict({
        "Normal": StreamMode.NORMAL,
        "Random shuffle": StreamMode.RANDOM_SHUFFLE,
    })

    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.load_run_btn = QPushButton("Load run")

        self.data_folder_le = SmartLineEdit()

        self.serve_start_btn = create_icon_button(
            "start.png", 18, description="Stream once")
        self.repeat_serve_start_btn = create_icon_button(
            "repeat.png", 18, description="Stream repeatedly")
        self.serve_terminate_btn = create_icon_button(
            "stop.png", 18, description="Stop stream")
        self.serve_terminate_btn.setEnabled(False)

        self.mode_cb = QComboBox()
        for v in self._available_modes:
            self.mode_cb.addItem(v)
        self.port_le = QLineEdit("*")
        self.port_le.setValidator(QIntValidator(0, 65535))

        lcd = QLCDNumber(10)
        lcd.setSegmentStyle(QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QColor(0, 0, 0))
        palette.setColor(palette.Background, QColor(0, 170, 255))
        lcd.setPalette(palette)
        lcd.setAutoFillBackground(True)
        lcd.display(None)
        self.curr_tid_lcd = lcd
        self.tid_start_lb = QLabel("")
        self.tid_start_sld = QSlider(Qt.Horizontal)
        self.tid_start_sld.setToolTip("First train ID")
        self.tid_start_sld.setRange(0, 0)
        self.tid_end_lb = QLabel("")
        self.tid_end_sld = QSlider(Qt.Horizontal)
        self.tid_end_sld.setToolTip("Last train ID")
        self.tid_end_sld.setRange(0, 0)
        self.tid_stride_le = QLineEdit("1")
        validator = QIntValidator()
        validator.setBottom(1)
        self.tid_stride_le.setValidator(validator)

        self.stream_rate_lb = QLabel(f"{0.0:<12.1f}")
        self.stream_rate_lb.setFixedWidth(
            self.stream_rate_lb.sizeHint().width())

        self.tid_progress_br = QProgressBar()

        self._run = None
        self._select_all_cb = QCheckBox("Select all")
        self._detector_src_tb = QTableWidget()
        self._instrument_src_tb = QTableWidget()
        self._control_src_tb = QTableWidget()
        for tb in (self._detector_src_tb,
                   self._instrument_src_tb,
                   self._control_src_tb):
            tb.setColumnCount(2)
            header = tb.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Interactive)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            tb.verticalHeader().hide()
            tb.setHorizontalHeaderLabels(['Device ID', 'Property'])
            header.resizeSection(0, int(0.6 * header.length()))

        self._gui_logger = GuiLogger(parent=self)
        self._gui_logger.widget.setFixedHeight(
            QFontMetrics(QApplication.font()).lineSpacing() * 1.8)
        self._gui_logger.widget.setMaximumBlockCount(1)
        logger.addHandler(self._gui_logger)

        self._non_reconfigurable_widgets = [
            self.data_folder_le,
            self._select_all_cb,
            self._detector_src_tb,
            self._instrument_src_tb,
            self._control_src_tb,
            self.tid_start_sld,
            self.tid_end_sld,
            self.tid_stride_le,
            self.load_run_btn,
            self.mode_cb,
            self.port_le,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        AR = Qt.AlignRight

        ctrl_layout = QGridLayout()
        ctrl_layout.addWidget(self.load_run_btn, 0, 0)
        ctrl_layout.addWidget(self.data_folder_le, 0, 1, 1, 11)

        ctrl_layout.addWidget(QLabel("Port: "), 1, 0, AR)
        ctrl_layout.addWidget(self.port_le, 1, 1)
        ctrl_layout.addWidget(self.serve_start_btn, 1, 2)
        ctrl_layout.addWidget(self.repeat_serve_start_btn, 1, 3)
        ctrl_layout.addWidget(self.serve_terminate_btn, 1, 4)
        ctrl_layout.addWidget(QLabel("Stride: "), 1, 5, AR)
        ctrl_layout.addWidget(self.tid_stride_le, 1, 6)
        ctrl_layout.addWidget(QLabel("Mode: "), 1, 7, AR)
        ctrl_layout.addWidget(self.mode_cb, 1, 8, AR)
        ctrl_layout.addWidget(QLabel("Rate: "), 1, 9, AR)
        ctrl_layout.addWidget(self.stream_rate_lb, 1, 10)
        ctrl_layout.addWidget(QLabel("Hz"), 1, 11, AR)

        progress = QWidget()
        progress_layout = QGridLayout()
        progress_layout.addWidget(self.tid_start_lb, 2, 0, AR)
        progress_layout.addWidget(self.curr_tid_lcd, 2, 1)
        progress_layout.addWidget(self.tid_end_lb, 2, 2)
        progress_layout.addWidget(self.tid_start_sld, 3, 0)
        progress_layout.addWidget(self.tid_progress_br, 3, 1)
        progress_layout.addWidget(self.tid_end_sld, 3, 2)
        progress.setLayout(progress_layout)
        progress.setFixedHeight(progress.minimumSizeHint().height())

        table_area = QSplitter()
        sp_sub = QSplitter(Qt.Vertical)
        sp_sub.addWidget(self._createListGroupBox(
            self._detector_src_tb, "Detector sources", self._select_all_cb))
        sp_sub.addWidget(self._createListGroupBox(
            self._instrument_src_tb,
            "Instrument sources (excluding detector sources)"))
        table_area.addWidget(sp_sub)
        table_area.addWidget(self._createListGroupBox(
            self._control_src_tb, "Control sources"))

        layout = QVBoxLayout()
        layout.addLayout(ctrl_layout)
        layout.addWidget(progress)
        layout.addWidget(self._gui_logger.widget)
        layout.addWidget(table_area)
        self.setLayout(layout)

    def initConnections(self):
        self.load_run_btn.clicked.connect(self.onRunFolderLoad)

        self.tid_start_sld.valueChanged.connect(self._onTidStartChanged)
        self.tid_end_sld.valueChanged.connect(self._onTidEndChanged)

        self._select_all_cb.toggled.connect(
            lambda x: self._setAllChecked(self._detector_src_tb, x))

    def onRunFolderLoad(self):
        folder_name = QFileDialog.getExistingDirectory(
            options=QFileDialog.ShowDirsOnly)

        if folder_name:
            self.data_folder_le.setText(folder_name)

    def _onTidStartChanged(self, tid: int):
        ub = self.tid_end_lb.text()
        if ub and tid > int(ub):
            tid = int(ub)
            self.tid_start_sld.setValue(tid)
            # The method will anyway be called again
            return

        self.tid_start_lb.setText(str(tid))
        self.tid_progress_br.setMinimum(tid)
        self.tid_progress_br.reset()

    def _onTidEndChanged(self, tid: int):
        lb = self.tid_start_lb.text()
        if lb and tid < int(lb):
            tid = int(lb)
            # The method will anyway be called again
            self.tid_end_sld.setValue(tid)
            return

        self.tid_end_lb.setText(str(tid))
        self.tid_progress_br.setMaximum(tid)
        self.tid_progress_br.reset()

    def initProgressControl(self, first_tid, last_tid):
        self.tid_progress_br.setRange(first_tid, last_tid)
        self.tid_end_sld.setRange(first_tid, last_tid)
        self.tid_start_sld.setRange(first_tid, last_tid)

        if first_tid == -1:
            self.tid_start_lb.setText("")
            self.tid_end_lb.setText("")
        else:
            # prevent slider from unable to set new value
            self.tid_start_lb.setText(str(first_tid))
            self.tid_end_lb.setText(str(last_tid))

        self.tid_start_sld.setValue(first_tid)
        self.tid_end_sld.setValue(last_tid)

    def fillSourceTables(self, run_dir_cal, run_dir_raw):
        detector_srcs, instrument_srcs, control_srcs = gather_sources(
            run_dir_cal, run_dir_raw)
        self._fillSourceTable(detector_srcs, self._detector_src_tb )
        self._fillSourceTable(instrument_srcs, self._instrument_src_tb)
        self._fillSourceTable(control_srcs, self._control_src_tb)

    def _fillSourceTable(self, srcs, table):
        table.setRowCount(len(srcs))
        for i, key in enumerate(sorted(srcs)):
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setText(key)
            table.setItem(i, 0, item)
            cb = QComboBox()
            cb.addItems(srcs[key])
            table.setCellWidget(i, 1, cb)

    def onFileServerStarted(self):
        logger.info("File server started")
        self.serve_start_btn.setEnabled(False)
        self.repeat_serve_start_btn.setEnabled(False)
        self.serve_terminate_btn.setEnabled(True)
        for w in self._non_reconfigurable_widgets:
            w.setEnabled(False)

    def onFileServerStopped(self):
        self.serve_start_btn.setEnabled(True)
        self.repeat_serve_start_btn.setEnabled(True)
        self.serve_terminate_btn.setEnabled(False)
        for w in self._non_reconfigurable_widgets:
            w.setEnabled(True)

    def _createListGroupBox(self, widget, title, ctrl_widget=None):
        gb = QGroupBox(title)
        gb.setStyleSheet(self.GROUP_BOX_STYLE_SHEET)
        layout = QVBoxLayout()
        if ctrl_widget is not None:
            layout.addWidget(ctrl_widget)
        layout.addWidget(widget)
        gb.setLayout(layout)
        return gb

    def getMode(self):
        return self._available_modes[self.mode_cb.currentText()]

    def getSourceLists(self):
        return (self._getSourceListFromTable(self._detector_src_tb),
                self._getSourceListFromTable(self._instrument_src_tb),
                self._getSourceListFromTable(self._control_src_tb))

    def getTidRange(self):
        return (self.tid_start_sld.value(), self.tid_end_sld.value() + 1,
                int(self.tid_stride_le.text()))

    def _getSourceListFromTable(self, table):
        ret = []
        for i in range(table.rowCount()):
            item = table.item(i, 0)
            if item.checkState() == Qt.Checked:
                ret.append((table.item(i, 0).text(),
                            table.cellWidget(i, 1).currentText()))
        return ret

    def _setAllChecked(self, table, checked):
        for i in range(table.rowCount()):
            table.item(i, 0).setCheckState(
                Qt.Checked if checked else Qt.Unchecked)

    def resetDisplay(self):
        self.curr_tid_lcd.display(None)
        self.tid_progress_br.reset()

    def close(self):
        """Override."""
        logger.removeHandler(self._gui_logger)
        return super().close()


class FileStreamWindow(_AbstractSatelliteWindow):

    _title = "File Streamer"

    file_server_started_sgn = pyqtSignal()
    file_server_stopped_sgn = pyqtSignal()

    def __init__(self, *args, port=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._file_server = None

        self._port = None

        self._rd_cal = None
        self._rd_raw = None

        self._latest_tid = Value('i', -1)
        self._rate = Value('f', 0.0)

        self._cw = QWidget()

        self._ctrl_widget = _FileStreamCtrlWidget(parent=self)
        port_le = self._ctrl_widget.port_le
        if self.parent() is None:
            # opened from the terminal
            if port_le.validator().validate(str(port), 0)[0] == \
                    QValidator.Acceptable:
                port_le.setText(str(port))
            else:
                raise ValueError(f"Invalid TCP port: {port}")
        else:
            # opened from the main GUI
            port_le.setReadOnly(True)

        self.initUI()
        self.initConnections()

        self.setMinimumSize(1440, 960)
        self.setCentralWidget(self._cw)

        self.show()

        self._timer = QTimer()
        self._timer.setInterval(100)  # update progress bar every 100 ms
        self._timer.timeout.connect(self._updateDisplay)
        self._timer.start()

    def initUI(self):
        """Override"""
        layout = QVBoxLayout()
        layout.addWidget(self._ctrl_widget)
        self._cw.setLayout(layout)

    def initConnections(self):
        """Override"""
        self._ctrl_widget.serve_start_btn.clicked.connect(
            lambda: self.startFileServer(False))
        self._ctrl_widget.repeat_serve_start_btn.clicked.connect(
            lambda: self.startFileServer(True))
        self.file_server_started_sgn.connect(
            self._ctrl_widget.onFileServerStarted)
        self._ctrl_widget.serve_terminate_btn.clicked.connect(
            self.stopFileServer)
        self.file_server_stopped_sgn.connect(
            self._ctrl_widget.onFileServerStopped)

        self._ctrl_widget.data_folder_le.value_changed_sgn.connect(
            self._populateSources)

        if self._mediator is not None:
            self._mediator.connection_change_sgn.connect(
                self._onTcpPortChange)
            self._mediator.file_stream_initialized_sgn.emit()

    def _populateSources(self, path):
        # reset old DataCollections
        self._rd_cal, self._rd_raw = None, None

        try:
            self._rd_cal, self._rd_raw = load_runs(path)
        except Exception as e:
            logger.error(str(e))
        finally:
            self._ctrl_widget.fillSourceTables(self._rd_cal, self._rd_raw)
            n_trains, first_tid, last_tid = run_info(self._rd_cal)
            if n_trains > 0:
                logger.info(f"Loaded a run from {path} with {n_trains} "
                            f"trains in total!")
            self._ctrl_widget.initProgressControl(first_tid, last_tid)

    def _onTcpPortChange(self, connections):
        endpoint = list(connections.keys())[0]
        self._port = int(endpoint.split(":")[-1])

    def _updateDisplay(self):
        tid = self._latest_tid.value
        if tid > 0:
            self._ctrl_widget.curr_tid_lcd.display(tid)
        self._ctrl_widget.tid_progress_br.setValue(tid)

        self._ctrl_widget.stream_rate_lb.setText(
            f"{round(self._rate.value, 1)}")

    def startFileServer(self, repeat=False):
        ctrl_widget = self._ctrl_widget

        if self._rd_cal is None:
            logger.error("Please load a valid run first!")
            return

        folder = ctrl_widget.data_folder_le.text()
        mode = ctrl_widget.getMode()
        tid_range = ctrl_widget.getTidRange()

        if self._port is None:
            port = int(ctrl_widget.port_le.text())
        else:
            port = self._port

        # Since it is not possible to catch the exception in ZMQStreamer which
        # runs in a thread of the file server process, we check the port
        # availability here.
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                logger.info("Port {} is already in use!".format(port))
                return

        detector_srcs, instrument_srcs, control_srcs = \
            ctrl_widget.getSourceLists()

        self._file_server = Process(
            target=serve_files,
            args=((self._rd_cal, self._rd_raw), port, self._latest_tid,
                  self._rate),
            kwargs={
                'tid_range': tid_range,
                'mode': mode,
                'detector_sources': detector_srcs,
                'instrument_sources': instrument_srcs,
                'control_sources': control_srcs,
                'repeat_stream': repeat,
                'require_all': False,
            })

        self._file_server.start()
        self.file_server_started_sgn.emit()
        logger.info("Streaming file in the folder {} through port {}"
                    .format(folder, port))

    def stopFileServer(self):
        if self._file_server is not None and self._file_server.is_alive():
            # a file_server does not have any shared object
            self._file_server.terminate()

        if self._file_server is not None:
            # this join may be redundant
            self._file_server.join()

        logger.info("File streaming stopped!")

        self.file_server_stopped_sgn.emit()
        self._latest_tid.value = -1
        self._rate.value = 0
        self._ctrl_widget.resetDisplay()

    def closeEvent(self, QCloseEvent):
        """Override"""
        # remove the logger handler first
        self._ctrl_widget.close()
        self.stopFileServer()
        super().closeEvent(QCloseEvent)
