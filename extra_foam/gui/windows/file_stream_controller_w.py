"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Ebad Kamil <ebad.kamil@xfel.eu> and Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
from multiprocessing import Process, Value

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIntValidator, QValidator
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
    QHeaderView, QGroupBox, QLabel, QLCDNumber, QSplitter, QTableWidget,
    QTableWidgetItem, QProgressBar, QPushButton, QVBoxLayout, QWidget
)

from zmq.error import ZMQError

from .base_window import _AbstractSatelliteWindow
from ..ctrl_widgets.smart_widgets import SmartLineEdit
from ...config import StreamerMode
from ...logger import logger
from ...offline import gather_sources, load_runs, run_info, serve_files


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
        "continuous": StreamerMode.CONTINUOUS,
    })

    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self.load_run_btn = QPushButton("Load run")

        self.data_folder_le = SmartLineEdit()

        self.serve_start_btn = QPushButton("Start")
        self.serve_terminate_btn = QPushButton("Terminate")
        self.serve_terminate_btn.setEnabled(False)

        self.mode_cb = QComboBox()
        for v in self._available_modes:
            self.mode_cb.addItem(v)
        self.port_le = SmartLineEdit("*")
        self.port_le.setValidator(QIntValidator(0, 65535))

        lcd = QLCDNumber(12)
        lcd.setSegmentStyle(QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QColor(0, 0, 0))
        palette.setColor(palette.Background, QColor(0, 170, 255))
        lcd.setPalette(palette)
        lcd.setAutoFillBackground(True)
        lcd.display(None)
        self.curr_tid_lcd = lcd
        self.tid_progress_br = QProgressBar()

        self.repeat_stream_cb = QCheckBox("Repeat Stream")
        self.repeat_stream_cb.setChecked(False)

        self._run = None
        self._detector_src_tb = QTableWidget()
        self._instrument_src_tb = QTableWidget()
        self._control_src_tb = QTableWidget()
        for tb in (self._detector_src_tb, self._instrument_src_tb, self._control_src_tb):
            tb.setColumnCount(2)
            header = tb.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Interactive)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            tb.verticalHeader().hide()
            tb.setHorizontalHeaderLabels(['Device ID', 'Property'])
            header.resizeSection(0, int(0.6 * header.length()))

        self._non_reconfigurable_widgets = [
            self.data_folder_le,
            self._detector_src_tb ,
            self._instrument_src_tb,
            self._control_src_tb,
            self.load_run_btn,
            self.repeat_stream_cb,
            self.mode_cb,
            self.port_le,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        AR = Qt.AlignRight

        ls_layout = QGridLayout()
        ls_gb = QGroupBox("Load and Stream")
        ls_gb.setStyleSheet(self.GROUP_BOX_STYLE_SHEET)
        ls_layout.addWidget(self.load_run_btn, 0, 0)
        ls_layout.addWidget(self.data_folder_le, 0, 1, 1, 9)
        ls_layout.addWidget(QLabel("Mode: "), 1, 1, AR)
        ls_layout.addWidget(self.mode_cb, 1, 2)
        ls_layout.addWidget(QLabel("Port: "), 1, 3, AR)
        ls_layout.addWidget(self.port_le, 1, 4)
        ls_layout.addWidget(self.serve_start_btn, 1, 5)
        ls_layout.addWidget(self.serve_terminate_btn, 1, 6)
        ls_layout.addWidget(self.repeat_stream_cb, 1, 7)
        ls_gb.setLayout(ls_layout)
        ls_gb.setFixedHeight(ls_gb.minimumSizeHint().height())

        progress_layout = QGridLayout()
        progress_layout.addWidget(self.curr_tid_lcd, 0, 0)
        progress_layout.addWidget(self.tid_progress_br, 0, 1)

        table_area = QSplitter()
        sp_sub = QSplitter(Qt.Vertical)
        sp_sub.addWidget(self._createListGroupBox(
            self._detector_src_tb, "Detector sources"))
        sp_sub.addWidget(self._createListGroupBox(
            self._instrument_src_tb,
            "Instrument sources (excluding detector sources)"))
        table_area.addWidget(sp_sub)
        table_area.addWidget(self._createListGroupBox(
            self._control_src_tb, "Control sources"))

        layout = QVBoxLayout()
        layout.addWidget(ls_gb)
        layout.addLayout(progress_layout)
        layout.addWidget(table_area)
        self.setLayout(layout)

    def initConnections(self):
        self.load_run_btn.clicked.connect(self.onRunFolderLoad)

    def onRunFolderLoad(self):
        folder_name = QFileDialog.getExistingDirectory(
            options=QFileDialog.ShowDirsOnly)

        if folder_name:
            self.data_folder_le.setText(folder_name)

    def initProgressBar(self, first_tid, last_tid):
        self.tid_progress_br.setRange(first_tid, last_tid)

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
        self.serve_terminate_btn.setEnabled(True)
        for w in self._non_reconfigurable_widgets:
            w.setEnabled(False)

    def onFileServerStopped(self):
        self.serve_start_btn.setEnabled(True)
        self.serve_terminate_btn.setEnabled(False)
        for w in self._non_reconfigurable_widgets:
            w.setEnabled(True)

    def _createListGroupBox(self, widget, title):
        gb = QGroupBox(title)
        gb.setStyleSheet(self.GROUP_BOX_STYLE_SHEET)
        layout = QHBoxLayout()
        layout.addWidget(widget)
        gb.setLayout(layout)
        return gb

    def getSourceLists(self):
        return (self._getSourceListFromTable(self._detector_src_tb),
                self._getSourceListFromTable(self._instrument_src_tb),
                self._getSourceListFromTable(self._control_src_tb))

    def _getSourceListFromTable(self, table):
        ret = []
        for i in range(table.rowCount()):
            item = table.item(i, 0)
            if item.checkState() == Qt.Checked:
                ret.append((table.item(i, 0).text(),
                            table.cellWidget(i, 1).currentText()))
        return ret


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

        self._timer = QTimer()
        self._timer.setInterval(100)  # update progress bar every 100 ms
        self._timer.timeout.connect(self._updateProgress)

        self.initUI()
        self.initConnections()

        self.setMinimumSize(1440, 960)
        self.setCentralWidget(self._cw)

        self.show()

    def initUI(self):
        """Override"""
        layout = QVBoxLayout()
        layout.addWidget(self._ctrl_widget)
        self._cw.setLayout(layout)

    def initConnections(self):
        """Override"""
        self._ctrl_widget.serve_start_btn.clicked.connect(
            self.startFileServer)
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
            n_trains, first_tid, last_tid = run_info(self._rd_cal)
            self._ctrl_widget.fillSourceTables(self._rd_cal, self._rd_raw)
            self._ctrl_widget.initProgressBar(first_tid, last_tid)

    def _onTcpPortChange(self, connections):
        endpoint = list(connections.keys())[0]
        self._port = int(endpoint.split(":")[-1])

    def _updateProgress(self):
        tid = self._latest_tid.value
        if tid > 0:
            self._ctrl_widget.curr_tid_lcd.display(tid)
            self._ctrl_widget.tid_progress_br.setValue(tid)

    def startFileServer(self):
        if self._rd_cal is None:
            logger.error("Please load a valid run first!")
            return

        folder = self._ctrl_widget.data_folder_le.text()

        if self._port is None:
            port = int(self._ctrl_widget.port_le.text())
        else:
            port = self._port

        detector_srcs, instrument_srcs, control_srcs = \
            self._ctrl_widget.getSourceLists()

        repeat_stream = self._ctrl_widget.repeat_stream_cb.isChecked()

        self._file_server = Process(
            target=serve_files,
            args=((self._rd_cal, self._rd_raw), port, self._latest_tid),
            kwargs={
                'detector_sources': detector_srcs,
                'instrument_sources': instrument_srcs,
                'control_sources': control_srcs,
                'repeat_stream': repeat_stream,
                'require_all': False,
            })

        try:
            self._file_server.start()
            self.file_server_started_sgn.emit()
            logger.info("Serving file in the folder {} through port {}"
                        .format(folder, port))
        except ZMQError:
            logger.info("Port {} is already in use!".format(port))

        self._timer.start()

    def stopFileServer(self):
        if self._file_server is not None and self._file_server.is_alive():
            # a file_server does not have any shared object
            logger.info("Shutting down File server")
            self._file_server.terminate()

        if self._file_server is not None:
            # this join may be redundant
            self._file_server.join()

        self.file_server_stopped_sgn.emit()
        self._latest_tid.value = -1
        self._ctrl_widget.curr_tid_lcd.display(None)
        self._ctrl_widget.tid_progress_br.reset()

    def closeEvent(self, QCloseEvent):
        """Override"""
        self.stopFileServer()
        super().closeEvent(QCloseEvent)
