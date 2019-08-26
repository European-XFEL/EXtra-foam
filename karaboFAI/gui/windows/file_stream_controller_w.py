"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

FileStreamControllerWindow

Author: Ebad Kamil <ebad.kamil@xfel.eu> and Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtCore, QtGui
from zmq.error import ZMQError

from .base_window import AbstractSatelliteWindow
from ...config import config
from ..ctrl_widgets.smart_widgets import SmartLineEdit
from ...logger import logger
from ...metadata import Metadata as mt
from ...metadata import MetaProxy
from ...offline import gather_sources, FileServer


class _FileStreamCtrlWidget(QtGui.QWidget):

    def __init__(self, *, parent=None):
        super().__init__(parent=parent)

        self._load_run_btn = QtGui.QPushButton("Load Run Folder")

        self._data_folder_le = SmartLineEdit()

        self._serve_start_btn = QtGui.QPushButton("Stream files")
        self._serve_terminate_btn = QtGui.QPushButton("Terminate")
        self._serve_terminate_btn.setEnabled(False)

        self._stream_files_once_cb = QtGui.QCheckBox("Repeat Stream")
        self._stream_files_once_cb.setChecked(False)
        self._slow_source_list_widget = QtGui.QListWidget()
        self._run_info_te = QtGui.QPlainTextEdit()
        self._run_info_te.setReadOnly(True)

        self._slow_source_list_widget.setMinimumHeight(60)
        self.initCtrlUI()

    def initCtrlUI(self):
        """Override."""
        GROUP_BOX_STYLE_SHEET = 'QGroupBox:title {'\
                            'color: #8B008B;' \
                            'border: 1px;' \
                            'subcontrol-origin: margin;' \
                            'subcontrol-position: top left;' \
                            'padding-left: 10px;' \
                            'padding-top: 10px;' \
                            'margin-top: 0.0em;}'
        layout = QtGui.QVBoxLayout()

        load_stream_layout = QtGui.QGridLayout()
        load_stream_gb = QtGui.QGroupBox("Load and Stream")
        load_stream_gb.setStyleSheet(GROUP_BOX_STYLE_SHEET)

        load_stream_layout.addWidget(self._load_run_btn, 0, 0)
        load_stream_layout.addWidget(self._data_folder_le, 0, 1, 1, 2)

        load_stream_layout.addWidget(self._stream_files_once_cb, 1, 0)
        load_stream_layout.addWidget(self._serve_start_btn, 1, 1)
        load_stream_layout.addWidget(self._serve_terminate_btn, 1, 2)

        load_stream_gb.setLayout(load_stream_layout)

        run_info_gb = QtGui.QGroupBox("Data Sources and Run Info")
        run_info_gb.setStyleSheet(GROUP_BOX_STYLE_SHEET)

        run_info_layout = QtGui.QHBoxLayout()
        run_info_layout.addWidget(self._slow_source_list_widget)
        run_info_layout.addWidget(self._run_info_te)

        run_info_gb.setLayout(run_info_layout)

        layout.addWidget(load_stream_gb)
        layout.addWidget(run_info_gb)
        self.setLayout(layout)

    def initConnections(self):
        self._data_folder_le.returnPressed.connect(
            lambda: self.populateSources(
                self._data_folder_le.text()))

        self._data_folder_le.returnPressed.emit()

        self._serve_start_btn.clicked.connect(
            self.onFileServerStarted)
        self._serve_terminate_btn.clicked.connect(
            self.onFileServerStopped)

        self._load_run_btn.clicked.connect(self.onRunFolderLoad)

    def onRunFolderLoad(self):
        folder_name = QtGui.QFileDialog.getExistingDirectory(
                        options=QtGui.QFileDialog.ShowDirsOnly)
        if folder_name:
            self._slow_source_list_widget.clear()
            self._data_folder_le.setText(folder_name)

    def populateSources(self, path):
        self._slow_source_list_widget.clear()
        self._run_info_te.clear()
        if path:
            sources, info = gather_sources(path)
            for src in sources:
                item = QtGui.QListWidgetItem()
                item.setCheckState(QtCore.Qt.Unchecked)
                item.setText(src)
                self._slow_source_list_widget.addItem(item)

            self._slow_source_list_widget.sortItems()
            self._run_info_te.appendPlainText(info)

    def onFileServerStarted(self):
        logger.info("File server started")
        self._serve_start_btn.setEnabled(False)
        self._serve_terminate_btn.setEnabled(True)
        self._data_folder_le.setEnabled(False)
        self._slow_source_list_widget.setEnabled(False)
        self._load_run_btn.setEnabled(False)
        self._stream_files_once_cb.setEnabled(False)

    def onFileServerStopped(self):
        self._serve_start_btn.setEnabled(True)
        self._serve_terminate_btn.setEnabled(False)
        self._data_folder_le.setEnabled(True)
        self._slow_source_list_widget.setEnabled(True)
        self._load_run_btn.setEnabled(True)
        self._stream_files_once_cb.setEnabled(True)


class FileStreamControllerWindow(AbstractSatelliteWindow):

    title = "File Streamer"

    def __init__(self, *args, detector=None, port=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._detector = detector
        self._port = port
        self._data_folder = None
        self._file_server = None
        self._repeat_stream = False
        self._slow_devices = set()

        self._meta = MetaProxy()

        self._cw = QtGui.QWidget()
        self._file_stream_ctrl_widget = _FileStreamCtrlWidget(parent=self)
        self._widget = self._file_stream_ctrl_widget

        self.initUI()
        self.setMinimumSize(800, 500)
        self.setCentralWidget(self._cw)

        self.initConnections()

        self.show()

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._widget)
        self._cw.setLayout(layout)

    def initConnections(self):
        """Override"""
        self._widget._data_folder_le.returnPressed.connect(
            lambda: self.onFileServerDataFolderChange(
                self._widget._data_folder_le.text()))

        self._widget._serve_start_btn.clicked.connect(
            self.startFileServer)
        self._widget._serve_terminate_btn.clicked.connect(
            self.stopFileServer)

        self._widget._slow_source_list_widget.itemClicked.connect(
            lambda x: self.onSlowDeviceChange(
                x.checkState(), x.text()))

        self._widget._stream_files_once_cb.toggled.connect(
            self.onRepeatStreamChange)

        self._widget.initConnections()

    def onFileServerDataFolderChange(self, path):
        self._slow_devices.clear()
        self._data_folder = path if path else None

    def startFileServer(self):
        folder = self._data_folder

        if folder is None:
            logger.error("No run folder specified")
            return

        if self._port is not None:
            port = self._port
        else:
            cfg = self._meta.get_all(mt.DATA_SOURCE)
            try:
                port = cfg['endpoint'].split(':')[-1]
            except KeyError as e:
                logger.error(repr(e))
                return

        slow_devices = list(self._slow_devices)
        repeat_stream = self._repeat_stream
        # process can only be start once
        self._file_server = FileServer(folder, port,
                                       detector=self._detector,
                                       slow_devices=slow_devices,
                                       repeat_stream=repeat_stream)
        try:
            self._file_server.start()

            logger.info("Serving file in the folder {} through port {}"
                        .format(folder, port))
        except FileNotFoundError:
            logger.info("{} does not exist!".format(folder))
        except ZMQError:
            logger.info("Port {} is already in use!".format(port))

    def stopFileServer(self):
        if self._file_server is not None and self._file_server.is_alive():
            # a file_server does not have any shared object
            logger.info("Shutting down File server")
            self._file_server.terminate()

        if self._file_server is not None:
            # this join may be redundant
            self._file_server.join()

    def onSlowDeviceChange(self, state, source):
        self._slow_devices.add((source, '*')) if state \
            else self._slow_devices.discard((source, '*'))

    def onRepeatStreamChange(self, state):
        self._repeat_stream = state

    def closeEvent(self, QCloseEvent):
        """Override"""
        self.stopFileServer()
        super().closeEvent(QCloseEvent)
