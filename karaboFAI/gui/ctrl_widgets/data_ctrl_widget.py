"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

DataCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ...config import config, DataSource


class DataCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the data source."""

    _available_sources = OrderedDict({
        "stream data from files": DataSource.FILES,
        "stream data from bridge": DataSource.BRIDGE,
    })

    server_tcp_sgn = QtCore.pyqtSignal(str, str)
    source_type_sgn = QtCore.pyqtSignal(object)

    source_name_sgn = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super().__init__("Data source", *args, **kwargs)

        self._hostname_le = QtGui.QLineEdit(config["SERVER_ADDR"])
        self._hostname_le.setMinimumWidth(120)
        self._port_le = QtGui.QLineEdit(str(config["SERVER_PORT"]))
        self._port_le.setValidator(QtGui.QIntValidator(0, 65535))

        self._source_name_cb = QtGui.QComboBox()
        for src in config["SOURCE_NAME"]:
            self._source_name_cb.addItem(src)
        self._source_name_cb.currentIndexChanged.connect(
            lambda i: self.source_name_sgn.emit(config["SOURCE_NAME"][i]))

        self._source_type_rbts = []
        for key in self._available_sources:
            self._source_type_rbts.append(QtGui.QRadioButton(key))

        source_type = int(config["SOURCE_TYPE"])
        self._source_type_rbts[source_type].setChecked(True)
        # this is a temporary solution to switch the two difference
        # source names for FastCCD at SCS
        if self._source_name_cb.count() > source_type:
            self._source_name_cb.setCurrentIndex(source_type)

        self._data_folder_le = QtGui.QLineEdit(config["DATA_FOLDER"])

        self._serve_start_btn = QtGui.QPushButton("Serve")
        self._serve_start_btn.clicked.connect(self.parent().onStartServeFile)
        self._serve_terminate_btn = QtGui.QPushButton("Terminate")
        self._serve_terminate_btn.setEnabled(False)
        self._serve_terminate_btn.clicked.connect(
            self.parent().onStopServeFile)

        self._disabled_widgets_during_daq = [
            self._hostname_le,
            self._port_le,
        ]
        self._disabled_widgets_during_daq.extend(self._source_type_rbts)

        self.initUI()

        self.parent().file_server_started_sgn.connect(self.onFileServerStarted)
        self.parent().file_server_stopped_sgn.connect(self.onFileServerStopped)

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        AR = QtCore.Qt.AlignRight

        src_layout = QtGui.QGridLayout()
        src_layout.addWidget(QtGui.QLabel("Hostname: "), 0, 0, AR)
        src_layout.addWidget(self._hostname_le, 0, 1)
        src_layout.addWidget(QtGui.QLabel("Port: "), 0, 2, AR)
        src_layout.addWidget(self._port_le, 0, 3)
        src_layout.addWidget(QtGui.QLabel("Detector source name: "), 1, 0, AR)
        src_layout.addWidget(self._source_name_cb, 1, 1, 1, 4)
        src_layout.addWidget(self._source_type_rbts[0], 2, 0)
        src_layout.addWidget(self._source_type_rbts[1], 2, 2)

        serve_file_layout = QtGui.QHBoxLayout()
        serve_file_layout.addWidget(self._serve_start_btn)
        serve_file_layout.addWidget(self._serve_terminate_btn)
        serve_file_layout.addWidget(self._data_folder_le)

        layout.addLayout(src_layout)
        layout.addLayout(serve_file_layout)
        self.setLayout(layout)

    def updateSharedParameters(self, log=False):
        """Override"""
        for btn in self._source_type_rbts:
            if btn.isChecked():
                source_type = self._available_sources[btn.text()]
                break

        self.source_type_sgn.emit(source_type)

        self.source_name_sgn.emit(self._source_name_cb.currentText())

        server_hostname = self._hostname_le.text().strip()
        server_port = self._port_le.text().strip()
        self.server_tcp_sgn.emit(server_hostname, server_port)

        return True

    @property
    def file_server(self):
        source_name = self._data_folder_le.text().strip()
        server_port = self._port_le.text().strip()
        return source_name, server_port

    @QtCore.pyqtSlot()
    def onFileServerStarted(self):
        self._serve_start_btn.setEnabled(False)
        self._serve_terminate_btn.setEnabled(True)
        self._data_folder_le.setEnabled(False)

    @QtCore.pyqtSlot()
    def onFileServerStopped(self):
        self._serve_start_btn.setEnabled(True)
        self._serve_terminate_btn.setEnabled(False)
        self._data_folder_le.setEnabled(True)
