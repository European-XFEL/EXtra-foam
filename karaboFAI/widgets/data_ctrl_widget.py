"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

DataCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_ctrl_widgets import AbstractCtrlWidget
from ..config import config
from ..data_processing import DataSource
from ..logger import logger
from ..widgets.pyqtgraph import QtCore, QtGui


class DataCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the data source."""

    server_tcp_sgn = QtCore.pyqtSignal(str, str)
    data_source_sgn = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__("Data source", *args, **kwargs)

        self._hostname_le = QtGui.QLineEdit(config["SERVER_ADDR"])
        self._port_le = QtGui.QLineEdit(str(config["SERVER_PORT"]))
        self._source_name_le = QtGui.QLineEdit(config["SOURCE_NAME"])

        self._data_src_rbts = []
        # the order must match the definition in the DataSource class
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@files"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@ZMQ bridge"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Processed data@ZMQ bridge"))
        self._data_src_rbts[int(config["SOURCE_TYPE"])].setChecked(True)

        self._server_start_btn = QtGui.QPushButton("Serve")
        self._server_start_btn.clicked.connect(self.parent().onStartServeFile)
        self._server_kill_btn = QtGui.QPushButton("Kill")
        self._server_kill_btn.setEnabled(False)
        self._server_kill_btn.clicked.connect(
            self.parent().onStopServeFile)

        self._disabled_widgets_during_file_serving = [
            self._source_name_le,
        ]

        self._disabled_widgets_during_daq = [
            self._hostname_le,
            self._port_le,
            self._source_name_le,
        ]
        self._disabled_widgets_during_daq.extend(self._data_src_rbts)

        self.initUI()

        self.parent().file_server_started_sgn.connect(self.onFileServerStarted)
        self.parent().file_server_stopped_sgn.connect(self.onFileServerStopped)

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        AR = QtCore.Qt.AlignRight

        src_layout = QtGui.QGridLayout()
        src_layout.addWidget(QtGui.QLabel("Hostname: "), 0, 0, AR)

        sub_layout = QtGui.QHBoxLayout()
        sub_layout.addWidget(self._hostname_le)
        sub_layout.addWidget(QtGui.QLabel("Port: "))
        sub_layout.addWidget(self._port_le)

        src_layout.addLayout(sub_layout, 0, 1)
        src_layout.addWidget(QtGui.QLabel("Source: "), 1, 0, AR)
        src_layout.addWidget(self._source_name_le, 1, 1)
        layout.addLayout(src_layout)

        sub_layout2 = QtGui.QHBoxLayout()
        sub_layout2.addWidget(self._server_start_btn)
        sub_layout2.addWidget(self._server_kill_btn)
        layout.addLayout(sub_layout2)

        for i, btn in enumerate(self._data_src_rbts):
            layout.addWidget(btn)

        self.setLayout(layout)

    def updateSharedParameters(self, log=False):
        """Override"""
        if self._data_src_rbts[DataSource.CALIBRATED_FILE].isChecked() is True:
            data_source = DataSource.CALIBRATED_FILE
        elif self._data_src_rbts[DataSource.CALIBRATED].isChecked() is True:
            data_source = DataSource.CALIBRATED
        else:
            data_source = DataSource.PROCESSED

        self.data_source_sgn.emit(data_source)

        server_hostname = self._hostname_le.text().strip()
        server_port = self._port_le.text().strip()
        self.server_tcp_sgn.emit(server_hostname, server_port)

        if log:
            logger.info("<Host name>, <Port>: {}, {}".
                        format(server_hostname, server_port))

        return True

    @property
    def file_server(self):
        source_name = self._source_name_le.text().strip()
        server_port = self._port_le.text().strip()
        return source_name, server_port

    @QtCore.pyqtSlot()
    def onFileServerStarted(self):
        self._server_start_btn.setEnabled(False)
        self._server_kill_btn.setEnabled(True)

    @QtCore.pyqtSlot()
    def onFileServerStopped(self):
        self._server_start_btn.setEnabled(True)
        self._server_kill_btn.setEnabled(False)
