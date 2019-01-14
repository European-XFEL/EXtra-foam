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
    pulse_range_sgn = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__("Data source", parent=parent)

        self._hostname_le = QtGui.QLineEdit(config["SERVER_ADDR"])
        self._port_le = QtGui.QLineEdit(str(config["SERVER_PORT"]))
        self._source_name_le = QtGui.QLineEdit(config["SOURCE_NAME"])
        self._pulse_range0_le = QtGui.QLineEdit(str(0))
        self._pulse_range1_le = QtGui.QLineEdit(str(2699))

        self._data_src_rbts = []
        # the order must match the definition in the DataSource class
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@files"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Calibrated data@ZMQ bridge"))
        self._data_src_rbts.append(
            QtGui.QRadioButton("Processed data@ZMQ bridge"))
        self._data_src_rbts[int(config["SOURCE_TYPE"])].setChecked(True)

        self._pulse_range0_le.setEnabled(False)

        self._server_start_btn = QtGui.QPushButton("Serve")
        self._server_start_btn.clicked.connect(self.parent().onStartServeFile)
        self._server_terminate_btn = QtGui.QPushButton("Terminate")
        self._server_terminate_btn.setEnabled(False)
        self._server_terminate_btn.clicked.connect(
            self.parent().onStopServeFile)

        self._disabled_widgets_during_file_serving = [
            self._source_name_le,
        ]

        self._disabled_widgets_during_daq = [
            self._hostname_le,
            self._port_le,
            self._source_name_le,
            self._pulse_range1_le,
        ]
        self._disabled_widgets_during_daq.extend(self._data_src_rbts)

        self.initUI()

        self.parent().file_server_started_sgn.connect(self.onFileServerStarted)
        self.parent().file_server_stopped_sgn.connect(self.onFileServerStopped)

    def initUI(self):
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
        layout.addLayout(sub_layout1)

        sub_layout2 = QtGui.QHBoxLayout()
        sub_layout2.addWidget(source_name_lb)
        sub_layout2.addWidget(self._source_name_le)
        layout.addLayout(sub_layout2)

        for i, btn in enumerate(self._data_src_rbts):
            if i == 0:
                sub_layout3 = QtGui.QHBoxLayout()
                sub_layout3.addWidget(btn)
                sub_layout3.addWidget(self._server_start_btn)
                sub_layout3.addWidget(self._server_terminate_btn)
                layout.addLayout(sub_layout3)
            else:
                layout.addWidget(btn)

        sub_layout4 = QtGui.QHBoxLayout()
        sub_layout4.addWidget(pulse_range_lb)
        sub_layout4.addWidget(self._pulse_range0_le)
        sub_layout4.addWidget(QtGui.QLabel(" to "))
        sub_layout4.addWidget(self._pulse_range1_le)
        sub_layout4.addStretch(2)
        layout.addLayout(sub_layout4)

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
            logger.info("<Host name>, <Port>: {}, {}".
                        format(server_hostname, server_port))
            logger.info("<Pulse range>: ({}, {})".format(*pulse_range))

        return True

    @property
    def file_server(self):
        source_name = self._source_name_le.text().strip()
        server_port = self._port_le.text().strip()
        return source_name, server_port

    @QtCore.pyqtSlot()
    def onFileServerStarted(self):
        self._server_start_btn.setEnabled(False)
        self._server_terminate_btn.setEnabled(True)

    @QtCore.pyqtSlot()
    def onFileServerStopped(self):
        self._server_start_btn.setEnabled(True)
        self._server_terminate_btn.setEnabled(False)
