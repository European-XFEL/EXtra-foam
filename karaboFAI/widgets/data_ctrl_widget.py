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

    vip_pulse_id1_sgn = QtCore.pyqtSignal(int)
    vip_pulse_id2_sgn = QtCore.pyqtSignal(int)

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
        self._server_terminate_btn = QtGui.QPushButton("Terminate")
        self._server_terminate_btn.setEnabled(False)
        self._server_terminate_btn.clicked.connect(
            self.parent().onStopServeFile)

        # We keep the definitions of attributes which are not used in the
        # PULSE_RESOLVED = True case. It makes sense since these attributes
        # also appear in the defined methods.

        if self._pulse_resolved:
            pulse_range0 = 0
            pulse_range1 = 2700
            vip_pulse_id1 = 0
            vip_pulse_id2 = 1
        else:
            pulse_range0 = 0
            pulse_range1 = 1  # not included, Python convention
            vip_pulse_id1 = 0
            vip_pulse_id2 = 0

        self._pulse_range0_le = QtGui.QLineEdit(str(pulse_range0))
        self._pulse_range0_le.setEnabled(False)
        self._pulse_range1_le = QtGui.QLineEdit(str(pulse_range1))
        self._vip_pulse_id1_le = QtGui.QLineEdit(str(vip_pulse_id1))
        self._vip_pulse_id1_le.returnPressed.connect(
            self.onVipPulse1Confirmed)
        self._vip_pulse_id2_le = QtGui.QLineEdit(str(vip_pulse_id2))
        self._vip_pulse_id2_le.returnPressed.connect(
            self.onVipPulse2Confirmed)

        self._disabled_widgets_during_file_serving = [
            self._source_name_le,
        ]

        self._disabled_widgets_during_daq = [
            self._hostname_le,
            self._port_le,
            self._source_name_le,
            self._pulse_range1_le
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
        vip_pulse1_lb = QtGui.QLabel("VIP pulse ID 1: ")
        vip_pulse2_lb = QtGui.QLabel("VIP pulse ID 2: ")

        layout = QtGui.QVBoxLayout()
        host_layout = QtGui.QHBoxLayout()
        host_layout.addWidget(hostname_lb)
        host_layout.addWidget(self._hostname_le)
        host_layout.addWidget(port_lb)
        host_layout.addWidget(self._port_le)
        layout.addLayout(host_layout)

        source_layout = QtGui.QHBoxLayout()
        source_layout.addWidget(source_name_lb)
        source_layout.addWidget(self._source_name_le)
        layout.addLayout(source_layout)

        for i, btn in enumerate(self._data_src_rbts):
            if i == 0:
                serve_layout = QtGui.QHBoxLayout()
                serve_layout.addWidget(btn)
                serve_layout.addWidget(self._server_start_btn)
                serve_layout.addWidget(self._server_terminate_btn)
                layout.addLayout(serve_layout)
            else:
                layout.addWidget(btn)

        if self._pulse_resolved:
            prange_layout = QtGui.QHBoxLayout()
            prange_layout.addWidget(pulse_range_lb)
            prange_layout.addWidget(self._pulse_range0_le)
            prange_layout.addWidget(QtGui.QLabel(" to "))
            prange_layout.addWidget(self._pulse_range1_le)
            layout.addLayout(prange_layout)

            vipp_layout = QtGui.QHBoxLayout()
            vipp_layout.addWidget(vip_pulse1_lb)
            vipp_layout.addWidget(self._vip_pulse_id1_le)
            vipp_layout.addWidget(vip_pulse2_lb)
            vipp_layout.addWidget(self._vip_pulse_id2_le)
            layout.addLayout(vipp_layout)

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

        pulse_range = (int(self._pulse_range0_le.text()),
                       int(self._pulse_range1_le.text()))
        if pulse_range[1] <= 0:
            logger.error("<Pulse range>: Invalid input!")
            return False
        self.pulse_range_sgn.emit(*pulse_range)

        self._emit_vip_pulse_id1()
        self._emit_vip_pulse_id2()

        if log:
            logger.info("<Host name>, <Port>: {}, {}".
                        format(server_hostname, server_port))
            if self._pulse_resolved:
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

    @QtCore.pyqtSlot()
    def onVipPulse1Confirmed(self):
        self._emit_vip_pulse_id1()

    def _emit_vip_pulse_id1(self):
        try:
            pulse_id = int(self._vip_pulse_id1_le.text().strip())
        except ValueError as e:
            logger.error("<VIP pulse ID 1>: " + str(e))
            return

        if pulse_id < 0:
            logger.error("<VIP pulse ID 1>: pulse ID must be non-negative!")
            return

        self.vip_pulse_id1_sgn.emit(pulse_id)

    @QtCore.pyqtSlot()
    def onVipPulse2Confirmed(self):
        self._emit_vip_pulse_id2()

    def _emit_vip_pulse_id2(self):
        try:
            pulse_id = int(self._vip_pulse_id2_le.text().strip())
        except ValueError as e:
            logger.error("<VIP pulse ID 2>: " + str(e))
            return

        if pulse_id < 0:
            logger.error("<VIP pulse ID 2>: pulse ID must be non-negative!")
            return

        self.vip_pulse_id2_sgn.emit(pulse_id)
