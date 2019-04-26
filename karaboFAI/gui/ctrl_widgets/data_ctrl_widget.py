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
from ..mediator import Mediator
from ...config import config, DataSource

mediator = Mediator()


class DataCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the data source."""

    _available_sources = OrderedDict({
        "Stream data from run directory": DataSource.FILES,
        "Stream data from bridge": DataSource.BRIDGE,
    })

    _mono_chromators = [
        "",
        "SA3_XTD10_MONO/MDL/PHOTON_ENERGY",
    ]

    _xgms = [
        "",
        "SA1_XTD2_XGM/DOOCS/MAIN",
        "SPB_XTD9_XGM/DOOCS/MAIN",
        "SA3_XTD10_XGM/XGM/DOOCS",
        "SCS_BLU_XGM/XGM/DOOCS",
    ]

    source_type_change_sgn = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__("Data source", *args, **kwargs)

        self._hostname_le = QtGui.QLineEdit(config["SERVER_ADDR"])
        self._hostname_le.setMinimumWidth(120)
        self._port_le = QtGui.QLineEdit(str(config["SERVER_PORT"]))
        self._port_le.setValidator(QtGui.QIntValidator(0, 65535))

        self._detector_src_cb = QtGui.QComboBox()
        for src in config["SOURCE_NAME"]:
            self._detector_src_cb.addItem(src)

        self._mono_src_cb = QtGui.QComboBox()
        for src in self._mono_chromators:
            self._mono_src_cb.addItem(src)

        self._xgm_src_cb = QtGui.QComboBox()
        for src in self._xgms:
            self._xgm_src_cb.addItem(src)

        self._source_type_rbts = []
        for key in self._available_sources:
            self._source_type_rbts.append(QtGui.QRadioButton(key))

        source_type = int(config["SOURCE_TYPE"])
        self._source_type_rbts[source_type].setChecked(True)
        # this is a temporary solution to switch the two difference
        # source names for FastCCD at SCS
        if self._detector_src_cb.count() > source_type:
            self._detector_src_cb.setCurrentIndex(source_type)

        self._data_folder_le = QtGui.QLineEdit(config["DATA_FOLDER"])

        self._serve_start_btn = QtGui.QPushButton("Stream files")
        self._serve_start_btn.clicked.connect(self.parent().onStartServeFile)
        self._serve_terminate_btn = QtGui.QPushButton("Terminate")
        self._serve_terminate_btn.setEnabled(False)

        self._disabled_widgets_during_daq = [
            self._detector_src_cb,
            self._hostname_le,
            self._port_le,
        ]
        self._disabled_widgets_during_daq.extend(self._source_type_rbts)

        self.initUI()
        self.initConnections()

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        AR = QtCore.Qt.AlignRight

        src_layout = QtGui.QGridLayout()
        src_layout.addWidget(QtGui.QLabel("Hostname: "), 0, 0, AR)
        src_layout.addWidget(self._hostname_le, 0, 1)
        src_layout.addWidget(QtGui.QLabel("Port: "), 0, 2, AR)
        src_layout.addWidget(self._port_le, 0, 3)
        src_layout.addWidget(QtGui.QLabel("Detector source name: "), 1, 0, AR)
        src_layout.addWidget(self._detector_src_cb, 1, 1, 1, 3)
        src_layout.addWidget(QtGui.QLabel("MonoChromator source name: "), 2, 0, AR)
        src_layout.addWidget(self._mono_src_cb, 2, 1, 1, 3)
        src_layout.addWidget(QtGui.QLabel("XGM source name: "), 3, 0, AR)
        src_layout.addWidget(self._xgm_src_cb, 3, 1, 1, 3)
        src_layout.addWidget(self._source_type_rbts[0], 4, 0)
        src_layout.addWidget(self._source_type_rbts[1], 4, 2)

        serve_file_layout = QtGui.QHBoxLayout()
        serve_file_layout.addWidget(self._serve_start_btn)
        serve_file_layout.addWidget(self._serve_terminate_btn)
        serve_file_layout.addWidget(self._data_folder_le)

        layout.addLayout(src_layout)
        layout.addLayout(serve_file_layout)
        self.setLayout(layout)

    def initConnections(self):
        self._hostname_le.editingFinished.connect(
            lambda: mediator.tcp_host_change_sgn.emit(self._hostname_le.text()))
        self._hostname_le.editingFinished.emit()
        self._port_le.editingFinished.connect(
            lambda: mediator.tcp_port_change_sgn.emit(int(self._port_le.text())))
        self._port_le.editingFinished.emit()

        self._detector_src_cb.currentIndexChanged.connect(
            lambda i: mediator.detector_source_change_sgn.emit(
                config["SOURCE_NAME"][i]))
        self._detector_src_cb.currentIndexChanged.emit(
            self._detector_src_cb.currentIndex())

        self._mono_src_cb.currentTextChanged.connect(
            lambda x: mediator.mono_source_change_sgn.emit(x))
        self._mono_src_cb.currentTextChanged.emit(
            self._mono_src_cb.currentText())

        self._xgm_src_cb.currentTextChanged.connect(
            lambda x: mediator.xgm_source_change_sgn.emit(x))
        self._xgm_src_cb.currentTextChanged.emit(
            self._mono_src_cb.currentText())

        self.source_type_change_sgn.connect(mediator.source_type_change_sgn)

        self._serve_terminate_btn.clicked.connect(self.parent().onStopServeFile)

        self.parent().file_server_started_sgn.connect(self.onFileServerStarted)
        self.parent().file_server_stopped_sgn.connect(self.onFileServerStopped)

    def updateSharedParameters(self, log=False):
        """Override"""
        for btn in self._source_type_rbts:
            if btn.isChecked():
                source_type = self._available_sources[btn.text()]
                break
        self.source_type_change_sgn.emit(source_type)

        return True

    @property
    def file_server(self):
        source_name = self._data_folder_le.text()
        server_port = self._port_le.text()
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
