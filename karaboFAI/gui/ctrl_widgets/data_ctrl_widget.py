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
from ..misc_widgets import SmartLineEdit
from ...config import config, DataSource


class DataCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the data source."""

    _available_sources = OrderedDict({
        "run directory": DataSource.FILE,
        "ZeroMQ bridge": DataSource.BRIDGE,
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

    _timing_motors = [
        "",
        "SA3_XTD10_MONO/MDL/PHOTON_ENERGY",
    ]

    # signal used in updateShareParameters since currentTextChanged
    # will affect other entries
    source_type_sgn = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__("Data source", *args, **kwargs)

        self._hostname_le = QtGui.QLineEdit()
        self._hostname_le.setMinimumWidth(150)
        self._port_le = QtGui.QLineEdit()
        self._port_le.setValidator(QtGui.QIntValidator(0, 65535))

        self._source_type_cb = QtGui.QComboBox()
        for src in self._available_sources:
            self._source_type_cb.addItem(src)
        self._source_type_cb.setCurrentIndex(config['DEFAULT_SOURCE_TYPE'])

        # fill the combobox in the run time based on the source type
        self._detector_src_cb = QtGui.QComboBox()

        self._mono_src_cb = QtGui.QComboBox()
        for src in self._mono_chromators:
            self._mono_src_cb.addItem(src)

        self._xgm_src_cb = QtGui.QComboBox()
        for src in self._xgms:
            self._xgm_src_cb.addItem(src)

        self._timing_src_cb = QtGui.QComboBox()
        for src in self._timing_motors:
            self._timing_src_cb.addItem(src)

        self._data_folder_le = SmartLineEdit(config["DATA_FOLDER"])

        self._serve_start_btn = QtGui.QPushButton("Stream files")
        self._serve_terminate_btn = QtGui.QPushButton("Terminate")
        self._serve_terminate_btn.setEnabled(False)

        self._non_reconfigurable_widgets = [
            self._source_type_cb,
            self._detector_src_cb,
            self._hostname_le,
            self._port_le,
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        AR = QtCore.Qt.AlignRight

        src_layout = QtGui.QGridLayout()
        src_layout.addWidget(QtGui.QLabel("Data streamed from: "), 0, 0, AR)
        src_layout.addWidget(self._source_type_cb, 0, 1)
        src_layout.addWidget(QtGui.QLabel("Hostname: "), 0, 2, AR)
        src_layout.addWidget(self._hostname_le, 0, 3)
        src_layout.addWidget(QtGui.QLabel("Port: "), 0, 4, AR)
        src_layout.addWidget(self._port_le, 0, 5)
        src_layout.addWidget(QtGui.QLabel("Detector source name: "), 1, 0, AR)
        src_layout.addWidget(self._detector_src_cb, 1, 1, 1, 5)
        src_layout.addWidget(QtGui.QLabel("MonoChromator source name: "), 2, 0, AR)
        src_layout.addWidget(self._mono_src_cb, 2, 1, 1, 5)
        src_layout.addWidget(QtGui.QLabel("XGM source name: "), 3, 0, AR)
        src_layout.addWidget(self._xgm_src_cb, 3, 1, 1, 5)
        src_layout.addWidget(QtGui.QLabel("Timing source name: "), 4, 0, AR)
        src_layout.addWidget(self._timing_src_cb, 4, 1, 1, 5)

        serve_file_layout = QtGui.QHBoxLayout()
        serve_file_layout.addWidget(self._serve_start_btn)
        serve_file_layout.addWidget(self._serve_terminate_btn)
        serve_file_layout.addWidget(self._data_folder_le)

        layout.addLayout(src_layout)
        layout.addLayout(serve_file_layout)
        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._source_type_cb.currentTextChanged.connect(
            lambda x: self.onSourceTypeChange(self._available_sources[x]))
        self._source_type_cb.currentTextChanged.connect(
            lambda x: mediator.onSourceTypeChange(
                self._available_sources[x]))
        # Emit once to fill the QLineEdit
        self._source_type_cb.currentTextChanged.emit(
            self._source_type_cb.currentText())
        self.source_type_sgn.connect(lambda x: mediator.onSourceTypeChange(
                self._available_sources[x]))

        # Note: use textChanged signal for non-reconfigurable QLineEdit
        self._hostname_le.textChanged.connect(self.onEndpointChange)
        self._port_le.textChanged.connect(self.onEndpointChange)

        self._data_folder_le.returnPressed.connect(
            lambda: mediator.onDataFolderChange(self._data_folder_le.text()))

        self._detector_src_cb.currentTextChanged.connect(
            mediator.onDetectorSourceNameChange)

        self._mono_src_cb.currentTextChanged.connect(
            mediator.onMonoSourceNameChange)

        self._xgm_src_cb.currentTextChanged.connect(
            mediator.onXgmSourceNameChange)

        self._timing_src_cb.currentTextChanged.connect(
            mediator.onTimingSourceNameChange)

        self._serve_start_btn.clicked.connect(self.updateSharedParameters)
        self._serve_start_btn.clicked.connect(mediator.start_file_server_sgn)
        self._serve_terminate_btn.clicked.connect(mediator.stop_file_server_sgn)
        mediator.file_server_started_sgn.connect(self.onFileServerStarted)
        mediator.file_server_stopped_sgn.connect(self.onFileServerStopped)

    def updateSharedParameters(self):
        self.source_type_sgn.emit(self._source_type_cb.currentText())

        self._data_folder_le.returnPressed.emit()

        self._port_le.textChanged.emit(self._port_le.text())

        self._detector_src_cb.currentTextChanged.emit(
            self._detector_src_cb.currentText())

        self._mono_src_cb.currentTextChanged.emit(
            self._mono_src_cb.currentText())

        self._xgm_src_cb.currentTextChanged.emit(
            self._xgm_src_cb.currentText())

        self._timing_src_cb.currentTextChanged.emit(
            self._timing_src_cb.currentText())

        return True

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

    @QtCore.pyqtSlot()
    def onEndpointChange(self):
        endpoint = f"tcp://{self._hostname_le.text()}:{self._port_le.text()}"
        self._mediator.onBridgeEndpointChange(endpoint)

    @QtCore.pyqtSlot(object)
    def onSourceTypeChange(self, source_type):
        while self._detector_src_cb.currentIndex() >= 0:
            self._detector_src_cb.removeItem(0)

        if source_type == DataSource.BRIDGE:
            sources = config["SOURCE_NAME_BRIDGE"]
            hostname = config["SERVER_ADDR"]
            port = config["SERVER_PORT"]
        else:
            sources = config["SOURCE_NAME_FILE"]
            hostname = config["LOCAL_HOST"]
            port = config["LOCAL_PORT"]

        self._hostname_le.setText(hostname)
        self._port_le.setText(str(port))

        self._detector_src_cb.addItems(sources)
