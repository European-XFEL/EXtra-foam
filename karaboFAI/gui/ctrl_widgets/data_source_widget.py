"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5 import QtCore, QtGui, QtWidgets

from .base_ctrl_widgets import AbstractCtrlWidget
from ...config import config, DataSource


class DeviceListModel(QtCore.QAbstractListModel):
    """List model interface for monitoring devices."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self._devices = []

    def data(self, index, role=None):
        """Override."""
        if not index.isValid() or index.row() > len(self._devices):
            return None

        if role == QtCore.Qt.DisplayRole:
            return self._devices[index.row()]

    def rowCount(self, parent=None, *args, **kwargs):
        """Override."""
        return len(self._devices)

    def setDeviceList(self, devices):
        if self._devices != devices:
            self.beginResetModel()
            self._devices = devices
            self.endResetModel()


class DeviceListWidget(AbstractCtrlWidget):
    """Widget used to visualize a DeviceListModel."""
    def __init__(self, *args, **kwargs):
        super().__init__("Monitor", *args, **kwargs)

        self._view = QtWidgets.QListView()
        self._model = DeviceListModel()
        self._view.setModel(self._model)

        self.initUI()

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._view)
        self.setLayout(layout)

    def setDeviceList(self, devices):
        self._model.setDeviceList(devices)


class ConnectionCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the TCP connection."""

    def __init__(self, *args, **kwargs):
        super().__init__("Connection", *args, **kwargs)

        self._hostname_le = QtGui.QLineEdit()
        self._hostname_le.setMinimumWidth(150)
        self._port_le = QtGui.QLineEdit()
        self._port_le.setValidator(QtGui.QIntValidator(0, 65535))

        self._source_type_cb = QtGui.QComboBox()
        self._source_type_cb.addItem("run directory", DataSource.FILE)
        self._source_type_cb.addItem("ZeroMQ bridge", DataSource.BRIDGE)
        self._source_type_cb.setCurrentIndex(config['DEFAULT_SOURCE_TYPE'])
        self._current_source_type = None

        # fill the combobox in the run time based on the source type
        self._detector_src_cb = QtGui.QComboBox()

        self._xgm_src_cb = QtGui.QComboBox()
        try:
            self._xgms = \
                self._TOPIC_DATA_CATEGORIES[config["TOPIC"]]["XGM"].device_ids
        except KeyError:
            self._xgms = [""]

        for src in self._xgms:
            self._xgm_src_cb.addItem(src)

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
        src_layout.addWidget(QtGui.QLabel("Hostname: "), 1, 0, AR)
        src_layout.addWidget(self._hostname_le, 1, 1)
        src_layout.addWidget(QtGui.QLabel("Port: "), 2, 0, AR)
        src_layout.addWidget(self._port_le, 2, 1)
        src_layout.addWidget(QtGui.QLabel("Detector source name: "), 3, 0, AR)
        src_layout.addWidget(self._detector_src_cb, 3, 1)
        src_layout.addWidget(QtGui.QLabel("XGM source name: "), 4, 0, AR)
        src_layout.addWidget(self._xgm_src_cb, 4, 1)

        layout.addLayout(src_layout)
        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._source_type_cb.currentIndexChanged.connect(
            lambda x: self.onSourceTypeChange(
                self._source_type_cb.itemData(x)))
        self._source_type_cb.currentIndexChanged.connect(
            lambda x: mediator.onSourceTypeChange(
                self._source_type_cb.itemData(x)))

        # Emit once to fill the QLineEdit
        self._source_type_cb.currentIndexChanged.emit(
            self._source_type_cb.currentIndex())

        # Note: use textChanged signal for non-reconfigurable QLineEdit
        self._hostname_le.textChanged.connect(self.onEndpointChange)
        self._port_le.textChanged.connect(self.onEndpointChange)

        self._detector_src_cb.currentTextChanged.connect(
            mediator.onDetectorSourceNameChange)

        self._xgm_src_cb.currentTextChanged.connect(
            mediator.onXgmSourceNameChange)

    def updateMetaData(self):
        self._source_type_cb.currentIndexChanged.emit(
            self._source_type_cb.currentIndex())

        self._hostname_le.textChanged.emit(self._hostname_le.text())
        self._port_le.textChanged.emit(self._port_le.text())

        self._detector_src_cb.currentTextChanged.emit(
            self._detector_src_cb.currentText())

        self._xgm_src_cb.currentTextChanged.emit(
            self._xgm_src_cb.currentText())

        return True

    @QtCore.pyqtSlot()
    def onEndpointChange(self):
        endpoint = f"tcp://{self._hostname_le.text()}:{self._port_le.text()}"
        self._mediator.onBridgeEndpointChange(endpoint)

    @QtCore.pyqtSlot(object)
    def onSourceTypeChange(self, source_type):
        if source_type == self._current_source_type:
            return
        self._current_source_type = source_type

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


class DataSourceWidget(QtWidgets.QWidget):
    """DataSourceWidget class.

    A widget container which holds ConnectionCtrlWidget, DeviceListWidget
    and DeviceTreeWidget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.connection_ctrl_widget = ConnectionCtrlWidget()
        self.device_list_widget = DeviceListWidget()

        self.initUI()

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.connection_ctrl_widget)
        layout.addWidget(self.device_list_widget)
        self.setLayout(layout)
