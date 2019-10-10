"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""

from PyQt5 import QtCore, QtGui, QtWidgets

from .base_ctrl_widgets import AbstractCtrlWidget


class DeviceListModel(QtCore.QAbstractListModel):
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
    def __init__(self, *args, **kwargs):
        super().__init__("Data source monitor", *args, **kwargs)

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


class DataSourceWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.device_list_widget = DeviceListWidget()

        self.initUI()

    def initUI(self):
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.device_list_widget)
        self.setLayout(layout)
