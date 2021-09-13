"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QLabel

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartLineEdit
from ...config import config
from ...utils import get_available_port


class ExtensionCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up extension pipe parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__("Extension setup", *args, **kwargs)

        self._host_le = SmartLineEdit('*')
        self._host_le.setEnabled(False)
        self._port_le = SmartLineEdit(str(get_available_port(config["EXTENSION_PORT"])))
        self._port_le.setValidator(QIntValidator(0, 65535))

        self._non_reconfigurable_widgets = [
            self._port_le,
        ]

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        layout.addWidget(QLabel("IP address"), 0, 0, AR)
        layout.addWidget(self._host_le, 0, 1)
        layout.addWidget(QLabel("Port"), 0, 2, AR)
        layout.addWidget(self._port_le, 0, 3)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        self._host_le.returnPressed.connect(self._onEndpointChange)
        self._port_le.returnPressed.connect(self._onEndpointChange)

    def updateMetaData(self):
        """Overload."""
        self._port_le.returnPressed.emit()
        return True

    def loadMetaData(self):
        """Override."""
        pass

    @pyqtSlot()
    def _onEndpointChange(self):
        self._mediator.onExtensionEndpointChange(
            f"tcp://{self._host_le.text()}:{self._port_le.text()}")
