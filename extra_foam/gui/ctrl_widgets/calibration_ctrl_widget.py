"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QGridLayout, QLabel

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartLineEdit


class CalibrationCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up calibration parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._gain_le = SmartLineEdit(str(1.0))
        self._gain_le.setValidator(QDoubleValidator())

        self._offset_le = SmartLineEdit(str(0.0))
        self._offset_le.setValidator(QDoubleValidator())

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        layout.addWidget(QLabel("Gain: "), 0, 0, AR)
        layout.addWidget(self._gain_le, 0, 1)
        layout.addWidget(QLabel("Offset: "), 1, 0, AR)
        layout.addWidget(self._offset_le, 1, 1)

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        self._gain_le.value_changed_sgn.connect(
            lambda x: mediator.onCaliGainChange(float(x)))
        self._offset_le.value_changed_sgn.connect(
            lambda x: mediator.onCaliOffsetChange(float(x)))

    def updateMetaData(self):
        """Override."""
        self._gain_le.returnPressed.emit()
        self._offset_le.returnPressed.emit()
        return True
