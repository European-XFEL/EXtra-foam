"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartLineEdit


class AnalysisCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up the general analysis parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__("Global setup", *args, **kwargs)

        self._ma_window_le = SmartLineEdit("1")
        self._ma_window_le.setValidator(QIntValidator(1, 99999))

        self._reset_ma_btn = QPushButton("Reset")

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        layout.addWidget(QLabel("Moving average window: "), 0, 0, AR)
        layout.addWidget(self._ma_window_le, 0, 1)
        layout.addWidget(self._reset_ma_btn, 0, 2, AR)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._ma_window_le.value_changed_sgn.connect(
            lambda x: mediator.onMaWindowChange(int(x)))

        self._reset_ma_btn.clicked.connect(mediator.onResetMa)

    def updateMetaData(self):
        """Override"""
        self._ma_window_le.returnPressed.emit()

        return True
