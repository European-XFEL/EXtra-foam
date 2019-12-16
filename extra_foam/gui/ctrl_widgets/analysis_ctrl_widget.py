"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QWidget

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartLineEdit
from ...config import config


class AnalysisCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up general analysis parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__("Global setup", *args, **kwargs)

        index_validator = QIntValidator(
            0, config["MAX_N_PULSES_PER_TRAIN"] - 1)
        self._poi_index_les = [SmartLineEdit(str(0)), SmartLineEdit(str(0))]
        for w in self._poi_index_les:
            w.setValidator(index_validator)
            if not self._pulse_resolved:
                w.setEnabled(False)

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

        row = 0
        layout.addWidget(QLabel("POI indices: "), row, 0, 1, 1, AR)
        layout.addWidget(self._poi_index_les[0], row, 1, 1, 1)
        layout.addWidget(self._poi_index_les[1], row, 2, 1, 1)
        layout.addWidget(QWidget(), 3, 1, 1, 1)

        row += 1
        layout.addWidget(QLabel("Moving average window: "), row, 0, AR)
        layout.addWidget(self._ma_window_le, row, 1)
        layout.addWidget(self._reset_ma_btn, row, 2, AR)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        # this cannot be done using a 'for' loop
        self._poi_index_les[0].value_changed_sgn.connect(
            lambda x: mediator.onPoiIndexChange(0, int(x)))
        self._poi_index_les[1].value_changed_sgn.connect(
            lambda x: mediator.onPoiIndexChange(1, int(x)))
        mediator.poi_window_initialized_sgn.connect(self.updateMetaData)

        self._ma_window_le.value_changed_sgn.connect(
            lambda x: mediator.onMaWindowChange(int(x)))

        self._reset_ma_btn.clicked.connect(mediator.onResetMa)

    def updateMetaData(self):
        """Override"""
        for w in self._poi_index_les:
            w.returnPressed.emit()
        self._ma_window_le.returnPressed.emit()
        return True
