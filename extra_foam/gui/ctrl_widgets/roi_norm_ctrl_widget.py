"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QGridLayout, QLabel

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from ...config import RoiCombo, RoiFom


class RoiNormCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up ROI normalizer parameters."""

    _available_combos = OrderedDict({
        "ROI3": RoiCombo.ROI3,
        "ROI4": RoiCombo.ROI4,
        "ROI3 - ROI4": RoiCombo.ROI3_SUB_ROI4,
        "ROI3 + ROI4": RoiCombo.ROI3_ADD_ROI4,
    })

    _available_types = OrderedDict({
        "SUM": RoiFom.SUM,
        "MEAN": RoiFom.MEAN,
        "MEDIAN": RoiFom.MEDIAN,
        "MAX": RoiFom.MAX,
        "MIN": RoiFom.MIN,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("ROI normalizer setup", *args, **kwargs)

        self._combo_cb = QComboBox()
        for v in self._available_combos:
            self._combo_cb.addItem(v)

        self._type_cb = QComboBox()
        for v in self._available_types:
            self._type_cb.addItem(v)

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        row = 0
        layout.addWidget(QLabel("Combo: "), row, 0, AR)
        layout.addWidget(self._combo_cb, row, 1)

        row += 1
        layout.addWidget(QLabel("FOM: "), row, 0, AR)
        layout.addWidget(self._type_cb, row, 1)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._combo_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiNormComboChange(self._available_combos[x]))

        self._type_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiNormTypeChange(self._available_types[x]))

    def updateMetaData(self):
        """Overload."""
        self._combo_cb.currentTextChanged.emit(self._combo_cb.currentText())
        self._type_cb.currentTextChanged.emit(self._type_cb.currentText())
        return True
