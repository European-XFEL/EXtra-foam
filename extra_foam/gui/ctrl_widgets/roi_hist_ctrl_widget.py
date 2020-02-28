"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QComboBox, QGridLayout, QLabel

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit, SmartLineEdit
from ...config import RoiCombo


class RoiHistCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up ROI pixel-wise histogram parameters."""

    _available_combos = OrderedDict({
        "": RoiCombo.UNDEFINED,
        "ROI1": RoiCombo.ROI1,
        "ROI2": RoiCombo.ROI2,
        "ROI1 - ROI2": RoiCombo.ROI1_SUB_ROI2,
        "ROI1 + ROI2": RoiCombo.ROI1_ADD_ROI2,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("ROI histogram setup", *args, **kwargs)

        self._combo_cb = QComboBox()
        for v in self._available_combos:
            self._combo_cb.addItem(v)

        self._n_bins_le = SmartLineEdit("10")
        self._n_bins_le.setValidator(QIntValidator(1, 999))

        self._bin_range_le = SmartBoundaryLineEdit("-Inf, Inf")

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
        layout.addWidget(QLabel("# of bins: "), row, 0, AR)
        layout.addWidget(self._n_bins_le, row, 1)

        row += 1
        layout.addWidget(QLabel("Bin range: "), row, 0, AR)
        layout.addWidget(self._bin_range_le, row, 1)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._combo_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiHistComboChange(self._available_combos[x]))

        self._n_bins_le.returnPressed.connect(
            lambda: mediator.onRoiHistNumBinsChange(self._n_bins_le.text()))

        self._bin_range_le.value_changed_sgn.connect(
            mediator.onRoiHistBinRangeChange)

    def updateMetaData(self):
        """Overload."""
        self._combo_cb.currentTextChanged.emit(self._combo_cb.currentText())
        self._n_bins_le.returnPressed.emit()
        self._bin_range_le.returnPressed.emit()
        return True
