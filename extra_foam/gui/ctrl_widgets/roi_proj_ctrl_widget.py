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

from .base_ctrl_widgets import _AbstractCtrlWidget, _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit
from ...config import RoiCombo, RoiProjType


class RoiProjCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up ROI 1D projection analysis parameters."""

    _available_norms = _AbstractCtrlWidget._available_norms

    _available_combos = OrderedDict({
        "ROI1": RoiCombo.ROI1,
        "ROI2": RoiCombo.ROI2,
        "ROI1 - ROI2": RoiCombo.ROI1_SUB_ROI2,
        "ROI1 + ROI2": RoiCombo.ROI1_ADD_ROI2,
    })

    _available_types = OrderedDict({
        "SUM": RoiProjType.SUM,
        "MEAN": RoiProjType.MEAN,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("ROI projection setup", *args, **kwargs)

        self._combo_cb = QComboBox()
        for v in self._available_combos:
            self._combo_cb.addItem(v)

        self._type_cb = QComboBox()
        for v in self._available_types:
            self._type_cb.addItem(v)

        self._direct_cb = QComboBox()
        for v in ['x', 'y']:
            self._direct_cb.addItem(v)

        self._norm_cb = QComboBox()
        for v in self._available_norms:
            self._norm_cb.addItem(v)

        self._auc_range_le = SmartBoundaryLineEdit("0, Inf")
        self._fom_integ_range_le = SmartBoundaryLineEdit("0, Inf")

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
        layout.addWidget(QLabel("Type: "), row, 0, AR)
        layout.addWidget(self._type_cb, row, 1)

        row += 1
        layout.addWidget(QLabel("Direction: "), row, 0, AR)
        layout.addWidget(self._direct_cb, row, 1)

        row += 1
        layout.addWidget(QLabel("Norm: "), row, 0, AR)
        layout.addWidget(self._norm_cb, row, 1)

        row += 1
        layout.addWidget(QLabel("AUC range: "), row, 0, AR)
        layout.addWidget(self._auc_range_le, row, 1)

        row += 1
        layout.addWidget(QLabel("FOM range: "), row, 0, AR)
        layout.addWidget(self._fom_integ_range_le, row, 1)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._combo_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiProjComboChange(self._available_combos[x]))

        self._type_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiProjTypeChange(self._available_types[x]))

        self._direct_cb.currentTextChanged.connect(
            mediator.onRoiProjDirectChange)

        self._norm_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiProjNormChange(self._available_norms[x]))

        self._auc_range_le.value_changed_sgn.connect(
            mediator.onRoiProjAucRangeChange)

        self._fom_integ_range_le.value_changed_sgn.connect(
            mediator.onRoiProjFomIntegRangeChange)

    def updateMetaData(self):
        """Overload."""
        self._combo_cb.currentTextChanged.emit(self._combo_cb.currentText())
        self._type_cb.currentTextChanged.emit(self._type_cb.currentText())
        self._direct_cb.currentTextChanged.emit(self._direct_cb.currentText())
        self._norm_cb.currentTextChanged.emit(self._norm_cb.currentText())
        self._auc_range_le.returnPressed.emit()
        self._fom_integ_range_le.returnPressed.emit()
        return True
