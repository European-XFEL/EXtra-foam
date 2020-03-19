"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from ..gui_helpers import invert_dict
from ...config import Normalizer, RoiCombo, RoiFom
from ...database import Metadata as mt


class RoiFomCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up ROI FOM analysis parameters."""

    _available_norms = OrderedDict({
        "": Normalizer.UNDEFINED,
        "XGM": Normalizer.XGM,
        "DIGITIZER": Normalizer.DIGITIZER,
        "ROI": Normalizer.ROI,
    })
    _available_norms_inv = invert_dict(_available_norms)

    _available_combos = OrderedDict({
        "ROI1": RoiCombo.ROI1,
        "ROI2": RoiCombo.ROI2,
        "ROI1 - ROI2": RoiCombo.ROI1_SUB_ROI2,
        "ROI1 + ROI2": RoiCombo.ROI1_ADD_ROI2,
        "ROI1 / ROI2": RoiCombo.ROI1_DIV_ROI2,
    })
    _available_combos_inv = invert_dict(_available_combos)

    _available_types = OrderedDict({
        "SUM": RoiFom.SUM,
        "MEAN": RoiFom.MEAN,
        "MEDIAN": RoiFom.MEDIAN,
        "MAX": RoiFom.MAX,
        "MIN": RoiFom.MIN,
        "STD": RoiFom.STD,
        "VAR": RoiFom.VAR,
        "STD (norm)": RoiFom.N_STD,
        "VAR (norm)": RoiFom.N_VAR,
    })
    _available_types_inv = invert_dict(_available_types)

    def __init__(self, *args, **kwargs):
        super().__init__("ROI FOM setup", *args, **kwargs)

        self._combo_cb = QComboBox()
        for v in self._available_combos:
            self._combo_cb.addItem(v)

        self._type_cb = QComboBox()
        for v in self._available_types:
            self._type_cb.addItem(v)

        self._norm_cb = QComboBox()
        for v in self._available_norms:
            self._norm_cb.addItem(v)

        self._master_slave_cb = QCheckBox("Master-slave")

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

        row += 1
        layout.addWidget(QLabel("Norm: "), row, 0, AR)
        layout.addWidget(self._norm_cb, row, 1)

        row += 1
        layout.addWidget(self._master_slave_cb, row, 1)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._combo_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiFomComboChange(self._available_combos[x]))

        self._type_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiFomTypeChange(self._available_types[x]))

        self._norm_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiFomNormChange(self._available_norms[x]))

        self._master_slave_cb.toggled.connect(
            mediator.onRoiFomMasterSlaveModeChange)
        self._master_slave_cb.toggled.connect(self.onMasterSlaveModeToggled)

    def updateMetaData(self):
        """Overload."""
        self._combo_cb.currentTextChanged.emit(self._combo_cb.currentText())
        self._type_cb.currentTextChanged.emit(self._type_cb.currentText())
        self._norm_cb.currentTextChanged.emit(self._norm_cb.currentText())
        self._master_slave_cb.toggled.emit(
            self._master_slave_cb.isChecked())
        return True

    def onMasterSlaveModeToggled(self, state):
        if state:
            self._combo_cb.setCurrentText("ROI1")
            self._combo_cb.setEnabled(False)
        else:
            self._combo_cb.setEnabled(True)

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.ROI_PROC)
        self._combo_cb.setCurrentText(
            self._available_combos_inv[int(cfg["fom:combo"])])
        self._type_cb.setCurrentText(
            self._available_types_inv[int(cfg["fom:type"])])
        self._norm_cb.setCurrentText(
            self._available_norms_inv[int(cfg["fom:norm"])])
        self._master_slave_cb.setChecked(cfg["fom:master_slave"] == 'True')
