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
from ..gui_helpers import invert_dict
from ...config import RoiCombo, RoiFom, config
from ...database import Metadata as mt


class RoiNormCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up ROI normalizer parameters."""

    _available_combos = OrderedDict({
        "ROI3": RoiCombo.ROI3,
        "ROI4": RoiCombo.ROI4,
        "ROI3 - ROI4": RoiCombo.ROI3_SUB_ROI4,
        "ROI3 + ROI4": RoiCombo.ROI3_ADD_ROI4,
    })
    _available_combos_inv = invert_dict(_available_combos)

    _available_types = OrderedDict({
        "SUM": RoiFom.SUM,
        "MEAN": RoiFom.MEAN,
        "MEDIAN": RoiFom.MEDIAN,
        "MAX": RoiFom.MAX,
        "MIN": RoiFom.MIN,
    })
    _available_types_inv = invert_dict(_available_types)

    def __init__(self, *args, **kwargs):
        super().__init__("ROI normalizer setup", *args, **kwargs)

        self._source_cb = QComboBox()
        self._source_cb.addItem(config["DETECTOR"])
        self._current_options = []

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
        layout.addWidget(QLabel("ROI source: "), row, 0, AR)
        layout.addWidget(self._source_cb, row, 1)

        row += 1
        layout.addWidget(QLabel("Combo: "), row, 0, AR)
        layout.addWidget(self._combo_cb, row, 1)

        row += 1
        layout.addWidget(QLabel("FOM: "), row, 0, AR)
        layout.addWidget(self._type_cb, row, 1)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._source_cb.currentTextChanged.connect(
            mediator.onRoiNormSourceChange
        )

        self._combo_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiNormComboChange(self._available_combos[x]))

        self._type_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiNormTypeChange(self._available_types[x]))

    def updateMetaData(self):
        """Overload."""
        self._source_cb.currentTextChanged.emit(self._source_cb.currentText())
        self._combo_cb.currentTextChanged.emit(self._combo_cb.currentText())
        self._type_cb.currentTextChanged.emit(self._type_cb.currentText())
        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.ROI_PROC)
        self._source_cb.setCurrentText(cfg["norm:source"])
        self._combo_cb.setCurrentText(
            self._available_combos_inv[int(cfg["norm:combo"])])
        self._type_cb.setCurrentText(
            self._available_types_inv[int(cfg["norm:type"])])

    def updateOptions(self, options):
        if options == self._current_options:
            return

        self._current_options = options
        self._source_cb.blockSignals(True)
        selected = self._source_cb.currentText()
        for i in range(1, self._source_cb.count()):
            self._source_cb.removeItem(1)

        for item in options:
            self._source_cb.addItem(item)

        self._source_cb.blockSignals(False)
        self._source_cb.setCurrentText(selected)

    @property
    def selected_source(self):
        return self._source_cb.currentText()
