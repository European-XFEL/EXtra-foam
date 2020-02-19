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
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QLabel, QPushButton
)

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartLineEdit
from ...config import AnalysisType


class HistogramCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up histogram analysis parameters."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "ROI FOM": AnalysisType.ROI_FOM,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Histogram setup", *args, **kwargs)

        self._analysis_type_cb = QComboBox()
        self._analysis_type_cb.addItems(self._analysis_types.keys())

        self._pulse_resolved_cb = QCheckBox("Pulse resolved")
        self._pulse_resolved_cb.setChecked(True)

        self._n_bins_le = SmartLineEdit("10")
        self._n_bins_le.setValidator(QIntValidator(1, 999))

        self._reset_btn = QPushButton("Reset")

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        layout.addWidget(QLabel("Analysis type: "), 0, 0, AR)
        layout.addWidget(self._analysis_type_cb, 0, 1)
        layout.addWidget(QLabel("# of bins: "), 0, 2, AR)
        layout.addWidget(self._n_bins_le, 0, 3)
        layout.addWidget(self._pulse_resolved_cb, 0, 4, AR)
        if not self._pulse_resolved:
            self._pulse_resolved_cb.setChecked(False)
            self._pulse_resolved_cb.setEnabled(False)

        layout.addWidget(self._reset_btn, 0, 5, AR)
        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onHistAnalysisTypeChange(
                self._analysis_types[x]))
        self._n_bins_le.returnPressed.connect(
            lambda: mediator.onHistNumBinsChange(self._n_bins_le.text()))
        self._pulse_resolved_cb.toggled.connect(
            mediator.onHistPulseResolvedChange)

        self._reset_btn.clicked.connect(mediator.onHistReset)

    def updateMetaData(self):
        """Overload."""
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._n_bins_le.returnPressed.emit()
        self._pulse_resolved_cb.toggled.emit(
            self._pulse_resolved_cb.isChecked())

        return True
