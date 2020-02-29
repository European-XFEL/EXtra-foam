"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit
from ...config import AnalysisType


class FomFilterCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up pulse-resolved filter parameters."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "ROI FOM": AnalysisType.ROI_FOM,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("FOM filter setup", *args, **kwargs)

        self._analysis_type_cb = QComboBox()
        self._analysis_type_cb.addItems(self._analysis_types.keys())
        self._fom_range_le = SmartBoundaryLineEdit("-Inf, Inf")

        self._pulse_resolved_cb = QCheckBox("Pulse resolved")
        if self._pulse_resolved:
            self._pulse_resolved_cb.setChecked(True)
        else:
            self._pulse_resolved_cb.setChecked(False)
            self._pulse_resolved_cb.setEnabled(False)

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        layout.addWidget(QLabel("Analysis type: "), 0, 0, AR)
        layout.addWidget(self._analysis_type_cb, 0, 1)
        layout.addWidget(QLabel("Fom range: "), 0, 2, AR)
        layout.addWidget(self._fom_range_le, 0, 3)
        layout.addWidget(self._pulse_resolved_cb, 0, 4)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onFomFilterAnalysisTypeChange(
                self._analysis_types[x]))
        self._fom_range_le.value_changed_sgn.connect(
            mediator.onFomFilterRangeChange)
        self._pulse_resolved_cb.toggled.connect(
            mediator.onFomFilterPulseResolvedChange)

    def updateMetaData(self):
        """Overload."""
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._fom_range_le.returnPressed.emit()
        self._pulse_resolved_cb.toggled.emit(
            self._pulse_resolved_cb.isChecked())
        return True
