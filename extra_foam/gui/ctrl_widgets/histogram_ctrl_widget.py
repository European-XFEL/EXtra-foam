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
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit, SmartLineEdit
from ..gui_helpers import invert_dict
from ...config import AnalysisType
from ...database import Metadata as mt


class HistogramCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up histogram analysis parameters."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "pump-probe": AnalysisType.PUMP_PROBE,
        "ROI FOM": AnalysisType.ROI_FOM,
    })
    _analysis_types_inv = invert_dict(_analysis_types)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._analysis_type_cb = QComboBox()
        self._analysis_type_cb.addItems(self._analysis_types.keys())

        self._pulse_resolved_cb = QCheckBox("Pulse resolved")
        if self._pulse_resolved:
            self._pulse_resolved_cb.setChecked(True)
        else:
            self._pulse_resolved_cb.setChecked(False)
            self._pulse_resolved_cb.setEnabled(False)

        self._n_bins_le = SmartLineEdit("10")
        self._n_bins_le.setValidator(QIntValidator(1, 999))

        self._bin_range_le = SmartBoundaryLineEdit("-Inf, Inf")

        self._reset_btn = QPushButton("Reset")

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        AR = Qt.AlignRight
        AT = Qt.AlignTop

        lwidget = QFrame()
        llayout = QGridLayout()
        llayout.addWidget(QLabel("Analysis type: "), 0, 0, AR)
        llayout.addWidget(self._analysis_type_cb, 0, 1)
        llayout.addWidget(self._pulse_resolved_cb, 1, 1)
        llayout.addWidget(self._reset_btn, 2, 1)
        lwidget.setLayout(llayout)

        rwidget = QFrame()
        rlayout = QGridLayout()
        rlayout.addWidget(QLabel("Bin range: "), 1, 0, AR)
        rlayout.addWidget(self._bin_range_le, 1, 1)
        rlayout.addWidget(QLabel("# of bins: "), 2, 0, AR)
        rlayout.addWidget(self._n_bins_le, 2, 1)
        rwidget.setLayout(rlayout)

        layout = QHBoxLayout()
        layout.addWidget(lwidget, alignment=AT)
        layout.addWidget(rwidget, alignment=AT)
        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onHistAnalysisTypeChange(
                self._analysis_types[x]))
        self._bin_range_le.value_changed_sgn.connect(
            mediator.onHistBinRangeChange)
        self._n_bins_le.returnPressed.connect(
            lambda: mediator.onHistNumBinsChange(self._n_bins_le.text()))
        self._pulse_resolved_cb.toggled.connect(
            mediator.onHistPulseResolvedChange)

        self._reset_btn.clicked.connect(mediator.onHistReset)

    def updateMetaData(self):
        """Overload."""
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._bin_range_le.returnPressed.emit()
        self._n_bins_le.returnPressed.emit()
        self._pulse_resolved_cb.toggled.emit(
            self._pulse_resolved_cb.isChecked())
        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.HISTOGRAM_PROC)
        if not cfg:
            # not initialized
            return

        self._analysis_type_cb.setCurrentText(
            self._analysis_types_inv[int(cfg["analysis_type"])])
        self._bin_range_le.setText(cfg["bin_range"][1:-1])
        self._n_bins_le.setText(cfg['n_bins'])
        if self._pulse_resolved:
            self._pulse_resolved_cb.setChecked(cfg["pulse_resolved"] == 'True')

    def resetAnalysisType(self):
        self._analysis_type_cb.setCurrentText(
            self._analysis_types_inv[AnalysisType.UNDEFINED])
