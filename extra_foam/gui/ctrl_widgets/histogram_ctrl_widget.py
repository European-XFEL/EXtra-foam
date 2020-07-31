"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QWidget
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .curve_fitting_ctrl_widget import _BaseFittingCtrlWidget, FittingType
from .smart_widgets import SmartBoundaryLineEdit, SmartLineEdit
from ..gui_helpers import invert_dict
from ...config import AnalysisType
from ...database import Metadata as mt


class _FittingCtrlWidget(_BaseFittingCtrlWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initUI()
        self.initConnections()

    def initUI(self):
        AR = Qt.AlignRight

        layout = QGridLayout()
        layout.addWidget(QLabel("Param a0 = "), 1, 0, AR)
        layout.addWidget(self._params[0], 1, 1)
        layout.addWidget(QLabel("Param b0 = "), 1, 2, AR)
        layout.addWidget(self._params[1], 1, 3)
        layout.addWidget(QLabel("Param c0 = "), 1, 4, AR)
        layout.addWidget(self._params[2], 1, 5)
        layout.addWidget(QLabel("Param d0 = "), 2, 0, AR)
        layout.addWidget(self._params[3], 2, 1)
        layout.addWidget(QLabel("Param e0 = "), 2, 2, AR)
        layout.addWidget(self._params[4], 2, 3)
        layout.addWidget(QLabel("Param f0 = "), 2, 4, AR)
        layout.addWidget(self._params[5], 2, 5)
        layout.addWidget(self.fit_btn, 3, 0, 1, 2)
        layout.addWidget(self.clear_btn, 3, 2, 1, 2)
        layout.addWidget(QLabel("Fit type: "), 3, 4, AR)
        layout.addWidget(self.fit_type_cb, 3, 5)
        layout.addWidget(self._output, 4, 0, 1, 6)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.setFixedWidth(self.minimumSizeHint().width())


class HistogramCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up histogram analysis parameters."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "pump-probe": AnalysisType.PUMP_PROBE,
        "ROI FOM": AnalysisType.ROI_FOM,
    })
    _analysis_types_inv = invert_dict(_analysis_types)

    fit_curve_sgn = pyqtSignal()
    clear_fitting_sgn = pyqtSignal()

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

        self._fitting = _FittingCtrlWidget()

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        AR = Qt.AlignRight

        ctrl_widget = QFrame()
        llayout = QGridLayout()
        llayout.addWidget(QLabel("Analysis type: "), 0, 0, AR)
        llayout.addWidget(self._analysis_type_cb, 0, 1)
        llayout.addWidget(self._pulse_resolved_cb, 0, 3, AR)
        llayout.addWidget(self._reset_btn, 0, 5)
        llayout.addWidget(QLabel("Bin range: "), 1, 2, AR)
        llayout.addWidget(self._bin_range_le, 1, 3)
        llayout.addWidget(QLabel("# of bins: "), 1, 4, AR)
        llayout.addWidget(self._n_bins_le, 1, 5)
        llayout.setContentsMargins(0, 0, 0, 0)
        ctrl_widget.setLayout(llayout)

        layout = QHBoxLayout()
        layout.addWidget(ctrl_widget, alignment=Qt.AlignTop)
        layout.addWidget(self._fitting)
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

        self._fitting.fit_btn.clicked.connect(self.fit_curve_sgn)
        self._fitting.clear_btn.clicked.connect(self.clear_fitting_sgn)

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
        if "analysis_type" not in cfg:
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

    def fit_curve(self, x, y):
        return self._fitting.fit(x, y)
