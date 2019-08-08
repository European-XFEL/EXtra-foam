"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

DataReductionCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5 import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit
from ...config import AnalysisType


class DataReductionCtrlWidget(AbstractCtrlWidget):
    """Parameters setup for data filtering and reduction."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "ROI1 (sum)": AnalysisType.ROI1,
        "ROI2 (sum)": AnalysisType.ROI2,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Data reduction setup", *args, **kwargs)

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(self._analysis_types.keys())

        self._pulse_resolved_cb = QtGui.QCheckBox("Pulse resolved")
        self._pulse_resolved_cb.setChecked(True)
        # For now train-resolved is not allowed
        self._pulse_resolved_cb.setEnabled(False)

        self._fom_range_le = SmartBoundaryLineEdit("-Inf, Inf")

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Analysis type: "), 0, 0, AR)
        layout.addWidget(self._analysis_type_cb, 0, 1)
        layout.addWidget(QtGui.QLabel("Fom range: "), 0, 2, AR)
        layout.addWidget(self._fom_range_le, 0, 3)
        layout.addWidget(self._pulse_resolved_cb, 0, 4, AR)
        placeholder = QtGui.QPushButton(' ' * len('Reset'))
        placeholder.setEnabled(False)
        layout.addWidget(placeholder, 0, 5, AR)
        if not self._pulse_resolved:
            self._pulse_resolved_cb.setChecked(False)
            self._pulse_resolved_cb.setEnabled(False)
            # disable it for train-resolved detectors for now
            self.setEnabled(False)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onDrAnalysisTypeChange(
                self._analysis_types[x]))
        self._fom_range_le.value_changed_sgn.connect(
            mediator.onDrFomRangeChange)

    def updateMetaData(self):
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._fom_range_le.returnPressed.emit()

        return True
