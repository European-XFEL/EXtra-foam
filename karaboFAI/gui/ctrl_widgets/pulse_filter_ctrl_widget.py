"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5 import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit
from ...config import AnalysisType


class PulseFilterCtrlWidget(AbstractCtrlWidget):
    """Parameters setup for pulse-resolved data filtering."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "ROI1 (sum)": AnalysisType.ROI1_PULSE,
        "ROI2 (sum)": AnalysisType.ROI2_PULSE,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG_PULSE,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Pulse filter setup", *args, **kwargs)

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(self._analysis_types.keys())
        self._fom_range_le = SmartBoundaryLineEdit("-Inf, Inf")

        self._xgm_intensity_range_le = SmartBoundaryLineEdit("0, Inf")

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

        layout.addWidget(QtGui.QLabel("Analysis type: "), 1, 0, AR)
        xgm_type_cb = QtGui.QComboBox()
        xgm_type_cb.addItem("XGM intensity")
        layout.addWidget(xgm_type_cb, 1, 1)
        layout.addWidget(QtGui.QLabel("Value range: "), 1, 2, AR)
        layout.addWidget(self._xgm_intensity_range_le, 1, 3)

        self.setLayout(layout)

        if not self._pulse_resolved:
            self.setEnabled(False)

    def initConnections(self):
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onPfAnalysisTypeChange(
                self._analysis_types[x]))
        self._fom_range_le.value_changed_sgn.connect(
            mediator.onPfFomRangeChange)

        self._xgm_intensity_range_le.value_changed_sgn.connect(
            mediator.onPfXgmIntensityRangeChange)

    def updateMetaData(self):
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._fom_range_le.returnPressed.emit()
        self._xgm_intensity_range_le.returnPressed.emit()

        return True
