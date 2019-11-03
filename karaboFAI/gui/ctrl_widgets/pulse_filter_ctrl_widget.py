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

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit
from ...config import AnalysisType


class PulseFilterCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Parameters setup for pulse-resolved data filtering."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "ROI1 (sum)": AnalysisType.ROI1_PULSE,
        "ROI2 (sum)": AnalysisType.ROI2_PULSE,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Pulse filter setup", *args, **kwargs)

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(self._analysis_types.keys())
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

    def updateMetaData(self):
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._fom_range_le.returnPressed.emit()

        return True
