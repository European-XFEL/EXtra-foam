"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5 import QtCore, QtGui

from .base_ctrl_widgets import GroupBoxCtrlWidgetBase
from .smart_widgets import SmartLineEdit
from ...config import AnalysisType


class StatisticsCtrlWidget(GroupBoxCtrlWidgetBase):
    """Analysis parameters setup for monitoring statistics of V(FOM)."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "ROI1 (sum)": AnalysisType.ROI1,
        "ROI2 (sum)": AnalysisType.ROI2,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Statistics setup", *args, **kwargs)

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(self._analysis_types.keys())

        self._pulse_resolved_cb = QtGui.QCheckBox("Pulse resolved")
        self._pulse_resolved_cb.setChecked(True)

        self._num_bins_le = SmartLineEdit("10")
        self._num_bins_le.setValidator(QtGui.QIntValidator(1, 10000))

        self._reset_btn = QtGui.QPushButton("Reset")
        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Analysis type: "), 0, 0, AR)
        layout.addWidget(self._analysis_type_cb, 0, 1)
        layout.addWidget(QtGui.QLabel("# of bins: "), 0, 2, AR)
        layout.addWidget(self._num_bins_le, 0, 3)
        layout.addWidget(self._pulse_resolved_cb, 0, 4, AR)
        if not self._pulse_resolved:
            self._pulse_resolved_cb.setChecked(False)
            self._pulse_resolved_cb.setEnabled(False)

        layout.addWidget(self._reset_btn, 0, 5, AR)
        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onStAnalysisTypeChange(
                self._analysis_types[x]))
        self._num_bins_le.returnPressed.connect(
            lambda: mediator.onStNumBinsChange(
                self._num_bins_le.text()))
        self._pulse_resolved_cb.toggled.connect(
            mediator.onStPulseOrTrainResolutionChange)

        self._reset_btn.clicked.connect(mediator.onStReset)

    def updateMetaData(self):
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._num_bins_le.returnPressed.emit()
        self._pulse_resolved_cb.toggled.emit(
            self._pulse_resolved_cb.isChecked())

        return True
