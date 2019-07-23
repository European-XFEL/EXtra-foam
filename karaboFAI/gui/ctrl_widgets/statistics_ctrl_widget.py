"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

StatisticsCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from PyQt5 import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ...config import AnalysisType


class StatisticsCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for monitoring statistics of V(FOM)."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG_PULSE,
        "ROI1 (sum)": AnalysisType.ROI1_PULSE,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Statistics setup", *args, **kwargs)

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(list(self._analysis_types.keys()))

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QHBoxLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Analysis type: "), 1, AR)
        layout.addWidget(self._analysis_type_cb, 1)
        layout.addStretch(4)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onStatisticsAnalysisTypeChange(
                self._analysis_types[x]))

    def updateMetaData(self):
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())

        return True
