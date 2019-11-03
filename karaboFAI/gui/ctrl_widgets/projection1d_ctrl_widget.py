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
from .smart_widgets import SmartBoundaryLineEdit
from ...config import Normalizer


class Projection1DCtrlWidget(GroupBoxCtrlWidgetBase):
    """Analysis parameters setup for ROI analysis."""

    _available_normalizers = OrderedDict({
        "": Normalizer.UNDEFINED,
        "AUC": Normalizer.AUC,
        "XGM": Normalizer.XGM,
        "ROI3 (sum)": Normalizer.ROI3,
        "ROI4 (sum)": Normalizer.ROI4,
        "ROI3 (sum) - ROI4 (sum)": Normalizer.ROI3_SUB_ROI4,
        "ROI3 (sum) + ROI4 (sum)": Normalizer.ROI3_ADD_ROI4,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("ROI 1D projection setup", *args, **kwargs)

        self._direct_cb = QtGui.QComboBox()
        for v in ['x', 'y']:
            self._direct_cb.addItem(v)

        self._normalizers_cb = QtGui.QComboBox()
        for v in self._available_normalizers:
            self._normalizers_cb.addItem(v)

        self._auc_range_le = SmartBoundaryLineEdit("0, Inf")
        self._fom_integ_range_le = SmartBoundaryLineEdit("0, Inf")

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Direction: "), 1, 0, AR)
        layout.addWidget(self._direct_cb, 1, 1)
        layout.addWidget(QtGui.QLabel("Normalizer: "), 1, 2, AR)
        layout.addWidget(self._normalizers_cb, 1, 3)
        layout.addWidget(QtGui.QLabel("AUC range: "), 2, 0, AR)
        layout.addWidget(self._auc_range_le, 2, 1)
        layout.addWidget(QtGui.QLabel("FOM range: "), 2, 2, AR)
        layout.addWidget(self._fom_integ_range_le, 2, 3)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._direct_cb.currentTextChanged.connect(
            mediator.onRoiProjDirectChange)

        self._normalizers_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiProjNormalizerChange(
                self._available_normalizers[x]))

        self._auc_range_le.value_changed_sgn.connect(
            mediator.onRoiProjAucRangeChange)

        self._fom_integ_range_le.value_changed_sgn.connect(
            mediator.onRoiProjFomIntegRangeChange)

    def updateMetaData(self):
        self._direct_cb.currentTextChanged.emit(
            self._direct_cb.currentText())

        self._normalizers_cb.currentTextChanged.emit(
            self._normalizers_cb.currentText())

        self._auc_range_le.returnPressed.emit()

        self._fom_integ_range_le.returnPressed.emit()

        return True
