"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

RoiCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ..misc_widgets import SmartBoundaryLineEdit
from ...config import Projection1dNormalizer, RoiFom


class RoiCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for ROI analysis."""

    _available_roi_foms = OrderedDict({
        "sum": RoiFom.SUM,
        "mean": RoiFom.MEAN,
    })

    _available_normalizers = OrderedDict({
        "AUC": Projection1dNormalizer.AUC,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("ROI analysis setup", *args, **kwargs)

        self._roi_fom_cb = QtGui.QComboBox()
        for v in self._available_roi_foms:
            self._roi_fom_cb.addItem(v)

        self._roi_hist_reset_btn = QtGui.QPushButton("Reset")

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

        layout.addWidget(QtGui.QLabel("FOM type: "), 0, 0, AR)
        layout.addWidget(self._roi_fom_cb, 0, 1)
        layout.addWidget(self._roi_hist_reset_btn, 0, 3, AR)
        layout.addWidget(QtGui.QLabel("Normalized by (Proj 1D): "), 1, 0, AR)
        layout.addWidget(self._normalizers_cb, 1, 1)
        layout.addWidget(QtGui.QLabel("AUC range (Proj 1D): "), 1, 2, AR)
        layout.addWidget(self._auc_range_le, 1, 3)
        layout.addWidget(QtGui.QLabel("FOM range (Proj 1D): "), 2, 2, AR)
        layout.addWidget(self._fom_integ_range_le, 2, 3)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._roi_fom_cb.currentTextChanged.connect(
            lambda x: mediator.onRoiFomChange(self._available_roi_foms[x]))
        self._roi_fom_cb.currentTextChanged.emit(
            self._roi_fom_cb.currentText())

        self._roi_hist_reset_btn.clicked.connect(mediator.onRoiReset)

        self._normalizers_cb.currentTextChanged.connect(
            lambda x: mediator.onProj1dNormalizerChange(
                self._available_normalizers[x]))
        self._normalizers_cb.currentTextChanged.emit(
            self._normalizers_cb.currentText())

        self._auc_range_le.value_changed_sgn.connect(
            mediator.onProj1dAucRangeChange)
        self._auc_range_le.returnPressed.emit()

        self._fom_integ_range_le.value_changed_sgn.connect(
            mediator.onProj1dFomIntegRangeChange)
        self._fom_integ_range_le.returnPressed.emit()
