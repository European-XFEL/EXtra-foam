"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Projection1dCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ..mediator import Mediator
from ..gui_helpers import parse_boundary
from ...config import Projection1dNormalizer


class Projection1dCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for ROI analysis."""

    _available_normalizers = OrderedDict({
        "AUC": Projection1dNormalizer.AUC,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("1D projection analysis setup", *args, **kwargs)

        self._normalizers_cb = QtGui.QComboBox()
        for v in self._available_normalizers:
            self._normalizers_cb.addItem(v)

        self._auc_x_range_le = QtGui.QLineEdit("0, Inf")

        self._fom_integ_range_le = QtGui.QLineEdit("0, Inf")

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Normalized by: "), 0, 0, AR)
        layout.addWidget(self._normalizers_cb, 0, 1)
        layout.addWidget(QtGui.QLabel("AUC x range: "), 0, 2, AR)
        layout.addWidget(self._auc_x_range_le, 0, 3)
        layout.addWidget(QtGui.QLabel("FOM integ range: "), 1, 2, AR)
        layout.addWidget(self._fom_integ_range_le, 1, 3)

        self.setLayout(layout)

    def initConnections(self):
        mediator = Mediator()

        self._normalizers_cb.currentTextChanged.connect(
            lambda x: mediator.proj1d_normalizer_change_sgn.emit(
                self._available_normalizers[x]))
        self._normalizers_cb.currentTextChanged.emit(
            self._normalizers_cb.currentText())

        self._auc_x_range_le.editingFinished.connect(
            lambda: mediator.proj1d_auc_x_range_change_sgn.emit(
                *parse_boundary(self._auc_x_range_le.text())))
        self._auc_x_range_le.editingFinished.emit()

        self._fom_integ_range_le.editingFinished.connect(
            lambda: mediator.proj1d_fom_integ_range_change_sgn.emit(
                *parse_boundary(self._fom_integ_range_le.text())))
        self._fom_integ_range_le.editingFinished.emit()
