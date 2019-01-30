"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

CorrelationCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..widgets.pyqtgraph import QtCore, QtGui
from .base_ctrl_widgets import AbstractCtrlWidget
from ..config import config


class CorrelationCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the correlation analysis parameters."""

    allowed_corr_params = ("photon energy",)

    normalizers = (
        "integrated curve", "reference signal"
    )

    figure_of_merits = (
        "single image", "on-off"
    )

    def __init__(self, *args, **kwargs):
        super().__init__("Correlation analysis setup", *args, **kwargs)

        self._figure_of_merit_cb = QtGui.QComboBox()
        for v in self.figure_of_merits:
            self._figure_of_merit_cb.addItem(v)

        self._normalizers_cb = QtGui.QComboBox()
        for v in self.normalizers:
            self._normalizers_cb.addItem(v)

        self._integration_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))

        self._correlation1_le = QtGui.QComboBox()
        self._correlation2_le = QtGui.QComboBox()
        for param in self.allowed_corr_params:
            self._correlation1_le.addItem(param)
            self._correlation2_le.addItem(param)

        self.initUI()

    def initUI(self):
        """Overload."""
        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        # correlation
        layout.addRow("Figure of merit (FOM): ", self._figure_of_merit_cb)
        layout.addRow("Normalized by: ", self._normalizers_cb)
        layout.addRow("Integration range: ", self._integration_range_le)
        layout.addRow("Correlation param 1: ", self._correlation1_le)
        layout.addRow("Correlation param 2: ", self._correlation2_le)

        self.setLayout(layout)

    def updateSharedParameters(self):
        """Override"""
        return ""
