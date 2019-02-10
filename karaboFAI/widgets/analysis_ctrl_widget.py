"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AnalysisCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from .base_ctrl_widgets import AbstractCtrlWidget
from ..widgets.pyqtgraph import QtCore, QtGui
from ..data_processing import AiNormalizer
from ..helpers import parse_boundary
from ..config import config
from ..logger import logger


class AnalysisCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the general analysis parameters."""

    _available_normalizers = OrderedDict({
        "curve": AiNormalizer.CURVE,
        "roi": AiNormalizer.ROI
    })

    pulse_id_range_sgn = QtCore.pyqtSignal(int, int)
    vip_pulse_id1_sgn = QtCore.pyqtSignal(int)
    vip_pulse_id2_sgn = QtCore.pyqtSignal(int)

    max_pulse_id_validator = QtGui.QIntValidator(0, 2699)
    vip_pulse_validator = QtGui.QIntValidator(0, 2699)

    ai_normalizer_sgn = QtCore.pyqtSignal(object)
    integration_range_sgn = QtCore.pyqtSignal(float, float)
    normalization_range_sgn = QtCore.pyqtSignal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__("General analysis setup", *args, **kwargs)

        # We keep the definitions of attributes which are not used in the
        # PULSE_RESOLVED = True case. It makes sense since these attributes
        # also appear in the defined methods.

        if self._pulse_resolved:
            min_pulse_id = 0
            max_pulse_id = self.max_pulse_id_validator.top()
            vip_pulse_id1 = 0
            vip_pulse_id2 = 1
        else:
            min_pulse_id = 0
            max_pulse_id = 0
            vip_pulse_id1 = 0
            vip_pulse_id2 = 0

        self._min_pulse_id_le = QtGui.QLineEdit(str(min_pulse_id))
        self._min_pulse_id_le.setEnabled(False)
        self._max_pulse_id_le = QtGui.QLineEdit(str(max_pulse_id))
        self._max_pulse_id_le.setValidator(self.max_pulse_id_validator)

        self._vip_pulse_id1_le = QtGui.QLineEdit(str(vip_pulse_id1))
        self._vip_pulse_id1_le.setValidator(self.vip_pulse_validator)
        self._vip_pulse_id1_le.returnPressed.connect(
            self.onVipPulseConfirmed)
        self._vip_pulse_id2_le = QtGui.QLineEdit(str(vip_pulse_id2))
        self._vip_pulse_id2_le.setValidator(self.vip_pulse_validator)
        self._vip_pulse_id2_le.returnPressed.connect(
            self.onVipPulseConfirmed)

        self.enable_ai_cb = QtGui.QCheckBox("Azimuthal integration")

        self._normalizers_cb = QtGui.QComboBox()
        for v in self._available_normalizers:
            self._normalizers_cb.addItem(v)
        self._normalizers_cb.currentTextChanged.connect(
            lambda x: self.ai_normalizer_sgn.emit(self._available_normalizers[x]))
        self._normalization_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._integration_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))

        self._disabled_widgets_during_daq = [
            self._max_pulse_id_le,
            self.enable_ai_cb,
            self._normalizers_cb,
            self._normalization_range_le,
            self._integration_range_le,
        ]

        self.initUI()

    def initUI(self):
        """Overload."""
        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        layout.addRow(self.enable_ai_cb)
        layout.addRow("Normalized by: ", self._normalizers_cb)
        layout.addRow("Normalization range (1/A): ",
                      self._normalization_range_le)
        layout.addRow("Integration range (1/A): ",
                      self._integration_range_le)

        if self._pulse_resolved:
            pid_layout = QtGui.QGridLayout()
            pid_layout.addWidget(QtGui.QLabel("Min. pulse ID: "), 0, 0, 1, 1)
            pid_layout.addWidget(self._min_pulse_id_le, 0, 1, 1, 1)
            pid_layout.addWidget(QtGui.QLabel("Max. pulse ID: "), 0, 2, 1, 1)
            pid_layout.addWidget(self._max_pulse_id_le, 0, 3, 1, 1)

            pid_layout.addWidget(QtGui.QLabel("VIP pulse ID 1: "), 1, 0, 1, 1)
            pid_layout.addWidget(self._vip_pulse_id1_le, 1, 1, 1, 1)
            pid_layout.addWidget(QtGui.QLabel("VIP pulse ID 2: "), 1, 2, 1, 1)
            pid_layout.addWidget(self._vip_pulse_id2_le, 1, 3, 1, 1)
            layout.addRow(pid_layout)

        self.setLayout(layout)

    def updateSharedParameters(self):
        """Override"""
        # Upper bound is not included, Python convention
        pulse_id_range = (int(self._min_pulse_id_le.text()),
                          int(self._max_pulse_id_le.text()) + 1)
        self.pulse_id_range_sgn.emit(*pulse_id_range)

        self._vip_pulse_id1_le.returnPressed.emit()
        self._vip_pulse_id2_le.returnPressed.emit()

        self._normalizers_cb.currentTextChanged.emit(
            self._normalizers_cb.currentText())

        try:
            normalization_range = parse_boundary(
                self._normalization_range_le.text())
            self.normalization_range_sgn.emit(*normalization_range)
        except ValueError as e:
            logger.error("<Normalization range>: " + str(e))
            return None

        try:
            integration_range = parse_boundary(
                self._integration_range_le.text())
            self.integration_range_sgn.emit(*integration_range)
        except ValueError as e:
            logger.error("<Integration range>: " + str(e))
            return None

        info = ''
        if self._pulse_resolved:
            info += "\n<Pulse ID range>: ({}, {})".format(*pulse_id_range)
            info += "\n<Normalization range>: ({}, {})".format(*normalization_range)
            info += "\n<Integration range>: ({}, {})".format(*integration_range)

        return info

    def onVipPulseConfirmed(self):
        sender = self.sender()
        if sender is self._vip_pulse_id1_le:
            sgn = self.vip_pulse_id1_sgn
        else:
            sgn = self.vip_pulse_id2_sgn

        sgn.emit(int(sender.text()))
