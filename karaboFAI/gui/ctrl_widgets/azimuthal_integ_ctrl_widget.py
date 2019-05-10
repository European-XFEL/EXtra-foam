"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AzimuthalIntegCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ..gui_helpers import parse_boundary
from ...config import AiNormalizer, config
from ...logger import logger


class AzimuthalIntegCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the azimuthal integration parameters."""

    _available_normalizers = OrderedDict({
        "AUC": AiNormalizer.AUC,
        "ROI1 - ROI2": AiNormalizer.ROI_SUB,
        "ROI1": AiNormalizer.ROI1,
        "ROI2": AiNormalizer.ROI2,
        "ROI1 + ROI2": AiNormalizer.ROI_SUM,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Azimuthal integration setup", *args, **kwargs)

        # default state is unchecked
        self._pulsed_integ_cb = QtGui.QCheckBox("Pulsed azimuthal integ")

        self._cx_le = QtGui.QLineEdit(str(config["CENTER_X"]))
        self._cx_le.setValidator(QtGui.QIntValidator())
        self._cy_le = QtGui.QLineEdit(str(config["CENTER_Y"]))
        self._cy_le.setValidator(QtGui.QIntValidator())
        self._itgt_method_cb = QtGui.QComboBox()
        for method in config["AZIMUTHAL_INTEG_METHODS"]:
            self._itgt_method_cb.addItem(method)
        self._itgt_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["AZIMUTHAL_INTEG_RANGE"]]))
        self._itgt_points_le = QtGui.QLineEdit(
            str(config["AZIMUTHAL_INTEG_POINTS"]))
        self._itgt_points_le.setValidator(QtGui.QIntValidator(1, 8192))

        self._normalizers_cb = QtGui.QComboBox()
        for v in self._available_normalizers:
            self._normalizers_cb.addItem(v)

        self._auc_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["AZIMUTHAL_INTEG_RANGE"]]))

        self._fom_integ_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["AZIMUTHAL_INTEG_RANGE"]]))

        self._non_reconfigurable_widgets = [
            self._pulsed_integ_cb,
            self._cx_le,
            self._cy_le,
            self._itgt_method_cb,
            self._itgt_range_le,
            self._itgt_points_le,
            self._normalizers_cb,
            self._auc_range_le,
            self._fom_integ_range_le,
        ]

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Cx (pixel): "), 0, 0, AR)
        layout.addWidget(self._cx_le, 0, 1)
        layout.addWidget(QtGui.QLabel("Cy (pixel): "), 0, 2, AR)
        layout.addWidget(self._cy_le, 0, 3)
        layout.addWidget(QtGui.QLabel("Integ method: "), 1, 0, AR)
        layout.addWidget(self._itgt_method_cb, 1, 1)
        layout.addWidget(QtGui.QLabel("Integ points: "), 1, 2, AR)
        layout.addWidget(self._itgt_points_le, 1, 3)
        layout.addWidget(QtGui.QLabel("Integ range (1/A): "), 2, 0, AR)
        layout.addWidget(self._itgt_range_le, 2, 1)
        layout.addWidget(QtGui.QLabel("Normalized by: "), 2, 2, AR)
        layout.addWidget(self._normalizers_cb, 2, 3)
        layout.addWidget(QtGui.QLabel("AUC x range: "), 3, 0, AR)
        layout.addWidget(self._auc_range_le, 3, 1)
        layout.addWidget(QtGui.QLabel("FOM integ range: "), 3, 2, AR)
        layout.addWidget(self._fom_integ_range_le, 3, 3)
        layout.addWidget(self._pulsed_integ_cb, 4, 0, 1, 4, AR)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._itgt_method_cb.currentTextChanged.connect(
            mediator.ai_integ_method_change_sgn)
        self._itgt_method_cb.currentTextChanged.emit(
            self._itgt_method_cb.currentText())

        self._normalizers_cb.currentTextChanged.connect(
            lambda x: mediator.ai_normalizer_change_sgn.emit(
                self._available_normalizers[x]))
        self._normalizers_cb.currentTextChanged.emit(
            self._normalizers_cb.currentText())

        self._pulsed_integ_cb.toggled.connect(
            mediator.ai_pulsed_integ_state_sgn)
        # Let toggled emit
        self._pulsed_integ_cb.setChecked(True)
        self._pulsed_integ_cb.setChecked(False)

    def updateSharedParameters(self):
        """Override"""
        mediator = self._mediator

        center_x = int(self._cx_le.text())
        center_y = int(self._cy_le.text())
        mediator.ai_integ_center_change_sgn.emit(center_x, center_y)

        integ_pts = int(self._itgt_points_le.text().strip())
        if integ_pts < 1:
            logger.error(
                "<Integ points>: Invalid input! Must be positive!")
            return False
        else:
            mediator.ai_integ_pts_change_sgn.emit(integ_pts)

        try:
            integ_range = parse_boundary(self._itgt_range_le.text())
            mediator.ai_integ_range_change_sgn.emit(*integ_range)
        except ValueError as e:
            logger.error("<Integ range>: " + repr(e))
            return False

        try:
            auc_range = parse_boundary(self._auc_range_le.text())
            mediator.ai_auc_range_change_sgn.emit(*auc_range)
        except ValueError as e:
            logger.error("<AUC range>: " + repr(e))
            return False

        try:
            fom_integ_range = parse_boundary(self._fom_integ_range_le.text())
            mediator.ai_fom_integ_range_change_sgn.emit(*fom_integ_range)
        except ValueError as e:
            logger.error("<FOM integ range>: " + repr(e))
            return False

        return True
