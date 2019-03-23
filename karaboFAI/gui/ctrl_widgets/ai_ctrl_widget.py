"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AiCtrlWidget.

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


class AiCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the azimuthal integration parameters."""
    photon_energy_sgn = QtCore.pyqtSignal(float)
    sample_distance_sgn = QtCore.pyqtSignal(float)
    poni_sgn = QtCore.pyqtSignal(int, int)  # (cy, cx)
    integration_method_sgn = QtCore.pyqtSignal(str)
    integration_range_sgn = QtCore.pyqtSignal(float, float)
    integration_points_sgn = QtCore.pyqtSignal(int)
    ai_normalizer_sgn = QtCore.pyqtSignal(object)
    auc_x_range_sgn = QtCore.pyqtSignal(float, float)
    fom_integration_range_sgn = QtCore.pyqtSignal(float, float)

    _available_normalizers = OrderedDict({
        "AUC": AiNormalizer.AUC,
        "ROI": AiNormalizer.ROI
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Azimuthal integration setup", *args, **kwargs)

        self._photon_energy_le = QtGui.QLineEdit(str(config["PHOTON_ENERGY"]))
        self._photon_energy_le.setValidator(QtGui.QDoubleValidator(0, 100, 6))
        self._sample_dist_le = QtGui.QLineEdit(str(config["DISTANCE"]))
        self._sample_dist_le.setValidator(QtGui.QDoubleValidator(0, 100, 6))
        self._cx_le = QtGui.QLineEdit(str(config["CENTER_X"]))
        self._cx_le.setValidator(QtGui.QIntValidator())
        self._cy_le = QtGui.QLineEdit(str(config["CENTER_Y"]))
        self._cx_le.setValidator(QtGui.QIntValidator())
        self._itgt_method_cb = QtGui.QComboBox()
        for method in config["INTEGRATION_METHODS"]:
            self._itgt_method_cb.addItem(method)
        self._itgt_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._itgt_points_le = QtGui.QLineEdit(
            str(config["INTEGRATION_POINTS"]))
        self._itgt_points_le.setValidator(QtGui.QIntValidator(1, 8192))

        self._normalizers_cb = QtGui.QComboBox()
        for v in self._available_normalizers:
            self._normalizers_cb.addItem(v)
        self._normalizers_cb.currentTextChanged.connect(
            lambda x: self.ai_normalizer_sgn.emit(self._available_normalizers[x]))

        self._auc_x_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))

        self._fom_itgt_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))

        self._disabled_widgets_during_daq = [
            self._photon_energy_le,
            self._sample_dist_le,
            self._cx_le,
            self._cy_le,
            self._itgt_method_cb,
            self._itgt_range_le,
            self._itgt_points_le,
            self._normalizers_cb,
            self._auc_x_range_le,
            self._fom_itgt_range_le,
        ]

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Photon energy (keV): "), 0, 0, AR)
        layout.addWidget(self._photon_energy_le, 0, 1)
        layout.addWidget(QtGui.QLabel("Sample distance (m): "), 1, 0, AR)
        layout.addWidget(self._sample_dist_le, 1, 1)
        layout.addWidget(QtGui.QLabel("Cx (pixel): "), 2, 0, AR)
        layout.addWidget(self._cx_le, 2, 1)
        layout.addWidget(QtGui.QLabel("Cy (pixel): "), 3, 0, AR)
        layout.addWidget(self._cy_le, 3, 1)
        layout.addWidget(QtGui.QLabel("Integration method: "), 4, 0, AR)
        layout.addWidget(self._itgt_method_cb, 4, 1)
        layout.addWidget(QtGui.QLabel("Integration points: "), 5, 0, AR)
        layout.addWidget(self._itgt_points_le, 5, 1)
        layout.addWidget(QtGui.QLabel("Integration range (1/A): "), 6, 0, AR)
        layout.addWidget(self._itgt_range_le, 6, 1)
        layout.addWidget(QtGui.QLabel("Normalized by: "), 7, 0, AR)
        layout.addWidget(self._normalizers_cb, 7, 1)
        layout.addWidget(QtGui.QLabel("AUC x range: "), 8, 0, AR)
        layout.addWidget(self._auc_x_range_le, 8, 1)
        layout.addWidget(QtGui.QLabel("FOM integration range: "), 9, 0, AR)
        layout.addWidget(self._fom_itgt_range_le, 9, 1)

        self.setLayout(layout)

    def updateSharedParameters(self):
        """Override"""
        photon_energy = float(self._photon_energy_le.text().strip())
        if photon_energy <= 0:
            logger.error("<Photon energy>: Invalid input! Must be positive!")
            return False
        else:
            self.photon_energy_sgn.emit(photon_energy)

        sample_distance = float(self._sample_dist_le.text().strip())
        if sample_distance <= 0:
            logger.error("<Sample distance>: Invalid input! Must be positive!")
            return False
        else:
            self.sample_distance_sgn.emit(sample_distance)

        center_x = int(self._cx_le.text().strip())
        center_y = int(self._cy_le.text().strip())
        self.poni_sgn.emit(center_y, center_x)

        integration_method = self._itgt_method_cb.currentText()
        self.integration_method_sgn.emit(integration_method)

        integration_points = int(self._itgt_points_le.text().strip())
        if integration_points < 1:
            logger.error(
                "<Integration points>: Invalid input! Must be positive!")
            return False
        else:
            self.integration_points_sgn.emit(integration_points)

        try:
            integration_range = parse_boundary(self._itgt_range_le.text())
            self.integration_range_sgn.emit(*integration_range)
        except ValueError as e:
            logger.error("<Integration range>: " + str(e))
            return False

        self._normalizers_cb.currentTextChanged.emit(
            self._normalizers_cb.currentText())

        try:
            auc_x_range = parse_boundary(self._auc_x_range_le.text())
            self.auc_x_range_sgn.emit(*auc_x_range)
        except ValueError as e:
            logger.error("<AUC x range>: " + str(e))
            return False

        try:
            fom_integration_range = parse_boundary(
                self._fom_itgt_range_le.text())
            self.fom_integration_range_sgn.emit(*fom_integration_range)
        except ValueError as e:
            logger.error("<FOM integration range>: " + str(e))
            return False

        return True
