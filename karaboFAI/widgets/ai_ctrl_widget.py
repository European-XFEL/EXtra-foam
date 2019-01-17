"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AiCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_ctrl_widgets import AbstractCtrlWidget
from ..config import config
from ..helpers import parse_boundary
from ..logger import logger
from ..widgets.pyqtgraph import QtCore, QtGui


class AiCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the azimuthal integration parameters."""

    sample_distance_sgn = QtCore.pyqtSignal(float)
    center_coordinate_sgn = QtCore.pyqtSignal(int, int)  # (cx, cy)
    integration_method_sgn = QtCore.pyqtSignal(str)
    integration_range_sgn = QtCore.pyqtSignal(float, float)
    integration_points_sgn = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("Azimuthal integration setup", parent=parent)

        self._sample_dist_le = QtGui.QLineEdit(str(config["DISTANCE"]))
        self._cx_le = QtGui.QLineEdit(str(config["CENTER_X"]))
        self._cy_le = QtGui.QLineEdit(str(config["CENTER_Y"]))
        self._itgt_method_cb = QtGui.QComboBox()
        for method in config["INTEGRATION_METHODS"]:
            self._itgt_method_cb.addItem(method)
        self._itgt_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))
        self._itgt_points_le = QtGui.QLineEdit(
            str(config["INTEGRATION_POINTS"]))

        self._disabled_widgets_during_daq = [
            self._sample_dist_le,
            self._cx_le,
            self._cy_le,
            self._itgt_method_cb,
            self._itgt_range_le,
            self._itgt_points_le
        ]

        self.initUI()

    def initUI(self):
        """Override."""
        sample_dist_lb = QtGui.QLabel("Sample distance (m): ")
        cx = QtGui.QLabel("Cx (pixel): ")
        cy = QtGui.QLabel("Cy (pixel): ")
        itgt_method_lb = QtGui.QLabel("Integration method: ")
        itgt_points_lb = QtGui.QLabel("Integration points: ")
        itgt_range_lb = QtGui.QLabel("Integration range (1/A): ")

        layout = QtGui.QHBoxLayout()

        key_layout = QtGui.QVBoxLayout()
        key_layout.addWidget(sample_dist_lb)
        key_layout.addWidget(cx)
        key_layout.addWidget(cy)
        key_layout.addWidget(itgt_method_lb)
        key_layout.addWidget(itgt_points_lb)
        key_layout.addWidget(itgt_range_lb)

        value_layout = QtGui.QVBoxLayout()
        value_layout.addWidget(self._sample_dist_le)
        value_layout.addWidget(self._cx_le)
        value_layout.addWidget(self._cy_le)
        value_layout.addWidget(self._itgt_method_cb)
        value_layout.addWidget(self._itgt_points_le)
        value_layout.addWidget(self._itgt_range_le)

        layout.addLayout(key_layout)
        layout.addLayout(value_layout)
        self.setLayout(layout)

    def updateSharedParameters(self, log=False):
        """Override"""

        sample_distance = float(self._sample_dist_le.text().strip())
        if sample_distance <= 0:
            logger.error("<Sample distance>: Invalid input! Must be positive!")
            return False
        else:
            self.sample_distance_sgn.emit(sample_distance)

        center_x = int(self._cx_le.text().strip())
        center_y = int(self._cy_le.text().strip())
        self.center_coordinate_sgn.emit(center_x, center_y)

        integration_method = self._itgt_method_cb.currentText()
        self.integration_method_sgn.emit(integration_method)

        integration_points = int(self._itgt_points_le.text().strip())
        if integration_points <= 0:
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

        if log:
            logger.info("<Sample distance (m)>: {}".format(sample_distance))
            logger.info("<Cx (pixel), Cy (pixel>: ({:d}, {:d})".
                        format(center_x, center_y))
            logger.info("<Cy (pixel)>: {:d}".format(center_y))
            logger.info("<Integration method>: '{}'".format(
                integration_method))
            logger.info("<Integration range (1/A)>: ({}, {})".
                        format(*integration_range))
            logger.info("<Number of integration points>: {}".
                        format(integration_points))

        return True
