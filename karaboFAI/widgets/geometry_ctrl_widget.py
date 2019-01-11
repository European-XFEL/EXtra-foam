"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

GeometryCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_ctrl_widgets import AbstractCtrlWidget
from ..config import config
from ..helpers import parse_table_widget
from ..logger import logger
from ..widgets.pyqtgraph import QtCore, QtGui
from ..widgets.misc_widgets import FixedWidthLineEdit


class GeometryCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the geometry parameters."""

    # (geometry file, quadrant positions)
    geometry_sgn = QtCore.pyqtSignal(str, list)

    def __init__(self, parent=None):
        super().__init__("Geometry setup", parent=parent)

        self._quad_positions_tb = QtGui.QTableWidget()
        self._geom_file_le = FixedWidthLineEdit(285, config["GEOMETRY_FILE"])

        self._disabled_widgets_during_daq = [
            self._quad_positions_tb,
            self._geom_file_le,
        ]

        self.initUI()

    def initUI(self):
        """Override."""
        geom_file_lb = QtGui.QLabel("Geometry file:")
        quad_positions_lb = QtGui.QLabel("Quadrant positions:")

        self.initQuadTable()

        layout = QtGui.QGridLayout()
        layout.addWidget(geom_file_lb, 0, 0, 1, 3)
        layout.addWidget(self._geom_file_le, 1, 0, 1, 3)
        layout.addWidget(quad_positions_lb, 2, 0, 1, 2)
        layout.addWidget(self._quad_positions_tb, 3, 0, 1, 2)

        self.setLayout(layout)

    def initQuadTable(self):
        n_row = 4
        n_col = 2
        widget = self._quad_positions_tb
        widget.setRowCount(n_row)
        widget.setColumnCount(n_col)
        try:
            for i in range(n_row):
                for j in range(n_col):
                    widget.setItem(i, j, QtGui.QTableWidgetItem(
                        str(config["QUAD_POSITIONS"][i][j])))
        except IndexError:
            pass

        widget.move(0, 0)
        widget.setHorizontalHeaderLabels(['x', 'y'])
        widget.setVerticalHeaderLabels(['1', '2', '3', '4'])
        widget.setColumnWidth(0, 80)
        widget.setColumnWidth(1, 80)

    def updateSharedParameters(self, log=False):
        """Override"""

        try:
            geom_file = self._geom_file_le.text()
            quad_positions = parse_table_widget(self._quad_positions_tb)
            self.geometry_sgn.emit(geom_file, quad_positions)
        except ValueError as e:
            logger.error("<Quadrant positions>: " + str(e))
            return False

        if log:
            logger.info("<Geometry file>: {}".format(geom_file))
            logger.info("<Quadrant positions>: [{}]".format(
                ", ".join(["[{}, {}]".format(p[0], p[1])
                           for p in quad_positions])))

        return True
