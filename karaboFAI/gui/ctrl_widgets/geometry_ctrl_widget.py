"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

GeometryCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import Qt, QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ..gui_helpers import parse_table_widget
from ...config import config
from ...logger import logger


class GeometryCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the geometry parameters."""
    # (geometry file, quadrant positions)
    geometry_sgn = QtCore.pyqtSignal(str, list)

    def __init__(self, *args, **kwargs):
        super().__init__("Geometry setup", *args, **kwargs)

        self._quad_positions_tb = QtGui.QTableWidget()
        self._geom_file_le = QtGui.QLineEdit(config["GEOMETRY_FILE"])
        self._geom_file_le.setMinimumWidth(180)
        self._geom_file_open_btn = QtGui.QPushButton("Select")
        self._geom_file_open_btn.clicked.connect(self.loadGeometryFile)

        self._disabled_widgets_during_daq = [
            self._quad_positions_tb,
            self._geom_file_le,
            self._geom_file_open_btn
        ]

        self.initUI()

    def initUI(self):
        """Override."""
        self.initQuadTable()

        layout = QtGui.QVBoxLayout()
        layout.addWidget(QtGui.QLabel("Geometry file:"))
        sub_layout = QtGui.QHBoxLayout()
        sub_layout.addWidget(self._geom_file_le)
        sub_layout.addWidget(self._geom_file_open_btn)
        layout.addLayout(sub_layout)
        layout.addWidget(QtGui.QLabel("Quadrant positions:"))
        layout.addWidget(self._quad_positions_tb)

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

        header = widget.horizontalHeader()
        for i in range(n_col):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

        header = widget.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

    def loadGeometryFile(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if filename:
            self._geom_file_le.setText(filename)

    def updateSharedParameters(self):
        """Override"""
        try:
            geom_file = self._geom_file_le.text()
            quad_positions = parse_table_widget(self._quad_positions_tb)
            self.geometry_sgn.emit(geom_file, quad_positions)
        except ValueError as e:
            logger.error("<Quadrant positions>: " + str(e))
            return None

        if config["REQUIRE_GEOMETRY"]:
            info = "\n<Geometry file>: {}".format(geom_file)
            info += ("\n<Quadrant positions>: [{}]".format(
                ", ".join(["[{}, {}]".format(p[0], p[1])
                           for p in quad_positions])))
        else:
            info = ""

        return info
