"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

GeometryCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

from ..pyqtgraph import Qt, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ..gui_helpers import parse_table_widget
from ...config import config
from ...logger import logger


class GeometryCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the geometry parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__("Geometry setup", *args, **kwargs)

        self._quad_positions_tb = QtGui.QTableWidget()
        self._geom_file_le = QtGui.QLineEdit(config["GEOMETRY_FILE"])
        self._geom_file_open_btn = QtGui.QPushButton("Load geometry file")
        self._geom_file_open_btn.clicked.connect(self.loadGeometryFile)

        self._non_reconfigurable_widgets = [
            self._quad_positions_tb,
            self._geom_file_le,
            self._geom_file_open_btn
        ]

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        self.initQuadTable()

        layout = QtGui.QVBoxLayout()
        sub_layout1 = QtGui.QHBoxLayout()
        sub_layout1.addWidget(self._geom_file_open_btn)
        sub_layout1.addWidget(self._geom_file_le)
        sub_layout2 = QtGui.QHBoxLayout()
        sub_layout2.addWidget(QtGui.QLabel("Quadrant positions:"))
        sub_layout2.addWidget(self._quad_positions_tb)
        layout.addLayout(sub_layout1)
        layout.addLayout(sub_layout2)
        self.setLayout(layout)

    def initQuadTable(self):
        n_row = 2
        n_col = 4
        widget = self._quad_positions_tb
        widget.setRowCount(n_row)
        widget.setColumnCount(n_col)
        try:
            for i in range(n_row):
                for j in range(n_col):
                    widget.setItem(i, j, QtGui.QTableWidgetItem(
                        str(config["QUAD_POSITIONS"][j][i])))
        except IndexError:
            pass

        widget.move(0, 0)
        widget.setVerticalHeaderLabels(['x', 'y'])
        widget.setHorizontalHeaderLabels(['1', '2', '3', '4'])

        header = widget.horizontalHeader()
        for i in range(n_col):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

        header = widget.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

        header_height = widget.horizontalHeader().height()
        widget.setMinimumHeight(header_height * (n_row + 1.5))
        widget.setMaximumHeight(header_height * (n_row + 2.0))

    def loadGeometryFile(self):
        filename = QtGui.QFileDialog.getOpenFileName()[0]
        if filename:
            self._geom_file_le.setText(filename)

    def updateMetaData(self):
        """Override"""
        if not config['REQUIRE_GEOMETRY']:
            return True

        geom_file = self._geom_file_le.text()
        if not osp.isfile(geom_file):
            logger.error(f"<Geometry file>: {geom_file} is not a valid file")
            return False

        self._mediator.onGeomFilenameChange(geom_file)

        try:
            quad_positions = parse_table_widget(self._quad_positions_tb)
        except ValueError as e:
            logger.error("<Quadrant positions>: " + repr(e))
            return False

        self._mediator.onGeomQuadPositionsChange(quad_positions)

        return True
