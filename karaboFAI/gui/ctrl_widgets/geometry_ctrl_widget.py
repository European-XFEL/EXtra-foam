"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QHeaderView, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from ..gui_helpers import parse_table_widget
from ...config import config
from ...logger import logger


class GeometryCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up the geometry parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._with_geometry_cb = QCheckBox("Assemble with geometry")
        self._with_geometry_cb.setChecked(True)

        self._quad_positions_tb = QTableWidget()
        self._geom_file_le = QLineEdit(config["GEOMETRY_FILE"])
        self._geom_file_open_btn = QPushButton("Load geometry file")
        self._geom_file_open_btn.clicked.connect(self.loadGeometryFile)

        self._non_reconfigurable_widgets = [
            self._with_geometry_cb,
            self._quad_positions_tb,
            self._geom_file_le,
            self._geom_file_open_btn
        ]

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        self.initQuadTable()

        layout = QVBoxLayout()
        sub_layout1 = QHBoxLayout()
        sub_layout1.addWidget(self._geom_file_open_btn)
        sub_layout1.addWidget(self._geom_file_le)
        sub_layout2 = QHBoxLayout()
        sub_layout2.addWidget(QLabel("Quadrant positions:"))
        sub_layout2.addWidget(self._quad_positions_tb)
        layout.addWidget(self._with_geometry_cb)
        layout.addLayout(sub_layout1)
        layout.addLayout(sub_layout2)
        self.setLayout(layout)

    def initConnections(self):
        pass

    def initQuadTable(self):
        n_row = 2
        n_col = 4
        widget = self._quad_positions_tb
        widget.setRowCount(n_row)
        widget.setColumnCount(n_col)
        try:
            for i in range(n_row):
                for j in range(n_col):
                    widget.setItem(i, j, QTableWidgetItem(
                        str(config["QUAD_POSITIONS"][j][i])))
        except IndexError:
            pass

        widget.move(0, 0)
        widget.setVerticalHeaderLabels(['x', 'y'])
        widget.setHorizontalHeaderLabels(['1', '2', '3', '4'])

        header = widget.horizontalHeader()
        for i in range(n_col):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        header = widget.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        header_height = widget.horizontalHeader().height()
        widget.setMinimumHeight(header_height * (n_row + 1.5))
        widget.setMaximumHeight(header_height * (n_row + 2.0))

    def loadGeometryFile(self):
        filename = QFileDialog.getOpenFileName()[0]
        if filename:
            self._geom_file_le.setText(filename)

    def updateMetaData(self):
        """Override"""
        if not config['REQUIRE_GEOMETRY']:
            return True

        self._mediator.onGeomAssembleWithGeometryChange(
            self._with_geometry_cb.isChecked())

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
