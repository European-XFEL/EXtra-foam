"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp
from collections import OrderedDict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHeaderView, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from ..gui_helpers import parse_table_widget
from ...config import config, GeomAssembler
from ...logger import logger


class GeometryCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up detector geometry parameters."""

    _assemblers = OrderedDict({
        "EXtra-foam": GeomAssembler.OWN,
        "EXtra-geom": GeomAssembler.EXTRA_GEOM,
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._assembler_cb = QComboBox()
        for item in self._assemblers:
            self._assembler_cb.addItem(item)

        self._stack_only_cb = QCheckBox("Stack only")
        self._stack_only_cb.setChecked(False)

        if config["DETECTOR"] == "AGIPD":
            # FIXME: native AGIPD geometry is not implemented yet
            self._assembler_cb.removeItem(0)
            self._stack_only_cb.setEnabled(False)

        self._quad_positions_tb = QTableWidget()

        self._geom_file_le = QLineEdit(config["GEOMETRY_FILE"])
        self._geom_file_open_btn = QPushButton("Load geometry file")
        self._geom_file_open_btn.clicked.connect(self.loadGeometryFile)

        self._non_reconfigurable_widgets = [
            self
        ]

        self.initUI()
        self.initConnections()
        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        self.initQuadTable()
        AR = Qt.AlignRight

        layout1 = QVBoxLayout()
        sub_layout1 = QHBoxLayout()
        sub_layout1.addWidget(self._geom_file_open_btn)
        sub_layout1.addWidget(self._geom_file_le)
        sub_layout2 = QHBoxLayout()
        sub_layout2.addWidget(QLabel("Quadrant positions:"))
        sub_layout2.addWidget(self._quad_positions_tb)
        layout1.addLayout(sub_layout1)
        layout1.addLayout(sub_layout2)

        layout2 = QGridLayout()
        layout2.addWidget(QLabel("Assembler: "), 0, 0, AR)
        layout2.addWidget(self._assembler_cb, 0, 1)
        layout2.addWidget(self._stack_only_cb, 1, 0, 1, 2, AR)

        layout = QHBoxLayout()
        layout.addLayout(layout1)
        layout.addLayout(layout2)
        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._stack_only_cb.toggled.connect(
            mediator.onGeomStackOnlyChange)

        self._assembler_cb.currentTextChanged.connect(
            lambda x: mediator.onGeomAssemblerChange(
                self._assemblers[x]))

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

        self._stack_only_cb.toggled.emit(self._stack_only_cb.isChecked())

        self._assembler_cb.currentTextChanged.emit(
            self._assembler_cb.currentText())

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
