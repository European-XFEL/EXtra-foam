"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHeaderView, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QVBoxLayout
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartLineEdit, SmartStringLineEdit
from ..gui_helpers import invert_dict
from ...config import config, GeomAssembler
from ...database import Metadata as mt
from ..items import GeometryItem


def _parse_table_widget(widget):
    ret = []
    for i in range(widget.columnCount()):
        ret.append([float(widget.cellWidget(j, i).text())
                    for j in range(widget.rowCount())])
    return ret


class GeometryCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up detector geometry parameters."""

    _assemblers = OrderedDict({
        "EXtra-foam": GeomAssembler.OWN,
        "EXtra-geom": GeomAssembler.EXTRA_GEOM,
    })
    _assemblers_inv = invert_dict(_assemblers)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._geom = GeometryItem()

        self._assembler_cb = QComboBox()
        for item in self._assemblers:
            self._assembler_cb.addItem(item)

        self._stack_only_cb = QCheckBox("Stack only")
        self._stack_only_cb.setChecked(False)

        self._quad_positions_tb = QTableWidget()
        if config["DETECTOR"] == "AGIPD":
            self._quad_positions_tb.setEnabled(False)

        self._geom_file_le = SmartStringLineEdit(config["GEOMETRY_FILE"])
        self._geom_file_open_btn = QPushButton("Load geometry file")
        self._geom_file_open_btn.clicked.connect(self._setGeometryFile)

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

        self._stack_only_cb.toggled.connect(self._onStackOnlyChange)

        self._assembler_cb.currentTextChanged.connect(
            lambda x: mediator.onGeomAssemblerChange(
                self._assemblers[x]))
        self._assembler_cb.currentTextChanged.connect(
            lambda x: self._onAssemblerChange(self._assemblers[x]))

        self._geom_file_open_btn.clicked.connect(self._setGeometryFile)

        self._geom_file_le.value_changed_sgn.connect(
            self._onGeometryFileChange)

    def initQuadTable(self):
        n_row = 2
        n_col = 4

        table = self._quad_positions_tb
        table.setRowCount(n_row)
        table.setColumnCount(n_col)

        for i in range(n_row):
            for j in range(n_col):
                if config["DETECTOR"] in ["LPD", "DSSC"]:
                    widget = SmartLineEdit(str(config["QUAD_POSITIONS"][j][i]))
                else:
                    widget = SmartLineEdit('0')
                widget.setValidator(QDoubleValidator(-999, 999, 6))
                widget.value_changed_sgn.connect(self._onQuadPositionsChange)
                table.setCellWidget(i, j, widget)

        table.move(0, 0)
        table.setVerticalHeaderLabels(['x', 'y'])
        table.setHorizontalHeaderLabels(['1', '2', '3', '4'])

        header = table.horizontalHeader()

        for i in range(n_col):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        header = table.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        header_height = table.horizontalHeader().height()
        table.setMinimumHeight(header_height * (n_row + 1.5))
        table.setMaximumHeight(header_height * (n_row + 2.0))

    def _setGeometryFile(self):
        filepath = QFileDialog.getOpenFileName()[0]
        if filepath:
            self._geom_file_le.setText(filepath)

    def _onGeometryFileChange(self, filepath):
        self._mediator.onGeomFileChange(filepath)
        self._geom.setFilepath(filepath)

    def _onStackOnlyChange(self, state):
        self._mediator.onGeomStackOnlyChange(state)
        self._geom.setStackOnly(state)

    def _onQuadPositionsChange(self):
        v = _parse_table_widget(self._quad_positions_tb)
        self._mediator.onGeomQuadPositionsChange(v)
        self._geom.setQuadPositions(v)

    def _onAssemblerChange(self, assembler):
        if assembler == GeomAssembler.EXTRA_GEOM:
            self._stack_only_cb.setChecked(False)
            self._stack_only_cb.setEnabled(False)
        else:
            self._stack_only_cb.setEnabled(True)

        self._geom.setAssembler(assembler)

    def updateMetaData(self):
        """Override"""
        if not self._require_geometry:
            return True

        self._stack_only_cb.toggled.emit(self._stack_only_cb.isChecked())

        self._assembler_cb.currentTextChanged.emit(
            self._assembler_cb.currentText())

        self._geom_file_le.returnPressed.emit()

        self._onQuadPositionsChange()

        return True

    def loadMetaData(self):
        """Override."""
        if not self._require_geometry:
            return True

        cfg = self._meta.hget_all(mt.GEOMETRY_PROC)

        self._assembler_cb.setCurrentText(
            self._assemblers_inv[int(cfg["assembler"])])
        self._stack_only_cb.setChecked(cfg["stack_only"] == 'True')
        self._geom_file_le.setText(cfg["geometry_file"])

        quad_positions = json.loads(cfg["quad_positions"], encoding='utf8')
        table = self._quad_positions_tb
        n_rows = table.rowCount()
        n_cols = table.columnCount()
        for j in range(n_cols):
            for i in range(n_rows):
                table.cellWidget(i, j).setText(str(quad_positions[j][i]))
