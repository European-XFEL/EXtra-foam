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
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartLineEdit
from ..gui_helpers import invert_dict
from ..items import GeometryItem
from ...config import config, GeomAssembler
from ...database import Metadata as mt
from ...geometries import module_indices
from ...logger import logger


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

        self._detector = config["DETECTOR"]

        self._geom = GeometryItem()

        self._assembler_cb = QComboBox()
        for item in self._assemblers:
            self._assembler_cb.addItem(item)

        self._coordinates_tb = QTableWidget()

        if self._detector in ("JungFrau", "ePix100"):
            self._coordinates_tb.setEnabled(False)
            self._assembler_cb.removeItem(1)  # no EXtra-geom
        elif self._detector == "AGIPD":
            self._coordinates_tb.setEnabled(False)

        self._geom_file_le = SmartLineEdit(config["GEOMETRY_FILE"])
        self._geom_file_open_btn = QPushButton("Load geometry file")

        self._stack_only_cb = QCheckBox("Stack without geometry file")
        if not config["GEOMETRY_FILE"]:
            self._stack_only_cb.setChecked(True)

        self._non_reconfigurable_widgets = [
            self
        ]

        self.initUI()
        self.initConnections()
        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Override."""
        AR = Qt.AlignRight

        layout = QGridLayout()

        i_row = 0
        layout.addWidget(QLabel("Assembler: "), i_row, 0, AR)
        layout.addWidget(self._assembler_cb, i_row, 1)
        layout.addWidget(self._geom_file_open_btn, i_row, 2)
        layout.addWidget(self._geom_file_le, i_row, 3, 1, 5)
        layout.addWidget(self._stack_only_cb, i_row, 8, AR)

        i_row += 1
        self.initCoordinatesTable()
        if self._detector in ("JungFrau", "ePix100"):
            label = QLabel("Module positions:")
        else:
            label = QLabel("Quadrant positions:")
        layout.addWidget(label, i_row, 0, AR)
        layout.addWidget(self._coordinates_tb, i_row, 1, 1, 8)

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

    def initCoordinatesTable(self):
        n_modules = config["NUMBER_OF_MODULES"]
        if self._detector in ("JungFrau", "ePix100"):
            coordinates = config["MODULE_POSITIONS"][:n_modules]
        else:
            coordinates = config["QUAD_POSITIONS"]

        n_row = len(coordinates[0])
        n_col = len(coordinates)

        table = self._coordinates_tb
        table.setRowCount(n_row)
        table.setColumnCount(n_col)

        for i in range(n_row):
            for j in range(n_col):
                if self._detector in ("LPD", "DSSC"):
                    widget = SmartLineEdit(str(coordinates[j][i]))
                else:
                    widget = SmartLineEdit('0')
                widget.setValidator(QDoubleValidator(-999, 999, 6))
                widget.value_changed_sgn.connect(self._onCoordinatesChange)
                table.setCellWidget(i, j, widget)

        table.move(0, 0)
        table.setVerticalHeaderLabels(['x', 'y'])
        if self._detector == "JungFrau":
            table.setHorizontalHeaderLabels(
                [str(i) for i in module_indices(n_modules,
                                                detector=self._detector)])

        header = table.horizontalHeader()

        for i in range(n_col):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        header = table.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        header_height = table.horizontalHeader().height()
        table.setMinimumHeight(int(header_height * (n_row + 1.5)))
        table.setMaximumHeight(int(header_height * (n_row + 2.0)))

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

    def _onCoordinatesChange(self):
        v = _parse_table_widget(self._coordinates_tb)
        self._mediator.onGeomCoordinatesChange(v)
        self._geom.setCoordinates(v)

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

        self._onCoordinatesChange()

        return True

    def loadMetaData(self):
        """Override."""
        if not self._require_geometry:
            return True

        cfg = self._meta.hget_all(mt.GEOMETRY_PROC)

        assembler = self._getMetaData(cfg, "assembler")
        if assembler is not None:
            self._assembler_cb.setCurrentText(
                self._assemblers_inv[int(cfg["assembler"])])

        self._updateWidgetValue(self._stack_only_cb, cfg, "stack_only")
        self._updateWidgetValue(self._geom_file_le, cfg, "geometry_file")

        # TODO: check number of modules for JungFrau
        coordinates = self._getMetaData(cfg, "coordinates")
        if coordinates is not None:
            coordinates = json.loads(coordinates)
            table = self._coordinates_tb
            n_rows = table.rowCount()
            n_cols = table.columnCount()
            for j in range(n_cols):
                for i in range(n_rows):
                    table.cellWidget(i, j).setText(str(coordinates[j][i]))
