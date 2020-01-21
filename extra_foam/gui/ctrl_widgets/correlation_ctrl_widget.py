"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
import functools

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QComboBox, QGridLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
)

from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget
from .smart_widgets import SmartLineEdit
from ...config import AnalysisType, config

_N_PARAMS = 2  # maximum number of correlated parameters
_DEFAULT_RESOLUTION = 0.0


class CorrelationCtrlWidget(_AbstractGroupBoxCtrlWidget):
    """Widget for setting up correlation analysis parameters."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "pump-probe": AnalysisType.PUMP_PROBE,
        "ROI FOM": AnalysisType.ROI_FOM,
        "ROI proj": AnalysisType.ROI_PROJ,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG,
    })

    _user_defined_key = config["SOURCE_USER_DEFINED_CATEGORY"]

    def __init__(self, *args, **kwargs):
        super().__init__("Correlation setup", *args, **kwargs)

        self._analysis_type_cb = QComboBox()
        for v in self._analysis_types:
            self._analysis_type_cb.addItem(v)

        self._reset_btn = QPushButton("Reset")

        self._table = QTableWidget()

        self._src_instrument = config.control_sources
        self._src_metadata = config.meta_sources

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        layout.addWidget(QLabel("Analysis type: "), 0, 0, AR)
        layout.addWidget(self._analysis_type_cb, 0, 1)
        layout.addWidget(self._reset_btn, 0, 5, AR)
        layout.addWidget(self._table, 1, 0, 1, 6)

        self.setLayout(layout)

        self.initParamTable()

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onCorrelationAnalysisTypeChange(
                self._analysis_types[x]))
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())

        self._reset_btn.clicked.connect(mediator.onCorrelationReset)

    def initParamTable(self):
        """Initialize the correlation parameter table widget."""
        table = self._table

        n_row = _N_PARAMS
        n_col = 4

        table.setColumnCount(n_col)
        table.setRowCount(n_row)
        table.setHorizontalHeaderLabels([
            'Category', 'Karabo Device ID', 'Property Name', 'Resolution'])
        table.setVerticalHeaderLabels([str(i+1) for i in range(n_row)])

        for i_row in range(_N_PARAMS):
            category_cb = QComboBox()
            category_cb.addItem('')  # default is empty
            for k, v in self._src_metadata.items():
                if v:
                    category_cb.addItem(k)
            for k, v in self._src_instrument.items():
                if v:
                    category_cb.addItem(k)
            category_cb.addItem(self._user_defined_key)
            table.setCellWidget(i_row, 0, category_cb)
            category_cb.currentTextChanged.connect(
                functools.partial(self.onCategoryChange, i_row))

            # Set up "device id" and "property" cells for category ''
            for i_col in [1, 2]:
                widget = SmartLineEdit()
                widget.setReadOnly(True)
                table.setCellWidget(i_row, i_col, widget)

            # Set up "resolution" cell for category ''
            widget = SmartLineEdit(str(_DEFAULT_RESOLUTION))
            widget.setReadOnly(True)
            table.setCellWidget(i_row, 3, widget)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        header = table.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        header_height = self._table.horizontalHeader().height()
        self._table.setMinimumHeight(header_height * (_N_PARAMS + 1.5))
        self._table.setMaximumHeight(header_height * (_N_PARAMS + 2.5))

    @pyqtSlot(str)
    def onCategoryChange(self, i_row, category):
        resolution_le = SmartLineEdit(str(_DEFAULT_RESOLUTION))
        validator = QDoubleValidator()
        validator.setBottom(0.0)
        resolution_le.setValidator(validator)

        if not category or category == self._user_defined_key:
            device_id_le = SmartLineEdit()
            property_le = SmartLineEdit()
            if not category:
                device_id_le.setReadOnly(True)
                property_le.setReadOnly(True)
                resolution_le.setReadOnly(True)
            else:
                device_id_le.returnPressed.connect(functools.partial(
                    self.onCorrelationParamChangeLe, i_row))
                property_le.returnPressed.connect(functools.partial(
                    self.onCorrelationParamChangeLe, i_row))
                resolution_le.returnPressed.connect(functools.partial(
                    self.onCorrelationParamChangeLe, i_row))

            self._table.setCellWidget(i_row, 1, device_id_le)
            self._table.setCellWidget(i_row, 2, property_le)
            self._table.setCellWidget(i_row, 3, resolution_le)

            self.onCorrelationParamChangeLe(i_row)
        else:
            srcs = self._src_metadata if category in config.meta_sources \
                else self._src_instrument
            category_srcs = srcs.get(category, dict())
            device_id_cb = QComboBox()
            property_cb = QComboBox()
            for device_id in category_srcs:
                device_id_cb.addItem(device_id)
                for ppt in category_srcs[device_id]:
                    property_cb.addItem(ppt)

            device_id_cb.currentTextChanged.connect(functools.partial(
                self.onCorrelationParamChangeCb, i_row))
            property_cb.currentTextChanged.connect(functools.partial(
                self.onCorrelationParamChangeCb, i_row))
            resolution_le.returnPressed.connect(functools.partial(
                self.onCorrelationParamChangeCb, i_row))

            self._table.setCellWidget(i_row, 1, device_id_cb)
            self._table.setCellWidget(i_row, 2, property_cb)
            self._table.setCellWidget(i_row, 3, resolution_le)

            self.onCorrelationParamChangeCb(i_row)

    @pyqtSlot()
    def onCorrelationParamChangeLe(self, i_row):
        device_id = self._table.cellWidget(i_row, 1).text()
        ppt = self._table.cellWidget(i_row, 2).text()
        res = float(self._table.cellWidget(i_row, 3).text())

        src = f"{device_id} {ppt}" if device_id and ppt else ""
        self._mediator.onCorrelationParamChange((i_row+1, src, res))

    @pyqtSlot(str)
    def onCorrelationParamChangeCb(self, i_row):
        device_id = self._table.cellWidget(i_row, 1).currentText()
        ppt = self._table.cellWidget(i_row, 2).currentText()
        res = float(self._table.cellWidget(i_row, 3).text())

        src = f"{device_id} {ppt}" if device_id and ppt else ""
        self._mediator.onCorrelationParamChange((i_row+1, src, res))

    def updateMetaData(self):
        """Overload."""
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())

        for i_row in range(_N_PARAMS):
            category = self._table.cellWidget(i_row, 0).currentText()
            if not category or category == self._user_defined_key:
                self.onCorrelationParamChangeLe(i_row)
            else:
                self.onCorrelationParamChangeCb(i_row)
        return True
