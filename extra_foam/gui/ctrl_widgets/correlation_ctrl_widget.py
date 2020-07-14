"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
import functools
from enum import IntEnum
import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGridLayout, QHeaderView, QHBoxLayout,
    QLabel, QPlainTextEdit, QPushButton, QTableWidget, QWidget
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartLineEdit
from ..gui_helpers import invert_dict
from ...config import AnalysisType, config
from ...database import Metadata as mt
from ...database import SourceCatalog

_N_PARAMS = 2  # maximum number of correlated parameters
_DEFAULT_RESOLUTION = 0.0


class FittingType(IntEnum):
    UNDEFINED = 0
    LINEAR = 1
    CUBIC = 2


class FittingCtrlWidget(QWidget):

    _available_types = OrderedDict({
        "": FittingType.UNDEFINED,
        "Linear": FittingType.LINEAR,
        "Cubic": FittingType.CUBIC,
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fit_type_cb = QComboBox()
        for name in self._available_types:
            self.fit_type_cb.addItem(name)

        self.fit_btn = QPushButton("Fit")
        self.clear_btn = QPushButton("Clear")
        self.corr1_cb = QCheckBox("Correlation 1")
        self.corr1_cb.setChecked(True)
        self.corr2_cb = QCheckBox("Correlation 2")

        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(100)

        self.initUI()
        self.initConnections()

    def initUI(self):
        llayout = QGridLayout()
        llayout.addWidget(self.corr1_cb, 0, 0)
        llayout.addWidget(self.corr2_cb, 0, 1)
        llayout.addWidget(QLabel("Type of fit: "), 1, 0)
        llayout.addWidget(self.fit_type_cb, 1, 1)
        llayout.addWidget(self.fit_btn, 2, 0)
        llayout.addWidget(self.clear_btn, 2, 1)

        layout = QHBoxLayout()
        layout.addLayout(llayout)
        layout.addWidget(self._output)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._output.setMinimumWidth(1.5 * llayout.minimumSize().width())
        self.setFixedWidth(self.minimumSizeHint().width())

    def initConnections(self):
        self.corr1_cb.toggled.connect(lambda x: self.corr2_cb.setChecked(not x))
        self.corr2_cb.toggled.connect(lambda x: self.corr1_cb.setChecked(not x))

    def fit(self, x, y, is_master=True):
        new_x, new_y = None, None
        if len(x) > 2:
            fit_type_str = self.fit_type_cb.currentText()
            fit_type = self._available_types[fit_type_str]
            info = ""

            # TODO: need to design to accommodate more fitting types
            if fit_type == FittingType.LINEAR:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                info = f"p0 = {z[0]:.4e}, p1 = {z[1]:.4e}"
                new_x, new_y = x, p(x)

            elif fit_type == FittingType.CUBIC:
                z = np.polyfit(x, y, 3)
                p = np.poly1d(z)
                info = f"p0 = {z[0]:.4e}, p1 = {z[1]:.4e}, " \
                       f"p2 = {z[2]:.4e}, p3 = {z[3]:.4e}"
                new_x, new_y = x, p(x)

            if info:
                if is_master:
                    self._output.appendPlainText(
                        f"\n{fit_type_str} fit for correlation "
                        f"{1 if self.corr1_cb.isChecked() else 2}: ")
                    self._output.appendPlainText("Master data - ")
                else:
                    self._output.appendPlainText("Slave data - ")
                self._output.appendPlainText(info)

        return new_x, new_y


class CorrelationCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up correlation analysis parameters."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "pump-probe": AnalysisType.PUMP_PROBE,
        "ROI FOM": AnalysisType.ROI_FOM,
        "ROI proj": AnalysisType.ROI_PROJ,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG,
    })
    _analysis_types_inv = invert_dict(_analysis_types)

    _user_defined_key = config["SOURCE_USER_DEFINED_CATEGORY"]

    _UNDEFINED_CATEGORY = ''
    _META_CATEGORY = 'Metadata'

    fit_curve_sgn = pyqtSignal(bool)
    clear_fitting_sgn = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._analysis_type_cb = QComboBox()
        for v in self._analysis_types:
            self._analysis_type_cb.addItem(v)

        self._reset_btn = QPushButton("Reset")

        self._table = QTableWidget()

        self._src_instrument = config.control_sources
        tid_key_split = SourceCatalog.TRAIN_ID.split(" ")
        self._src_metadata = {
            self._META_CATEGORY: {
                tid_key_split[0]: [tid_key_split[1]],
            }
        }

        self._fitting = FittingCtrlWidget()

        self.initParamTable()
        self.initUI()
        self.initConnections()

    def initUI(self):
        """Overload."""
        AR = Qt.AlignRight
        AT = Qt.AlignTop
        layout = QHBoxLayout()

        ctrl_widget = QFrame()
        llayout = QGridLayout()
        llayout.addWidget(QLabel("Analysis type: "), 0, 0, AR)
        llayout.addWidget(self._analysis_type_cb, 0, 1)
        llayout.addWidget(self._reset_btn, 1, 1)
        ctrl_widget.setLayout(llayout)
        ctrl_widget.setFixedWidth(ctrl_widget.minimumSizeHint().width())

        layout.addWidget(ctrl_widget, 1, alignment=AT)
        layout.addWidget(self._table, 2, alignment=AT)
        layout.addWidget(self._fitting, 1, alignment=AT)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onCorrelationAnalysisTypeChange(
                self._analysis_types[x]))

        self._reset_btn.clicked.connect(mediator.onCorrelationReset)

        self._fitting.fit_btn.clicked.connect(
            lambda: self.fit_curve_sgn.emit(
                self._fitting.corr1_cb.isChecked()))
        self._fitting.clear_btn.clicked.connect(
            lambda: self.clear_fitting_sgn.emit(
                self._fitting.corr1_cb.isChecked()))

    def initParamTable(self):
        """Initialize the correlation parameter table widget."""
        table = self._table

        h_labels = [
            'Category', 'Karabo Device ID', 'Property Name', 'Resolution']
        n_row = len(h_labels)
        n_col = _N_PARAMS

        table.setColumnCount(n_col)
        table.setRowCount(n_row)
        table.setVerticalHeaderLabels(h_labels)
        table.setHorizontalHeaderLabels([str(i+1) for i in range(n_col)])

        for i_col in range(n_col):
            category_cb = QComboBox()
            category_cb.addItem(self._UNDEFINED_CATEGORY)
            for k, v in self._src_metadata.items():
                if v:
                    category_cb.addItem(k)
            for k, v in self._src_instrument.items():
                if v:
                    category_cb.addItem(k)
            category_cb.addItem(self._user_defined_key)
            table.setCellWidget(0, i_col, category_cb)
            category_cb.currentTextChanged.connect(
                functools.partial(self.onCategoryChange, i_col))

            # Set up "device id" and "property" cells for category ''
            for i_row in [1, 2]:
                widget = SmartLineEdit()
                widget.setReadOnly(True)
                table.setCellWidget(i_row, i_col, widget)

            # Set up "resolution" cell for category ''
            widget = SmartLineEdit(str(_DEFAULT_RESOLUTION))
            widget.setReadOnly(True)
            table.setCellWidget(3, i_col, widget)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        header = table.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        header_height = self._table.horizontalHeader().height()
        self._table.setMinimumHeight(header_height * (n_row + 2))
        self._table.setMaximumHeight(header_height * (n_row + 3))

    @pyqtSlot(str)
    def onCategoryChange(self, i_col, category):
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
                    self.onCorrelationParamChangeLe, i_col))
                property_le.returnPressed.connect(functools.partial(
                    self.onCorrelationParamChangeLe, i_col))
                resolution_le.returnPressed.connect(functools.partial(
                    self.onCorrelationParamChangeLe, i_col))

            self._table.setCellWidget(1, i_col, device_id_le)
            self._table.setCellWidget(2, i_col, property_le)
            self._table.setCellWidget(3, i_col, resolution_le)

            self.onCorrelationParamChangeLe(i_col)
        else:
            srcs = self._src_metadata if category in self._src_metadata \
                else self._src_instrument
            category_srcs = srcs.get(category, dict())
            device_id_cb = QComboBox()
            property_cb = QComboBox()
            for device_id in category_srcs:
                device_id_cb.addItem(device_id)
                for ppt in category_srcs[device_id]:
                    property_cb.addItem(ppt)

            device_id_cb.currentTextChanged.connect(functools.partial(
                self.onCorrelationParamChangeCb, i_col))
            property_cb.currentTextChanged.connect(functools.partial(
                self.onCorrelationParamChangeCb, i_col))
            resolution_le.returnPressed.connect(functools.partial(
                self.onCorrelationParamChangeCb, i_col))

            self._table.setCellWidget(1, i_col, device_id_cb)
            self._table.setCellWidget(2, i_col, property_cb)
            self._table.setCellWidget(3, i_col, resolution_le)

            self.onCorrelationParamChangeCb(i_col)

    @pyqtSlot()
    def onCorrelationParamChangeLe(self, i_col):
        device_id = self._table.cellWidget(1, i_col).text()
        ppt = self._table.cellWidget(2, i_col).text()
        res = float(self._table.cellWidget(3, i_col).text())

        src = f"{device_id} {ppt}" if device_id and ppt else ""
        self._mediator.onCorrelationParamChange((i_col + 1, src, res))

    @pyqtSlot(str)
    def onCorrelationParamChangeCb(self, i_col):
        device_id = self._table.cellWidget(1, i_col).currentText()
        ppt = self._table.cellWidget(2, i_col).currentText()
        res = float(self._table.cellWidget(3, i_col).text())

        src = f"{device_id} {ppt}" if device_id and ppt else ""
        self._mediator.onCorrelationParamChange((i_col + 1, src, res))

    def updateMetaData(self):
        """Overload."""
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())

        for i_col in range(_N_PARAMS):
            category = self._table.cellWidget(0, i_col).currentText()
            if not category or category == self._user_defined_key:
                self.onCorrelationParamChangeLe(i_col)
            else:
                self.onCorrelationParamChangeCb(i_col)
        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.CORRELATION_PROC)
        if not cfg:
            # not initialized
            return

        self._analysis_type_cb.setCurrentText(
            self._analysis_types_inv[int(cfg["analysis_type"])])

        for i in range(_N_PARAMS):
            src = cfg[f'source{i+1}']
            if not src:
                self._table.cellWidget(0, i).setCurrentText(
                    self._UNDEFINED_CATEGORY)
                self.onCategoryChange(i, self._UNDEFINED_CATEGORY)
            else:
                device_id, ppt = src.split(' ')
                resolution = cfg[f'resolution{i+1}']
                ctg = self._find_category(device_id, ppt)

                self._table.cellWidget(0, i).setCurrentText(ctg)
                self.onCategoryChange(i, ctg)
                if ctg == self._user_defined_key:
                    self._table.cellWidget(1, i).setText(device_id)
                    self._table.cellWidget(2, i).setText(ppt)
                else:
                    self._table.cellWidget(1, i).setCurrentText(device_id)
                    self._table.cellWidget(2, i).setCurrentText(ppt)
                self._table.cellWidget(3, i).setText(resolution)

    def _find_category(self, device_id, ppt):
        for ctg in self._src_instrument:
            ctg_srcs = self._src_instrument[ctg]
            if device_id in ctg_srcs and ppt in ctg_srcs[device_id]:
                return ctg

        for ctg in self._src_metadata:
            ctg_srcs = self._src_metadata[ctg]
            if device_id in ctg_srcs and ppt in ctg_srcs[device_id]:
                return ctg

        return self._user_defined_key

    def resetAnalysisType(self):
        self._analysis_type_cb.setCurrentText(
            self._analysis_types_inv[AnalysisType.UNDEFINED])

    def fit_curve(self, x, y, is_master):
        return self._fitting.fit(x, y, is_master)
