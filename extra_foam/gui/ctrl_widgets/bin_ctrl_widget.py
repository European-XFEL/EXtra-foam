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
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTableWidget
)

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit, SmartLineEdit
from ..gui_helpers import invert_dict
from ...config import AnalysisType, BinMode, config
from ...database import Metadata as mt
from ...database import SourceCatalog


_N_PARAMS = 2
_DEFAULT_N_BINS = 20
_DEFAULT_BIN_RANGE = "-inf, inf"
_MAX_N_BINS = 999


class BinCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up binning analysis parameters."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "pump-probe": AnalysisType.PUMP_PROBE,
        "ROI FOM": AnalysisType.ROI_FOM,
        "ROI proj": AnalysisType.ROI_PROJ,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG,
    })
    _analysis_types_inv = invert_dict(_analysis_types)

    _bin_modes = OrderedDict({
        "average": BinMode. AVERAGE,
        "accumulate": BinMode.ACCUMULATE,
    })
    _bin_modes_inv = invert_dict(_bin_modes)

    _user_defined_key = config["SOURCE_USER_DEFINED_CATEGORY"]

    _UNDEFINED_CATEGORY = ''
    _META_CATEGORY = 'Metadata'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._reset_btn = QPushButton("Reset")

        self._table = QTableWidget()

        self._analysis_type_cb = QComboBox()
        self._analysis_type_cb.addItems(list(self._analysis_types.keys()))

        self._mode_cb = QComboBox()
        self._mode_cb.addItems(list(self._bin_modes.keys()))

        self._auto_level_btn = QPushButton("Auto level")

        self._src_instrument = config.control_sources
        tid_key_split = SourceCatalog.TRAIN_ID.split(" ")
        self._src_metadata = {
            self._META_CATEGORY: {
                tid_key_split[0]: [tid_key_split[1]],
            }
        }

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        AR = Qt.AlignRight
        AT = Qt.AlignTop
        layout = QHBoxLayout()

        ctrl_widget = QFrame()
        llayout = QGridLayout()
        llayout.addWidget(QLabel("Analysis type: "), 0, 0, AR)
        llayout.addWidget(self._analysis_type_cb, 0, 1)
        llayout.addWidget(QLabel("Mode: "), 1, 0, AR)
        llayout.addWidget(self._mode_cb, 1, 1)
        llayout.addWidget(self._auto_level_btn, 2, 0, 1, 2)
        llayout.addWidget(self._reset_btn, 3, 0, 1, 2)
        ctrl_widget.setLayout(llayout)

        layout.addWidget(ctrl_widget, alignment=AT)
        layout.addWidget(self._table, alignment=AT)

        self.setLayout(layout)

        self.initParamTable()

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._reset_btn.clicked.connect(mediator.onBinReset)

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onBinAnalysisTypeChange(
                self._analysis_types[x]))

        self._mode_cb.currentTextChanged.connect(
            lambda x: mediator.onBinModeChange(self._bin_modes[x]))

        self._auto_level_btn.clicked.connect(
            mediator.bin_heatmap_autolevel_sgn)

    def initParamTable(self):
        """Initialize the correlation parameter table widget."""
        table = self._table

        h_labels = [
            'Category', 'Karabo Device ID', 'Property Name',
            'Bin range', '# of bins'
        ]

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

            # Set up "value range" and "# of bins" cell for category ''
            widget = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
            widget.setReadOnly(True)
            table.setCellWidget(3, i_col, widget)
            widget = SmartLineEdit(str(_DEFAULT_N_BINS))
            widget.setReadOnly(True)
            table.setCellWidget(4, i_col, widget)

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
        bin_range_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)

        n_bins_le = SmartLineEdit(str(_DEFAULT_N_BINS))
        n_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        # i_row is the row number in the QTableWidget
        if not category or category == self._user_defined_key:
            device_id_le = SmartLineEdit()
            property_le = SmartLineEdit()

            if not category:
                device_id_le.setReadOnly(True)
                property_le.setReadOnly(True)
                bin_range_le.setReadOnly(True)
                n_bins_le.setReadOnly(True)
            else:
                device_id_le.returnPressed.connect(functools.partial(
                    self.onBinParamChangeLe, i_col))
                property_le.returnPressed.connect(functools.partial(
                    self.onBinParamChangeLe, i_col))
                bin_range_le.returnPressed.connect(functools.partial(
                    self.onBinParamChangeLe, i_col))
                n_bins_le.returnPressed.connect(functools.partial(
                    self.onBinParamChangeLe, i_col))

            self._table.setCellWidget(1, i_col, device_id_le)
            self._table.setCellWidget(2, i_col, property_le)
            self._table.setCellWidget(3, i_col, bin_range_le)
            self._table.setCellWidget(4, i_col, n_bins_le)

            self.onBinParamChangeLe(i_col)
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
                self.onBinParamChangeCb, i_col))
            property_cb.currentTextChanged.connect(functools.partial(
                self.onBinParamChangeCb, i_col))
            bin_range_le.returnPressed.connect(functools.partial(
                self.onBinParamChangeCb, i_col))
            n_bins_le.returnPressed.connect(functools.partial(
                self.onBinParamChangeCb, i_col))

            self._table.setCellWidget(1, i_col, device_id_cb)
            self._table.setCellWidget(2, i_col, property_cb)
            self._table.setCellWidget(3, i_col, bin_range_le)
            self._table.setCellWidget(4, i_col, n_bins_le)

            self.onBinParamChangeCb(i_col)

    @pyqtSlot()
    def onBinParamChangeLe(self, i_col):
        device_id = self._table.cellWidget(1, i_col).text()
        ppt = self._table.cellWidget(2, i_col).text()
        bin_range = self._table.cellWidget(3, i_col).value()
        n_bins = int(self._table.cellWidget(4, i_col).text())

        src = f"{device_id} {ppt}" if device_id and ppt else ""
        self._mediator.onBinParamChange((i_col + 1, src, bin_range, n_bins))

    @pyqtSlot(str)
    def onBinParamChangeCb(self, i_col):
        device_id = self._table.cellWidget(1, i_col).currentText()
        ppt = self._table.cellWidget(2, i_col).currentText()
        bin_range = self._table.cellWidget(3, i_col).value()
        n_bins = int(self._table.cellWidget(4, i_col).text())

        src = f"{device_id} {ppt}" if device_id and ppt else ""
        self._mediator.onBinParamChange((i_col + 1, src, bin_range, n_bins))

    def updateMetaData(self):
        """Override."""
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._mode_cb.currentTextChanged.emit(self._mode_cb.currentText())

        for i_col in range(_N_PARAMS):
            category = self._table.cellWidget(0, i_col).currentText()
            if not category or category == self._user_defined_key:
                self.onBinParamChangeLe(i_col)
            else:
                self.onBinParamChangeCb(i_col)

        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.BINNING_PROC)
        if "analysis_type" not in cfg:
            # not initialized
            return

        self._analysis_type_cb.setCurrentText(
            self._analysis_types_inv[int(cfg["analysis_type"])])

        self._mode_cb.setCurrentText(self._bin_modes_inv[int(cfg["mode"])])

        for i in range(_N_PARAMS):
            src = cfg[f'source{i+1}']
            if not src:
                self._table.cellWidget(0, i).setCurrentText(
                    self._UNDEFINED_CATEGORY)
                self.onCategoryChange(i, self._UNDEFINED_CATEGORY)
            else:
                device_id, ppt = src.split(' ')
                bin_range = cfg[f'bin_range{i+1}'][1:-1]
                n_bins = cfg[f'n_bins{i+1}']
                ctg = self._find_category(device_id, ppt)

                self._table.cellWidget(0, i).setCurrentText(ctg)
                self.onCategoryChange(i, ctg)
                if ctg == self._user_defined_key:
                    self._table.cellWidget(1, i).setText(device_id)
                    self._table.cellWidget(2, i).setText(ppt)
                else:
                    self._table.cellWidget(1, i).setCurrentText(device_id)
                    self._table.cellWidget(2, i).setCurrentText(ppt)
                self._table.cellWidget(3, i).setText(bin_range)
                self._table.cellWidget(4, i).setText(n_bins)

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
