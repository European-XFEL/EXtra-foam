"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
import functools

from PyQt5 import QtCore, QtGui, QtWidgets

from .base_ctrl_widgets import AbstractCtrlWidget
from .smart_widgets import SmartBoundaryLineEdit, SmartLineEdit
from ...config import AnalysisType, BinMode, config


_N_PARAMS = 2
_DEFAULT_BIN_RANGE = "-1, 1"
_DEFAULT_N_BINS = "10"
_MAX_N_BINS = 1e5


class BinCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for ROI analysis."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "pump-probe": AnalysisType.PUMP_PROBE,
        "ROI1 (sum)": AnalysisType.ROI1,
        "ROI2 (sum)": AnalysisType.ROI2,
        "ROI1 - ROI2 (sum)": AnalysisType.ROI1_SUB_ROI2,
        "ROI1 + ROI2 (sum)": AnalysisType.ROI1_ADD_ROI2,
        "ROI1 (proj)": AnalysisType.PROJ_ROI1,
        "ROI2 (proj)": AnalysisType.PROJ_ROI2,
        "ROI1 - ROI2 (proj)": AnalysisType.PROJ_ROI1_SUB_ROI2,
        "ROI1 + ROI2 (proj)": AnalysisType.PROJ_ROI1_ADD_ROI2,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG,
    })

    _bin_modes = OrderedDict({
        "average": BinMode. AVERAGE,
        "accumulcate": BinMode.ACCUMULATE,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Binning setup", *args, **kwargs)

        self._reset_btn = QtGui.QPushButton("Reset")

        self._table = QtGui.QTableWidget()

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(list(self._analysis_types.keys()))

        self._mode_cb = QtGui.QComboBox()
        self._mode_cb.addItems(list(self._bin_modes.keys()))

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Analysis type: "), 0, 0, AR)
        layout.addWidget(self._analysis_type_cb, 0, 1)
        layout.addWidget(QtGui.QLabel("Mode: "), 0, 2, AR)
        layout.addWidget(self._mode_cb, 0, 3)
        layout.addWidget(self._reset_btn, 0, 4, 1, 2, AR)
        layout.addWidget(self._table, 2, 0, 1, 6)

        self.setLayout(layout)

        self.initParamTable()

    def initConnections(self):
        mediator = self._mediator

        self._reset_btn.clicked.connect(mediator.onBinReset)

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onBinAnalysisTypeChange(
                self._analysis_types[x]))

        self._mode_cb.currentTextChanged.connect(
            lambda x: mediator.onBinModeChange(self._bin_modes[x]))

    def initParamTable(self):
        """Initialize the correlation parameter table widget."""
        table = self._table

        n_row = _N_PARAMS
        n_col = 5

        table.setColumnCount(n_col)
        table.setRowCount(n_row)
        table.setHorizontalHeaderLabels([
            'Category',
            'Karabo Device ID',
            'Property Name',
            'Value range',
            '# of bins'
        ])
        table.setVerticalHeaderLabels(['1', '2'])
        for i_row in range(n_row):
            combo = QtGui.QComboBox()
            for t in self._TOPIC_DATA_CATEGORIES[config["TOPIC"]].keys():
                combo.addItem(t)
            table.setCellWidget(i_row, 0, combo)
            combo.currentTextChanged.connect(
                functools.partial(self.onCategoryChange, i_row))

            for i_col in range(1, 3):
                widget = SmartLineEdit()
                table.setCellWidget(i_row, i_col, widget)
                widget.setReadOnly(True)

            widget = self._get_default_bin_range_widget()
            table.setCellWidget(i_row, 3, widget)

            widget = self._get_default_n_bins_widget()
            table.setCellWidget(i_row, 4, widget)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

        header = table.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)

        header_height = self._table.horizontalHeader().height()
        self._table.setMinimumHeight(header_height * 3.5)
        self._table.setMaximumHeight(header_height * 4.5)

    @QtCore.pyqtSlot(str)
    def onCategoryChange(self, i_row, text):
        range_le = self._get_default_bin_range_widget()
        n_bins_le = self._get_default_n_bins_widget()

        # i_row is the row number in the QTableWidget
        if not text or text == "User defined":
            # '' or 'User defined'
            device_id_le = SmartLineEdit()
            property_le = SmartLineEdit()

            if not text:
                device_id_le.setReadOnly(True)
                property_le.setReadOnly(True)
            else:
                device_id_le.returnPressed.connect(functools.partial(
                    self.onBinGroupChangeLe, i_row))
                property_le.returnPressed.connect(functools.partial(
                    self.onBinGroupChangeLe, i_row))
                range_le.returnPressed.connect(functools.partial(
                    self.onBinGroupChangeLe, i_row))
                n_bins_le.returnPressed.connect(functools.partial(
                    self.onBinGroupChangeLe, i_row))

            self._table.setCellWidget(i_row, 1, device_id_le)
            self._table.setCellWidget(i_row, 2, property_le)
            self._table.setCellWidget(i_row, 3, range_le)
            self._table.setCellWidget(i_row, 4, n_bins_le)

            self.onBinGroupChangeLe(i_row)
        else:
            combo_device_ids = QtGui.QComboBox()
            for device_id in self._TOPIC_DATA_CATEGORIES[config["TOPIC"]][text].device_ids:
                combo_device_ids.addItem(device_id)
            combo_device_ids.currentTextChanged.connect(functools.partial(
                self.onBinGroupChangeCb, i_row))
            self._table.setCellWidget(i_row, 1, combo_device_ids)

            combo_properties = QtGui.QComboBox()
            for ppt in self._TOPIC_DATA_CATEGORIES[config["TOPIC"]][text].properties:
                combo_properties.addItem(ppt)
            combo_properties.currentTextChanged.connect(functools.partial(
                self.onBinGroupChangeCb, i_row))
            self._table.setCellWidget(i_row, 2, combo_properties)

            range_le.returnPressed.connect(functools.partial(
                self.onBinGroupChangeCb, i_row))
            self._table.setCellWidget(i_row, 3, range_le)

            n_bins_le.returnPressed.connect(functools.partial(
                self.onBinGroupChangeCb, i_row))
            self._table.setCellWidget(i_row, 4, n_bins_le)

            self.onBinGroupChangeCb(i_row)

    @QtCore.pyqtSlot()
    def onBinGroupChangeLe(self, i_row):
        device_id = self._table.cellWidget(i_row, 1).text()
        ppt = self._table.cellWidget(i_row, 2).text()
        bin_range = self._table.cellWidget(i_row, 3).value()
        n_bins = int(self._table.cellWidget(i_row, 4).text())

        self._mediator.onBinGroupChange(
            (i_row+1, device_id, ppt, bin_range, n_bins))

    @QtCore.pyqtSlot(str)
    def onBinGroupChangeCb(self, i_row):
        device_id = self._table.cellWidget(i_row, 1).currentText()
        ppt = self._table.cellWidget(i_row, 2).currentText()
        bin_range = self._table.cellWidget(i_row, 3).value()
        n_bins = int(self._table.cellWidget(i_row, 4).text())

        self._mediator.onBinGroupChange(
            (i_row+1, device_id, ppt, bin_range, n_bins))

    def updateMetaData(self):
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._mode_cb.currentTextChanged.emit(self._mode_cb.currentText())

        for i_row in range(_N_PARAMS):
            category = self._table.cellWidget(i_row, 0).currentText()
            if not category or category == "User defined":
                self.onBinGroupChangeLe(i_row)
            else:
                self.onBinGroupChangeCb(i_row)

        return True

    def _get_default_bin_range_widget(self):
        return SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)

    def _get_default_n_bins_widget(self):
        widget = SmartLineEdit(_DEFAULT_N_BINS)
        widget.setValidator(QtGui.QIntValidator(1, _MAX_N_BINS))
        return widget
