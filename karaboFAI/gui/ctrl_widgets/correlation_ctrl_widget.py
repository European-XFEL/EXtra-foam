"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

CorrelationCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
import functools

from ..pyqtgraph import Qt, QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ...config import FomName


class CorrelationCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the correlation analysis parameters."""

    class CorrelationParam:
        def __init__(self, device_ids=None, properties=None):
            if device_ids is None:
                self.device_ids = []
            else:
                self.device_ids = device_ids

            if properties is None:
                self.properties = []
            else:
                self.properties = properties

    _n_params = 4  # maximum number of correlated parameters

    _available_foms = OrderedDict({
        "A.I. mean": FomName.AI_MEAN,
        "A.I. on - A.I. off": FomName.AI_ON_OFF,
        "ROI1": FomName.ROI1,
        "ROI2": FomName.ROI2,
        "ROI1 + ROI2": FomName.ROI_SUM,
        "ROI1 - ROI2": FomName.ROI_SUB
    })

    # Leave the default device ID empty since the available devices
    # in different instruments are different.
    #
    # TODO: move this to a separate config file
    _available_categories = OrderedDict({
        "": CorrelationParam(),
        "XGM": CorrelationParam(
            device_ids=[
                "",
                "SA1_XTD2_XGM/DOOCS/MAIN",
                "SPB_XTD9_XGM/DOOCS/MAIN",
                "SA3_XTD10_XGM/XGM/DOOCS",
                "SCS_BLU_XGM/XGM/DOOCS"
            ],
            properties=["data.intensityTD"],
        ),
        "MonoChromator": CorrelationParam(
            device_ids=[
                "",
                "SA3_XTD10_MONO/MDL/PHOTON_ENERGY"
            ],
            properties=["actualEnergy"],
        ),
        "Digitizer": CorrelationParam(
            device_ids=[
                "",
                "SCS_UTC1_ADQ/ADC/1"
            ],
            properties=["MCP1", "MCP2", "MCP3", "MCP4"],
        ),
        "Motor": CorrelationParam(
            device_ids=[
                ""
            ],
            properties=["actualPosition"],
        ),
        "Train ID": CorrelationParam(
            device_ids=["", "Any"],
            properties=["timestamp.tid"]
        ),
        "User defined": CorrelationParam()
    })

    # index, device ID, property name
    correlation_param_sgn = QtCore.pyqtSignal(int, str, str)

    correlation_fom_sgn = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__("Correlation analysis setup", *args, **kwargs)

        self._figure_of_merit_cb = QtGui.QComboBox()
        for v in self._available_foms:
            self._figure_of_merit_cb.addItem(v)
        self._figure_of_merit_cb.currentTextChanged.connect(
            lambda x: self.correlation_fom_sgn.emit(self._available_foms[x]))

        self._table = QtGui.QTableWidget()

        self._disabled_widgets_during_daq = [
            self._figure_of_merit_cb,
        ]

        self.initUI()

    def initUI(self):
        """Overload."""
        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        # correlation
        layout.addRow("Figure of merit (FOM): ", self._figure_of_merit_cb)
        layout.addRow(self._table)

        self.setLayout(layout)

        self.initParamTable()

    def initParamTable(self):
        """Initialize the correlation parameter table widget."""
        table = self._table

        n_row = self._n_params
        n_col = 3

        table.setColumnCount(n_col)
        table.setRowCount(n_row)
        table.setHorizontalHeaderLabels(['Category', 'Device ID', 'Property'])
        table.setVerticalHeaderLabels(['1', '2', '3', '4'])
        for i_row in range(self._n_params):
            combo = QtGui.QComboBox()
            for t in self._available_categories.keys():
                combo.addItem(t)
            table.setCellWidget(i_row, 0, combo)
            combo.currentTextChanged.connect(
                functools.partial(self.onCategoryChange, i_row))

            table.setCellWidget(i_row, 1, QtGui.QLineEdit())
            table.cellWidget(i_row, 1).setReadOnly(True)
            table.setCellWidget(i_row, 2, QtGui.QLineEdit())
            table.cellWidget(i_row, 2).setReadOnly(True)

        header = table.horizontalHeader()
        for i in range(n_col):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

        header = table.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

    def updateSharedParameters(self):
        """Override"""
        self._figure_of_merit_cb.currentTextChanged.emit(
            self._figure_of_merit_cb.currentText())
        return ""

    @QtCore.pyqtSlot(str)
    def onCategoryChange(self, i_row, text):
        # i_row is the row number in the QTableWidget
        if not text or text == "User defined":
            # '' or 'User defined'
            device_id_le = QtGui.QLineEdit()
            property_le = QtGui.QLineEdit()

            if not text:
                device_id_le.setReadOnly(True)
                property_le.setReadOnly(True)
            else:
                device_id_le.editingFinished.connect(functools.partial(
                    self.onCorrelationParamChangeLe, i_row, 1))
                property_le.editingFinished.connect(functools.partial(
                    self.onCorrelationParamChangeLe, i_row, 2))

            self._table.setCellWidget(i_row, 1, device_id_le)
            self._table.setCellWidget(i_row, 2, property_le)
        else:
            combo_device_ids = QtGui.QComboBox()
            for device_id in self._available_categories[text].device_ids:
                combo_device_ids.addItem(device_id)
            combo_device_ids.currentTextChanged.connect(functools.partial(
                self.onCorrelationParamChangeCb, i_row, 1))
            self._table.setCellWidget(i_row, 1, combo_device_ids)

            combo_properties = QtGui.QComboBox()
            for ppt in self._available_categories[text].properties:
                combo_properties.addItem(ppt)
            combo_properties.currentTextChanged.connect(functools.partial(
                self.onCorrelationParamChangeCb, i_row, 2))
            self._table.setCellWidget(i_row, 2, combo_properties)

        # we always have invalid (empty) input when the category changes
        self.correlation_param_sgn.emit(i_row, '', '')

    @QtCore.pyqtSlot()
    def onCorrelationParamChangeLe(self, i_row, i_col):
        if i_col == 1:
            # device ID changed
            device_id = self.sender().text()
            ppt = self._table.cellWidget(i_row, 2).text()
        elif i_col == 2:
            # property changed
            device_id = self._table.cellWidget(i_row, 1).text()
            ppt = self.sender().text()

        self.correlation_param_sgn.emit(i_row, device_id, ppt)

    @QtCore.pyqtSlot(str)
    def onCorrelationParamChangeCb(self, i_row, i_col, text):
        if i_col == 1:
            # device ID changed
            device_id = text
            ppt = self._table.cellWidget(i_row, 2).currentText()
        elif i_col == 2:
            # property changed
            device_id = self._table.cellWidget(i_row, 1).currentText()
            ppt = text

        self.correlation_param_sgn.emit(i_row, device_id, ppt)
