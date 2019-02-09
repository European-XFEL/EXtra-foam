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

from ..widgets.pyqtgraph import Qt, QtCore, QtGui
from .base_ctrl_widgets import AbstractCtrlWidget
from ..config import config


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

    normalizers = (
        "integrated curve", "reference signal"
    )

    figure_of_merits = (
        "single image", "on-off"
    )

    _available_categories = OrderedDict({
        "": CorrelationParam(),
        "XGM": CorrelationParam(
            device_ids=["device name", "very long device name"],
            properties=["property1", "property2"],
        ),
        "MonoChromator": CorrelationParam(
            device_ids=["device name", "very long device name"],
            properties=["property1", "property2"],
        ),
        "Motor": CorrelationParam(
            device_ids=["device name", "very long device name"],
            properties=["property1", "property2"],
        ),
        "User defined": CorrelationParam()
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Correlation analysis setup", *args, **kwargs)

        self._figure_of_merit_cb = QtGui.QComboBox()
        for v in self.figure_of_merits:
            self._figure_of_merit_cb.addItem(v)

        self._normalizers_cb = QtGui.QComboBox()
        for v in self.normalizers:
            self._normalizers_cb.addItem(v)

        self._integration_range_le = QtGui.QLineEdit(
            ', '.join([str(v) for v in config["INTEGRATION_RANGE"]]))

        self._table = QtGui.QTableWidget()

        self._disabled_widgets_during_daq = [
            self._figure_of_merit_cb,
            self._normalizers_cb,
            self._integration_range_le,
            self._table
        ]

        self.initUI()

    def initUI(self):
        """Overload."""
        layout = QtGui.QFormLayout()
        layout.setLabelAlignment(QtCore.Qt.AlignRight)

        # correlation
        layout.addRow("Figure of merit (FOM): ", self._figure_of_merit_cb)
        layout.addRow("Normalized by: ", self._normalizers_cb)
        layout.addRow("Integration range (1/A): ", self._integration_range_le)
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

            # the rest columns will be set automatically

        header = table.horizontalHeader()
        for i in range(n_col):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

        header = table.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

    def updateSharedParameters(self):
        """Override"""
        return ""

    @QtCore.pyqtSlot(str)
    def onCategoryChange(self, i_row, text):
        # i_row is the row number in the QTableWidget
        if not text or text == "User defined":
            # '' or 'User defined'
            le1 = QtGui.QLineEdit()
            le2 = QtGui.QLineEdit()
            if not text:
                le1.setReadOnly(True)
                le2.setReadOnly(True)
            self._table.setCellWidget(i_row, 1, le1)
            self._table.setCellWidget(i_row, 2, le2)
        else:
            combo_device_ids = QtGui.QComboBox()
            for device_id in self._available_categories[text].device_ids:
                combo_device_ids.addItem(device_id)
            self._table.setCellWidget(i_row, 1, combo_device_ids)

            combo_properties = QtGui.QComboBox()
            for property in self._available_categories[text].properties:
                combo_properties.addItem(property)
            self._table.setCellWidget(i_row, 2, combo_properties)
