"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

CorrelationCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
from ..widgets.pyqtgraph import Qt, QtCore, QtGui
from .base_ctrl_widgets import AbstractCtrlWidget
from ..config import config


class CorrelationParam(abc.ABC):
    """Base class for correlation parameters."""
    def __init__(self):
        self.device_id = QtGui.QComboBox()
        for device_id in self._available_device_ids:
            self.device_id.addItem(device_id)

        self.property = QtGui.QComboBox()
        for property in self._available_properties:
            self.property.addItem(property)


class CorrelationParamXGM(CorrelationParam):
    _available_device_ids = [
        'SCS_BLU_XGM/XGM/DOOCS',
        'SA3_XTD10_XGM/XGM/DOOCS'
    ]

    _available_properties = [
        'data.intensityTD'
    ]

    def __init__(self):
        super().__init__()


class CorrelationParamMono(CorrelationParam):
    _available_device_ids = [
    ]

    _available_properties = [
        "actualEnergy"
    ]

    def __init__(self):
        super().__init__()


class CorrelationParamMotor(CorrelationParam):
    _available_device_ids = [
    ]

    _available_properties = [
    ]

    def __init__(self):
        super().__init__()



class CorrelationCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the correlation analysis parameters."""

    _n_params = 4  # maximum number of correlated parameters

    normalizers = (
        "integrated curve", "reference signal"
    )

    figure_of_merits = (
        "single image", "on-off"
    )

    param1_sgn = QtCore.pyqtSignal(str, str)
    param2_sgn = QtCore.pyqtSignal(str, str)

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
        """"""
        table = self._table

        n_row = self._n_params
        n_col = 3
        table.setColumnCount(3)
        data1 = ['', '', '', '']
        data2 = ['', '', '', '']
        avaible_param_types = ["", "XGM", "Mono", "Motor", "User defined"]

        table.setRowCount(4)

        for i_row in range(self._n_params):
            combo = QtGui.QComboBox()
            for t in avaible_param_types:
                combo.addItem(t)
            table.setCellWidget(i_row, 0, combo)

            item1 = QtGui.QTableWidgetItem(data1[i_row])
            table.setItem(i_row, 1, item1)

            item2 = QtGui.QTableWidgetItem(data2[i_row])
            table.setItem(i_row, 2, item2)

        table.cellChanged.connect(self.onTableCellChanged)

        table.setHorizontalHeaderLabels(
            ['Category', 'Device ID', 'Property'])
        table.setVerticalHeaderLabels(['1', '2', '3', '4'])

        header = table.horizontalHeader()
        for i in range(n_col):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

        header = table.verticalHeader()
        for i in range(n_row):
            header.setSectionResizeMode(i, Qt.QtWidgets.QHeaderView.Stretch)

    def updateSharedParameters(self):
        """Override"""
        return ""

    @QtCore.pyqtSlot(int, int)
    def onTableCellChanged(self, row, col):
        table = self._table
        if col == 0:
            # category changed
            category = table.itemAt(row, col).currentText()
