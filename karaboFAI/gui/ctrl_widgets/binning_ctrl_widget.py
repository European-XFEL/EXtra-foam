"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

BinningCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ..misc_widgets import SmartBoundaryLineEdit, SmartLineEdit
from ...config import BinningType


class BinningCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for ROI analysis."""

    _analysis_types = OrderedDict({
        "": BinningType.UNDEFINED,
        "azimuthal integ": BinningType.AZIMUTHAL_INTEG,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Binning analysis setup", *args, **kwargs)

        self._hist_reset_btn = QtGui.QPushButton("Reset")

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(list(self._analysis_types.keys()))

        self._n_bins_le = SmartLineEdit("10")
        self._n_bins_le.setValidator(QtGui.QIntValidator(1, 100))
        self._bin_range_le = SmartBoundaryLineEdit("-1.0, 1.0")

        self._non_reconfigurable_widgets = [
            self._n_bins_le,
            self._bin_range_le,
        ]

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(QtGui.QLabel("Analysis type: "), 0, 0, AR)
        layout.addWidget(self._analysis_type_cb, 0, 1)
        layout.addWidget(self._hist_reset_btn, 0, 3, AR)
        layout.addWidget(QtGui.QLabel("Bin range: "), 1, 0, AR)
        layout.addWidget(self._bin_range_le, 1, 1)
        layout.addWidget(QtGui.QLabel("# of bins: "), 1, 2, AR)
        layout.addWidget(self._n_bins_le, 1, 3)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._hist_reset_btn.clicked.connect(mediator.onBinningReset)

        self._n_bins_le.returnPressed.connect(
            mediator.onBinningBinsChange)
        self._n_bins_le.returnPressed.emit()

        self._bin_range_le.value_changed_sgn.connect(
            mediator.onBinningRangeChange)
        self._bin_range_le.returnPressed.emit()

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onBinningAnalysisTypeChange(
                self._analysis_types[x]))
        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
