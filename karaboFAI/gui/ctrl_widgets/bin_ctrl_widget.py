"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

BinCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict

from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from ..misc_widgets import SmartBoundaryLineEdit, SmartLineEdit
from ...config import AnalysisType, BinMode


class BinCtrlWidget(AbstractCtrlWidget):
    """Analysis parameters setup for ROI analysis."""

    _analysis_types = OrderedDict({
        "": AnalysisType.UNDEFINED,
        "azimuthal integ": AnalysisType.AZIMUTHAL_INTEG,
    })

    _bin_modes = OrderedDict({
        "accumulcate": BinMode.ACCUMULATE,
        # "average": BinMode. AVERAGE,
    })

    def __init__(self, *args, **kwargs):
        super().__init__("Binning analysis setup", *args, **kwargs)

        self._reset_btn = QtGui.QPushButton("Reset")

        self._analysis_type_cb = QtGui.QComboBox()
        self._analysis_type_cb.addItems(list(self._analysis_types.keys()))

        self._mode_cb = QtGui.QComboBox()
        self._mode_cb.addItems(list(self._bin_modes.keys()))

        self._n_bins_le = SmartLineEdit("10")
        self._n_bins_le.setValidator(QtGui.QIntValidator(1, 20))
        self._bin_range_le = SmartBoundaryLineEdit("-1.0, 1.0")

        self.initUI()

        self.setFixedHeight(self.minimumSizeHint().height())

        self.initConnections()

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        layout.addWidget(self._reset_btn, 0, 3, AR)
        layout.addWidget(QtGui.QLabel("Analysis type: "), 1, 0, AR)
        layout.addWidget(self._analysis_type_cb, 1, 1)
        layout.addWidget(QtGui.QLabel("Mode: "), 1, 2, AR)
        layout.addWidget(self._mode_cb, 1, 3)
        layout.addWidget(QtGui.QLabel("Bin range (arb): "), 2, 0, AR)
        layout.addWidget(self._bin_range_le, 2, 1)
        layout.addWidget(QtGui.QLabel("# of bins: "), 2, 2, AR)
        layout.addWidget(self._n_bins_le, 2, 3)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._reset_btn.clicked.connect(mediator.onBinReset)

        self._n_bins_le.returnPressed.connect(
            lambda: mediator.onBinBinsChange(int(self._n_bins_le.text())))

        self._bin_range_le.value_changed_sgn.connect(
            mediator.onBinBinRangeChange)

        self._analysis_type_cb.currentTextChanged.connect(
            lambda x: mediator.onBinAnalysisTypeChange(
                self._analysis_types[x]))

        self._mode_cb.currentTextChanged.connect(
            lambda x: mediator.onBinModeChange(self._bin_modes[x]))

    def updateSharedParameters(self):
        self._n_bins_le.returnPressed.emit()

        self._bin_range_le.returnPressed.emit()

        self._analysis_type_cb.currentTextChanged.emit(
            self._analysis_type_cb.currentText())
        self._mode_cb.currentTextChanged.emit(self._mode_cb.currentText())

        return True

