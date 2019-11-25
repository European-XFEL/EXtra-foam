"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton

from .base_ctrl_widgets import _AbstractCtrlWidget
from .smart_widgets import (
    SmartBoundaryLineEdit, SmartLineEdit, SmartStringLineEdit
)
from .scan_button_set import ScanButtonSet
from ...config import AnalysisType


_DEFAULT_N_BINS = "10"
_DEFAULT_BIN_RANGE = "0, 1e9"
_MAX_N_BINS = 9999


class TrXasCtrlWidget(_AbstractCtrlWidget):
    """Widget for setting up tr-XAS analysis parameters.

    tr-XAS stands for Time-resolved X-ray Absorption Spectroscopy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._delay_device_le = SmartStringLineEdit("Any")
        self._delay_ppt_le = SmartStringLineEdit("timestamp.tid")

        # self._energy_device_le = SmartLineEdit("SA3_XTD10_MONO/MDL/PHOTON_ENERGY")
        # self._energy_ppt_le = SmartLineEdit("actualEnergy")
        self._energy_device_le = SmartStringLineEdit("Any")
        self._energy_ppt_le = SmartStringLineEdit("timestamp.tid")

        self._delay_range_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self._n_delay_bins_le = SmartLineEdit(_DEFAULT_N_BINS)
        self._n_delay_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self._energy_range_le = SmartBoundaryLineEdit(_DEFAULT_BIN_RANGE)
        self._n_energy_bins_le = SmartLineEdit(_DEFAULT_N_BINS)
        self._n_energy_bins_le.setValidator(QIntValidator(1, _MAX_N_BINS))

        self._swap_btn = QPushButton("Swap delay and energy")

        self._scan_btn_set = ScanButtonSet()

        self._non_reconfigurable_widgets = [
            self._delay_device_le,
            self._delay_ppt_le,
            self._energy_device_le,
            self._energy_ppt_le,
            self._swap_btn
        ]

        self.initUI()
        self.initConnections()

        # required for non-registered ctrl widgets
        self.updateMetaData()

    def initUI(self):
        """Overload."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        i_row = 0
        layout.addWidget(QLabel("Delay device ID: "), i_row, 0, AR)
        layout.addWidget(self._delay_device_le, i_row, 1, 1, 3)

        i_row += 1
        layout.addWidget(QLabel("Delay device property: "), i_row, 0, AR)
        layout.addWidget(self._delay_ppt_le, i_row, 1, 1, 3)

        i_row += 1
        layout.addWidget(QLabel("Mono device ID: "), i_row, 0, AR)
        layout.addWidget(self._energy_device_le, i_row, 1, 1, 3)

        i_row += 1
        layout.addWidget(QLabel("Mono device property: "), i_row, 0, AR)
        layout.addWidget(self._energy_ppt_le, i_row, 1, 1, 3)

        i_row += 1
        layout.addWidget(QLabel("Delay range: "), i_row, 0, AR)
        layout.addWidget(self._delay_range_le, i_row, 1)
        layout.addWidget(QLabel("# of delay bins: "), i_row, 2, AR)
        layout.addWidget(self._n_delay_bins_le, i_row, 3)

        i_row += 1
        layout.addWidget(QLabel("Energy range: "), i_row, 0, AR)
        layout.addWidget(self._energy_range_le, i_row, 1)
        layout.addWidget(QLabel("# of energy bins: "), i_row, 2, AR)
        layout.addWidget(self._n_energy_bins_le, i_row, 3)

        i_row += 1
        layout.addWidget(self._swap_btn, i_row, 3)

        i_row += 1
        layout.addWidget(self._scan_btn_set, i_row, 0, 1, 4)

        self.setLayout(layout)

    def initConnections(self):
        """Overload."""
        mediator = self._mediator

        self._delay_device_le.value_changed_sgn.connect(
            mediator.onTrXasDelayDeviceChange)
        self._delay_ppt_le.value_changed_sgn.connect(
            mediator.onTrXasDelayPropertyChange)

        self._energy_device_le.value_changed_sgn.connect(
            mediator.onTrXasEnergyDeviceChange)
        self._energy_ppt_le.value_changed_sgn.connect(
            mediator.onTrXasEnergyPropertyChange)

        self._n_delay_bins_le.value_changed_sgn.connect(
            mediator.onTrXasNoDelayBinsChange)
        self._delay_range_le.value_changed_sgn.connect(
            mediator.onTrXasDelayRangeChange)

        self._n_energy_bins_le.value_changed_sgn.connect(
            mediator.onTrXasNoEnergyBinsChange)
        self._energy_range_le.value_changed_sgn.connect(
            mediator.onTrXasEnergyRangeChange)

        self._scan_btn_set.scan_toggled_sgn.connect(
            self._onScanStateToggled)

        self._scan_btn_set.reset_sgn.connect(mediator.onTrXasReset)

        self._swap_btn.clicked.connect(self._swapEnergyDelay)

    def updateMetaData(self):
        """Overload."""
        self._delay_device_le.returnPressed.emit()
        self._delay_ppt_le.returnPressed.emit()

        self._energy_device_le.returnPressed.emit()
        self._energy_ppt_le.returnPressed.emit()

        self._n_delay_bins_le.returnPressed.emit()
        self._delay_range_le.returnPressed.emit()

        self._n_energy_bins_le.returnPressed.emit()
        self._energy_range_le.returnPressed.emit()

        return True

    def _onScanStateToggled(self, state):
        if state:
            if not self.updateMetaData():
                return
            self.onStart()
        else:
            self.onStop()

        self._mediator.onTrXasScanStateToggled(AnalysisType.TR_XAS, state)

    def _swapEnergyDelay(self):
        self._swapLineEditContent(self._delay_device_le, self._energy_device_le)
        self._swapLineEditContent(self._delay_ppt_le, self._energy_ppt_le)
        self._swapLineEditContent(self._delay_range_le, self._energy_range_le)
        self._swapLineEditContent(self._n_delay_bins_le, self._n_energy_bins_le)
        self._scan_btn_set.reset_sgn.emit()

    def _swapLineEditContent(self, edit1, edit2):
        text1 = edit1.text()
        text2 = edit2.text()
        edit1.setText(text2)
        edit2.setText(text1)
