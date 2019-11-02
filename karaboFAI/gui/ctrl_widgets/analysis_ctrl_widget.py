"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from .smart_widgets import SmartLineEdit
from ...config import config


class AnalysisCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the general analysis parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__("Global setup", *args, **kwargs)

        self._photon_energy_le = SmartLineEdit(str(config["PHOTON_ENERGY"]))
        self._photon_energy_le.setValidator(
            QtGui.QDoubleValidator(0.001, 100, 6))

        self._sample_dist_le = SmartLineEdit(str(config["SAMPLE_DISTANCE"]))
        self._sample_dist_le.setValidator(QtGui.QDoubleValidator(0.001, 100, 6))

        self._ma_window_le = SmartLineEdit("1")
        self._ma_window_le.setValidator(QtGui.QIntValidator(1, 99999))

        self._reset_ma_btn = QtGui.QPushButton("Reset moving average")

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        row = 0
        layout.addWidget(QtGui.QLabel("Photon energy (keV): "), row, 0, AR)
        layout.addWidget(self._photon_energy_le, row, 1)
        layout.addWidget(QtGui.QLabel("Sample distance (m): "), row, 2, AR)
        layout.addWidget(self._sample_dist_le, row, 3)

        row += 1
        layout.addWidget(QtGui.QLabel("Moving average window: "), row, 0, AR)
        layout.addWidget(self._ma_window_le, row, 1)
        layout.addWidget(self._reset_ma_btn, row, 3, AR)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._photon_energy_le.returnPressed.connect(
            lambda: mediator.onPhotonEnergyChange(
                float(self._photon_energy_le.text())))

        self._sample_dist_le.returnPressed.connect(
            lambda: mediator.onSampleDistanceChange(
                float(self._sample_dist_le.text())))

        self._ma_window_le.returnPressed.connect(
            lambda: mediator.onMaWindowChange(
                int(self._ma_window_le.text())))

        self._reset_ma_btn.clicked.connect(mediator.onResetMa)

    def updateMetaData(self):
        """Override"""
        self._photon_energy_le.returnPressed.emit()
        self._sample_dist_le.returnPressed.emit()
        self._ma_window_le.returnPressed.emit()

        return True
