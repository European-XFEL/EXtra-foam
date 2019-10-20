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
from .smart_widgets import SmartLineEdit, SmartSliceLineEdit
from ...config import config


class AnalysisCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the general analysis parameters."""

    _pulse_index_validator = QtGui.QIntValidator(
        0, config["MAX_N_PULSES_PER_TRAIN"] - 1)

    def __init__(self, *args, **kwargs):
        super().__init__("Global setup", *args, **kwargs)

        poi_index1 = 0
        poi_index2 = 0

        self._pulse_slicer_le = SmartSliceLineEdit(":")
        self._xgm_pulse_slicer_le = SmartSliceLineEdit(":")

        self._poi_index1_le = SmartLineEdit(str(poi_index1))
        self._poi_index1_le.setValidator(self._pulse_index_validator)

        self._poi_index2_le = SmartLineEdit(str(poi_index2))
        self._poi_index2_le.setValidator(self._pulse_index_validator)

        if not self._pulse_resolved:
            self._pulse_slicer_le.setEnabled(False)
            self._poi_index1_le.setEnabled(False)
            self._poi_index2_le.setEnabled(False)

        self._photon_energy_le = SmartLineEdit(str(config["PHOTON_ENERGY"]))
        self._photon_energy_le.setValidator(
            QtGui.QDoubleValidator(0.001, 100, 6))

        self._sample_dist_le = SmartLineEdit(str(config["SAMPLE_DISTANCE"]))
        self._sample_dist_le.setValidator(QtGui.QDoubleValidator(0.001, 100, 6))

        self._ma_window_le = SmartLineEdit("1")
        self._ma_window_le.setValidator(QtGui.QIntValidator(1, 99999))

        self._ma_reset_btn = QtGui.QPushButton("Reset M.A.")

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        row = 0
        layout.addWidget(QtGui.QLabel("Pulse slicer: "), row, 0, AR)
        layout.addWidget(self._pulse_slicer_le, row, 1)
        layout.addWidget(QtGui.QLabel("XGM pulse slicer: "), row, 2, AR)
        layout.addWidget(self._xgm_pulse_slicer_le, row, 3)

        row += 1
        layout.addWidget(QtGui.QLabel("POI index 1: "), row, 0, AR)
        layout.addWidget(self._poi_index1_le, row, 1)
        layout.addWidget(QtGui.QLabel("POI index 2: "), row, 2, AR)
        layout.addWidget(self._poi_index2_le, row, 3)

        row += 1
        layout.addWidget(QtGui.QLabel("Photon energy (keV): "), row, 0, AR)
        layout.addWidget(self._photon_energy_le, row, 1)
        layout.addWidget(QtGui.QLabel("Sample distance (m): "), row, 2, AR)
        layout.addWidget(self._sample_dist_le, row, 3)

        row += 1
        layout.addWidget(QtGui.QLabel("M.A. window: "), row, 0, AR)
        layout.addWidget(self._ma_window_le, row, 1)
        layout.addWidget(self._ma_reset_btn, row, 3, AR)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._poi_index1_le.returnPressed.connect(
            lambda: mediator.onPoiPulseIndexChange(
                1, int(self._poi_index1_le.text())))

        self._poi_index2_le.returnPressed.connect(
            lambda: mediator.onPoiPulseIndexChange(
                2, int(self._poi_index2_le.text())))

        mediator.poi_indices_connected_sgn.connect(
            self.updateVipPulseIDs)

        self._photon_energy_le.returnPressed.connect(
            lambda: mediator.onPhotonEnergyChange(
                float(self._photon_energy_le.text())))

        self._sample_dist_le.returnPressed.connect(
            lambda: mediator.onSampleDistanceChange(
                float(self._sample_dist_le.text())))

        self._pulse_slicer_le.value_changed_sgn.connect(
            mediator.onPulseIndexSelectorChange)
        self._xgm_pulse_slicer_le.value_changed_sgn.connect(
            mediator.onXgmPulseIndexSelectorChange)

        self._ma_window_le.returnPressed.connect(
            lambda: mediator.onMaWindowChange(
                int(self._ma_window_le.text())))

        self._ma_reset_btn.clicked.connect(mediator.onMaReset)

    def updateMetaData(self):
        """Override"""
        self._poi_index1_le.returnPressed.emit()
        self._poi_index2_le.returnPressed.emit()

        self._photon_energy_le.returnPressed.emit()

        self._sample_dist_le.returnPressed.emit()

        self._pulse_slicer_le.returnPressed.emit()
        self._xgm_pulse_slicer_le.returnPressed.emit()

        self._ma_window_le.returnPressed.emit()

        return True

    def updateVipPulseIDs(self):
        """Called when OverviewWindow is opened."""
        self._poi_index1_le.returnPressed.emit()
        self._poi_index2_le.returnPressed.emit()
