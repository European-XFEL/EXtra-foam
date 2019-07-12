"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AnalysisCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore, QtGui

from .base_ctrl_widgets import AbstractCtrlWidget
from .smart_widgets import SmartLineEdit, SmartRangeLineEdit
from ...config import config


class AnalysisCtrlWidget(AbstractCtrlWidget):
    """Widget for setting up the general analysis parameters."""

    _pulse_index_validator = QtGui.QIntValidator(
        0, config["MAX_N_PULSES_PER_TRAIN"] - 1)

    def __init__(self, *args, **kwargs):
        super().__init__("General setup", *args, **kwargs)

        # We keep the definitions of attributes which are not used in the
        # PULSE_RESOLVED = True case. It makes sense since these attributes
        # also appear in the defined methods.
        if self._pulse_resolved:
            poi_index1 = 0
            poi_index2 = 1
        else:
            poi_index1 = 0
            poi_index2 = 0

        self._pulse_index_filter_le = SmartRangeLineEdit(":")

        self._poi_index1_le = SmartLineEdit(str(poi_index1))
        self._poi_index1_le.setValidator(self._pulse_index_validator)

        self._poi_index2_le = SmartLineEdit(str(poi_index2))
        self._poi_index2_le.setValidator(self._pulse_index_validator)

        self._photon_energy_le = SmartLineEdit(str(config["PHOTON_ENERGY"]))
        self._photon_energy_le.setValidator(
            QtGui.QDoubleValidator(0.001, 100, 6))

        self._sample_dist_le = SmartLineEdit(str(config["SAMPLE_DISTANCE"]))
        self._sample_dist_le.setValidator(QtGui.QDoubleValidator(0.001, 100, 6))

        self.initUI()
        self.initConnections()

        self.setFixedHeight(self.minimumSizeHint().height())

    def initUI(self):
        """Overload."""
        layout = QtGui.QGridLayout()
        AR = QtCore.Qt.AlignRight

        if self._pulse_resolved:
            layout.addWidget(QtGui.QLabel("Pulse index filter: "), 0, 0, AR)
            layout.addWidget(self._pulse_index_filter_le, 0, 1, 1, 3)

            layout.addWidget(QtGui.QLabel("POI index 1: "), 1, 0, AR)
            layout.addWidget(self._poi_index1_le, 1, 1)
            layout.addWidget(QtGui.QLabel("POI index 2: "), 1, 2, AR)
            layout.addWidget(self._poi_index2_le, 1, 3)

        layout.addWidget(QtGui.QLabel("Photon energy (keV): "), 2, 0, AR)
        layout.addWidget(self._photon_energy_le, 2, 1)
        layout.addWidget(QtGui.QLabel("Sample distance (m): "), 2, 2, AR)
        layout.addWidget(self._sample_dist_le, 2, 3)

        self.setLayout(layout)

    def initConnections(self):
        mediator = self._mediator

        self._poi_index1_le.returnPressed.connect(
            lambda: mediator.onVipPulseIndexChange(
                1, int(self._poi_index1_le.text())))

        self._poi_index2_le.returnPressed.connect(
            lambda: mediator.onVipPulseIndexChange(
                2, int(self._poi_index2_le.text())))

        mediator.poi_indices_connected_sgn.connect(
            self.updateVipPulseIDs)

        self._photon_energy_le.returnPressed.connect(
            lambda: mediator.onPhotonEnergyChange(
                float(self._photon_energy_le.text())))

        self._sample_dist_le.returnPressed.connect(
            lambda: mediator.onSampleDistanceChange(
                float(self._sample_dist_le.text())))

        self._pulse_index_filter_le.value_changed_sgn.connect(
            mediator.onPulseIndexSelectorChange)

    def updateMetaData(self):
        """Override"""
        self._poi_index1_le.returnPressed.emit()
        self._poi_index2_le.returnPressed.emit()

        self._photon_energy_le.returnPressed.emit()

        self._sample_dist_le.returnPressed.emit()

        self._pulse_index_filter_le.returnPressed.emit()

        return True

    def updateVipPulseIDs(self):
        """Called when OverviewWindow is opened."""
        self._poi_index1_le.returnPressed.emit()
        self._poi_index2_le.returnPressed.emit()
