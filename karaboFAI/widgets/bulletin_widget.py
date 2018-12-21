"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

BulletinWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .pyqtgraph import QtGui


class BulletinWidget(QtGui.QWidget):
    """BulletinWidget class."""
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        parent.registerPlotWidget(self)

        self._trainid_lb = QtGui.QLabel("")
        self._trainid_lb.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))

        self._npulses_lb = QtGui.QLabel("")
        self._npulses_lb.setFont(QtGui.QFont("Times", 20, QtGui.QFont.Bold))

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._trainid_lb)
        layout.addWidget(self._npulses_lb)
        self.setLayout(layout)

        self.clear()

    def clear(self):
        self._trainid_lb.setText("Train ID: ")
        self._npulses_lb.setText("Number of pulses per train: ")

    def update(self, data):
        """Override."""
        self._trainid_lb.setText("Train ID: {}".format(data.tid))
        self._npulses_lb.setText(
            "Number of pulses per train: {}".format(len(data.intensity)))
