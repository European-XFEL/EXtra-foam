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
    def __init__(self, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(parent=parent)
        parent.registerPlotWidget(self)

        self._pulse_resolved = pulse_resolved

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
        self._set_text()

    def update(self, data):
        """Override."""
        self._set_text(data.tid, data.image.n_images)

    def _set_text(self, tid="", n=""):
        self._trainid_lb.setText("Train ID: {}".format(tid))
        if self._pulse_resolved:
            self._npulses_lb.setText(f"Number of images per train: {n}")
