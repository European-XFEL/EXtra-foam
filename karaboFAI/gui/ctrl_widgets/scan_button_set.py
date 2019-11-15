"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtCore, QtGui, QtWidgets


class ScanButtonSet(QtWidgets.QFrame):

    scan_toggled_sgn = QtCore.pyqtSignal(bool)
    reset_sgn = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._scan_btn = QtWidgets.QPushButton("Scan")
        self._pause_btn = QtWidgets.QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        self._reset_btn = QtWidgets.QPushButton("Reset")

        self.initUI()
        self.initConnections()

    def initUI(self):
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._scan_btn)
        layout.addWidget(self._pause_btn)
        layout.addWidget(self._reset_btn)
        self.setLayout(layout)

    def initConnections(self):
        self._scan_btn.clicked.connect(self._onStartScan)
        self._pause_btn.clicked.connect(self._onStopScan)
        self._reset_btn.clicked.connect(self.reset_sgn)

    def _onStartScan(self):
        self._scan_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self.scan_toggled_sgn.emit(True)

    def _onStopScan(self):
        self._pause_btn.setEnabled(False)
        self._scan_btn.setEnabled(True)
        self.scan_toggled_sgn.emit(False)
