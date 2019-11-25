"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton


class ScanButtonSet(QFrame):

    scan_toggled_sgn = pyqtSignal(bool)
    reset_sgn = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._scan_btn = QPushButton("Scan")
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        self._reset_btn = QPushButton("Reset")

        self.initUI()
        self.initConnections()

    def initUI(self):
        layout = QHBoxLayout()
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
