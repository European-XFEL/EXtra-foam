"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ProcessMonitor

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QPlainTextEdit

from .base_window import AbstractSatelliteWindow


class ProcessMonitor(AbstractSatelliteWindow):
    title = "Process monitor"

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._cw = QPlainTextEdit()
        logger_font = QtGui.QFont()
        logger_font.setPointSize(11)
        self._cw.setFont(logger_font)

        self.setCentralWidget(self._cw)

        self.setMinimumSize(900, 150)
        self.show()

    def initUI(self):
        """Override."""
        pass

    def initConnections(self):
        """Override."""
        pass

    @QtCore.pyqtSlot(object)
    def onProcessInfoUpdate(self, info):
        self._cw.clear()
        for item in info:
            self._cw.appendPlainText(str(item))
