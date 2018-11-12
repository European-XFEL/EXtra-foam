"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Abstract class worker.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .widgets.pyqtgraph import QtCore


class Worker(QtCore.QThread):
    # post messages in the main GUI
    message = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self._running = False

    def log(self, msg):
        """Log information in the main GUI.

        :param str msg: message string.
        """
        self.message.emit(msg)

    def terminate(self):
        self._running = False
