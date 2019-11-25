"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPlainTextEdit

from .base_window import _AbstractSatelliteWindow
from ...config import config
from ...processes import list_fai_processes


class ProcessMonitor(_AbstractSatelliteWindow):
    _title = "Process monitor"

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._cw = QPlainTextEdit()
        self._cw.setReadOnly(True)
        logger_font = QFont("monospace")
        logger_font.setStyleHint(QFont.TypeWriter)
        logger_font.setPointSize(11)
        self._cw.setFont(logger_font)

        self.setCentralWidget(self._cw)

        self._timer = QTimer()
        self._timer.timeout.connect(self.updateProcessInfo)
        self._timer.start(config["PROCESS_MONITOR_HEART_BEAT"])

        self.setMinimumSize(900, 150)
        self.show()

    def initUI(self):
        """Override."""
        pass

    def initConnections(self):
        """Override."""
        pass

    def updateProcessInfo(self):
        self._cw.clear()
        info = "{:<20s}{:<16s}{:<16s}{:<12s}{:<16s}\n".format(
            "Process name", "FAI name", "FAI type", "pid", "status")
        info += "-" * 80 + "\n"
        for p in list_fai_processes():
            info += f"{p.name:<20s}" \
                f"{p.fai_name:<16s}" \
                f"{p.fai_type:<16s}" \
                f"{p.pid:<12d}" \
                f"{p.status:<16s}" \
                f"\n"

        self._cw.appendPlainText(info)
