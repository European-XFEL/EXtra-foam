"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

ProcessMonitorWidget

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5 import QtCore
from PyQt5.QtWidgets import QListWidgetItem, QListWidget, QLayout

from .base_window import AbstractSatelliteWindow


class ProcessMonitorWidget(AbstractSatelliteWindow):
    title = "Process monitor"

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._cw = QListWidget()

    def initUI(self):
        """Override."""
        list_widget = QListWidget(self)
        list_widget.addItem(QListWidgetItem("abdfaafdafaf"))
        layout = QLayout()
        layout.addWidget(list_widget)
        self.setLayout(layout)
