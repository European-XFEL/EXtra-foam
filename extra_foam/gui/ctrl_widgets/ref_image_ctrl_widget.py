"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy,
    QSpacerItem, QWidget)

from ..ctrl_widgets import _AbstractCtrlWidget
from ..gui_helpers import create_icon_button


class RefImageCtrlWidget(_AbstractCtrlWidget):
    """Widget for manipulating reference image in the ImageToolWindow."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.load_btn = QPushButton("Load reference")
        self.filepath_le = QLineEdit()
        self.filepath_le.setEnabled(False)
        self.remove_btn = create_icon_button('remove.png', 20)

        self.set_current_btn = QPushButton("Set current as reference")

        # Record
        self.record_label = QLabel()
        self.record_label.setFixedWidth(430)
        self.record_btn = QPushButton("Record")
        self.record_btn.setCheckable(True)
        self.save_btn = QPushButton("Save")
        self.save_btn.setDisabled(True)

        self._non_reconfigurable_widgets = [
            self.load_btn
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()

        layout.addWidget(self.load_btn, 0, 0)
        layout.addWidget(self.filepath_le, 0, 1)
        layout.addWidget(self.remove_btn, 0, 3)

        layout.addWidget(self.set_current_btn, 1, 0)

        # Add widgets for recording reference
        spacer = QSpacerItem(0, 0, QSizePolicy.Expanding)
        record_layout = QHBoxLayout()
        record_layout.setContentsMargins(0, 0, 0, 0)
        record_layout.addSpacerItem(spacer)
        record_layout.addWidget(self.record_label)
        record_layout.addWidget(self.record_btn)
        record_layout.addWidget(self.save_btn)
        widget = QWidget()
        widget.setLayout(record_layout)
        layout.addWidget(widget, 1, 1)

        self.setLayout(layout)

    def initConnections(self):
        self.record_btn.toggled.connect(self._enableButtonsOnRecord)

    def onStart(self):
        super().onStart()
        self.record_btn.setEnabled(True)

    def onStop(self):
        super().onStop()
        self.record_btn.setEnabled(False)

    def updateMetaData(self):
        """Override."""
        return True

    def loadMetaData(self):
        """Override."""
        pass

    @pyqtSlot(bool)
    def _enableButtonsOnRecord(self, is_recording):
        # Disable/enable buttons when (not) recording
        self.save_btn.setDisabled(is_recording)
        self.set_current_btn.setDisabled(is_recording)

        # Update record button text
        self.record_btn.setText("Stop" if is_recording else "Record")
