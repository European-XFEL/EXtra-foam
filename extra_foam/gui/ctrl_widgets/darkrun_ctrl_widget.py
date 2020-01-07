"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QGridLayout

from ..ctrl_widgets import _AbstractCtrlWidget
from ..gui_helpers import create_icon_button


class DarkRunCtrlWidget(_AbstractCtrlWidget):
    """Widget for manipulating dark run in the ImageToolWindow."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._collect_btn = create_icon_button('record.png', 30)
        self._collect_btn.setCheckable(True)
        self._remove_btn = create_icon_button('remove.png', 30)

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()

        row = 0
        layout.addWidget(self._collect_btn, row, 0)
        layout.addWidget(self._remove_btn, row, 1)
        layout.setColumnStretch(2, 2)

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        mediator = self._mediator
        self._collect_btn.toggled.connect(mediator.onRdStateChange)
        self._collect_btn.toggled.emit(self._collect_btn.isChecked())
        self._remove_btn.clicked.connect(mediator.onRdRemoveDark)

    def updateMetaData(self):
        """Override."""
        return True

    def onDeactivated(self):
        if self._collect_btn.isChecked():
            self._collect_btn.setChecked(False)
