"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QGridLayout, QLineEdit, QPushButton

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

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateMetaData(self):
        """Override."""
        return True

    def loadMetaData(self):
        """Override."""
        pass
