"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QComboBox, QGridLayout, QLabel

from ..ctrl_widgets import _AbstractCtrlWidget


class ImageTransformCtrlWidget(_AbstractCtrlWidget):
    """Widget for transforming the current image in the ImageToolWindow."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trans_categories = QComboBox()
        self.trans_names = QComboBox()

        self._non_reconfigurable_widgets = [
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()

        layout.addWidget(QLabel("Image transformation"), 0, 0)
        layout.addWidget(self.trans_categories, 0, 1)
        layout.addWidget(self.trans_names, 0, 1)

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
