"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QGridLayout, QLabel, QPushButton
)

from ..ctrl_widgets import _AbstractCtrlWidget
from ..ctrl_widgets.smart_widgets import (
    SmartLineEdit, SmartBoundaryLineEdit
)
from ...database import Metadata as mt


class ImageCtrlWidget(_AbstractCtrlWidget):
    """Widget for manipulating images in the ImageToolWindow."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.auto_update_cb = QCheckBox("Update automatically")
        self.auto_update_cb.setChecked(True)
        self.update_image_btn = QPushButton("Update image")
        self.update_image_btn.setEnabled(False)

        # It is just a placeholder
        self.moving_avg_le = SmartLineEdit(str(1))
        self.moving_avg_le.setValidator(QIntValidator(1, 9999999))
        self.moving_avg_le.setMinimumWidth(60)
        self.moving_avg_le.setEnabled(False)

        self.threshold_mask_le = SmartBoundaryLineEdit('-1e5, 1e5')
        # avoid collapse on online and maxwell clusters
        self.threshold_mask_le.setMinimumWidth(160)

        self.auto_level_btn = QPushButton("Auto level")
        self.save_image_btn = QPushButton("Save image")

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        row = 0
        layout.addWidget(self.update_image_btn, row, 0)
        layout.addWidget(self.auto_update_cb, row, 1, AR)

        row += 1
        layout.addWidget(self.auto_level_btn, row, 0)

        row += 1
        layout.addWidget(QLabel("Moving average: "), row, 0, AR)
        layout.addWidget(self.moving_avg_le, row, 1)

        row += 1
        layout.addWidget(QLabel("Threshold mask: "), row, 0, AR)
        layout.addWidget(self.threshold_mask_le, row, 1)

        row += 1
        layout.addWidget(self.save_image_btn, row, 0)

        layout.setVerticalSpacing(20)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        self.auto_update_cb.toggled.connect(
            lambda: self.update_image_btn.setEnabled(
                not self.sender().isChecked()))

    def updateMetaData(self):
        """Override."""
        self.threshold_mask_le.returnPressed.emit()
        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.IMAGE_PROC)
        self.threshold_mask_le.setText(cfg["threshold_mask"][1:-1])
