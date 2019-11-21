"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox, QGridLayout, QLabel, QPushButton
)

from ..ctrl_widgets import _AbstractCtrlWidget
from ..ctrl_widgets.smart_widgets import (
    SmartLineEdit, SmartBoundaryLineEdit
)
from ...config import config


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

        self.threshold_mask_le = SmartBoundaryLineEdit(
            ', '.join([str(v) for v in config["MASK_RANGE"]]))
        # avoid collapse on online and maxwell clusters
        self.threshold_mask_le.setMinimumWidth(160)

        self.darksubtraction_cb = QCheckBox("Subtract dark")
        self.darksubtraction_cb.setChecked(True)

        self.bkg_le = SmartLineEdit(str(0.0))
        self.bkg_le.setValidator(QDoubleValidator())

        self.auto_level_btn = QPushButton("Auto level")
        self.save_image_btn = QPushButton("Save image")
        self.load_ref_btn = QPushButton("Load reference")
        self.set_ref_btn = QPushButton("Set reference")
        self.remove_ref_btn = QPushButton("Remove reference")

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
        layout.addWidget(self.darksubtraction_cb, row, 0, AR)

        row += 1
        layout.addWidget(QLabel("Subtract background: "), row, 0, AR)
        layout.addWidget(self.bkg_le, row, 1)

        row += 1
        layout.addWidget(self.save_image_btn, row, 0)
        layout.addWidget(self.load_ref_btn, row, 1)

        row += 1
        layout.addWidget(self.set_ref_btn, row, 0)
        layout.addWidget(self.remove_ref_btn, row, 1)

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
        self.darksubtraction_cb.toggled.emit(
            self.darksubtraction_cb.isChecked())
        self.bkg_le.returnPressed.emit()
        return True
