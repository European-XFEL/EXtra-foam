"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
)

from ..ctrl_widgets import _AbstractCtrlWidget
from ..ctrl_widgets.smart_widgets import SmartBoundaryLineEdit
from ..gui_helpers import create_icon_button
from ...config import GeomAssembler
from ...database import Metadata as mt


class MaskCtrlWidget(_AbstractCtrlWidget):
    """Widget for masking image in the ImageToolWindow."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.threshold_mask_le = SmartBoundaryLineEdit('-1e5, 1e5')
        # avoid collapse on online and maxwell clusters
        self.threshold_mask_le.setMinimumWidth(160)

        self.mask_tile_cb = QCheckBox("Mask tile edges")

        icon_size = 30
        self.draw_mask_btn = create_icon_button(
            "draw_mask.png", icon_size, description="Draw mask")
        self.draw_mask_btn.setCheckable(True)
        self.erase_mask_btn = create_icon_button(
            "erase_mask.png", icon_size, description="Erase mask")
        self.erase_mask_btn.setCheckable(True)
        self.remove_btn = create_icon_button(
            "remove_mask.png", icon_size, description="Remove mask")

        self.load_btn = QPushButton("Load mask")
        self.save_btn = QPushButton("Save mask")
        self.mask_save_in_modules_cb = QCheckBox("Save mask in modules")

        self._exclusive_btns = {self.erase_mask_btn, self.draw_mask_btn}

        self._non_reconfigurable_widgets = [
            self.save_btn,
            self.load_btn,
        ]

        if not self._require_geometry:
            self.mask_tile_cb.setDisabled(True)
            self.mask_save_in_modules_cb.setDisabled(True)
        else:
            self._non_reconfigurable_widgets.append(
                self.mask_save_in_modules_cb)

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        row = 0
        layout.addWidget(QLabel("Threshold mask: "), row, 0, AR)
        layout.addWidget(self.threshold_mask_le, row, 1)

        row += 1
        layout.addWidget(self.mask_tile_cb, row, 0, AR)

        row += 1
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.draw_mask_btn)
        sub_layout.addWidget(self.erase_mask_btn)
        sub_layout.addWidget(self.remove_btn)
        layout.addLayout(sub_layout, row, 0)

        row += 1
        layout.addWidget(self.load_btn, row, 0)
        layout.addWidget(self.save_btn, row, 1)

        row += 1
        layout.addWidget(self.mask_save_in_modules_cb, row, 0, 1, 2, AR)

        layout.setVerticalSpacing(20)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        self.threshold_mask_le.value_changed_sgn.connect(
            mediator.onImageThresholdMaskChange)

        self.mask_tile_cb.toggled.connect(
            mediator.onImageMaskTileEdgeChange)
        mediator.assembler_change_sgn.connect(self._onAssemblerChange)

        self.erase_mask_btn.toggled.connect(self._updateExclusiveBtns)
        self.draw_mask_btn.toggled.connect(self._updateExclusiveBtns)
        self.remove_btn.clicked.connect(
            lambda: self._updateExclusiveBtns(True))

        # required for loading metadata
        self.mask_save_in_modules_cb.toggled.connect(
            mediator.onImageMaskSaveInModulesToggled)

    @pyqtSlot(object)
    def _onAssemblerChange(self, assembler):
        if assembler == GeomAssembler.EXTRA_GEOM:
            self.mask_tile_cb.setChecked(False)
            self.mask_tile_cb.setEnabled(False)

            self.mask_save_in_modules_cb.setChecked(False)
            self.mask_save_in_modules_cb.setEnabled(False)
        else:
            self.mask_tile_cb.setEnabled(True)
            self.mask_save_in_modules_cb.setEnabled(True)

    def updateMetaData(self):
        """Override."""
        self.threshold_mask_le.returnPressed.emit()
        self.mask_tile_cb.toggled.emit(self.mask_tile_cb.isChecked())
        self.mask_save_in_modules_cb.toggled.emit(
            self.mask_save_in_modules_cb.isChecked())
        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.IMAGE_PROC)
        self.threshold_mask_le.setText(cfg["threshold_mask"][1:-1])
        if self._require_geometry:
            self.mask_tile_cb.setChecked(cfg["mask_tile"] == 'True')
            self.mask_save_in_modules_cb.setChecked(
                cfg["mask_save_in_modules"] == 'True')

    @pyqtSlot(bool)
    def _updateExclusiveBtns(self, checked):
        if checked:
            for at in self._exclusive_btns:
                if at != self.sender():
                    at.setChecked(False)

    def setInteractiveButtonsEnabled(self, state):
        self.draw_mask_btn.setEnabled(state)
        self.erase_mask_btn.setEnabled(state)
