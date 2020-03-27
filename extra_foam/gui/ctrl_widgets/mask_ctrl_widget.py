"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QCheckBox, QGridLayout, QLabel

from ..ctrl_widgets import _AbstractCtrlWidget
from ..ctrl_widgets.smart_widgets import SmartBoundaryLineEdit
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

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        AR = Qt.AlignRight

        row = 0
        layout.addWidget(QLabel("Threshold mask: "), row, 0, AR)
        layout.addWidget(self.threshold_mask_le, row, 1)

        if self._require_geometry:
            row += 1
            layout.addWidget(self.mask_tile_cb, row, 0, AR)

        layout.setVerticalSpacing(20)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        self.threshold_mask_le.value_changed_sgn.connect(
            lambda x: mediator.onImageThresholdMaskChange(x))

        self.mask_tile_cb.toggled.connect(
            mediator.onImageMaskTileEdgeChange)
        mediator.assembler_change_sgn.connect(self._onAssemblerChange)

    @pyqtSlot(object)
    def _onAssemblerChange(self, assembler):
        if assembler == GeomAssembler.EXTRA_GEOM:
            self.mask_tile_cb.setChecked(False)
            self.mask_tile_cb.setEnabled(False)
        else:
            self.mask_tile_cb.setEnabled(True)

    def updateMetaData(self):
        """Override."""
        self.threshold_mask_le.returnPressed.emit()
        self.mask_tile_cb.toggled.emit(self.mask_tile_cb.isChecked())
        return True

    def loadMetaData(self):
        """Override."""
        cfg = self._meta.hget_all(mt.IMAGE_PROC)
        self.threshold_mask_le.setText(cfg["threshold_mask"][1:-1])
        if self._require_geometry:
            self.mask_tile_cb.setChecked(cfg["mask_tile"] == 'True')
