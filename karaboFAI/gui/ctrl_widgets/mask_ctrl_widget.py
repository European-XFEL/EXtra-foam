"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

MaskCtrlWidget.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from ..pyqtgraph import QtCore, QtGui

from ...config import config


class MaskCtrlWidget(QtGui.QWidget):
    """Widget inside the action bar for masking image."""

    threshold_mask_sgn = QtCore.pyqtSignal(float, float)

    _double_validator = QtGui.QDoubleValidator()

    def __init__(self, parent=None):
        """Initialization"""
        super().__init__(parent=parent)

        self._min_pixel_le = QtGui.QLineEdit(str(config["MASK_RANGE"][0]))
        self._min_pixel_le.setValidator(self._double_validator)
        # avoid collapse on online and maxwell clusters
        self._min_pixel_le.setMinimumWidth(60)
        self._min_pixel_le.returnPressed.connect(
            self.onThresholdMaskChanged)
        self._max_pixel_le = QtGui.QLineEdit(str(config["MASK_RANGE"][1]))
        self._max_pixel_le.setValidator(self._double_validator)
        self._max_pixel_le.setMinimumWidth(60)
        self._max_pixel_le.returnPressed.connect(
            self.onThresholdMaskChanged)

        self.initUI()

        self.setFixedSize(self.minimumSizeHint())

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        layout.addWidget(QtGui.QLabel("Min. mask: "))
        layout.addWidget(self._min_pixel_le)
        layout.addWidget(QtGui.QLabel("Max. mask: "))
        layout.addWidget(self._max_pixel_le)

        self.setLayout(layout)
        self.layout().setContentsMargins(2, 1, 2, 1)

    def onThresholdMaskChanged(self):
        self.threshold_mask_sgn.emit(float(self._min_pixel_le.text()),
                                     float(self._max_pixel_le.text()))