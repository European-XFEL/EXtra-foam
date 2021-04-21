"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: James Wrigley <james.wrigley@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QCheckBox, QLabel, QGridLayout

from .smart_widgets import SmartLineEdit
from .base_ctrl_widgets import _AbstractGroupBoxCtrlWidget


class PhotonBinningCtrlWidget(_AbstractGroupBoxCtrlWidget):
    def __init__(self, *args, **kwargs):
        super().__init__("Photon binning", *args, **kwargs)

        self._binning_cb = QCheckBox("Enable")
        self._adu_count_le = SmartLineEdit("65")
        self._adu_count_le.setValidator(QIntValidator(0, 1_000_000))

        self.initUI()
        self.initConnections()

    def initUI(self):
        layout = QGridLayout()
        layout.addWidget(self._binning_cb, 0, 0)
        layout.addWidget(QLabel("ADUs/photon:"), 1, 0)
        layout.addWidget(self._adu_count_le, 1, 1)

        self.setLayout(layout)

    def initConnections(self):
        self._binning_cb.toggled.connect(self._mediator.onPhotonBinningChange)
        self._adu_count_le.value_changed_sgn.connect(self._mediator.onAduThresholdChanged)

    def updateMetaData(self):
        self._binning_cb.toggled.emit(self._binning_cb.isChecked())
        self._adu_count_le.returnPressed.emit()

        return True
