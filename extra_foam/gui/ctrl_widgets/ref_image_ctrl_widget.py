"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os.path as osp

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QGridLayout, QLineEdit, QPushButton

from ..ctrl_widgets import _AbstractCtrlWidget
from ..gui_helpers import create_icon_button
from ...file_io import read_image
from ...ipc import ReferencePub
from ...logger import logger


class RefImageCtrlWidget(_AbstractCtrlWidget):
    """Widget for manipulating reference image in the ImageToolWindow."""

    set_reference_sgn = pyqtSignal()

    def __init__(self, view, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # a reference to the corrected view in the parent widget
        self._view = view

        self._load_ref_btn = QPushButton("Load reference")
        self._ref_fp_le = QLineEdit()
        self._ref_fp_le.setEnabled(False)
        self._remove_ref_btn = create_icon_button('remove.png', 20)

        self._set_ref_btn = QPushButton("Set current as reference")

        self._pub = ReferencePub()

        self._non_reconfigurable_widgets = [
            self._load_ref_btn
        ]

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()

        layout.addWidget(self._load_ref_btn, 0, 0)
        layout.addWidget(self._ref_fp_le, 0, 1)
        layout.addWidget(self._remove_ref_btn, 0, 3)

        layout.addWidget(self._set_ref_btn, 1, 0)

        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        self._load_ref_btn.clicked.connect(self._loadReference)
        self._set_ref_btn.clicked.connect(self._setReference)
        self._remove_ref_btn.clicked.connect(self._removeReference)

    def updateMetaData(self):
        """Override."""
        return True

    @pyqtSlot()
    def _loadReference(self):
        """Load the reference image from a file."""
        filepath = QFileDialog.getOpenFileName(
            caption="Load reference image",
            directory=osp.expanduser("~"))[0]

        try:
            img = read_image(filepath)
        except ValueError as e:
            logger.error(f"[Image tool] {str(e)}")
            return

        self._ref_fp_le.setText(filepath)
        logger.info(f"[Image tool] Loaded reference image from {filepath}")
        self._pub.set(img)

    @pyqtSlot()
    def _setReference(self):
        """Set the current corrected image as reference."""
        img = self._view.image
        if img is not None:
            self._pub.set(img)

    @pyqtSlot()
    def _removeReference(self):
        self._ref_fp_le.setText('')
        self._pub.remove()
