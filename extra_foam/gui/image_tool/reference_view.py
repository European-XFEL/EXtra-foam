"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import os.path as osp

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QGridLayout

from .base_view import _AbstractImageToolView, create_imagetool_view
from ..ctrl_widgets import RefImageCtrlWidget
from ..plot_widgets import ImageViewF
from ...file_io import write_image
from ...ipc import ReferencePub
from ...logger import logger
from ... import ROOT_PATH


@create_imagetool_view(RefImageCtrlWidget)
class ReferenceView(_AbstractImageToolView):
    """ReferenceView class.

    Widget for visualizing the reference image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageViewF(hide_axis=False)
        self._corrected.setTitle("Current")
        self._reference = ImageViewF(hide_axis=False)
        self._reference.setTitle("Reference")

        self._pub = ReferencePub()

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        layout.addWidget(self._corrected, 0, 0)
        layout.addWidget(self._reference, 0, 1)
        layout.addWidget(self._ctrl_widget, 1, 0, 1, 2)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        self._ctrl_widget.load_btn.clicked.connect(self._loadReference)
        self._ctrl_widget.set_current_btn.clicked.connect(self._setReference)
        self._ctrl_widget.remove_btn.clicked.connect(self._removeReference)

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImage(data.image.masked_mean)
            # Removing and displaying of the currently displayed image
            # is deferred.
            self._reference.setImage(data.image.reference)

    @pyqtSlot()
    def _loadReference(self):
        """Load the reference image from a file."""
        filepath = QFileDialog.getOpenFileName(
            caption="Load reference image",
            directory=osp.expanduser("~"))[0]

        # do not remove reference if the user meant to cancel the selection
        if filepath:
            self._pub.set(filepath)
            self._ctrl_widget.filepath_le.setText(filepath)

    @pyqtSlot()
    def _setReference(self):
        """Set the current corrected image as reference."""
        img = self._corrected.image
        if img is not None:
            filepath = osp.join(ROOT_PATH, "tmp", ".reference.npy")
            if not osp.exists(osp.dirname(filepath)):
                os.mkdir(osp.dirname(filepath))

            try:
                write_image(filepath, img)
            except ValueError as e:
                logger.error(str(e))

            self._pub.set(filepath)
            self._ctrl_widget.filepath_le.setText(filepath)

    @pyqtSlot()
    def _removeReference(self):
        """Remove the reference image."""
        self._pub.set("")
        self._ctrl_widget.filepath_le.setText("")
