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
from ..ctrl_widgets import ImageTransformCtrlWidget
from ..plot_widgets import ImageViewF


@create_imagetool_view(ImageTransformCtrlWidget)
class ImageTransformView(_AbstractImageToolView):
    """ImageTransformView class.

    Widget for visualizing the corrected image and its transformation
    side-by-side.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageViewF()
        self._transformed = ImageViewF()
        self._transformed.setTitle("Transformed")

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        layout.addWidget(self._corrected, 0, 0)
        layout.addWidget(self._transformed, 0, 1)
        layout.addWidget(self._ctrl_widget, 1, 0, 1, 2)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImage(data.image.masked_mean)
            self._transformed.setImage(data.image.transformed)
