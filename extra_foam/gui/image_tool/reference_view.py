"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QGridLayout

from .base_view import _AbstractImageToolView
from .simple_image_data import _SimpleImageData
from ..ctrl_widgets import RefImageCtrlWidget
from ..plot_widgets import ImageAnalysis, ImageViewF


class ReferenceView(_AbstractImageToolView):
    """ReferenceView class.

    Widget for visualizing the reference image.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageAnalysis()
        self._corrected.setTitle("Corrected")
        self._reference = ImageViewF()
        self._reference.setTitle("Reference")

        self._ctrl_widget = self.parent().createCtrlWidget(
            RefImageCtrlWidget, self._corrected)

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
        pass

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImageData(_SimpleImageData(data.image))
            # Removing and displaying of the currently displayed image
            # is deferred.
            self._reference.setImage(data.image.reference)
