"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QGridLayout

from .base_view import _AbstractImageToolView, create_imagetool_view
from ..ctrl_widgets import ImageTransformCtrlWidget
from ..plot_widgets import ImageViewF


@create_imagetool_view(ImageTransformCtrlWidget)
class TransformView(_AbstractImageToolView):
    """TransformView class.

    Widget for image transform.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageViewF(hide_axis=False)
        self._corrected.setTitle("Origin")
        self._transformed = ImageViewF(hide_axis=False)
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
        transformed = data.image.transformed
        if auto_update or self._corrected.image is None:
            self._corrected.setImage(transformed.origin)
            self._transformed.setImage(transformed.transformed)

    def onActivated(self):
        """Override."""
        self._ctrl_widget.registerTransformType()

    def onDeactivated(self):
        """Override."""
        self._ctrl_widget.unregisterTransformType()

    def _extractFeature(self):
        marked, transformed = self._ctrl_widget.extractFeature(
            self._corrected.image)
        self._corrected.setImage(marked)
        self._transformed.setImage(transformed)
