"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QVBoxLayout

from .base_view import _AbstractImageToolView, create_imagetool_view
from ..ctrl_widgets import GeometryCtrlWidget
from ..plot_widgets import ImageViewF


@create_imagetool_view(GeometryCtrlWidget)
class GeometryView(_AbstractImageToolView):
    """GeometryView class.

    Widget for tweaking geometry parameters for assembling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageViewF(hide_axis=False)

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QVBoxLayout()
        layout.addWidget(self._corrected)
        layout.addWidget(self._ctrl_widget)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImage(data.image.masked_mean)
