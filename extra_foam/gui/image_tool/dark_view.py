"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QVBoxLayout

from .base_view import _AbstractImageToolView
from ..plot_widgets import ImageViewF


class DarkView(_AbstractImageToolView):
    """DarkView class.

    Widget for visualizing the recorded dark images.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._dark_view = ImageViewF()

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QVBoxLayout()
        layout.addWidget(self._dark_view)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateF(self, data, auto_update):
        """Override."""
        # always update when activated
        if data.image.dark_mean is None:
            self._dark_view.clear()
        else:
            self._dark_view.setImage(data.image.dark_mean)
