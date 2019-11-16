"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QVBoxLayout

from .simple_image_data import _SimpleImageData
from .base_view import _AbstractImageToolView
from ..ctrl_widgets import GeometryCtrlWidget
from ..plot_widgets import ImageAnalysis


class GeometryView(_AbstractImageToolView):
    """GeometryView class.

    Widget for tweaking geometry parameters for assembling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._image_view = ImageAnalysis(hide_axis=False)
        self._ctrl_widget = self.parent().createCtrlWidget(
            GeometryCtrlWidget)

        self.initUI()

    def initUI(self):
        """Override."""
        layout = QVBoxLayout()
        layout.addWidget(self._image_view)
        layout.addWidget(self._ctrl_widget)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._image_view.image is None:
            self._image_view.setImageData(_SimpleImageData(data.image))
