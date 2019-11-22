"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout

from .simple_image_data import _SimpleImageData
from .base_view import _AbstractImageToolView
from ..plot_widgets import ImageAnalysis
from ..ctrl_widgets import Projection1DCtrlWidget, RoiCtrlWidget


class CorrectedView(_AbstractImageToolView):
    """CorrectedView class.

    Widget for visualizing the corrected (masked, dark subtracted, etc.)
    image. ROI control widgets and 1D projection analysis control widget
    are included.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._image_view = ImageAnalysis(hide_axis=False, parent=self)

        self._roi_ctrl_widget = self.parent().createCtrlWidget(
            RoiCtrlWidget, self._image_view.rois)
        self._proj1d_ctrl_widget = self.parent().createCtrlWidget(
            Projection1DCtrlWidget)

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(self._roi_ctrl_widget)
        ctrl_layout.addWidget(self._proj1d_ctrl_widget)

        layout = QVBoxLayout()
        layout.addWidget(self._image_view)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._image_view.image is None:
            self._image_view.setImageData(_SimpleImageData(data.image))
