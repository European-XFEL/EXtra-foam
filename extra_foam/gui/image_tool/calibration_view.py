"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout

from .base_view import _AbstractImageToolView
from .simple_image_data import _SimpleImageData
from ..ctrl_widgets import CalibrationCtrlWidget
from ..plot_widgets import ImageViewF, ImageAnalysis


class CalibrationView(_AbstractImageToolView):
    """CalibrationView class.

    Widget for visualizing image calibration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._raw = ImageViewF(hide_axis=False)
        self._raw.setTitle("Raw")
        self._corrected = ImageAnalysis(hide_axis=False)
        self._corrected.setTitle("Corrected")
        self._gain = ImageViewF(hide_axis=False)
        self._gain.setTitle("Gain")
        self._offset = ImageViewF(hide_axis=False)
        self._offset.setTitle("Offset")

        self._ctrl_widget = self.parent().createCtrlWidget(
            CalibrationCtrlWidget)

        self.initUI()

    def initUI(self):
        """Override."""
        view_splitter = QGridLayout()
        view_splitter.addWidget(self._raw, 0, 0)
        view_splitter.addWidget(self._corrected, 1, 0)
        view_splitter.addWidget(self._gain, 0, 1)
        view_splitter.addWidget(self._offset, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(view_splitter)
        layout.addWidget(self._ctrl_widget)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImageData(_SimpleImageData(data.image))
