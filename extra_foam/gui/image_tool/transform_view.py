"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QVBoxLayout, QSplitter

from .base_view import _AbstractImageToolView, create_imagetool_view
from ..misc_widgets import FColor
from ..ctrl_widgets import ImageTransformCtrlWidget
from ..plot_widgets import ImageViewF, RingItem
from ...config import ImageTransformType


@create_imagetool_view(ImageTransformCtrlWidget)
class TransformView(_AbstractImageToolView):
    """TransformView class.

    Widget for image transform and feature extraction.
    """

    transform_type_changed_sgn = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageViewF(hide_axis=False)
        self._corrected.setTitle("Original")
        self._transformed = ImageViewF(hide_axis=False)
        self._transformed.setTitle("Transformed")

        self._ring_item = RingItem(pen=FColor.mkPen('g', alpha=180, width=10))
        self._transformed.addItem(self._ring_item)

        self._transform_type = ImageTransformType.UNDEFINED

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        view_splitter = QSplitter(Qt.Horizontal)
        view_splitter.addWidget(self._corrected)
        view_splitter.addWidget(self._transformed)

        layout = QVBoxLayout()
        layout.addWidget(view_splitter)
        layout.addWidget(self._ctrl_widget)
        layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        self._ctrl_widget.extract_concentric_rings_sgn.connect(
            self._extractConcentricRings)
        self._ctrl_widget.transform_type_changed_sgn.connect(
            self._onTransformTypeChanged)

    def updateF(self, data, auto_update):
        """Override."""
        image = data.image
        if auto_update or self._corrected.image is None:
            self._corrected.setImage(image.masked_mean_ma)

        if self._transform_type == ImageTransformType.CONCENTRIC_RINGS:
            self._transformed.setImage(image.masked_mean_ma)
        elif image.transform_type == self._transform_type:
            self._transformed.setImage(image.transformed)
        else:
            self._transformed.setImage(None)

    def onActivated(self):
        """Override."""
        self._ctrl_widget.registerTransformType()

    def onDeactivated(self):
        """Override."""
        self._ctrl_widget.unregisterTransformType()

    @pyqtSlot()
    def _extractConcentricRings(self):
        img = self._corrected.image
        if img is None:
            return

        cx, cy, radials = self._ctrl_widget.extractFeature(img)
        self._ring_item.setGeometry(cx, cy, radials)

    @pyqtSlot(int)
    def _onTransformTypeChanged(self, tp):
        self._transform_type = ImageTransformType(tp)

        if self._transform_type == ImageTransformType.CONCENTRIC_RINGS:
            self._ring_item.show()
        else:
            self._ring_item.hide()

        self.transform_type_changed_sgn.emit(tp)
