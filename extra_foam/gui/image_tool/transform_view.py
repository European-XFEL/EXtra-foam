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
from ..plot_widgets import ImageViewF, RingItem, RectROI
from ...config import ImageTransformType


class DynamicRoiImageView(ImageViewF):
    roi_added_sgn = pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._rois = { }
        self.new_roi_requested = False

        self._image_item.draw_started_sgn.connect(self.addRoi)

    def addRoi(self, x, y):
        if self.new_roi_requested:
            # Create and add new ROI
            idx = len(self.rois) + 1
            roi = RectROI(idx,
                          size=(100, 100),
                          pos=(x, y),
                          label=f"B{idx}",
                          pen=FColor.mkPen("r", width=2))
            roi.setLocked(False)
            self.rois[roi.label()] = roi
            self.addItem(roi)

            # Cleanup
            self.new_roi_requested = False
            self._image_item.drawing = False
            self.setCursor(Qt.ArrowCursor)
            self.roi_added_sgn.emit(roi)

    def removeRoi(self, label):
        roi = self._rois.pop(label)
        self.removeItem(roi)
        return roi


@create_imagetool_view(ImageTransformCtrlWidget)
class TransformView(_AbstractImageToolView):
    """TransformView class.

    Widget for image transform and feature extraction.
    """

    transform_type_changed_sgn = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = DynamicRoiImageView(hide_axis=False)
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
        self._ctrl_widget.roi_requested_sgn.connect(
            self._onRoiRequested
        )
        self._ctrl_widget.delete_roi_sgn.connect(
            self._onDeleteRoi
        )

        self._corrected.roi_added_sgn.connect(
            self._onRoiAdded
        )
        self._corrected.roi_added_sgn.connect(
            self._ctrl_widget.onRoiAdded
        )

    def updateF(self, data, auto_update):
        """Override."""
        image = data.image
        if auto_update or self._corrected.image is None:
            self._corrected.setImage(image.masked_mean)

        if self._transform_type == ImageTransformType.CONCENTRIC_RINGS:
            self._transformed.setImage(image.masked_mean)
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

        bragg_peak_analysis = self._transform_type == ImageTransformType.BRAGG_PEAK_ANALYSIS

        self._ring_item.setVisible(self._transform_type == ImageTransformType.CONCENTRIC_RINGS)
        self._transformed.setVisible(not bragg_peak_analysis)

        for roi in self._corrected.rois.values():
            roi.setVisible(bragg_peak_analysis)

        self.transform_type_changed_sgn.emit(tp)

    @pyqtSlot()
    def _onRoiRequested(self):
        self._ctrl_widget._bragg_peak.add_roi_btn.setEnabled(False)
        self._corrected.new_roi_requested = True
        self._corrected._image_item.drawing = True
        self._corrected.setCursor(Qt.CrossCursor)

    @pyqtSlot(object)
    def _onRoiAdded(self, roi):
        self._ctrl_widget._bragg_peak.add_roi_btn.setEnabled(True)

        roi.sigRegionChangeFinished.connect(self._onRoiChanged)
        self._onRoiChanged(roi)

    @pyqtSlot(str)
    def _onDeleteRoi(self, label):
        roi = self._corrected.removeRoi(label)
        self._mediator.onItBraggPeakRoiDeletion(roi.label())
        self._ctrl_widget._bragg_peak.onRoiDeleted(roi.label())

    @pyqtSlot(object)
    def _onRoiChanged(self, roi):
        x = int(roi.x())
        y = int(roi.y())
        width, height = int(roi.size().x()), int(roi.size().y())

        self._mediator.onItBraggPeakRoiChange(roi.label(), (x, y, width, height))
