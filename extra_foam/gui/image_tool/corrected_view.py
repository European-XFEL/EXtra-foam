"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QSplitter, QVBoxLayout
)

from .simple_image_data import _SimpleImageData
from .base_view import _AbstractImageToolView
from ..plot_widgets import ImageAnalysis, PlotWidgetF
from ..ctrl_widgets import (
    RoiProjCtrlWidget, RoiCtrlWidget, RoiFomCtrlWidget,
    RoiNormCtrlWidget, RoiHistCtrlWidget
)
from ..misc_widgets import FColor
from ...config import AnalysisType, plot_labels


class RoiProjPlot(PlotWidgetF):
    """RoiProjPlot class.

    Widget for visualizing the average ROI projection over a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        x_label, y_label = plot_labels[AnalysisType.ROI_PROJ]
        self.setLabel('bottom', x_label)
        self.setLabel('left', y_label)
        self.setTitle('ROI projection (average over train)')

        self._plot = self.plotCurve(pen=FColor.mkPen("p"))

    def updateF(self, data):
        """Override."""
        proj = data.roi.proj
        x = proj.x
        y = proj.y
        if y is None:
            return
        self._plot.setData(x, y)


class InTrainRoiFomPlot(PlotWidgetF):
    """InTrainRoiFomPlot class.

    Widget for visualizing the pulse-resolved ROI FOM in a train.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setLabel('bottom', "Pulse index")
        self.setLabel('left', "ROI FOM")
        self.setTitle('Pulse-resolved ROI FOMs (per train)')

        self._plot = self.plotScatter(brush=FColor.mkBrush('b'))

    def updateF(self, data):
        """Override."""
        fom = data.pulse.roi.fom
        if fom is None:
            return
        self._plot.setData(range(len(fom)), fom)


class CorrectedView(_AbstractImageToolView):
    """CorrectedView class.

    Widget for visualizing the corrected (masked, dark subtracted, etc.)
    image. ROI control widgets and 1D projection analysis control widget
    are included.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageAnalysis(hide_axis=False)
        self._roi_proj_plot = RoiProjPlot()
        self._roi_fom_plot = InTrainRoiFomPlot()

        self._roi_ctrl_widget = self.parent().createCtrlWidget(
            RoiCtrlWidget, self._corrected.rois)
        self._roi_fom_ctrl_widget = self.parent().createCtrlWidget(
            RoiFomCtrlWidget)
        self._roi_hist_ctrl_widget = self.parent().createCtrlWidget(
            RoiHistCtrlWidget)
        self._roi_norm_ctrl_widget = self.parent().createCtrlWidget(
            RoiNormCtrlWidget)
        self._roi_proj_ctrl_widget = self.parent().createCtrlWidget(
            RoiProjCtrlWidget)

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        ctrl_layout = QHBoxLayout()
        AT = Qt.AlignTop
        ctrl_layout.addWidget(self._roi_ctrl_widget)
        ctrl_layout.addWidget(self._roi_fom_ctrl_widget, alignment=AT)
        ctrl_layout.addWidget(self._roi_hist_ctrl_widget, alignment=AT)
        ctrl_layout.addWidget(self._roi_norm_ctrl_widget, alignment=AT)
        ctrl_layout.addWidget(self._roi_proj_ctrl_widget, alignment=AT)

        subview_splitter = QSplitter(Qt.Vertical)
        subview_splitter.setHandleWidth(9)
        subview_splitter.setChildrenCollapsible(False)
        subview_splitter.addWidget(self._roi_proj_plot)
        subview_splitter.addWidget(self._roi_fom_plot)

        view_splitter = QSplitter(Qt.Horizontal)
        view_splitter.setHandleWidth(9)
        view_splitter.setChildrenCollapsible(False)
        view_splitter.addWidget(self._corrected)
        view_splitter.addWidget(subview_splitter)

        layout = QVBoxLayout()
        layout.addWidget(view_splitter)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImageData(_SimpleImageData(data.image))
            self._roi_proj_plot.updateF(data)
            self._roi_fom_plot.updateF(data)

    @property
    def imageView(self):
        return self._corrected
