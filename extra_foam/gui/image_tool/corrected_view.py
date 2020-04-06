"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from string import Template

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QSplitter, QVBoxLayout, QWidget
)

from .simple_image_data import _SimpleImageData
from .base_view import _AbstractImageToolView, create_imagetool_view
from ..plot_widgets import HistMixin, ImageAnalysis, PlotWidgetF
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
        super().__init__(parent=parent, show_indicator=True)

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


class RoiHist(HistMixin, PlotWidgetF):
    """RoiHist class.

    Widget for visualizing the ROI histogram.
    """
    def __init__(self, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._plot = self.plotBar()

        self._title_template = Template(
            f"ROI Histogram (mean: $mean, median: $median, std: $std)")
        self.updateTitle()
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'Pixel value')

    def updateF(self, data):
        """Override."""
        hist = data.roi.hist
        if hist.hist is None:
            self.reset()
        else:
            self._plot.setData(hist.bin_centers, hist.hist)
            self.updateTitle(hist.mean, hist.median, hist.std)


@create_imagetool_view(_roi_ctrl_widget=RoiCtrlWidget,
                       _roi_fom_ctrl_widget=RoiFomCtrlWidget,
                       _roi_hist_ctrl_widget=RoiHistCtrlWidget,
                       _roi_norm_ctrl_widget=RoiNormCtrlWidget,
                       _roi_proj_ctrl_widget=RoiProjCtrlWidget)
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
        self._roi_hist = RoiHist()

        self._roi_ctrl_widget.setRois(self._corrected.rois)

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        ctrl_widget = QWidget()
        ctrl_layout = QHBoxLayout()
        AT = Qt.AlignTop
        ctrl_layout.addWidget(self._roi_ctrl_widget)
        ctrl_layout.addWidget(self._roi_fom_ctrl_widget, alignment=AT)
        ctrl_layout.addWidget(self._roi_hist_ctrl_widget, alignment=AT)
        ctrl_layout.addWidget(self._roi_norm_ctrl_widget, alignment=AT)
        ctrl_layout.addWidget(self._roi_proj_ctrl_widget, alignment=AT)
        ctrl_widget.setLayout(ctrl_layout)
        ctrl_widget.setFixedHeight(
            self._roi_proj_ctrl_widget.minimumSizeHint().height())

        subview_splitter = QSplitter(Qt.Vertical)
        subview_splitter.setHandleWidth(9)
        subview_splitter.setChildrenCollapsible(False)
        subview_splitter.addWidget(self._roi_proj_plot)
        subview_splitter.addWidget(self._roi_hist)

        view_splitter = QSplitter(Qt.Horizontal)
        view_splitter.setHandleWidth(9)
        view_splitter.setChildrenCollapsible(False)
        view_splitter.addWidget(self._corrected)
        view_splitter.addWidget(subview_splitter)

        layout = QVBoxLayout()
        layout.addWidget(view_splitter)
        layout.addWidget(ctrl_widget)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        pass

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImageData(_SimpleImageData(data.image))
            self._roi_proj_plot.updateF(data)
            self._roi_hist.updateF(data)

    @property
    def imageView(self):
        return self._corrected
