"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from string import Template

import numpy as np

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QHBoxLayout, QSplitter, QVBoxLayout, QWidget, QTabWidget, QStackedWidget
)

from .base_view import _AbstractImageToolView, create_imagetool_view
from ..plot_widgets import HistMixin, ImageAnalysis, ImageViewF, PlotWidgetF
from ..ctrl_widgets import (
    RoiProjCtrlWidget, RoiCtrlWidget, RoiFomCtrlWidget,
    RoiNormCtrlWidget, RoiHistCtrlWidget, PhotonBinningCtrlWidget
)
from ..misc_widgets import FColor
from ...config import AnalysisType, MaskState, plot_labels, config, KaraboType


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
                       _roi_norm_widget=RoiCtrlWidget,
                       _roi_fom_ctrl_widget=RoiFomCtrlWidget,
                       _roi_hist_ctrl_widget=RoiHistCtrlWidget,
                       _roi_norm_ctrl_widget=RoiNormCtrlWidget,
                       _roi_proj_ctrl_widget=RoiProjCtrlWidget,
                       _photon_binning_ctrl_widget=PhotonBinningCtrlWidget)
class CorrectedView(_AbstractImageToolView):
    """CorrectedView class.

    Widget for visualizing the corrected (masked, dark subtracted, etc.)
    image. ROI control widgets and 1D projection analysis control widget
    are included.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ctrl_tabwidget = QTabWidget()
        self._tab_views_widget = QStackedWidget()

        self._corrected = ImageAnalysis(hide_axis=False)
        self._corrected.setTitle("Averaged over train")

        self._roi_norm_image = ImageViewF(has_roi=True, hide_axis=False)
        self._roi_norm_image.setTitle("Image source for ROI normalizer")
        for roi in self._roi_norm_image.rois:
            roi.setLocked(False)

        self._roi_proj_plot = RoiProjPlot()
        self._roi_hist = RoiHist()

        # Split the four ROIs between the FOM control widget and the ROI
        # normalization control widget.
        self._roi_ctrl_widget.setRois(self._corrected.rois[:2])
        self._roi_norm_widget.setRois(self._roi_norm_image.rois[2:])

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
        ctrl_layout.addWidget(self._roi_proj_ctrl_widget, alignment=AT)
        ctrl_layout.addWidget(self._photon_binning_ctrl_widget, alignment=AT)
        ctrl_layout.setContentsMargins(1, 1, 1, 1)
        ctrl_widget.setLayout(ctrl_layout)
        self._ctrl_tabwidget.addTab(ctrl_widget, "General settings")

        roi_norm_widget = QWidget()
        roi_norm_layout = QHBoxLayout()
        roi_norm_layout.addWidget(self._roi_norm_widget)
        roi_norm_layout.addWidget(self._roi_norm_ctrl_widget, alignment=AT)
        roi_norm_layout.addStretch()
        roi_norm_layout.setContentsMargins(1, 1, 1, 1)
        roi_norm_widget.setLayout(roi_norm_layout)
        self._ctrl_tabwidget.addTab(roi_norm_widget, "ROI normalizer settings")

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
        view_splitter.setSizes([int(1e6), int(1e6)])

        self._tab_views_widget.addWidget(view_splitter)
        self._tab_views_widget.addWidget(self._roi_norm_image)

        layout = QVBoxLayout()
        layout.addWidget(self._tab_views_widget)
        layout.addWidget(self._ctrl_tabwidget)
        layout.setStretch(0, 1)
        layout.setStretch(0, 2)
        layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        self._ctrl_tabwidget.currentChanged.connect(self._tab_views_widget.setCurrentIndex)

    def updateF(self, data, auto_update):
        """Override."""
        raw = data["raw"]
        catalog = data["catalog"]
        processed = data["processed"]

        # Update the corrected image
        if auto_update or self._corrected.image is None:
            self._corrected.setImage(processed.image)
            self._roi_proj_plot.updateF(processed)
            self._roi_hist.updateF(processed)

        # Update the ROI normalization source image
        if self._roi_norm_ctrl_widget.selected_source == config["DETECTOR"]:
            self._roi_norm_image.setImage(processed.image.masked_mean)
        else:
            self._roi_norm_image.setImage(raw[self._roi_norm_ctrl_widget.selected_source])

        # Update the ROI normalization options list with 2D pipeline data sources
        norm_options = []
        for source in catalog.from_category(config["SOURCE_USER_DEFINED_CATEGORY"]):
            if catalog.get_type(source) == KaraboType.PIPELINE_DATA and source in raw and \
               isinstance(raw[source], np.ndarray) and raw[source].ndim == 2:
                norm_options.append(source)
        self._roi_norm_ctrl_widget.updateOptions(norm_options)

    @property
    def imageView(self):
        return self._corrected

    @pyqtSlot()
    def onSaveImage(self):
        self._corrected.writeImage()

    @pyqtSlot(bool)
    def onDrawMask(self, state):
        self._corrected.setMaskingState(MaskState.MASK, state)

    @pyqtSlot(bool)
    def onEraseMask(self, state):
        self._corrected.setMaskingState(MaskState.UNMASK, state)

    @pyqtSlot()
    def onLoadMask(self):
        self._corrected.loadImageMask()

    @pyqtSlot()
    def onSaveMask(self):
        self._corrected.saveImageMask()

    @pyqtSlot()
    def onRemoveMask(self):
        self._corrected.removeMask()

    @pyqtSlot(bool)
    def onMaskSaveInModulesChange(self, state):
        self._corrected.setMaskSaveInModules(state)

    @pyqtSlot(bool)
    def onImageSaveInModulesChange(self, state):
        self._corrected.setImageSaveInModules(state)
