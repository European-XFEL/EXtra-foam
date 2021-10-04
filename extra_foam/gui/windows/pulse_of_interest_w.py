"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from string import Template

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from .base_window import _AbstractPlotWindow
from ..misc_widgets import FColor
from ..plot_widgets import HistMixin, ImageViewF, PlotWidgetF, TimedPlotWidgetF
from ...config import config


class PoiImageView(ImageViewF):
    """PoiImageView class.

    Widget for displaying the assembled image of pulse-of-interest.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(has_roi=True, parent=parent)

        self._index = idx

    def updateF(self, data):
        """Override."""
        processed = data["processed"]

        try:
            img = processed.image.images[self._index]
            self.setImage(img, auto_levels=(not self._is_initialized))
        except (IndexError, TypeError):
            self.clear()
            return

        self.updateROI(processed)

        if not self._is_initialized:
            self._is_initialized = True

    def setPulseIndex(self, idx):
        self._index = idx
        self.setTitle(f"Pulse-of-interest {idx}")


class PoiFomHist(HistMixin, TimedPlotWidgetF):
    """PoiFomHist class.

    A widget which monitors the histogram of the FOM of the POI pulse.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._index = idx
        self._plot = self.plotBar(brush=FColor.mkBrush('p'))

        self._title_template = Template(
            f"FOM Histogram (mean: $mean, median: $median, std: $std)")
        self.updateTitle()
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'FOM')

    def refresh(self):
        """Override."""
        try:
            hist = self._data["processed"].pulse.hist[self._index]
        except KeyError:
            self.reset()
        else:
            self._plot.setData(hist.bin_centers, hist.hist)
            self.updateTitle(hist.mean, hist.median, hist.std)

    def setPulseIndex(self, idx):
        self._index = idx


class PoiRoiHist(HistMixin, PlotWidgetF):
    """PoiRoiHist class.

    A widget which monitors the pixel-wised histogram of the ROI of
    the POI pulse.
    """
    def __init__(self, idx, *, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self._index = idx
        self._plot = self.plotBar()

        self._title_template = Template(
            f"ROI Histogram (mean: $mean, median: $median, std: $std)")
        self.updateTitle()
        self.setLabel('left', 'Counts')
        self.setLabel('bottom', 'Pixel value')

    def updateF(self, data):
        """Override."""
        try:
            hist = data["processed"].pulse.roi.hist[self._index]
        except KeyError:
            self.reset()
        else:
            self._plot.setData(hist.bin_centers, hist.hist)
            self.updateTitle(hist.mean, hist.median, hist.std)

    def setPulseIndex(self, idx):
        self._index = idx


class PulseOfInterestWindow(_AbstractPlotWindow):
    """PulseOfInterestWindow class."""
    _title = "Pulse-of-interest"

    _TOTAL_W, _TOTAL_H = config['GUI_PLOT_WINDOW_SIZE']

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._poi_imgs = [PoiImageView(0, parent=self),
                          PoiImageView(0, parent=self)]
        self._poi_fom_hists = [PoiFomHist(0, parent=self),
                               PoiFomHist(0, parent=self)]
        self._poi_roi_hists = [PoiRoiHist(0, parent=self),
                               PoiRoiHist(0, parent=self)]

        self.initUI()
        self.initConnections()

        self.resize(self._TOTAL_W, self._TOTAL_H)

        self.setMinimumSize(int(0.6 * self._TOTAL_W), int(0.6 * self._TOTAL_H))

        self.update()

    def initUI(self):
        """Override."""
        self._cw = QSplitter(Qt.Vertical)

        for img, fom_hist, roi_hist in zip(self._poi_imgs,
                                           self._poi_fom_hists,
                                           self._poi_roi_hists):
            rw = QSplitter(Qt.Vertical)
            rw.addWidget(fom_hist)
            rw.addWidget(roi_hist)

            w = QSplitter()
            w.addWidget(img)
            w.addWidget(rw)

            w.setSizes([int(self._TOTAL_W / 2), int(self._TOTAL_W / 2)])
            self._cw.addWidget(w)

        self.setCentralWidget(self._cw)
        self._cw.setHandleWidth(self._SPLITTER_HANDLE_WIDTH)

    def initConnections(self):
        """Override."""
        mediator = self._mediator

        mediator.poi_index_change_sgn.connect(self._updatePoiIndex)
        mediator.poi_window_initialized_sgn.emit()

    def _updatePoiIndex(self, poi_idx, pulse_idx):
        self._poi_imgs[poi_idx].setPulseIndex(pulse_idx)
        self._poi_fom_hists[poi_idx].setPulseIndex(pulse_idx)
        self._poi_roi_hists[poi_idx].setPulseIndex(pulse_idx)
