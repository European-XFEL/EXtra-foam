"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

Plot widgets module.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc

import numpy as np

from .pyqtgraph.Qt import QtGui
from .pyqtgraph import (
    GraphicsLayoutWidget, ImageItem, mkPen, ColorMap, LinearRegionItem,
    ScatterPlotItem, mkBrush, SpinBox
)
from .pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from .config import Config as cfg
from .data_processing import integrate_curve, sub_array_with_range


X_LABEL = "Momentum transfer (1/A)"
Y_LABEL = "Scattering signal (arb. u.)"

COLOR_MAP = ColorMap(*zip(*Gradients["thermal"]["ticks"]))


class Pen:
    _w = 3
    red = mkPen((255, 0, 0), width=_w)
    green = mkPen((0, 255, 0), width=_w)
    blue = mkPen((0, 0, 255), width=_w)
    cyan = mkPen((0, 255, 255), width=_w)
    purple = mkPen((255, 0, 255), width=_w)
    yellow = mkPen((255, 255, 0), width=_w)


class MainGuiLinePlotWidget(GraphicsLayoutWidget):
    def __init__(self, parent=None, **kwargs):
        """Initialization."""
        super().__init__(parent, **kwargs)

        w = cfg.MAIN_WINDOW_WIDTH - cfg.MAIN_LINE_PLOT_HEIGHT - 25
        h = cfg.MAIN_LINE_PLOT_HEIGHT
        self.setFixedSize(w, h)

        self._plot = self.addPlot()
        self._plot.setTitle("")
        self._plot.setLabel('bottom', X_LABEL)
        self._plot.setLabel('left', Y_LABEL)

    def set_title(self, text):
        self._plot.setTitle(text)

    def clear_(self):
        self._plot.clear()

    def update(self, *args, **kwargs):
        self._plot.plot(*args, **kwargs)


class MainGuiImageViewWidget(GraphicsLayoutWidget):
    def __init__(self, parent=None, **kwargs):
        """Initialization."""
        super().__init__(parent, **kwargs)

        self.setFixedSize(cfg.MAIN_LINE_PLOT_HEIGHT, cfg.MAIN_LINE_PLOT_HEIGHT)

        self._img = ImageItem(border='w')
        self._img.setLookupTable(COLOR_MAP.getLookupTable())

        self._view = self.addViewBox(lockAspect=True)
        self._view.addItem(self._img)

    def clear_(self):
        self._img.clear()

    def update(self, *args, **kwargs):
        self._img.setImage(autoLevels=False, *args, **kwargs)
        self._view.autoRange()


class PlotWindow(QtGui.QMainWindow):
    """Base class for various plot windows."""
    def __init__(self, window_id, parent=None, title=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setWindowTitle(title)
        self._id = window_id
        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        self._gl_widget = GraphicsLayoutWidget()

        self.image_items = []
        self.plot_items = []

    @abc.abstractmethod
    def initUI(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, data):
        """Update plots"""
        raise NotImplementedError

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        # avoid memory leak
        self.parent().remove_window(self._id)

    def clear(self):
        for item in self.plot_items:
            item.clear()
        for item in self.image_items:
            item.clear()


class IndividualPulseWindow(PlotWindow):
    def __init__(self, window_id, pulse_ids, *,
                 parent=None,
                 show_image=False,
                 title="FXE Azimuthal integration"):
        """Initialization."""
        super().__init__(window_id, parent=parent, title=title)

        self._pulse_ids = pulse_ids
        self._show_image = show_image

        self.initUI()

    def initUI(self):
        g_layout = self._gl_widget.ci.layout
        g_layout.setColumnStretchFactor(0, 1)
        if self._show_image:
            g_layout.setColumnStretchFactor(1, 3)
        w = cfg.LINE_PLOT_WIDTH + self._show_image*(cfg.LINE_PLOT_HEIGHT - 20)
        h = min(4, len(self._pulse_ids))*cfg.LINE_PLOT_HEIGHT
        self._gl_widget.setFixedSize(w, h)

        for pulse_id in self._pulse_ids:
            if self._show_image is True:
                img = ImageItem(border='w')
                img.setLookupTable(COLOR_MAP.getLookupTable())
                self.image_items.append(img)

                vb = self._gl_widget.addViewBox(lockAspect=True)
                vb.addItem(img)

                line = self._gl_widget.addPlot()
            else:
                line = self._gl_widget.addPlot()

            line.setTitle("Pulse No. {:04d}".format(pulse_id))
            line.setLabel('left', Y_LABEL)
            if pulse_id == self._pulse_ids[-1]:
                # all plots share one x label
                line.setLabel('bottom', X_LABEL)
            else:
                line.setLabel('bottom', '')

            self.plot_items.append(line)
            self._gl_widget.nextRow()

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._gl_widget)
        self._cw.setLayout(layout)

    def update(self, data):
        for i, pulse_id in enumerate(self._pulse_ids):
            p = self.plot_items[i]
            if data is not None:
                p.plot(data.momentum, data.intensity[pulse_id],
                       name="origin",
                       pen=Pen.purple)

                ave = np.mean(data.intensity, axis=0)
                p.plot(data.momentum, ave, name="mean", pen=Pen.green)

                p.plot(data.momentum, data.intensity[pulse_id] - ave,
                       name="difference",
                       pen=Pen.yellow)

                if i == 0:
                    p.addLegend(offset=cfg.LINE_PLOT_LEGEND_OFFSET)

            if data is not None and self._show_image is True:
                self.image_items[i].setImage(data.image[pulse_id],
                                             autoLevels=False)


class LaserOnOffWindow(PlotWindow):
    def __init__(self,
                 window_id,
                 on_pulse_ids,
                 off_pulse_ids,
                 normalization_range,
                 fom_range, *,
                 parent=None,
                 title="FXE Azimuthal integration"):
        """Initialization."""
        super().__init__(window_id, parent=parent, title=title)

        self._on_pulse_ids = on_pulse_ids
        self._off_pulse_ids = off_pulse_ids

        # The No. of trains received
        self._count = 0
        # The x-data
        self._momentum = None
        # The average data
        self._on_pulse = None
        self._off_pulse = None
        # The history of integrated difference between on and off pulses
        self._fom_hist = []

        # *************************************************************
        # control parameters
        # *************************************************************
        self._diff_scale_sp = SpinBox(value=10)
        self._diff_scale_sp.setRange(1, 20)
        self._diff_scale_sp.setSingleStep(1)

        self._normalization_range = normalization_range
        self._fom_range = fom_range
        self._normalization_range_lri = LinearRegionItem(normalization_range)
        self._fom_range_lri = LinearRegionItem(fom_range)

        self._show_normalization_range_cb = QtGui.QCheckBox(
            "Show normalization range")
        self._show_normalization_range_cb.setChecked(False)
        self._show_fom_range_cb = QtGui.QCheckBox("Show FOM range")
        self._show_fom_range_cb.setChecked(False)

        self.initUI()

    def initUI(self):
        control_widget = self.initCtrlUI()
        self.initPlotUI()

        layout = QtGui.QHBoxLayout()
        layout.addWidget(control_widget, 1)
        layout.addWidget(self._gl_widget, 3)
        self._cw.setLayout(layout)

    def initPlotUI(self):
        self._gl_widget.setFixedSize(cfg.MA_PLOT_WIDTH, cfg.MA_PLOT_HEIGHT)

        self._gl_widget.addLabel(
            "On -pulse IDs: {}<br>Off-pulse IDs: {}".
            format(', '.join(str(i) for i in self._on_pulse_ids),
                   ', '.join(str(i) for i in self._off_pulse_ids)))

        self._gl_widget.nextRow()

        p1 = self._gl_widget.addPlot()
        self.plot_items.append(p1)
        p1.setLabel('left', Y_LABEL)
        p1.setLabel('bottom', X_LABEL)
        p1.setTitle(' ')

        self._gl_widget.nextRow()

        p2 = self._gl_widget.addPlot()
        self.plot_items.append(p2)
        p2.setLabel('left', "Integrated difference (arb.)")
        p2.setLabel('bottom', "Trains No.")
        p2.setTitle(' ')

    def initCtrlUI(self):
        scale_lb = QtGui.QLabel("Scale difference plot")
        self._diff_scale_sp.setFixedHeight(30)

        control_widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(scale_lb)
        layout.addWidget(self._diff_scale_sp)
        layout.addWidget(self._show_normalization_range_cb)
        layout.addWidget(self._show_fom_range_cb)
        layout.addStretch()
        control_widget.setLayout(layout)

        return control_widget

    def update(self, data):
        if data is None:
            return

        self._count += 1

        self._momentum = data.momentum
        if self._count == 1:
            self._on_pulse = \
                data.intensity[self._on_pulse_ids].mean(axis=0)
            self._off_pulse = \
                data.intensity[self._off_pulse_ids].mean(axis=0)
        else:
            self._on_pulse += \
                data.intensity[self._on_pulse_ids].mean(axis=0) \
                / self._count - self._on_pulse / (self._count - 1)
            self._off_pulse += \
                data.intensity[self._off_pulse_ids].mean(axis=0) \
                / self._count - self._off_pulse / (self._count - 1)

        # normalize curve

        normalized_on_pulse = self._on_pulse / integrate_curve(
            self._on_pulse, self._momentum, self._normalization_range)
        normalized_off_pulse = self._off_pulse / integrate_curve(
            self._off_pulse, self._momentum, self._fom_range)

        # update plots

        # upper plot
        p = self.plot_items[0]
        p.plot(self._momentum, normalized_on_pulse, name="On", pen=Pen.purple)

        p.plot(self._momentum, normalized_off_pulse, name="Off", pen=Pen.green)

        diff = normalized_on_pulse - normalized_off_pulse
        p.plot(self._momentum, self._diff_scale_sp.value() * diff,
               name="difference",
               pen=Pen.yellow)

        p.addLegend()

        # visualize normalization range
        if self._show_normalization_range_cb.isChecked():
            p.addItem(self._normalization_range_lri)
        # normalize FOM range
        if self._show_fom_range_cb.isChecked():
            p.addItem(self._fom_range_lri)

        # update history
        fom = sub_array_with_range(diff, self._momentum, self._fom_range)[0]
        self._fom_hist.append(np.sum(np.abs(fom)))

        # lower plot
        p = self.plot_items[1]
        p.clear()

        s = ScatterPlotItem(size=20, pen=mkPen(None),
                            brush=mkBrush(120, 255, 255, 255))
        s.addPoints([{'pos': (i, v), 'data': 1}
                     for i, v in enumerate(self._fom_hist)])
        p.addItem(s)

        p.plot(self._fom_hist, pen=Pen.yellow)

    def clear(self):
        """Overload.

        The history of FOM should be untouched when the new data is invalid.
        """
        self.plot_items[0].clear()
