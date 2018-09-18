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

from .pyqtgraph.Qt import QtCore, QtGui
from .pyqtgraph import (
    BarGraphItem, ColorMap, GraphicsLayoutWidget, ImageItem, intColor,
    LinearRegionItem, mkBrush, mkPen, ScatterPlotItem, SpinBox,
)
from .pyqtgraph.graphicsItems.GradientEditorItem import Gradients
from .config import Config as cfg
from .data_processing import integrate_curve, sub_array_with_range


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
        self._plot.setLabel('bottom', "Momentum transfer (1/A)")
        self._plot.setLabel('left', "Scattering signal (arb. u.)")

    def clear_(self):
        self._plot.clear()

    def update(self, data):
        momentum = data.momentum
        for i, intensity in enumerate(data.intensity):
            self._plot.plot(momentum, intensity,
                            pen=mkPen(intColor(i, hues=9, values=5), width=2))
        self._plot.setTitle("Train ID: {}, No. pulses: {}".
                            format(data.tid, len(data.intensity)))


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

    def update(self, data):
        self._img.setImage(np.flip(data.image_avg, axis=0), autoLevels=False)
        self._view.autoRange()


class PlotWindow(QtGui.QMainWindow):
    """Base class for various pop-out plot windows."""
    def __init__(self, window_id, parent=None):
        """Initialization."""
        super().__init__(parent=parent)

        self.setWindowTitle("FXE Azimuthal integration")
        self._id = window_id
        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

        self._gl_widget = GraphicsLayoutWidget()

        self._plot_items = []  # bookkeeping PlotItem objects
        self._image_items = []  # bookkeeping ImageItem objects

    @abc.abstractmethod
    def initUI(self):
        raise NotImplementedError

    @abc.abstractmethod
    def initCtrlUI(self):
        pass

    @abc.abstractmethod
    def initPlotUI(self):
        pass

    @abc.abstractmethod
    def update(self, data):
        """Update plots"""
        raise NotImplementedError

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        # avoid memory leak
        self.parent().remove_window(self._id)

    def clear(self):
        """Clear all plots in the window."""
        for item in self._plot_items:
            item.clear()
        for item in self._image_items:
            item.clear()


class IndividualPulseWindow(PlotWindow):
    plot_w = 800
    plot_h = 280
    max_plots = 4

    def __init__(self, window_id, pulse_ids, *, parent=None, show_image=False):
        """Initialization."""
        super().__init__(window_id, parent=parent)

        self._pulse_ids = pulse_ids
        self._show_image = show_image

        self.initUI()

    def initUI(self):
        layout = self._gl_widget.ci.layout
        layout.setColumnStretchFactor(0, 1)
        if self._show_image:
            layout.setColumnStretchFactor(1, 3)
        w = self.plot_w - self.plot_h + self._show_image*(self.plot_h - 20)
        h = min(self.max_plots, len(self._pulse_ids))*self.plot_h
        self._gl_widget.setFixedSize(w, h)

        count = 0
        for pulse_id in self._pulse_ids:
            count += 1
            if count > self.max_plots:
                break
            if self._show_image is True:
                img = ImageItem(border='w')
                img.setLookupTable(COLOR_MAP.getLookupTable())
                self._image_items.append(img)

                vb = self._gl_widget.addViewBox(lockAspect=True)
                vb.addItem(img)

                line = self._gl_widget.addPlot()
            else:
                line = self._gl_widget.addPlot()

            line.setTitle("Pulse No. {:04d}".format(pulse_id))
            line.setLabel('left', "Scattering signal (arb. u.)")
            if pulse_id == self._pulse_ids[-1]:
                # all plots share one x label
                line.setLabel('bottom', "Momentum transfer (1/A)")
            else:
                line.setLabel('bottom', '')

            self._plot_items.append(line)
            self._gl_widget.nextRow()

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._gl_widget)
        self._cw.setLayout(layout)

    def update(self, data):
        for i, pulse_id in enumerate(self._pulse_ids):
            p = self._plot_items[i]
            if i == 0:
                p.addLegend(offset=(-40, 20))

            if data is not None:
                p.plot(data.momentum, data.intensity[pulse_id],
                       name="origin",
                       pen=Pen.purple)

                ave = np.mean(data.intensity, axis=0)
                p.plot(data.momentum, ave, name="mean", pen=Pen.green)

                p.plot(data.momentum, data.intensity[pulse_id] - ave,
                       name="difference",
                       pen=Pen.yellow)

            if data is not None and self._show_image is True:
                self._image_items[i].setImage(
                    np.flip(data.image[pulse_id], axis=0), autoLevels=False)


class LaserOnOffWindow(PlotWindow):
    plot_w = 800
    plot_h = 450
    def __init__(self,
                 window_id,
                 on_pulse_ids,
                 off_pulse_ids,
                 normalization_range,
                 fom_range, *,
                 parent=None):
        """Initialization."""
        super().__init__(window_id, parent=parent)

        self._on_pulse_ids = on_pulse_ids
        self._off_pulse_ids = off_pulse_ids

        self._count = 0  # The number of trains received

        # The average data
        self._on_pulse = None
        self._off_pulse = None
        # The history of integrated difference between on and off pulses
        self._fom_hist = []

        # *************************************************************
        # control parameters
        # *************************************************************
        self._diff_scale_sp = SpinBox(value=10)
        self._diff_scale_sp.setRange(1, 100)
        self._diff_scale_sp.setSingleStep(1)

        self._normalization_range = normalization_range
        self._fom_range = fom_range

        pen1 = mkPen(QtGui.QColor(
            255, 255, 255, 255), width=1, style=QtCore.Qt.DashLine)
        brush1 = QtGui.QBrush(QtGui.QColor(0, 0, 255, 30))
        self._normalization_range_lri = LinearRegionItem(
            normalization_range, pen=pen1, brush=brush1, movable=False)
        self._normalization_range_cb = QtGui.QCheckBox("Normalization range (w)")
        self._normalization_range_cb.setChecked(True)

        pen2 = mkPen(QtGui.QColor(
            255, 0, 0, 255), width=1, style=QtCore.Qt.DashLine)
        brush2 = QtGui.QBrush(QtGui.QColor(0, 0, 255, 30))
        self._fom_range_lri = LinearRegionItem(
            fom_range, pen=pen2, brush=brush2, movable=False)
        self._fom_range_cb = QtGui.QCheckBox("FOM range (r)")
        self._fom_range_cb.setChecked(True)

        self.initUI()

    def initUI(self):
        control_widget = self.initCtrlUI()
        self.initPlotUI()

        layout = QtGui.QHBoxLayout()
        layout.addWidget(control_widget, 1)
        layout.addWidget(self._gl_widget, 3)
        self._cw.setLayout(layout)

    def initPlotUI(self):
        self._gl_widget.setFixedSize(self.plot_w, 2*self.plot_h)

        self._gl_widget.addLabel(
            "On -pulse IDs: {}<br>Off-pulse IDs: {}".
            format(', '.join(str(i) for i in self._on_pulse_ids),
                   ', '.join(str(i) for i in self._off_pulse_ids)))

        self._gl_widget.nextRow()

        p1 = self._gl_widget.addPlot()
        self._plot_items.append(p1)
        p1.setLabel('left', "Scattering signal (arb. u.)")
        p1.setLabel('bottom', "Momentum transfer (1/A)")
        p1.setTitle(' ')

        self._gl_widget.nextRow()

        p2 = self._gl_widget.addPlot()
        self._plot_items.append(p2)
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
        layout.addWidget(self._normalization_range_cb)
        layout.addWidget(self._fom_range_cb)
        layout.addStretch()
        control_widget.setLayout(layout)

        return control_widget

    def update(self, data):
        # TODO: think it twice about how to deal with None data here
        self._count += 1

        momentum = data.momentum
        if self._count == 1:
            # the first data point
            self._on_pulse = \
                data.intensity[self._on_pulse_ids].mean(axis=0)
            self._off_pulse = \
                data.intensity[self._off_pulse_ids].mean(axis=0)
        else:
            # apply moving average
            self._on_pulse += \
                data.intensity[self._on_pulse_ids].mean(axis=0) \
                / self._count - self._on_pulse / (self._count - 1)
            self._off_pulse += \
                data.intensity[self._off_pulse_ids].mean(axis=0) \
                / self._count - self._off_pulse / (self._count - 1)

        # normalize azimuthal integration curves
        normalized_on_pulse = self._on_pulse / integrate_curve(
            self._on_pulse, momentum, self._normalization_range)
        normalized_off_pulse = self._off_pulse / integrate_curve(
            self._off_pulse, momentum, self._normalization_range)

        # then calculate the difference between on- and off pulse curves
        diff = normalized_on_pulse - normalized_off_pulse

        # ------------
        # update plots
        # ------------

        # upper one
        # plot curves of on-, off- pulses and their difference
        p = self._plot_items[0]
        p.addLegend(offset=(-60, 20))
        p.plot(momentum, normalized_on_pulse, name="On", pen=Pen.purple)
        p.plot(momentum, normalized_off_pulse, name="Off", pen=Pen.green)
        p.plot(momentum, self._diff_scale_sp.value() * diff,
               name="difference",
               pen=Pen.yellow)

        # visualize normalization range
        if self._normalization_range_cb.isChecked():
            p.addItem(self._normalization_range_lri)
        # normalize FOM range
        if self._fom_range_cb.isChecked():
            p.addItem(self._fom_range_lri)

        # calculate figure-of-merit (FOM) and update history
        fom = sub_array_with_range(diff, momentum, self._fom_range)[0]
        self._fom_hist.append(np.sum(np.abs(fom)))

        # lower one
        # plot the evolution of fom
        s = ScatterPlotItem(size=20, pen=mkPen(None),
                            brush=mkBrush(120, 255, 255, 255))
        s.addPoints([{'pos': (i, v), 'data': 1}
                     for i, v in enumerate(self._fom_hist)])

        p = self._plot_items[1]
        p.clear()
        p.addItem(s)
        p.plot(self._fom_hist, pen=Pen.yellow)

    def clear(self):
        """Overload.

        The history of FOM should be untouched when the new data is invalid.
        """
        self._plot_items[0].clear()


class SanityCheckWindow(PlotWindow):
    plot_w = 800
    plot_h = 450
    def __init__(self, window_id, normalization_range, fom_range, *,
                 parent=None):
        """Initialization."""
        super().__init__(window_id, parent=parent)

        self._normalization_range = normalization_range
        self._fom_range = fom_range

        self.initUI()

    def initUI(self):
        self.initPlotUI()

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._gl_widget)
        self._cw.setLayout(layout)

    def initPlotUI(self):
        self._gl_widget.setFixedSize(self.plot_w, self.plot_h)

        self._gl_widget.nextRow()

        p = self._gl_widget.addPlot()
        self._plot_items.append(p)
        p.setLabel('left', "Integrated difference (arb.)")
        p.setLabel('bottom', "Pulse No.")
        p.setTitle(' ')

    def update(self, data):
        """Override."""
        momentum = data.momentum

        # normalize azimuthal integration curves for each pulse
        normalized_pulse_intensities = []
        for pulse_intensity in data.intensity:
            normalized = pulse_intensity / integrate_curve(
                pulse_intensity, momentum, self._normalization_range)
            normalized_pulse_intensities.append(normalized)

        # calculate the different between each pulse and the first one
        diffs = [p - normalized_pulse_intensities[0]
                 for p in normalized_pulse_intensities]

        # calculate the FOM for each pulse
        foms = []
        for diff in diffs:
            fom = sub_array_with_range(diff, momentum, self._fom_range)[0]
            foms.append(np.sum(np.abs(fom)))

        bar = BarGraphItem(x=range(len(foms)), height=foms, width=0.6, brush='b')

        p = self._plot_items[0]
        p.addItem(bar)
        p.setTitle("Train ID: {}".format(data.tid))
        p.plot()
