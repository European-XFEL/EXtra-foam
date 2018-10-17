"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold classes of various stand-alone windows.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque, OrderedDict

import numpy as np

import silx
from silx.gui.plot.MaskToolsWidget import MaskToolsWidget
from silx.gui.colors import Colormap as SilxColormap

from .pyqtgraph import (
    BarGraphItem, GraphicsLayoutWidget, ImageItem,
    LinearRegionItem, mkBrush, mkPen, QtCore, QtGui, ScatterPlotItem
)
from .pyqtgraph import parametertree as ptree

from ..logger import logger
from ..config import config
from ..data_processing.proc_utils import (
    normalize_curve, slice_curve
)
from .misc_widgets import PenFactory, lookupTableFactory


class SingletonWindow:
    def __init__(self, instance_type):
        self.instance = None
        self.instance_type = instance_type

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.instance_type(*args, **kwargs)
        else:
            try:
                self.instance.updatePlots()
            except AttributeError:
                pass
        return self.instance


class AbstractWindow(QtGui.QMainWindow):
    """Base class for various stand-alone windows.

    All the stand-alone windows should follow the interface defined
    in this abstract class.
    """
    def __init__(self, data, *, parent=None):
        """Initialization.

        :param Data4Visualization data: the data shared by widgets
            and windows.
        """
        super().__init__(parent=parent)
        self._data = data
        try:
            self.setWindowTitle(parent.title)
        except AttributeError:
            # for unit test where parent is None
            self.setWindowTitle("")

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

    def initUI(self):
        """Initialization of UI.

        This method should call 'initCtrlUI' and 'initPlotUI'.
        """
        pass

    def initCtrlUI(self):
        """Initialization of ctrl UI.

        Initialization of the ctrl UI should take place in this method.
        """
        pass

    def initPlotUI(self):
        """Initialization of plot UI.

        Initialization of the plot UI should take place in this method.
        """
        pass

    def updatePlots(self):
        """Update plots.

        This method is called by the main GUI.
        """
        raise NotImplementedError

    def clearPlots(self):
        """Clear plots.

        This method is called by the main GUI.
        """
        raise NotImplementedError

    def closeEvent(self, QCloseEvent):
        """Update the book-keeping in the main GUI."""
        super().closeEvent(QCloseEvent)
        self.parent().removeWindow(self)


class PlotWindow(AbstractWindow):
    """Base class for stand-alone windows."""
    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)

        self._gl_widget = GraphicsLayoutWidget()
        self._ctrl_widget = None

        self._plot_items = []  # bookkeeping PlotItem objects
        self._image_items = []  # bookkeeping ImageItem objects

    def initUI(self):
        """Override."""
        self.initCtrlUI()
        self.initPlotUI()

        layout = QtGui.QHBoxLayout()
        if self._ctrl_widget is not None:
            layout.addWidget(self._ctrl_widget)
        layout.addWidget(self._gl_widget)
        self._cw.setLayout(layout)

    def clearPlots(self):
        """Override."""
        for item in self._plot_items:
            item.clear()
        for item in self._image_items:
            item.clear()


class IndividualPulseWindow(PlotWindow):
    """IndividualPulseWindow class.

    A window which allows user to visualize the detector image and the
    azimuthal integration result of individual pulses. The azimuthal
    integration result is also compared with the average azimuthal
    integration of all the pulses. Visualization of the detector image
    is optional.
    """
    plot_w = 800
    plot_h = 280
    max_plots = 4

    def __init__(self, data, pulse_ids, *, parent=None, show_image=False):
        """Initialization."""
        super().__init__(data, parent=parent)

        self._pulse_ids = pulse_ids
        self._show_image = show_image

        self.initUI()
        self.updatePlots()

        logger.info("Open IndividualPulseWindow ({})".
                    format(", ".join(str(i) for i in pulse_ids)))

    def initPlotUI(self):
        """Override."""
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
                img.setLookupTable(lookupTableFactory[config["COLOR_MAP"]])
                self._image_items.append(img)

                vb = self._gl_widget.addViewBox(lockAspect=True)
                vb.addItem(img)

                line = self._gl_widget.addPlot()
            else:
                line = self._gl_widget.addPlot()

            line.setTitle("Pulse ID {:04d}".format(pulse_id))
            line.setLabel('left', "Scattering signal (arb. u.)")
            if pulse_id == self._pulse_ids[-1]:
                # all plots share one x label
                line.setLabel('bottom', "Momentum transfer (1/A)")
            else:
                line.setLabel('bottom', '')

            self._plot_items.append(line)
            self._gl_widget.nextRow()

    def updatePlots(self):
        """Override."""
        data = self._data.get()
        if data.empty():
            return

        for i, pulse_id in enumerate(self._pulse_ids):
            p = self._plot_items[i]
            if i == 0:
                p.addLegend(offset=(-40, 20))

            if data is not None:
                p.plot(data.momentum, data.intensity[pulse_id],
                       name="origin",
                       pen=PenFactory.purple)

                p.plot(data.momentum, data.intensity_mean,
                       name="mean",
                       pen=PenFactory.green)

                p.plot(data.momentum,
                       data.intensity[pulse_id] - data.intensity_mean,
                       name="difference",
                       pen=PenFactory.yellow)

            if data is not None and self._show_image is True:
                # in-place operation is faster
                np.clip(data.image[pulse_id],
                        config["MASK_RANGE"][0],
                        config["MASK_RANGE"][1],
                        data.image[pulse_id])
                self._image_items[i].setImage(
                    np.flip(data.image[pulse_id], axis=0))


class LaserOnOffWindow(PlotWindow):
    """LaserOnOffWindow class.

    A window which visualizes the moving average of the average of the
    azimuthal integration of all laser-on and laser-off pulses, as well
    as their difference, in the upper plot. It also visualize the
    evolution of the figure of merit (FOM), which is integration of the
    absolute difference between the moving average of the laser-on and
    laser-off results, for each pair of laser-on and laser-off trains,
    in the lower plot.
    """
    modes = OrderedDict({
        "normal": "Laser-on/off pulses in the same train",
        "even/odd": "Laser-on/off pulses in even/odd train",
        "odd/even": "Laser-on/off pulses in odd/even train"
    })

    plot_w = 800
    plot_h = 450

    def __init__(self,
                 data,
                 on_pulse_ids,
                 off_pulse_ids,
                 normalization_range,
                 fom_range,
                 laser_mode, *,
                 parent=None,
                 ma_window_size=9999):
        """Initialization."""
        super().__init__(data, parent=parent)

        self._ptree = ptree.ParameterTree(showHeader=True)
        params = [
            {'name': 'Experimental setups', 'type': 'group',
             'children': [
                {'name': 'Optical laser mode', 'type': 'str', 'readonly': True,
                 'value': self.modes[laser_mode]},
                {'name': 'Laser-on pulse ID(s)', 'type': 'str', 'readonly': True,
                 'value': ', '.join([str(x) for x in on_pulse_ids])},
                {'name': 'Laser-off pulse ID(s)', 'type': 'str', 'readonly': True,
                 'value': ', '.join([str(x) for x in  off_pulse_ids])}]},
            {'name': 'Data processing parameters', 'type': 'group',
             'children': [
                 {'name': 'Normalization range', 'type': 'str', 'readonly': True,
                  'value': ', '.join([str(x) for x in normalization_range])},
                 {'name': 'FOM range (1/A)', 'type': 'str', 'readonly': True,
                  'value': ', '.join([str(x) for x in fom_range])},
                 {'name': 'M.A. window size', 'type': 'int', 'readonly': True,
                  'value': ma_window_size}]},
            {'name': 'Visualization options', 'type': 'group',
             'children': [
                 {'name': 'Difference scale', 'type': 'int', 'value': 20},
                 {'name': 'Show normalization range', 'type': 'bool', 'value': False},
                 {'name': 'Show FOM range', 'type': 'bool', 'value': False}]},
            {'name': 'Actions', 'type': 'group',
             'children': [
                {'name': 'Clear history', 'type': 'action'}]},
        ]
        p = ptree.Parameter.create(name='params', type='group', children=params)
        self._ptree.setParameters(p, showTop=False)

        self._exp_setups = p.param('Experimental setups')

        self._vis_setups = p.param('Visualization options')

        self._proc_setups = p.param('Data processing parameters')

        p.param('Actions', 'Clear history').sigActivated.connect(self._reset)

        # for visualization of normalization range
        pen1 = mkPen(QtGui.QColor(
            255, 255, 255, 255), width=1, style=QtCore.Qt.DashLine)
        brush1 = QtGui.QBrush(QtGui.QColor(0, 0, 255, 30))
        self._normalization_range_lri = LinearRegionItem(
            normalization_range, pen=pen1, brush=brush1, movable=False)

        # for visualization of FOM range
        pen2 = mkPen(QtGui.QColor(
            255, 0, 0, 255), width=1, style=QtCore.Qt.DashLine)
        brush2 = QtGui.QBrush(QtGui.QColor(0, 0, 255, 30))
        self._fom_range_lri = LinearRegionItem(
            fom_range, pen=pen2, brush=brush2, movable=False)

        # -------------------------------------------------------------
        # parameters from the control panel of the main GUI
        # -------------------------------------------------------------
        self._laser_mode = laser_mode
        self._on_pulse_ids = on_pulse_ids
        self._off_pulse_ids = off_pulse_ids
        self._normalization_range = normalization_range
        self._fom_range = fom_range
        self._ma_window_size = ma_window_size

        # -------------------------------------------------------------
        # volatile parameters
        # -------------------------------------------------------------
        self._on_train_received = False
        self._off_train_received = False

        # if an on-pulse is followed by an on-pulse, drop the previous one
        self._drop_last_on_pulse = False

        # moving average
        self._on_pulses_ma = None
        self._off_pulses_ma = None
        # The histories of on/off pulses by train, which are used in
        # calculating moving average (MA)
        self._on_pulses_hist = deque()
        self._off_pulses_hist = deque()

        # The history of integrated difference (FOM) between on and off pulses
        self._fom_hist_train_id = []
        self._fom_hist = []

        self.initUI()
        self.updatePlots()

        logger.info("Open LaserOnOffWindow (on-pulse(s): {}, off-pulse(s): {})".
                    format(", ".join(str(i) for i in on_pulse_ids),
                           ", ".join(str(i) for i in off_pulse_ids)))

    def initPlotUI(self):
        """Override."""
        self._gl_widget.setFixedSize(self.plot_w, 2*self.plot_h)

        p1 = self._gl_widget.addPlot()
        self._plot_items.append(p1)
        p1.setLabel('left', "Scattering signal (arb. u.)")
        p1.setLabel('bottom', "Momentum transfer (1/A)")
        p1.setTitle(' ')

        self._gl_widget.nextRow()

        p2 = self._gl_widget.addPlot()
        self._plot_items.append(p2)
        p2.setLabel('left', "Integrated difference (arb.)")
        p2.setLabel('bottom', "Train ID")
        p2.setTitle(' ')

    def initCtrlUI(self):
        """Override."""
        self._ctrl_widget = QtGui.QWidget()
        self._ctrl_widget.setMinimumWidth(500)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._ptree)
        self._ctrl_widget.setLayout(layout)

    def _update(self, data):
        """Process incoming data and update history.

        :return: (normalized moving average for on-pulses,
                  normalized moving average for off-pulses)
        :rtype: (1D numpy.ndarray / None, 1D numpy.ndarray / None)
        """
        available_modes = list(self.modes.keys())
        if self._laser_mode == available_modes[0]:
            # compare laser-on/off pulses in the same train
            self._on_train_received = True
            self._off_train_received = True
        else:
            # compare laser-on/off pulses in different trains

            if self._laser_mode == available_modes[1]:
                flag = 0  # on-train has even train ID
            elif self._laser_mode == available_modes[2]:
                flag = 1  # on-train has odd train ID
            else:
                raise ValueError("Unknown laser mode!")

            # Off-train will only be acknowledged when an on-train
            # was received! This ensures that in the visualization
            # it always shows the on-train plot alone first, which
            # is followed by a combined plots if the next train is
            # an off-train pulse.
            if self._on_train_received:
                if data.tid % 2 == 1 ^ flag:
                    # an on-pulse is followed by an off-pulse
                    self._off_train_received = True
                else:
                    # an on-pulse is followed by an on-pulse
                    self._drop_last_on_pulse = True
            else:
                # an off-pulse is followed by an on-pulse
                if data.tid % 2 == flag:
                    self._on_train_received = True

        # update and plot

        momentum = data.momentum
        normalized_on_pulse = None
        normalized_off_pulse = None

        if self._on_train_received:
            # update on-pulse

            if self._laser_mode == available_modes[0] or \
                    not self._off_train_received:

                this_on_pulses = data.intensity[self._on_pulse_ids].mean(axis=0)
                if self._drop_last_on_pulse:
                    length = len(self._on_pulses_hist)
                    self._on_pulses_ma += \
                        (this_on_pulses - self._on_pulses_hist.pop()) / length
                    self._drop_last_on_pulse = False
                else:
                    if self._on_pulses_ma is None:
                        self._on_pulses_ma = np.copy(this_on_pulses)
                    elif len(self._on_pulses_hist) < self._ma_window_size:
                        self._on_pulses_ma += \
                                (this_on_pulses - self._on_pulses_ma) \
                                / (len(self._on_pulses_hist) + 1)
                    elif len(self._on_pulses_hist) == self._ma_window_size:
                        self._on_pulses_ma += \
                            (this_on_pulses - self._on_pulses_hist.popleft()) \
                            / self._ma_window_size
                    else:
                        raise ValueError  # should never reach here

                self._on_pulses_hist.append(this_on_pulses)

            normalized_on_pulse = normalize_curve(
                self._on_pulses_ma, momentum, *self._normalization_range)

        if self._off_train_received:
            # update off-pulse

            this_off_pulses = data.intensity[self._off_pulse_ids].mean(axis=0)
            self._off_pulses_hist.append(this_off_pulses)

            if self._off_pulses_ma is None:
                self._off_pulses_ma = np.copy(this_off_pulses)
            elif len(self._off_pulses_hist) <= self._ma_window_size:
                self._off_pulses_ma += \
                        (this_off_pulses - self._off_pulses_ma) \
                        / len(self._off_pulses_hist)
            elif len(self._off_pulses_hist) == self._ma_window_size + 1:
                self._off_pulses_ma += \
                    (this_off_pulses - self._off_pulses_hist.popleft()) \
                    / self._ma_window_size
            else:
                raise ValueError  # should never reach here

            normalized_off_pulse = normalize_curve(
                self._off_pulses_ma, momentum, *self._normalization_range)

            diff = normalized_on_pulse - normalized_off_pulse

            # calculate figure-of-merit (FOM) and update history
            fom = slice_curve(diff, momentum, *self._fom_range)[0]
            self._fom_hist.append(np.sum(np.abs(fom)))
            # always append the off-pulse id
            self._fom_hist_train_id.append(data.tid)

            # an extra check
            if len(self._on_pulses_hist) != len(self._off_pulses_hist):
                raise ValueError("Length of on-pulse history {} != length "
                                 "of off-pulse history {}".
                                 format(len(self._on_pulses_hist),
                                        len(self._off_pulses_hist)))

            # reset flags
            self._on_train_received = False
            self._off_train_received = False

        return normalized_on_pulse, normalized_off_pulse

    def updatePlots(self):
        """Override."""
        data = self._data.get()
        if data.empty():
            return

        normalized_on_pulse, normalized_off_pulse = self._update(data)

        momentum = data.momentum

        # upper plot
        p = self._plot_items[0]

        # visualize normalization/FOM range if requested
        if self._vis_setups.param("Show normalization range").value():
            p.addItem(self._normalization_range_lri)
        if self._vis_setups.param("Show FOM range").value():
            p.addItem(self._fom_range_lri)

        p.addLegend(offset=(-60, 20))

        p.setTitle("Train ID: {}".format(data.tid))
        if normalized_on_pulse is not None:
            # plot on-pulse
            p.plot(momentum, normalized_on_pulse,
                   name="On", pen=PenFactory.purple)

        if normalized_off_pulse is not None:
            assert normalized_on_pulse is not None

            # plot off-pulse
            p.plot(momentum, normalized_off_pulse,
                   name="Off", pen=PenFactory.green)

            # plot difference between on-/off- pulses
            diff_scale = self._vis_setups.param('Difference scale').value()
            p.plot(momentum,
                   diff_scale * (normalized_on_pulse - normalized_off_pulse),
                   name="difference", pen=PenFactory.yellow)

        # lower plot
        p = self._plot_items[1]
        p.clear()

        s = ScatterPlotItem(size=20,
                            pen=mkPen(None),
                            brush=mkBrush(120, 255, 255, 255))
        s.addPoints([{'pos': (i, v), 'data': 1} for i, v in
                     zip(self._fom_hist_train_id, self._fom_hist)])

        p.addItem(s)
        p.plot(self._fom_hist_train_id, self._fom_hist,
               pen=PenFactory.yellow)

    def clearPlots(self):
        """Override.

        The lower plot should be untouched when the new data cannot be
        used to update the history of FOM.
        """
        self._plot_items[0].clear()

    def _reset(self):
        """Clear history and internal states."""
        for p in self._plot_items:
            p.clear()

        self._on_train_received = False
        self._off_train_received = False
        self._drop_last_on_pulse = False
        self._on_pulses_ma = None
        self._off_pulses_ma = None
        self._on_pulses_hist.clear()
        self._off_pulses_hist.clear()
        self._fom_hist.clear()
        self._fom_hist_train_id.clear()


@SingletonWindow
class SampleDegradationMonitor(PlotWindow):
    """SampleDegradationMonitor class.

    A window which allows users to monitor the degradation of the sample
    within a train.
    """
    plot_w = 800
    plot_h = 450

    def __init__(self, data, normalization_range, fom_range, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)

        self._normalization_range = normalization_range
        self._fom_range = fom_range

        self.initUI()
        self.updatePlots()

        logger.info("Open SampleDegradationMonitor")

    def initPlotUI(self):
        """Override."""
        self._gl_widget.setFixedSize(self.plot_w, self.plot_h)

        self._gl_widget.nextRow()

        p = self._gl_widget.addPlot()
        self._plot_items.append(p)
        p.setLabel('left', "Integrated difference (arb.)")
        p.setLabel('bottom', "Pulse ID")
        p.setTitle(' ')

    def updatePlots(self):
        """Override."""
        data = self._data.get()
        if data.empty():
            return

        momentum = data.momentum

        # normalize azimuthal integration curves for each pulse
        normalized_pulse_intensities = []
        for pulse_intensity in data.intensity:
            normalized = normalize_curve(
                pulse_intensity, momentum, self._normalization_range)
            normalized_pulse_intensities.append(normalized)

        # calculate the different between each pulse and the first one
        diffs = [p - normalized_pulse_intensities[0]
                 for p in normalized_pulse_intensities]

        # calculate the FOM for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self._fom_range)[0]
            foms.append(np.sum(np.abs(fom)))

        bar = BarGraphItem(x=range(len(foms)), height=foms, width=0.6, brush='b')

        p = self._plot_items[0]
        p.addItem(bar)
        p.setTitle("Train ID: {}".format(data.tid))
        p.plot()


class BraggSpotsWindow(PlotWindow):
    """BraggSpotsWindow class."""
    def updatePlots(self):
        pass


@SingletonWindow
class DrawMaskWindow(AbstractWindow):
    """DrawMaskWindow class.

    A window which allows users to have a better visualization of the
    detector image and draw a mask for further azimuthal integration.
    The mask must be saved and then loaded in the main GUI manually.
    """
    def __init__(self, data, *, parent=None):
        super().__init__(data, parent=parent)

        self._image = silx.gui.plot.Plot2D()
        self._mask_panel = MaskToolsWidget(plot=self._image)

        self.initUI()
        self._updateImage()

        logger.info("Open DrawMaskWindow")

    def initUI(self):
        """Override."""
        self.initPlotUI()

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._image)
        layout.setStretch(0, 1)
        layout.addLayout(self.initCtrlUI())
        self._cw.setLayout(layout)

    def initPlotUI(self):
        """Override."""
        self._image.setKeepDataAspectRatio(True)
        self._image.setYAxisInverted(True)
        # normalization options: LINEAR or LOGARITHM
        self._image.setDefaultColormap(
            SilxColormap('viridis', normalization=SilxColormap.LINEAR))

    def initCtrlUI(self):
        """Override."""
        self._image.getMaskAction().setVisible(False)

        self._mask_panel.setDirection(QtGui.QBoxLayout.TopToBottom)
        self._mask_panel.setMultipleMasks("single")

        update_image_btn = QtGui.QPushButton("Update image")
        update_image_btn.clicked.connect(self._updateImage)
        update_image_btn.setMinimumHeight(60)

        ctrl_widget = QtGui.QVBoxLayout()
        ctrl_widget.addWidget(self._mask_panel)
        ctrl_widget.addWidget(update_image_btn)
        ctrl_widget.addStretch(1)

        return ctrl_widget

    def _updateImage(self):
        """For updating image manually."""
        data = self._data.get()
        if data.empty():
            return

        # TODO: apply the mask to data processing on the fly!
        # self._mask_panel.getSelectionMask()

        self._image.addImage(data.image_mean)

    def updatePlots(self):
        """Override."""
        pass

    def clearPlots(self):
        """Override"""
        pass
