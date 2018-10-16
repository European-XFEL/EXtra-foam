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
from scipy import ndimage

from silx.gui.colors import Colormap as SilxColormap

from .pyqtgraph import (
    BarGraphItem, GraphicsLayoutWidget, ImageItem,
    LinearRegionItem, mkBrush, mkPen, QtCore, QtGui, ScatterPlotItem,
    RectROI,LineROI, LineSegmentROI
)
from .pyqtgraph import parametertree as ptree

from ..logger import logger
from ..config import config
from ..data_processing.proc_utils import (
    integrate_curve, sub_array_with_range
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

        self._cw = QtGui.QWidget()
        self.setCentralWidget(self._cw)

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

            normalized_on_pulse = self._on_pulses_ma / integrate_curve(
                self._on_pulses_ma, momentum, self._normalization_range)

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

            normalized_off_pulse = self._off_pulses_ma / integrate_curve(
                self._off_pulses_ma, momentum, self._normalization_range)

            diff = normalized_on_pulse - normalized_off_pulse

            # calculate figure-of-merit (FOM) and update history
            fom = sub_array_with_range(diff, momentum, self._fom_range)[0]
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


class BraggSpotsWindow(PlotWindow):

    modes = OrderedDict({
        "normal": "Laser-on/off pulses in the same train",
        "even/odd": "Laser-on/off pulses in even/odd train",
        "odd/even": "Laser-on/off pulses in odd/even train"
    })

    def __init__(self,
                 data,
                 on_pulse_ids,
                 off_pulse_ids,
                 laser_mode,
                 mask_range,
                 *,
                 parent=None
                 ):
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
                     'value': ', '.join([str(x) for x in off_pulse_ids])}]},

            {'name': 'Analysis options', 'type': 'group',
             'children': [
                 # {'name': 'COM Analysis', 'type': 'bool', 'value': True},
                 {'name': 'Profile Analysis', 'type': 'bool', 'value': False}

             ]},
            {'name': 'Actions', 'type': 'group',
             'children': [
                 {'name': 'Clear history', 'type': 'action'}]},
        ]
        p = ptree.Parameter.create(
            name='params', type='group', children=params)
        self._ptree.setParameters(p, showTop=False)
        self._vis_setups = p.param('Analysis options')
        p.param('Actions', 'Clear history').sigActivated.connect(self._reset)
        p.param('Analysis options', 'Profile Analysis').sigStateChanged.connect(
            self._profile)

        self.setGeometry(100, 100, 1400, 800)

        self._rois = []  # bookeeping Region of interests.
        self._on_pulse_ids = on_pulse_ids
        self._off_pulse_ids = off_pulse_ids
        self._laser_mode = laser_mode
        self._mask_range = mask_range
        self._hist_train_on_id = []
        self._hist_train_off_id = []
        self._hist_com_on = []
        self._hist_com_off = []

        self._profile_plot_items = []
        self._profile_line_rois = []

        self._on_train_received = False
        self._off_train_received = False

        self._drop_last_on_pulse = False

        self._on_pulses_ma = None
        self._off_pulses_ma = None

        self._on_pulses_hist = deque()
        self._off_pulses_hist = deque()

        self.initUI()
        self.updatePlots()

        logger.info("Open COM Analysis Window")

    def initCtrlUI(self):
        """Override"""
        self._ctrl_widget = QtGui.QWidget()
        self._ctrl_widget.setMaximumWidth(400)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._ptree)
        self._ctrl_widget.setLayout(layout)

    def initPlotUI(self):

        img = ImageItem(border='w')
        img.setLookupTable(lookupTableFactory[config['COLOR_MAP']])
        self._image_items.append(img)
        self._main_vb = self._gl_widget.addPlot(
            row=0, col=0, rowspan=2, colspan=2, lockAspect=True, enableMouse=False)
        self._main_vb.addItem(img)

        # Define First Region of interests.Around Brag Data
        roi = RectROI([config['CENTER_X'], config['CENTER_Y']], [
                      100, 100], pen=mkPen((0, 255, 0), width=3))

        self._rois.append(roi)
        # Define Second Region of interests.Around Background
        roi = RectROI([config['CENTER_X'] - 100, config['CENTER_Y'] -
                       100], [100, 100], pen=mkPen((255, 0, 0), width=3))
        self._rois.append(roi)

        for roi in self._rois:
            self._main_vb.addItem(roi)

        # View Boxes vb1 and vb2 in lower left panels for images in selected ROIs
        vb1 = self._gl_widget.addViewBox(
            row=2, col=0, rowspan=2, colspan=1,  lockAspect=True, enableMouse=False)
        img1 = ImageItem()
        img1.setLookupTable(lookupTableFactory[config['COLOR_MAP']])
        vb1.addItem(img1)
        self._image_items.append(img1)

        vb2 = self._gl_widget.addViewBox(
            row=2, col=1, rowspan=2, colspan=1,  lockAspect=True, enableMouse=False)
        img2 = ImageItem(border='w')
        img2.setLookupTable(lookupTableFactory[config['COLOR_MAP']])
        vb2.addItem(img2)
        self._image_items.append(img2)

        self._gl_widget.ci.layout.setColumnStretchFactor(2, 2)
        # Plot regions for COM moving averages and history over different trains
        p1 = self._gl_widget.addPlot(
            row=0, col=2, rowspan=2, colspan=2, lockAspect=True)
        self._plot_items.append(p1)
        p1.setLabel('left', "COM position")
        p1.setLabel('bottom', "Pulse ids")

        p2 = self._gl_widget.addPlot(row=2, col=2, rowspan=2, colspan=2)
        self._plot_items.append(p2)
        p2.setLabel('left', "Average COM")
        p2.setLabel('bottom', "Train ID")
        p2.setTitle(' ')

    def _update(self, data):

        # Same logic as LaserOnOffWindow.
        available_modes = list(self.modes.keys())
        if self._laser_mode == available_modes[0]:
            self._on_train_received = True
            self._off_train_received = True
        else:

            if self._laser_mode == available_modes[1]:
                flag = 0
            elif self._laser_mode == available_modes[2]:
                flag = 1
            else:
                raise ValueError("Unknown laser mode!")

            if self._on_train_received:
                if data.tid % 2 == 1 ^ flag:
                    self._off_train_received = True
                else:
                    self._drop_last_on_pulse = True
            else:
                if data.tid % 2 == flag:
                    self._on_train_received = True

        # slices dictionary is used to store array region selected by two ROIs around
        # brag data and background
        keys = ['brag_data', 'background_data']
        slices = dict.fromkeys(keys)

        com_on = None
        com_off = None
        max_count = 9999  # ma_windowsize to be included later from cntrl panel
        if self._on_train_received:

            if self._laser_mode == available_modes[0] or \
                    not self._off_train_received:

                this_on_pulses = []
                # Collects centre of mass for each pulse in this_on_pulses list
                for pid in self._on_pulse_ids:

                    index = 0
                    for key in slices.keys():
                        # slices of regions selected by two ROIs.
                        # One around brag spot and one around background
                        # key : brag_data stores array region around brag spot ROI
                        # key : background_data stores array region around background ROI

                        slices[key] = self._rois[index].getArrayRegion(
                            data.image[pid], self._image_items[0])
                        index += 1
                        (slices[key])[np.isnan(slices[key])] = - \
                            np.inf  # convert nan to -inf
                        np.clip(slices[key],
                                self._mask_range[0], self._mask_range[1], out=slices[key])
                        # clip to restrict between mask values 0-2500

                    # background subtraction from Brag_data. Resulting image to be used for
                    # COM evaluation.
                    mass_from_data = slices['brag_data'] - \
                        slices['background_data']
                    np.clip(mass_from_data,
                            self._mask_range[0], self._mask_range[1], out=mass_from_data)

                    mass = ndimage.measurements.center_of_mass(mass_from_data)
                    # centre of mass (x,y) converted to distance wrt some origin (0,0)
                    # r = sqrt(x**2+y**2)
                    r = np.linalg.norm(mass)
                    this_on_pulses.append(r)

                # Same logic as LaserOnOffWindow. Running averages over trains
                if self._drop_last_on_pulse:
                    length = len(self._on_pulses_hist)
                    self._on_pulses_ma += \
                        (this_on_pulses - self._on_pulses_hist.pop()) / length
                    self._drop_last_on_pulse = False
                else:
                    if self._on_pulses_ma is None:
                        self._on_pulses_ma = np.copy(this_on_pulses)
                    elif len(self._on_pulses_hist) < max_count:
                        self._on_pulses_ma += \
                            (this_on_pulses - self._on_pulses_ma) \
                            / (len(self._on_pulses_hist) + 1)
                    elif len(self._on_pulses_hist) == max_count + 1:
                        self._on_pulses_ma += \
                            (this_on_pulses - self._on_pulses_hist.popleft()) \
                            / max_count
                    else:
                        raise ValueError

                self._on_pulses_hist.append(this_on_pulses)

            com_on = self._on_pulses_ma

            # This part at the moment makes no physical sense. Atleast to me.
            # To be discussed with Dmitry. I added it here for some kind of
            # history book keeping
            self._hist_train_on_id.append(data.tid)
            self._hist_com_on.append(np.mean(np.array(com_on)))

        if self._off_train_received:

            this_off_pulses = []
            for pid in self._off_pulse_ids:

                index = 0
                for key in slices.keys():
                    # slices of regions selected by two ROIs.
                    # One around brag spot and one around background
                    # key : brag_data stores array region around brag spot ROI
                    # key : background stores array region around background ROI
                    slices[key] = self._rois[index].getArrayRegion(
                        data.image[pid], self._image_items[0])
                    index += 1
                    (slices[key])[np.isnan(slices[key])] = - \
                        np.inf  # convert nan to -inf
                    np.clip(slices[key],
                            self._mask_range[0], self._mask_range[1], out=slices[key])
                    # clip to restrict between mask values 0-2500

                # background subtraction from Brag_data. Resulting image to be used for
                # COM evaluation.
                mass_from_data = slices['brag_data'] - \
                    slices['background_data']

                np.clip(mass_from_data,
                        self._mask_range[0], self._mask_range[1], out=mass_from_data)

                mass = ndimage.measurements.center_of_mass(mass_from_data)
                # centre of mass (x,y) converted to distance wrt some origin (0,0)
                # r = sqrt(x**2+y**2)
                r = np.linalg.norm(mass)
                this_off_pulses.append(r)

            self._off_pulses_hist.append(this_off_pulses)
            # Same logic as LaserOnOffWindow. Running averages over trains
            if self._off_pulses_ma is None:
                self._off_pulses_ma = np.copy(this_off_pulses)
            elif len(self._off_pulses_hist) <= max_count:
                self._off_pulses_ma += \
                    (this_off_pulses - self._off_pulses_ma) \
                    / len(self._off_pulses_hist)
            elif len(self._off_pulses_hist) == max_count + 1:
                self._off_pulses_ma += \
                    (this_off_pulses - self._off_pulses_hist.popleft()) \
                    / max_count
            else:
                raise ValueError

            com_off = self._off_pulses_ma

            # This part at the moment makes no physical sense. Atleast to me.
            # To be discussed with Dmitry. I added it here for some kind of
            # history book keeping
            self._hist_train_off_id.append(data.tid)
            self._hist_com_off.append(np.mean(np.array(com_off)))

            self._on_train_received = False
            self._off_train_received = False

        return com_on, com_off

    def updatePlots(self):
        data = self._data.get()
        if data.empty():
            return
        self._main_vb.setMouseEnabled(x=False,y=False)
        self._image_items[0].setImage(
            np.flip(data.image_mean, axis=0), autoLevels=False, levels=(0, data.image_mean.max()))
        # Size of two region of interests should stay same.
        # Important when Backgorund has to be subtracted from Brag data
        # TOFIX: Size of ROI should not be independent
        size_brag = (self._rois[0]).size()
        self._rois[1].setSize(size_brag)

        # Profile analysis (Histogram) along a line
        # To ADD Here
        if self._vis_setups.param('Profile Analysis').value():

            if len(self._profile_line_rois) > 0:
                for line in self._profile_line_rois:
                    index = self._profile_line_rois.index(line)

                    slice_hist = line.getArrayRegion(
                        data.image_mean, self._image_items[0])
                    y, x = np.histogram(slice_hist, bins=np.linspace(
                        slice_hist.min(), slice_hist.max(), 50))
                    self._profile_plot_items[index].plot(
                        x, y, stepMode=True, fillLevel=0, brush=(255, 0, 255, 150))

        # Plot average image around two region of interests.
        # Selected Brag region and Background
        for roi in self._rois:
            index = self._rois.index(roi)
            self._image_items[index+1].setImage(roi.getArrayRegion(
                np.flip(data.image_mean, axis=0), self._image_items[0]), levels=(0, data.image_mean.max()))

        p = self._plot_items[0]

        com_on, com_off = self._update(data)

        p.addLegend()
        p.setTitle(' TrainId :: {}'.format(data.tid))
        if com_on is not None:
            p.plot(self._on_pulse_ids, com_on, name='On',
                   pen=PenFactory.green, symbol='o', symbolBrush=mkBrush(0, 255, 0, 255))
        if com_off is not None:
            p.plot(self._off_pulse_ids, com_off, name="Off",
                   pen=PenFactory.purple, symbol='o', symbolBrush=mkBrush(255, 0, 255, 255))

        p = self._plot_items[1]
        p.clear()

        s = ScatterPlotItem(size=10,
                            pen=mkPen(None),
                            brush=mkBrush(120, 255, 255, 255))
        s.addPoints([{'pos': (i, v), 'data': 1} for i, v in
                     zip(self._hist_train_off_id, self._hist_com_off)])

        p.addItem(s)
        s = ScatterPlotItem(size=10,
                            pen=mkPen(None),
                            brush=mkBrush(240, 255, 255, 255))
        s.addPoints([{'pos': (i, v), 'data': 1} for i, v in
                     zip(self._hist_train_on_id, self._hist_com_on)])

        p.addItem(s)
        p.plot(self._hist_train_off_id, self._hist_com_off,
               pen=PenFactory.red, name='Off')
        p.plot(self._hist_train_on_id, self._hist_com_on,
               pen=PenFactory.green, name='On')
        p.addLegend()

    def _profile(self):
        if self._vis_setups.param('Profile Analysis').value():
            self._gl_widget.ci.layout.setRowStretchFactor(0, 2)
            profile_plot = self._gl_widget.addPlot(
            row=4, col=0, rowspan=3, colspan=2)

            self._profile_plot_items.append(profile_plot)
            profile_plot = self._gl_widget.addPlot(
            row=4, col=2, rowspan=3, colspan=2)

            self._profile_plot_items.append(profile_plot)

            self._image_items[0].mouseClickEvent = self._click

        else:
            self._gl_widget.ci.layout.setRowStretchFactor(0, 1)
            if len(self._profile_plot_items) > 0:
                for item in self._profile_plot_items:
                    self._gl_widget.removeItem(item)
                self._profile_plot_items.clear()
            if len(self._profile_line_rois) > 0:
                for line in self._profile_line_rois:
                    self._main_vb.removeItem(line)
                self._profile_line_rois.clear()


    def _click(self,event):
        data = self._data.get()
        if data.empty():
            return

        event.accept()

        pos = event.pos()
        x = int(pos.x())
        y = int(pos.y())
        x_pos,y_pos = data.image_mean.shape

        if len(self._profile_line_rois) > 0:
            for line in self._profile_line_rois:
                self._main_vb.removeItem(line)
            self._profile_line_rois.clear()


        line_roi = LineSegmentROI([[0, y], [y_pos,y]], pen=mkPen((255, 255, 255), width=3))
        self._profile_line_rois.append(line_roi)

        line_roi = LineSegmentROI([[x, 0], [x,x_pos]], pen=mkPen((255, 255, 255), width=3))
        self._profile_line_rois.append(line_roi)
        for line in self._profile_line_rois:
            self._main_vb.addItem(line)

    def clearPlots(self):
        """Override."""
        for item in self._image_items:
            item.clear()
        self._plot_items[0].clear()
        if len(self._profile_plot_items) > 0:
            for plot in self._profile_plot_items:
                plot.clear()

    def _reset(self):
        self._plot_items[1].clear()

        self._on_train_received = False
        self._off_train_received = False
        self._drop_last_on_pulse = False
        self._on_pulses_ma = None
        self._off_pulses_ma = None
        self._on_pulses_hist.clear()
        self._off_pulses_hist.clear()
        # TODO: Fix hostory
        self._hist_com_on.clear()
        self._hist_com_off.clear()
        self._hist_train_on_id.clear()
        self._hist_train_off_id.clear()

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

@SingletonWindow
class DrawMaskWindow(AbstractWindow):
    """DrawMaskWindow class.

    A window which allows users to have a better visualization of the
    detector image and draw a mask for further azimuthal integration.
    The mask must be saved and then loaded in the main GUI manually.
    """
    def __init__(self, data, *, parent=None):
        super().__init__(data, parent=parent)

        from pyFAI.app.drawmask import MaskImageWidget

        self._cw = MaskImageWidget()
        self._cw._MaskImageWidget__plot2D.setYAxisInverted(True)
        # normalization options: LINEAR or LOGARITHM
        self._cw._MaskImageWidget__plot2D.setDefaultColormap(
            SilxColormap('viridis', normalization=SilxColormap.LINEAR))
        self.setCentralWidget(self._cw)
        self.updatePlots()

        logger.info("Open DrawMaskWindow")

    def updatePlots(self):
        """Override."""
        data = self._data.get()
        if data.empty():
            return

        self._cw.setImageData(data.image_mean)

    def clearPlots(self):
        """Override"""
        pass
