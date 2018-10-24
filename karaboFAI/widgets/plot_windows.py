"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold classes of various stand-alone windows.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time

from collections import deque, OrderedDict

import numpy as np
from scipy import ndimage

import silx
from silx.gui.plot.MaskToolsWidget import MaskToolsWidget
from silx.gui.colors import Colormap as SilxColormap

from .pyqtgraph import (
    BarGraphItem, GraphicsLayoutWidget, ImageItem,
    LineSegmentROI, mkBrush, mkPen, QtCore, QtGui, RectROI,
    ScatterPlotItem
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
                self.instance.show()
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

        self.show()

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


class PlotWindow(AbstractWindow):
    """Base class for stand-alone windows."""

    available_modes = OrderedDict({
        "normal": "Laser-on/off pulses in the same train",
        "even/odd": "Laser-on/off pulses in even/odd train",
        "odd/even": "Laser-on/off pulses in odd/even train"
    })

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)
        self.parent().registerPlotWidget(self)

        self._gl_widget = GraphicsLayoutWidget()
        self._ctrl_widget = None

        self._plot_items = []  # bookkeeping PlotItem objects
        self._image_items = []  # bookkeeping ImageItem objects

        # -------------------------------------------------------------
        # define parameter tree
        # -------------------------------------------------------------

        self._ptree = ptree.ParameterTree(showHeader=True)

        # parameters are grouped into 4 groups
        self._exp_params = ptree.Parameter.create(
            name='Experimental setups', type='group')
        self._pro_params = ptree.Parameter.create(
            name='Data processing parameters', type='group')
        self._vis_params = ptree.Parameter.create(
            name='Visualization options', type='group')
        self._act_params = ptree.Parameter.create(
            name='Actions', type='group')
        self._ana_params = ptree.Parameter.create(
            name='Analysis options', type='group')
        self._ins_params = ptree.Parameter.create(
            name='General', type='group')

        # -------------------------------------------------------------
        # define slots' behaviors
        # -------------------------------------------------------------

        # shared parameters are updated by signal-slot
        # Note: shared parameters should end with '_sp'
        self.mask_range_sp = None
        self.fom_range_sp = None
        self.normalization_range_sp = None
        self.ma_window_size_sp = None
        self.laser_mode_sp = None
        self.on_pulse_ids_sp = None
        self.off_pulse_ids_sp = None

        self.parent().mask_range_sgn.connect(self.onMaskRangeChanged)
        self.parent().on_off_pulse_ids_sgn.connect(self.onOffPulseIdChanged)
        self.parent().fom_range_sgn.connect(self.onFomRangeChanged)
        self.parent().normalization_range_sgn.connect(
            self.onNormalizationRangeChanged)
        self.parent().ma_window_size_sgn.connect(self.onMAWindowSizeChanged)

        # -------------------------------------------------------------
        # available Parameters (shared parameters and actions)
        # -------------------------------------------------------------

        self.mask_range_param = ptree.Parameter.create(
            name='Mask range', type='str', readonly=True
        )
        self.optical_laser_mode_param = ptree.Parameter.create(
            name='Optical laser mode', type='str', readonly=True
        )
        self.laser_on_pulse_ids_param = ptree.Parameter.create(
            name='Laser-on pulse ID(s)', type='str', readonly=True
        )
        self.laser_off_pulse_ids_param = ptree.Parameter.create(
            name='Laser-off pulse ID(s)', type='str', readonly=True
        )
        self.normalization_range_param = ptree.Parameter.create(
            name="Normalization range", type='str', readonly=True
        )
        self.fom_range_param = ptree.Parameter.create(
            name="FOM range", type='str', readonly=True
        )
        self.ma_window_size_param = ptree.Parameter.create(
            name='M.A. window size', type='int', readonly=True
        )
        self.reset_action_param = ptree.Parameter.create(
            name='Clear history', type='action'
        )
        self.reset_action_param.sigActivated.connect(self._reset)

        # this method inject parameters into the parameter tree
        self.updateParameterTree()

        # tell MainGUI to emit signals in order to update shared parameters
        self.parent().updateSharedParameters()

    def initUI(self):
        """Override."""
        self.initCtrlUI()
        self.initPlotUI()

        layout = QtGui.QHBoxLayout()
        if self._ctrl_widget is not None:
            layout.addWidget(self._ctrl_widget)
        layout.addWidget(self._gl_widget)
        self._cw.setLayout(layout)

    def updatePlots(self):
        """Update plots.

        This method is called by the main GUI.
        """
        raise NotImplementedError

    def clearPlots(self):
        """Clear plots.

        This method is called by the main GUI.
        """
        for item in self._plot_items:
            item.clear()
        for item in self._image_items:
            item.clear()

    @QtCore.pyqtSlot(str, list, list)
    def onOffPulseIdChanged(self, mode, on_pulse_ids, off_pulse_ids):
        self.laser_mode_sp = mode
        self.on_pulse_ids_sp = on_pulse_ids
        self.off_pulse_ids_sp = off_pulse_ids
        # then update the parameter tree
        try:
            self._exp_params.child('Optical laser mode').setValue(
                self.available_modes[mode])
            self._exp_params.child('Laser-on pulse ID(s)').setValue(
                ', '.join([str(x) for x in on_pulse_ids]))
            self._exp_params.child('Laser-off pulse ID(s)').setValue(
                ', '.join([str(x) for x in off_pulse_ids]))
        except KeyError:
            pass

    @QtCore.pyqtSlot(float, float)
    def onMaskRangeChanged(self, lb, ub):
        self.mask_range_sp = (lb, ub)
        # then update the parameter tree
        try:
            self._pro_params.child('Mask range').setValue(
                '{}, {}'.format(lb, ub))
        except KeyError:
            pass

    @QtCore.pyqtSlot(float, float)
    def onNormalizationRangeChanged(self, lb, ub):
        self.normalization_range_sp = (lb, ub)
        # then update the parameter tree
        try:
            self._pro_params.child('Normalization range').setValue(
                '{}, {}'.format(lb, ub))
        except KeyError:
            pass

    @QtCore.pyqtSlot(float, float)
    def onFomRangeChanged(self, lb, ub):
        self.fom_range_sp = (lb, ub)
        # then update the parameter tree
        try:
            self._pro_params.child('FOM range').setValue(
                '{}, {}'.format(lb, ub))
        except KeyError:
            pass

    @QtCore.pyqtSlot(int)
    def onMAWindowSizeChanged(self, value):
        self.ma_window_size_sp = value
        # then update the parameter tree
        try:
            self._pro_params.child('M.A. window size').setValue(str(value))
        except KeyError:
            pass

    def updateParameterTree(self):
        """Update the parameter tree.

        In this method, one should and only should have codes like

        self._exp_params.addChildren(...)
        self._pro_params.addChildren(...)
        self._vis_params.addChildren(...)
        self._act_params.addChildren(...)

        params = ptree.Parameter.create(name='params', type='group',
                                        children=[self._exp_params,
                                                  self._pro_params,
                                                  self._vis_params,
                                                  self._act_params])

        self._ptree.setParameters(params, showTop=False)

        Here '...' is a list of Parameter instances or dictionaries which
        can be used to instantiate Parameter instances.
        """
        pass

    def _reset(self):
        """Reset all internal states/histories."""
        pass

    def closeEvent(self, QCloseEvent):
        super().closeEvent(QCloseEvent)
        self.parent().unregisterPlotWidget(self)


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
            if pulse_id >= data.intensity.shape[0]:
                logger.error("Pulse ID {} out of range (0 - {})!".
                             format(pulse_id, data.intensity.shape[0] - 1))
                continue

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
                        self.mask_range_sp[0],
                        self.mask_range_sp[1],
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
    plot_w = 800
    plot_h = 450

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)

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

        logger.info("Open LaserOnOffWindow")

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

    def updateParameterTree(self):
        """Override."""
        self._exp_params.addChildren([
            self.optical_laser_mode_param,
            self.laser_on_pulse_ids_param,
            self.laser_off_pulse_ids_param
        ])

        self._pro_params.addChildren([
            self.normalization_range_param,
            self.fom_range_param,
            self.ma_window_size_param
        ])

        self._vis_params.addChildren([
            {'name': 'Difference scale', 'type': 'int', 'value': 20}
        ])

        self._act_params.addChildren([
            self.reset_action_param
        ])

        params = ptree.Parameter.create(name='params', type='group',
                                        children=[self._exp_params,
                                                  self._pro_params,
                                                  self._vis_params,
                                                  self._act_params])

        self._ptree.setParameters(params, showTop=False)

    def _update(self, data):
        """Process incoming data and update history.

        :return: (normalized moving average for on-pulses,
                  normalized moving average for off-pulses)
        :rtype: (1D numpy.ndarray / None, 1D numpy.ndarray / None)
        """
        available_modes = list(self.available_modes.keys())
        if self.laser_mode_sp == available_modes[0]:
            # compare laser-on/off pulses in the same train
            self._on_train_received = True
            self._off_train_received = True
        else:
            # compare laser-on/off pulses in different trains

            if self.laser_mode_sp == available_modes[1]:
                flag = 0  # on-train has even train ID
            elif self.laser_mode_sp == available_modes[2]:
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

            if self.laser_mode_sp == available_modes[0] or \
                    not self._off_train_received:

                this_on_pulses = data.intensity[self.on_pulse_ids_sp].mean(axis=0)
                if self._drop_last_on_pulse:
                    length = len(self._on_pulses_hist)
                    self._on_pulses_ma += \
                        (this_on_pulses - self._on_pulses_hist.pop()) / length
                    self._drop_last_on_pulse = False
                else:
                    if self._on_pulses_ma is None:
                        self._on_pulses_ma = np.copy(this_on_pulses)
                    elif len(self._on_pulses_hist) < self.ma_window_size_sp:
                        self._on_pulses_ma += \
                                (this_on_pulses - self._on_pulses_ma) \
                                / (len(self._on_pulses_hist) + 1)
                    elif len(self._on_pulses_hist) == self.ma_window_size_sp:
                        self._on_pulses_ma += \
                            (this_on_pulses - self._on_pulses_hist.popleft()) \
                            / self.ma_window_size_sp
                    else:
                        raise ValueError  # should never reach here

                self._on_pulses_hist.append(this_on_pulses)

            normalized_on_pulse = normalize_curve(
                self._on_pulses_ma, momentum, *self.normalization_range_sp)

        if self._off_train_received:
            # update off-pulse

            this_off_pulses = data.intensity[self.off_pulse_ids_sp].mean(axis=0)
            self._off_pulses_hist.append(this_off_pulses)

            if self._off_pulses_ma is None:
                self._off_pulses_ma = np.copy(this_off_pulses)
            elif len(self._off_pulses_hist) <= self.ma_window_size_sp:
                self._off_pulses_ma += \
                        (this_off_pulses - self._off_pulses_ma) \
                        / len(self._off_pulses_hist)
            elif len(self._off_pulses_hist) == self.ma_window_size_sp + 1:
                self._off_pulses_ma += \
                    (this_off_pulses - self._off_pulses_hist.popleft()) \
                    / self.ma_window_size_sp
            else:
                raise ValueError  # should never reach here

            normalized_off_pulse = normalize_curve(
                self._off_pulses_ma, momentum, *self.normalization_range_sp)

            diff = normalized_on_pulse - normalized_off_pulse

            # calculate figure-of-merit (FOM) and update history
            fom = slice_curve(diff, momentum, *self.fom_range_sp)[0]
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

        if max(self.on_pulse_ids_sp) > data.intensity.shape[0]:
            logger.error("On-pulse ID {} out of range (0 - {})".
                         format(max(self.on_pulse_ids_sp),
                                data.intensity.shape[0] - 1))
            return

        if max(self.off_pulse_ids_sp) > data.intensity.shape[0]:
            logger.error("Off-pulse ID {} out of range (0 - {})".
                         format(max(self.off_pulse_ids_sp),
                                data.intensity.shape[0] - 1))
            return

        normalized_on_pulse, normalized_off_pulse = self._update(data)

        momentum = data.momentum

        # upper plot
        p = self._plot_items[0]

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
            diff_scale = self._vis_params.child('Difference scale').value()
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
        """Override."""
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
    ''' BraggSpotsClass:
    
    This window is used to visualize the moving average of the position
    of the centre of mass for the user selected region of interest. 
    User can drag the scalable ROI around the Bragg spot in the image
    on top left corner of the window. A second ROI is also provided for
    background subtraction. Two plots on top right corner shows the X 
    and Y coordinates of the centre of mass for the selected region as
    a function of On and Off pulseIds provided by the users from 
    control panel in the main window. Botton left images are the zoomed-
    in images of the selcted regions. Bottom right plot analyse the 
    pulsed averaged X and Y coordinates of centre of mass with trainIds.

    There is also an option for profile analysis of the image. 
    By checking in the option of "Profile Analysis", user 
    can click anywhere in the image and two histograms will appear in 
    the bottom that provides profile of image along the horizontal and 
    vertical line segments passing through the position where mouse 
    click event happened.

    Another option "Normalized Intensity Plot" when checked in, replaces
    the Moving average plot on top right corner with the normalized 
    intensity plot of the region of interest.
        I = \\sum (ROI_bragg - ROI_background)/(\\sum ROI_background)     

    Author : Ebad Kamil
    Email  : ebad.kamil@xfel.eu
    '''
    instructions = \
        ("Green ROI: Place it around Bragg peak.\n\n"
         "White ROI: Place it around Background.\n\n"
         "Scale the Green ROI using handle on top right corner.\n\n"
         "To analyse the profile of image check the Profile "
         "analysis box and then click on the image on top-left corner.\n\n"
         "Always Clear History when ROIs positions are changed or parameters " 
         "in the control panel in the main-gui are modified."
         )

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)

        self.setGeometry(100, 100, 1600, 1000)

        self._rois = []  # bookkeeping Region of interests.
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

    def updateParameterTree(self):
        """Override."""
        self._exp_params.addChildren([
            self.optical_laser_mode_param,
            self.laser_on_pulse_ids_param,
            self.laser_off_pulse_ids_param
        ])

        self._pro_params.addChildren([
            self.ma_window_size_param
        ])

        self._ana_params.addChildren([
           {'name': 'Profile Analysis', 'type': 'bool', 'value': False},
           {'name': 'Normalized Intensity Plot',
                     'type': 'bool', 'value': False}

        ])

        self._act_params.addChildren([
            self.reset_action_param
        ])

        self._ins_params.addChildren([
           {'name': 'Instructions', 'type': 'text', 'readonly': True,
                  'value': self.instructions}
        ])

        params = ptree.Parameter.create(name='params', type='group',
                                        children=[self._exp_params,
                                                  self._pro_params,
                                                  self._ana_params,
                                                  self._act_params,
                                                  self._ins_params])
        # Profile check button needed to avoid clash while moving
        # brad and background region of interests. Click based.
        self._ana_params.child('Profile Analysis').sigStateChanged.connect(
            self._profile)
        self._ana_params.child('Normalized Intensity Plot').\
            sigStateChanged.connect(self._intensity)

        self._ptree.setParameters(params, showTop=False)

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
            row=0, col=0, rowspan=2, colspan=2, 
            lockAspect=True, enableMouse=False)
        self._main_vb.addItem(img)

        data = self._data.get()
        if data.empty():
            # Define First Region of interests.Around Brag Data
            roi = RectROI([config['CENTER_X'], config['CENTER_Y']], [50, 50], 
                          pen=mkPen((0, 255, 0), width=2))
            self._rois.append(roi)
            # Define Second Region of interests.Around Background
            roi = RectROI([config['CENTER_X']-100, config['CENTER_Y']-100], 
                          [50, 50], pen=mkPen((255, 255, 255), width=2))
            self._rois.append(roi)
        else:
            centre_x, centre_y = data.image_mean.shape
            # Define First Region of interests.Around Brag Data
            # Max Bounds for region of interest defined
            roi = RectROI([int(centre_x/2), int(centre_y/2)], [50, 50], 
                           maxBounds=QtCore.QRectF(0, 0, centre_y, centre_x), 
                           pen=mkPen((0, 255, 0), width=2))
            self._rois.append(roi)
            # Define Second Region of interests.Around Background
            roi = RectROI([int(centre_x/2)-100, int(centre_y/2)-100],[50, 50], 
                          maxBounds=QtCore.QRectF(0, 0, centre_y, centre_x),
                          pen=mkPen((255, 255, 255), width=2))
            self._rois.append(roi)

        for roi in self._rois:
            self._main_vb.addItem(roi)

        for handle in self._rois[1].getHandles():
            self._rois[1].removeHandle(handle)

        # View Boxes vb1 and vb2 in lower left panels for images in 
        # selected ROIs
        vb1 = self._gl_widget.addViewBox(row=2, col=0, rowspan=2, colspan=1,  
                                         lockAspect=True, enableMouse=False)
        img1 = ImageItem()
        img1.setLookupTable(lookupTableFactory[config['COLOR_MAP']])
        vb1.addItem(img1)
        self._image_items.append(img1)

        vb2 = self._gl_widget.addViewBox(row=2, col=1, rowspan=2, colspan=1, 
                                         lockAspect=True, enableMouse=False)
        img2 = ImageItem(border='w')
        img2.setLookupTable(lookupTableFactory[config['COLOR_MAP']])
        vb2.addItem(img2)
        self._image_items.append(img2)

        self._gl_widget.ci.layout.setColumnStretchFactor(2, 2)
        
        # Plot regions for COM moving averages and history over 
        # different trains
        p = self._gl_widget.addPlot(
            row=0, col=2, rowspan=1, colspan=2, lockAspect=True)
        self._plot_items.append(p)
        p.setLabel('left',  "<span style='text-decoration: overline'>R</span>\
            <sub>x</sub>")

        p = self._gl_widget.addPlot(
            row=1, col=2, rowspan=1, colspan=2, lockAspect=True)
        self._plot_items.append(p)
        p.setLabel('left', "<span style='text-decoration: overline'>R</span>\
            <sub>y</sub>")
        p.setLabel('bottom', "Pulse ids")

        p = self._gl_widget.addPlot(
            row=2, col=2, rowspan=1, colspan=2, lockAspect=True)
        self._plot_items.append(p)
        p.setLabel('left', '&lt;<span style="text-decoration:\
            overline">R</span><sub>x</sub>&gt;<sub>pulse-avg</sub>')
        p.setTitle(' ')

        p = self._gl_widget.addPlot(
            row=3, col=2, rowspan=1, colspan=2, lockAspect=True)
        self._plot_items.append(p)
        p.setLabel('left',  "&lt;<span style='text-decoration:\
            overline'>R</span><sub>y</sub>&gt;<sub>pulse-avg</sub>")
        p.setLabel('bottom', "Train ID")
        p.setTitle(' ')

    def _update(self, data):

        # Same logic as LaserOnOffWindow.
        available_modes = list(self.available_modes.keys())
        if self.laser_mode_sp == available_modes[0]:
            self._on_train_received = True
            self._off_train_received = True
        else:

            if self.laser_mode_sp == available_modes[1]:
                flag = 0
            elif self.laser_mode_sp == available_modes[2]:
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

        # slices dictionary is used to store array region selected by 
        # two ROIs around brag data and background
        keys = ['brag_data', 'background_data']
        slices = dict.fromkeys(keys)

        com_on = None
        com_off = None
        if self._on_train_received:

            if self.laser_mode_sp == available_modes[0] or \
                    not self._off_train_received:

                this_on_pulses = []
                # Collects centre of mass for each pulse in 
                # this_on_pulses list
                for pid in self.on_pulse_ids_sp:
                    if pid >= data.image.shape[0]:
                        logger.error("Pulse ID {} out of range (0 - {})!".
                                     format(pid, data.image.shape[0] - 1))
                        continue
                    index = 0
                    for key in slices.keys():
                        # slices of regions selected by two ROIs.
                        # One around brag spot and one around background
                        # key : brag_data stores array region around 
                        #       brag spot ROI
                        # key : background_data stores array region 
                        #       around background ROI

                        slices[key] = self._rois[index].getArrayRegion(
                            data.image[pid], self._image_items[0])
                        index += 1
                        (slices[key])[np.isnan(slices[key])] = - \
                            np.inf  # convert nan to -inf
                        np.clip(slices[key],self.mask_range_sp[0], 
                                self.mask_range_sp[1], out=slices[key])
                        # clip to restrict between mask values 0-2500

                    # background subtraction from Brag_data. 
                    # Resulting image to be used for COM evaluation.
                    mass_from_data = slices['brag_data'] - \
                        slices['background_data']
                    np.clip(mass_from_data, self.mask_range_sp[0], 
                            self.mask_range_sp[1], out=mass_from_data)
                    # normalization = \sum ROI_background
                    # Ńormalized intensity: 
                    # \sum (ROI_brag - ROI_background)/ normalization
                    intensity = np.sum(
                        mass_from_data/np.sum(slices['background_data']))
                    # Centre of mass
                    mass = ndimage.measurements.center_of_mass(mass_from_data)

                    this_on_pulses.append(np.append(np.array(mass), intensity))

                this_on_pulses = np.array(this_on_pulses)
                # Same logic as LaserOnOffWindow. Running averages over 
                # trains.
                if self._drop_last_on_pulse:
                    length = len(self._on_pulses_hist)
                    self._on_pulses_ma += \
                        (this_on_pulses - self._on_pulses_hist.pop()) / length
                    self._drop_last_on_pulse = False
                else:
                    if self._on_pulses_ma is None:
                        self._on_pulses_ma = np.copy(this_on_pulses)
                    elif len(self._on_pulses_hist) < self.ma_window_size_sp:
                        self._on_pulses_ma += \
                            (this_on_pulses - self._on_pulses_ma) \
                            / (len(self._on_pulses_hist) + 1)
                    elif len(self._on_pulses_hist) == self.ma_window_size_sp:
                        self._on_pulses_ma += \
                            (this_on_pulses - self._on_pulses_hist.popleft()) \
                            / self.ma_window_size_sp
                    else:
                        raise ValueError

                self._on_pulses_hist.append(this_on_pulses)

            com_on = self._on_pulses_ma

            # This part at the moment makes no physical sense. 
            # Atleast to me. To be discussed with Dmitry. I added it 
            # here for some kind of history book keeping
            self._hist_train_on_id.append(data.tid)
            self._hist_com_on.append(np.mean(np.array(com_on), axis=0))

        if self._off_train_received:

            this_off_pulses = []
            for pid in self.off_pulse_ids_sp:
                if pid > data.image.shape[0]-1:
                    logger.error("Pulse ID {} out of range (0 - {})!".
                                 format(pid, data.image.shape[0] - 1))
                    continue
                index = 0
                for key in slices.keys():
                    # slices of regions selected by two ROIs.
                    # One around brag spot and one around background
                    # key : brag_data stores array region around brag 
                    #       spot ROI
                    # key : background stores array region around 
                    #       background ROI
                    slices[key] = self._rois[index].getArrayRegion(
                        data.image[pid], self._image_items[0])
                    index += 1
                    (slices[key])[np.isnan(slices[key])] = - \
                        np.inf  # convert nan to -inf
                    np.clip(slices[key], self.mask_range_sp[0], 
                            self.mask_range_sp[1], out=slices[key])
                    # clip to restrict between mask values 0-2500

                # background subtraction from Brag_data. Resulting image
                # to be used for COM evaluation.
                mass_from_data = slices['brag_data'] - \
                    slices['background_data']

                np.clip(mass_from_data, self.mask_range_sp[0],
                        self.mask_range_sp[1], out=mass_from_data)
                # normalization = \sum ROI_background
                # Ńormalized intensity:
                # \sum (ROI_brag - ROI_background)/ normalization
                intensity = np.sum(
                    mass_from_data/np.sum(slices['background_data']))
                # Centre of mass
                mass = ndimage.measurements.center_of_mass(mass_from_data)

                this_off_pulses.append(np.append(np.array(mass), intensity))

            this_off_pulses = np.array(this_off_pulses)
            self._off_pulses_hist.append(this_off_pulses)
            # Same logic as LaserOnOffWindow. Running averages over 
            # trains.
            if self._off_pulses_ma is None:
                self._off_pulses_ma = np.copy(this_off_pulses)
            elif len(self._off_pulses_hist) <= self.ma_window_size_sp:
                self._off_pulses_ma += \
                    (this_off_pulses - self._off_pulses_ma) \
                    / len(self._off_pulses_hist)
            elif len(self._off_pulses_hist) == self.ma_window_size_sp + 1:
                self._off_pulses_ma += \
                    (this_off_pulses - self._off_pulses_hist.popleft()) \
                    / self.ma_window_size_sp
            else:
                raise ValueError

            com_off = self._off_pulses_ma

            # This part at the moment makes no physical sense. Atleast to me.
            # To be discussed with Dmitry. I added it here for some kind of
            # history book keeping
            self._hist_train_off_id.append(data.tid)
            self._hist_com_off.append(np.mean(np.array(com_off), axis=0))

            self._on_train_received = False
            self._off_train_received = False
        return com_on, com_off

    def updatePlots(self):
        data = self._data.get()
        if data.empty():
            return
        self._main_vb.setMouseEnabled(x=False, y=False)
        self._image_items[0].setImage(np.flip(data.image_mean, axis=0),
                                      autoLevels=False, 
                                      levels=(0, data.image_mean.max()))
        # Size of two region of interests should stay same.
        # Important when Backgorund has to be subtracted from Brag data
        # TODO: Size of ROI should not be independent
        size_brag = (self._rois[0]).size()
        self._rois[1].setSize(size_brag)

        # Profile analysis (Histogram) along a line
        # Horizontal and vertical line region of interests
        # Histograms along these lines plotted in the bottom panel
        if self._ana_params.child('Profile Analysis').value():

            if len(self._profile_line_rois) > 0:
                for line in self._profile_line_rois:
                    index = self._profile_line_rois.index(line)

                    slice_hist = line.getArrayRegion(
                        data.image_mean, self._image_items[0])
                    y, x = np.histogram(slice_hist, bins=np.linspace(
                        slice_hist.min(), slice_hist.max(), 50))
                    self._profile_plot_items[index].plot(
                        x, y, stepMode=True, fillLevel=0, 
                        brush=(255, 0, 255, 150))


        # Plot average image around two region of interests.
        # Selected Brag region and Background
        for roi in self._rois:
            index = self._rois.index(roi)
            self._image_items[index+1].setImage(roi.getArrayRegion(
                np.flip(data.image_mean, axis=0), 
                self._image_items[0]), levels=(0, data.image_mean.max()))
        # com_on and com_off are of shape (num_pulses,3)
        # contains (pulse_index, com_x, com_y, normalized intensity)
        t0 = time.perf_counter()
        
        com_on, com_off = self._update(data)

        logger.debug("Time for centre of mass evaluation: {:.1f} ms\n"
                     .format(1000 * (time.perf_counter() - t0)))
        # If Normalized intensity plot Checkbox is not checked then
        # just plot COM X and Y as a function of pulseIds
        if not self._ana_params.child('Normalized Intensity Plot').value():
            for p in self._plot_items[:-2]:
                index = self._plot_items.index(p)
                p.addLegend()
                if index == 0:
                    p.setTitle(' TrainId :: {}'.format(data.tid))
                if com_on is not None:
                    p.plot(self.on_pulse_ids_sp[0:com_on.shape[0]], 
                           com_on[:, index], name='On',pen=PenFactory.green, 
                           symbol='o', symbolBrush=mkBrush(0, 255, 0, 255))
                if com_off is not None:
                    p.plot(self.off_pulse_ids_sp[0:com_off.shape[0]], 
                           com_off[:, index], name="Off",pen=PenFactory.purple, 
                           symbol='o', symbolBrush=mkBrush(255, 0, 255, 255))
        # Else plot Normalized intensity.
        else:
            p = self._plot_items[0]
            p.setTitle(' TrainId :: {}'.format(data.tid))
            if com_on is not None:
                p.plot(self.on_pulse_ids_sp[0:com_on.shape[0]], 
                       com_on[:, -1], name='On', pen=PenFactory.green, 
                       symbol='o', symbolBrush=mkBrush(0, 255, 0, 255))
            if com_off is not None:
                p.plot(self.off_pulse_ids_sp[0:com_off.shape[0]], 
                       com_off[:, -1], name="Off", pen=PenFactory.purple, 
                       symbol='o', symbolBrush=mkBrush(255, 0, 255, 255))

        idx = 0
        for p in self._plot_items[-2:]:
            p.clear()
            if self._hist_com_off:
                s = ScatterPlotItem(size=10,
                                    pen=mkPen(None),
                                    brush=mkBrush(120, 255, 255, 255))
                s.addPoints([{'pos': (i, v), 'data': 1} for i, v in
                             zip(self._hist_train_off_id,
                                 np.array(self._hist_com_off)[:, idx])])

                p.addItem(s)
                p.plot(self._hist_train_off_id, 
                       np.array(self._hist_com_off)[:, idx],
                       pen=PenFactory.purple, name='Off')
            if self._hist_com_on:
                s = ScatterPlotItem(size=10,
                                    pen=mkPen(None),
                                    brush=mkBrush(240, 255, 255, 255))
                s.addPoints([{'pos': (i, v), 'data': 1} for i, v in
                             zip(self._hist_train_on_id,
                                 np.array(self._hist_com_on)[:, idx])])

                p.addItem(s)

                p.plot(self._hist_train_on_id,
                       np.array(self._hist_com_on)[:, idx],
                       pen=PenFactory.green, name='On')
                p.addLegend()
            idx += 1

    # Profile state change triggers this function
    # If profile is checked, adds bottom panels to plot histograms.
    def _profile(self):
        if self._ana_params.child('Profile Analysis').value():

            self._gl_widget.ci.layout.setRowStretchFactor(0, 2)
            self._gl_widget.ci.layout.setRowStretchFactor(1, 2)
            profile_plot = self._gl_widget.addPlot(
                row=4, col=0, rowspan=3, colspan=2)

            self._profile_plot_items.append(profile_plot)
            profile_plot = self._gl_widget.addPlot(
                row=4, col=2, rowspan=3, colspan=2)
            self._gl_widget.ci.layout.setRowStretchFactor(4, 2)

            self._profile_plot_items.append(profile_plot)

            self._image_items[0].mouseClickEvent = self._click

        else:
            self._gl_widget.ci.layout.setRowStretchFactor(0, 1)
            self._gl_widget.ci.layout.setRowStretchFactor(1, 1)

            if len(self._profile_plot_items) > 0:
                for item in self._profile_plot_items:
                    self._gl_widget.removeItem(item)
                self._profile_plot_items.clear()
            if len(self._profile_line_rois) > 0:
                for line in self._profile_line_rois:
                    self._main_vb.removeItem(line)
                self._profile_line_rois.clear()

    # Mouse click on image in top left panel creates two line
    # region of interests. One horizontal and one vertical.
    def _click(self, event):
        data = self._data.get()
        if data.empty():
            return

        event.accept()

        pos = event.pos()
        x = int(pos.x())
        y = int(pos.y())
        x_pos, y_pos = data.image_mean.shape

        if len(self._profile_line_rois) > 0:
            for line in self._profile_line_rois:
                self._main_vb.removeItem(line)
            self._profile_line_rois.clear()

        line_roi = LineSegmentROI(
            [[0, y], [y_pos, y]], 
            pen=mkPen((255, 255, 255), width=3), movable=False)
        self._profile_line_rois.append(line_roi)

        line_roi = LineSegmentROI(
            [[x, 0], [x, x_pos]], 
            pen=mkPen((255, 255, 255), width=3), movable=False)
        self._profile_line_rois.append(line_roi)
        for line in self._profile_line_rois:
            self._main_vb.addItem(line)

        if self._ana_params.child('Profile Analysis').value():
            for line in self._profile_line_rois:
                index = self._profile_line_rois.index(line)
                self._profile_plot_items[index].clear()

                slice_hist = line.getArrayRegion(
                    data.image_mean, self._image_items[0])
                y, x = np.histogram(slice_hist, bins=np.linspace(
                    slice_hist.min(), slice_hist.max(), 50))
                self._profile_plot_items[index].plot(
                    x, y, stepMode=True, fillLevel=0, 
                    brush=(255, 0, 255, 150))

    # Normalized intensity plot. When state changes in the checkbox
    # it removes Centre of Mass X and Y plots and replace it with
    # intensity plot.

    def _intensity(self):
        if self._ana_params.child('Normalized Intensity Plot').value():
            for plot in self._plot_items[:-2]:
                self._gl_widget.removeItem(plot)
                self._plot_items.remove(plot)

            p = self._gl_widget.addPlot(
                row=0, col=2, rowspan=2, colspan=2, lockAspect=True)
            self._plot_items.insert(0, p)
            p.setLabel('left', "Intensity")
            p.setLabel('bottom', "Pulse ids")
        else:
            for plot in self._plot_items[:-2]:
                self._gl_widget.removeItem(plot)
                self._plot_items.remove(plot)

            p = self._gl_widget.addPlot(
                row=0, col=2, rowspan=1, colspan=2, lockAspect=True)
            self._plot_items.insert(0, p)
            p.setLabel('left',"<span style='text-decoration: overline'>R</span>\
            <sub>x</sub>")

            p = self._gl_widget.addPlot(
                row=1, col=2, rowspan=1, colspan=2, lockAspect=True)
            self._plot_items.insert(1, p)
            p.setLabel('left',"<span style='text-decoration: overline'>R</span>\
            <sub>x</sub>")
            p.setLabel('bottom', "Pulse ids")

    def clearPlots(self):
        """Override."""
        for item in self._image_items:
            item.clear()
        for plot in self._plot_items[:-2]:
            plot.clear()
        if len(self._profile_plot_items) > 0:
            for plot in self._profile_plot_items:
                plot.clear()

    def _reset(self):
        for plot in self._plot_items[-2:]:
            plot.clear()

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

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)

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

    def initCtrlUI(self):
        """Override."""
        self._ctrl_widget = QtGui.QWidget()
        self._ctrl_widget.setMinimumWidth(250)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._ptree)
        self._ctrl_widget.setLayout(layout)

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
                pulse_intensity, momentum, *self.normalization_range_sp)
            normalized_pulse_intensities.append(normalized)

        # calculate the different between each pulse and the first one
        diffs = [p - normalized_pulse_intensities[0]
                 for p in normalized_pulse_intensities]

        # calculate the FOM for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self.fom_range_sp)[0]
            foms.append(np.sum(np.abs(fom)))

        bar = BarGraphItem(
            x=range(len(foms)), height=foms, width=0.6, brush='b')

        p = self._plot_items[0]
        p.addItem(bar)
        p.setTitle("Train ID: {}".format(data.tid))
        p.plot()

    def updateParameterTree(self):
        """Override."""
        self._pro_params.addChildren([
            self.normalization_range_param,
            self.fom_range_param,
        ])

        params = ptree.Parameter.create(name='params', type='group',
                                        children=[self._pro_params])

        self._ptree.setParameters(params, showTop=False)


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
