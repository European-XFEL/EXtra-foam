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
    mkBrush, mkPen, QtCore, QtGui, ScatterPlotItem
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

        bar = BarGraphItem(x=range(len(foms)), height=foms, width=0.6, brush='b')

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
