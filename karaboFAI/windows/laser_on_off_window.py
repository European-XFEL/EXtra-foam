"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

LaserOnOffWindow.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque

import numpy as np

from ..widgets.pyqtgraph import mkBrush, mkPen, QtGui, ScatterPlotItem

from ..widgets.pyqtgraph import parametertree as ptree

from .base_window import PlotWindow
from ..logger import logger
from ..data_processing.proc_utils import normalize_curve, slice_curve
from ..widgets.misc_widgets import PenFactory


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
    plot_h = 320

    title = "optical laser on/off"

    def __init__(self, data, *, parent=None):
        """Initialization."""
        super().__init__(data, parent=parent)

        # -------------------------------------------------------------
        # connect signal and slot
        # -------------------------------------------------------------
        self.parent().analysis_ctrl_widget.mask_range_sgn.connect(
            self.onMaskRangeChanged)
        self.parent().analysis_ctrl_widget.on_off_pulse_ids_sgn.connect(
            self.onOffPulseIdChanged)
        self.parent().analysis_ctrl_widget.diff_integration_range_sgn.connect(
            self.onDiffIntegrationRangeChanged)
        self.parent().analysis_ctrl_widget.normalization_range_sgn.connect(
            self.onNormalizationRangeChanged)
        self.parent().analysis_ctrl_widget.ma_window_size_sgn.connect(
            self.onMAWindowSizeChanged)

        # tell MainGUI to emit signals in order to update shared parameters
        self.parent().updateSharedParameters()

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
        self.update()

        logger.info("Open LaserOnOffWindow")

    def initPlotUI(self):
        """Override."""
        self._gl_widget.setFixedSize(self.plot_w, 3*self.plot_h)

        # on- and off- pulses
        p1 = self._gl_widget.addPlot()
        self._plot_items.append(p1)
        p1.setLabel('left', "Scattering signal (arb. u.)")
        p1.setLabel('bottom', "Momentum transfer (1/A)")
        p1.setTitle('Moving average of on- and off- pulses')

        self._gl_widget.nextRow()

        # difference curve
        p2 = self._gl_widget.addPlot()
        self._plot_items.append(p2)
        p2.setLabel('left', "Scattering signal (arb. u.)")
        p2.setLabel('bottom', "Momentum transfer (1/A)")
        p2.setTitle("'On - Off'")

        self._gl_widget.nextRow()

        # history of integrated absolute difference
        p3 = self._gl_widget.addPlot()
        self._plot_items.append(p3)
        p3.setLabel('left', "Integrated abs. difference (arb.)")
        p3.setLabel('bottom', "Train ID")
        p3.setTitle("History of integrated absolute 'On - Off'")

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
            self.diff_integration_range_param,
            self.ma_window_size_param
        ])

        self._act_params.addChildren([
            self.reset_action_param
        ])

        params = ptree.Parameter.create(name='params', type='group',
                                        children=[self._exp_params,
                                                  self._pro_params,
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

            # calculate figure-of-merit and update history
            fom = slice_curve(diff, momentum, *self.diff_integration_range_sp)[0]
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

    def update(self):
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
        p1 = self._plot_items[0]

        p1.addLegend(offset=(-60, 20))

        if normalized_on_pulse is not None:
            # plot on-pulse
            p1.plot(momentum, normalized_on_pulse,
                    name="On", pen=PenFactory.purple)

        if normalized_off_pulse is not None:
            assert normalized_on_pulse is not None

            # plot off-pulse
            p1.plot(momentum, normalized_off_pulse,
                    name="Off", pen=PenFactory.green)

            # plot difference between on-/off- pulses
            p2 = self._plot_items[1]
            p2.clear()
            p2.plot(momentum,
                    (normalized_on_pulse - normalized_off_pulse),
                    name="on - off", pen=PenFactory.yellow)

        # lower plot
        p3 = self._plot_items[2]
        p3.clear()

        s = ScatterPlotItem(size=20,
                            pen=mkPen(None),
                            brush=mkBrush(120, 255, 255, 255))
        s.addPoints([{'pos': (i, v), 'data': 1} for i, v in
                     zip(self._fom_hist_train_id, self._fom_hist)])

        p3.addItem(s)
        p3.plot(self._fom_hist_train_id, self._fom_hist,
                pen=PenFactory.yellow)

    def clearPlots(self):
        """Override.

        The second and third plot should stay there if no valid data
        is received.
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