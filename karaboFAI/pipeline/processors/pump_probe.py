"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data processors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum

import numpy as np

from .base_processor import AbstractProcessor
from ..exceptions import ProcessingError
from ...algorithms import slice_curve
from ...config import PumpProbeMode


class BasePumpProbeProcessor(AbstractProcessor):
    """BasePumpProbeProcessor class.

    Base class for pump-probe processors.

    Attributes:
        mode (int): Pump-probe mode.
        on_pulse_ids (list): a list of laser-on pulse IDs.
        off_pulse_ids (list): a list of laser-off pulse IDs.
        _ma_window (int): moving average window size.
        _ma_count (int): moving average window count.
    """
    class State(IntEnum):
        ON_OFF = 1
        OFF_ON = 2
        ON_ON = 3
        OFF_OFF = 4

    def __init__(self):
        super().__init__()

        self.mode = None
        self.fom_type = None
        self.on_pulse_ids = None
        self.off_pulse_ids = None

        self._ma_window = 1
        self._ma_count = 0

        self._ma_on = None
        self._ma_off = None
        self._ma_on_off = None
        self._prev_on = None
        self._state = self.State.OFF_OFF

    @property
    def ma_window(self):
        return self._ma_window

    @ma_window.setter
    def ma_window(self, v):
        if not isinstance(v, int) or v < 0:
            v = 1

        if v < self._ma_window:
            # if the new window size is smaller than the current one,
            # we reset the moving average result
            self._ma_window = v
            self.reset()

        self._ma_window = v

    def process(self, proc_data, raw_data=None):
        """Override."""
        # TODO: implement
        pass

    def _pre_process(self, proc_data):
        if self.mode in (PumpProbeMode.PRE_DEFINED_OFF, PumpProbeMode.SAME_TRAIN):
            self._state = self.State.ON_OFF
        else:
            # compare laser-on/off pulses in different trains
            if self.mode == PumpProbeMode.EVEN_TRAIN_ON:
                flag = 0  # on-train has even train ID
            elif self.mode == PumpProbeMode.ODD_TRAIN_ON:
                flag = 1  # on-train has odd train ID
            else:
                raise ProcessingError(f"Unknown laser mode: {self.mode}")

            if proc_data.tid % 2 == 1 ^ flag:
                # off received
                if self._state in (self.State.OFF_ON, self.State.ON_ON):
                    self._state = self.State.ON_OFF
                else:
                    self._state = self.State.OFF_OFF
            else:
                # on received
                if self._state in (self.State.ON_OFF, self.State.OFF_OFF):
                    self._state = self.State.OFF_ON
                else:
                    self._state = self.State.ON_ON

    def reset(self):
        """Override."""
        self._ma_count = 0
        self._ma_on = None
        self._ma_off = None
        self._ma_on_off = None
        self._prev_on = None

        self._state = self.State.OFF_OFF


class PumpProbeProcessor(BasePumpProbeProcessor):
    """PumpProbeProcessor class.

    A processor which calculated the moving average of the average azimuthal
    integration of all pump/probe (on/off) pulses, as well as their difference.
    It also calculates the the figure of merit (FOM), which is integration
    of the absolute aforementioned difference.

    Attributes:
        abs_difference (bool): True for calculating the absolute value of
            difference between laser-on and laser-off.
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """
    def __init__(self):
        super().__init__()

        self.abs_difference = True
        self.fom_itgt_range = None

    def process(self, proc_data, raw_data=None):
        """Override."""
        on_pulse_ids = self.on_pulse_ids
        off_pulse_ids = self.off_pulse_ids
        momentum = proc_data.ai.momentum
        intensities = proc_data.ai.intensities
        ref_intensity = proc_data.ai.reference_intensity

        n_pulses = intensities.shape[0]
        max_on_pulse_id = max(on_pulse_ids)
        if max_on_pulse_id >= n_pulses:
            raise ProcessingError(f"On-pulse ID {max_on_pulse_id} out of range "
                                  f"(0 - {n_pulses - 1})")

        if self.mode != PumpProbeMode.PRE_DEFINED_OFF:
            max_off_pulse_id = max(off_pulse_ids)
            if max_off_pulse_id >= n_pulses:
                raise ProcessingError(f"Off-pulse ID {max_off_pulse_id} out of "
                                      f"range (0 - {n_pulses - 1})")

        self._pre_process(proc_data)

        # Off-train will only be acknowledged when an on-train
        # was received! This ensures that in the visualization
        # it always shows the on-train plot alone first, which
        # is followed by a combined plots if the next train is
        # an off-train pulse.

        fom = None

        if self._state in (self.State.ON_ON, self.State.OFF_ON):
            self._prev_on = intensities[on_pulse_ids].mean(axis=0)
        elif self._state == self.State.ON_OFF:
            if self._prev_on is None:
                this_on = intensities[on_pulse_ids].mean(axis=0)
            else:
                this_on = self._prev_on
                self._prev_on = None

            if self.mode == PumpProbeMode.PRE_DEFINED_OFF:
                this_off = ref_intensity
            else:
                this_off = intensities[off_pulse_ids].mean(axis=0)

            this_on_off = this_on - this_off

            if self.ma_window > 1 and self._ma_count > 0:
                if self._ma_count < self.ma_window:
                    self._ma_count += 1
                    denominator = self._ma_count
                else:  # self._ma_count == self._ma_window
                    # this is an approximation
                    denominator = self._ma_window

                self._ma_on += (this_on - self._ma_on) / denominator
                self._ma_off += (this_off - self._ma_off) / denominator
                self._ma_on_off += (this_on_off - self._ma_on_off) / denominator

            else:
                self._ma_on = this_on
                self._ma_off = this_off
                self._ma_on_off = this_on_off
                if self._ma_window > 1:
                    self._ma_count = 1  # 0 -> 1

            # calculate figure-of-merit and update history
            fom = slice_curve(
                self._ma_on_off, momentum, *self.fom_itgt_range)[0]
            if self.abs_difference:
                fom = np.sum(np.abs(fom))
            else:
                fom = np.sum(fom)

        # do nothing if self._state = self.State.OFF_OFF

        if fom is not None:
            proc_data.ai.on_intensity_mean = self._ma_on
            proc_data.ai.off_intensity_mean = self._ma_off
            proc_data.ai.on_off_intensity_mean = self._ma_on_off
            proc_data.ai.on_off_fom = (proc_data.tid, fom)

            proc_data.pp.on_data = self._ma_on
            proc_data.pp.off_data = self._ma_off
            proc_data.pp.on_off_data = self._ma_on_off
            proc_data.pp.fom = (proc_data.tid, fom)
        else:
            proc_data.ai.on_intensity_mean = self._prev_on

            proc_data.pp.on_data = self._prev_on
