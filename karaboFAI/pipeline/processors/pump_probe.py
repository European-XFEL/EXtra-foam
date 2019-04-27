"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeProcessors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum

import numpy as np

from .base_processor import LeafProcessor
from ..exceptions import ProcessingError
from ...algorithms import slice_curve
from ...config import PumpProbeMode, PumpProbeType
from ...helpers import profiler


class _BasePumpProbeProcessor(LeafProcessor):
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
        """Representation of the previous/current pulse type."""
        OFF_OFF = 0  # 0b00
        OFF_ON = 1  # 0b01
        ON_OFF = 2  # 0b10
        ON_ON = 3  # 0b11

    def __init__(self, scheduler):
        super().__init__(scheduler)

        self.mode = None
        self.analysis_type = None
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

    def run(self, processed, raw=None):
        """Override."""
        # TODO: implement
        pass

    def _pre_process(self, processed):
        n_pulses = processed.n_pulses

        max_on_pulse_id = max(self.on_pulse_ids)
        if max_on_pulse_id >= n_pulses:
            raise ProcessingError(
                f"Out of range: on-pulse ID = {max_on_pulse_id}, "
                f"total number of pulses = {n_pulses}")

        if self.mode != PumpProbeMode.PRE_DEFINED_OFF:
            max_off_pulse_id = max(self.off_pulse_ids)
            if max_off_pulse_id >= n_pulses:
                raise ProcessingError(
                    f"Out of range: off-pulse ID = {max_off_pulse_id}, "
                    f"total number of pulses = {n_pulses}")

        if self.mode in (
                PumpProbeMode.PRE_DEFINED_OFF, PumpProbeMode.SAME_TRAIN):
            self._state = self.State.ON_OFF
        else:
            # compare laser-on/off pulses in different trains
            if self.mode == PumpProbeMode.EVEN_TRAIN_ON:
                flag = 0  # on-train has even train ID
            elif self.mode == PumpProbeMode.ODD_TRAIN_ON:
                flag = 1  # on-train has odd train ID
            else:
                raise ProcessingError(f"Unknown pump-probe mode: {self.mode}")

            if processed.tid % 2 == 1 ^ flag:
                # off received
                if self._is_previous_on():
                    self._state = self.State.ON_OFF
                else:
                    self._state = self.State.OFF_OFF
            else:
                # on received
                if self._is_previous_on():
                    self._state = self.State.ON_ON
                else:
                    self._state = self.State.OFF_ON

    def _is_previous_on(self):
        return 0b01 & self._state == 1

    def reset(self):
        """Override."""
        self._ma_count = 0
        self._ma_on = None
        self._ma_off = None
        self._ma_on_off = None
        self._prev_on = None

        self._state = self.State.OFF_OFF


class PumpProbeProcessorFactory:

    class PumpProbeAiProcessor(_BasePumpProbeProcessor):
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

        def __init__(self, scheduler):
            super().__init__(scheduler)
            self.abs_difference = True
            self.fom_itgt_range = None

        @profiler("Pump-probe processor")
        def run(self, processed, raw=None):
            """Override."""
            if self.mode == PumpProbeMode.UNDEFINED:
                return

            self._pre_process(processed)

            on_pulse_ids = self.on_pulse_ids
            off_pulse_ids = self.off_pulse_ids
            momentum = processed.ai.momentum
            intensities = processed.ai.intensities
            ref_intensity = processed.ai.reference_intensity

            if momentum is None:
                raise ProcessingError(
                    "Azimuthal integration result is not available")

            # Off-train will only be acknowledged when an on-train
            # was received! This ensures that in the visualization
            # it always shows the on-train plot alone first, which
            # is followed by a combined plots if the next train is
            # an off-train pulse.

            fom = None

            if self._is_previous_on():
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
                processed.pp.on_data = self._ma_on
                processed.pp.off_data = self._ma_off
                processed.pp.on_off_data = self._ma_on_off
                processed.pp.fom = (processed.tid, fom)
            else:
                processed.pp.on_data = self._prev_on

    class PumpProbeRoiProcessor(_BasePumpProbeProcessor):
        def __init__(self, scheduler):
            super().__init__(scheduler)

    class PumpProbe1DProjProcessor(_BasePumpProbeProcessor):
        def __init__(self, scheduler, diret='x'):
            super().__init__(scheduler)

            diret = diret.lower()
            if diret not in ('x', 'y'):
                raise ValueError(
                    f"Not understandable projection direction: {diret}")
            self._direction = diret

    @classmethod
    def create(cls, analysis_type, scheduler=None):
        if analysis_type == PumpProbeType.AZIMUTHAL_INTEGRATION:
            return cls.PumpProbeAiProcessor(scheduler)

        if analysis_type == PumpProbeType.ROI:
            return cls.PumpProbeRoiProcessor(scheduler)

        if analysis_type == PumpProbeType.ROI_PROJECTION_X:
            return cls.PumpProbe1DProjProcessor(scheduler, 'x')

        if analysis_type == PumpProbeType.ROI_PROJECTION_X:
            return cls.PumpProbe1DProjProcessor(scheduler, 'y')

        raise NotImplementedError(
            f"Unknown pump-probe analysis type: {analysis_type}!")
