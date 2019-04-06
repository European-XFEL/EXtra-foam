"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Base classes in pipeline

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import IntEnum

from .exceptions import ProcessingError
from ..config import PumpProbeMode


class AbstractProcessor:
    """Base class for specific data processor."""

    def __init__(self):
        self.__enabled = True

        self.next = None  # next processor in the pipeline

    def setEnabled(self, state):
        self.__enabled = state

    def isEnabled(self):
        return self.__enabled

    def process(self, proc_data, raw_data=None):
        """Process data.

        :param ProcessedData proc_data: processed data.
        :param dict raw_data: raw data received from the bridge.

        :return str: error message.
        """
        raise NotImplementedError


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
