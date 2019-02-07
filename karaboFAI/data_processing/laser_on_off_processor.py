from collections import deque

import numpy as np

from ..data_processing.proc_utils import normalize_curve, slice_curve
from ..data_processing import OpLaserMode
from ..logger import logger


class LaserOnOffProcessor:
    """LaserOnOffProcessor class.

    A processor which calculated the moving average of the average of the
    azimuthal integration of all laser-on and laser-off pulses, as well
    as their difference. It also calculates the the figure of merit (FOM),
    which is integration of the absolute aforementioned difference.
    """
    def __init__(self):
        self.laser_mode = None
        self.on_pulse_ids = None
        self.off_pulse_ids = None

        self.moving_average_window = 1
        self.normalization_range = None
        self.integration_range = None

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

    def process(self, data):
        """Process the data

        :param ProcessedData data: data after the assembling and azimuthal
            integration for individual pulses.

        :return: (normalized moving average for on-pulses,
          normalized moving average for off-pulses)
        :rtype: (1D numpy.ndarray / None, 1D numpy.ndarray / None)
        """
        if self.laser_mode == OpLaserMode.INACTIVE:
            return

        momentum = data.momentum
        intensities = data.intensities

        n_pulses = intensities.shape[0]
        max_on_pulse_id = max(self.on_pulse_ids)
        if max_on_pulse_id >= n_pulses:
            logger.debug(f"On-pulse ID {max_on_pulse_id} out of range "
                         f"(0 - {n_pulses - 1})")
            return

        max_off_pulse_id = max(self.off_pulse_ids)
        if max_off_pulse_id >= n_pulses:
            logger.debug(f"Off-pulse ID {max_off_pulse_id} out of range "
                         f"(0 - {n_pulses - 1})")
            return

        if self.laser_mode == OpLaserMode.NORMAL:
            # compare laser-on/off pulses in the same train
            self._on_train_received = True
            self._off_train_received = True
        else:
            # compare laser-on/off pulses in different trains
            if self.laser_mode == OpLaserMode.NORMAL.EVEN_ON:
                flag = 0  # on-train has even train ID
            elif self.laser_mode == OpLaserMode.ODD_ON:
                flag = 1  # on-train has odd train ID
            else:
                logger.debug(f"Unexpected laser mode! {self.laser_mode}")
                return

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

        normalized_on_pulse = None
        normalized_off_pulse = None

        if self._on_train_received:
            # update on-pulse
            if self.laser_mode == OpLaserMode.NORMAL or not self._off_train_received:

                this_on_pulses = intensities[self.on_pulse_ids].mean(axis=0)

                if self._drop_last_on_pulse:
                    length = len(self._on_pulses_hist)
                    self._on_pulses_ma += \
                        (this_on_pulses - self._on_pulses_hist.pop()) / length
                    self._drop_last_on_pulse = False
                else:
                    if self._on_pulses_ma is None:
                        self._on_pulses_ma = np.copy(this_on_pulses)
                    elif len(self._on_pulses_hist) < self.moving_average_window:
                        self._on_pulses_ma += \
                                (this_on_pulses - self._on_pulses_ma) \
                                / (len(self._on_pulses_hist) + 1)
                    elif len(self._on_pulses_hist) == self.moving_average_window:
                        self._on_pulses_ma += \
                            (this_on_pulses - self._on_pulses_hist.popleft()) \
                            / self.moving_average_window
                    else:
                        raise ValueError  # should never reach here

                self._on_pulses_hist.append(this_on_pulses)

            normalized_on_pulse = normalize_curve(
                self._on_pulses_ma, momentum, *self.normalization_range)

        diff = None
        fom = None
        if self._off_train_received:
            # update off-pulse

            this_off_pulses = intensities[self.off_pulse_ids].mean(axis=0)

            self._off_pulses_hist.append(this_off_pulses)

            if self._off_pulses_ma is None:
                self._off_pulses_ma = np.copy(this_off_pulses)
            elif len(self._off_pulses_hist) <= self.moving_average_window:
                self._off_pulses_ma += \
                        (this_off_pulses - self._off_pulses_ma) \
                        / len(self._off_pulses_hist)
            elif len(self._off_pulses_hist) == self.moving_average_window + 1:
                self._off_pulses_ma += \
                    (this_off_pulses - self._off_pulses_hist.popleft()) \
                    / self.moving_average_window
            else:
                raise ValueError  # should never reach here

            normalized_off_pulse = normalize_curve(
                self._off_pulses_ma, momentum, *self.normalization_range)

            diff = normalized_on_pulse - normalized_off_pulse

            # calculate figure-of-merit and update history
            fom = slice_curve(diff, momentum, *self.integration_range)[0]
            fom = np.sum(np.abs(fom))

            # an extra check
            # TODO: check whether it is necessary
            if len(self._on_pulses_hist) != len(self._off_pulses_hist):
                raise ValueError("Length of on-pulse history {} != length "
                                 "of off-pulse history {}".
                                 format(len(self._on_pulses_hist),
                                        len(self._off_pulses_hist)))

            # reset flags
            self._on_train_received = False
            self._off_train_received = False

        data.on_off.on_pulse = normalized_on_pulse
        data.on_off.off_pulse = normalized_off_pulse
        data.on_off.diff = diff
        data.on_off.update_hist(data.tid, fom)

    def reset(self):
        """Override."""
        self._on_train_received = False
        self._off_train_received = False
        self._drop_last_on_pulse = False
        self._on_pulses_ma = None
        self._off_pulses_ma = None
        self._on_pulses_hist.clear()
        self._off_pulses_hist.clear()
