"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data model for analysis and visualization.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from enum import IntEnum

import numpy as np

from ..logger import logger


class DataSource(IntEnum):
    CALIBRATED_FILE = 0  # calibrated data from files
    CALIBRATED = 1  # calibrated data from Karabo-bridge
    ASSEMBLED = 2  # assembled data from Karabo-bridge
    PROCESSED = 3  # processed data from the Middle-layer device


class ProcessedData:
    """A class which stores the processed data.

    Attributes:
        tid (int): train ID.
        momentum (numpy.ndarray): x-axis of azimuthal integration result.
            Shape = (momentum,)
        intensity (numpy.ndarray): y-axis of azimuthal integration result.
            Shape = (pulse_id, intensity)
        intensity_mean (numpy.ndarray): average of the y-axis of azimuthal
            integration result over pulses. Shape = (intensity,)
        image (numpy.ndarray): assembled images for all the pulses.
            Shape = (pulse_id, y, x)
        image_mean (numpy.ndarray): average of the assembled images over
            pulses. Shape = (y, x)
    """
    def __init__(self, tid, *, momentum=None, intensity=None, assembled=None):
        """Initialization."""
        t0 = time.perf_counter()

        if not isinstance(tid, int):
            raise ValueError("Train ID must be an integer!")
        # tid is not allowed to be modified once initialized.
        self._tid = tid

        self.momentum = momentum
        self.intensity = intensity
        self.intensity_mean = None
        if self.intensity is not None:
            self.intensity_mean = np.mean(intensity, axis=0)

        self.image = None
        self.image_mean = None
        if assembled is not None:
            self.image = assembled
            self.image_mean = np.nanmean(assembled, axis=0)

        logger.debug("Time for pre-processing: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

    @property
    def tid(self):
        return self._tid

    def empty(self):
        """Check the goodness of the data."""
        if self.intensity is None or self.momentum is None \
                or self.image is None:
            return True
        return False
