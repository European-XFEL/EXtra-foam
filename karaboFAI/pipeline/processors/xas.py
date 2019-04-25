"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

XasProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import AbstractProcessor
from ..exceptions import ProcessingError
from ...algorithms import compute_spectrum


class XasProcessor(AbstractProcessor):
    """XasProcessor class.

    A processor which calculate absorption spectra based on different
    ROIs specified by the user.
    """

    def __init__(self):
        super().__init__()

        self.n_bins = 10

        self._energies = []
        self._xgm = []
        self._I0 = []
        self._I1 = []
        self._I2 = []

        self._bin_center = None
        self._absorptions = None
        self._bin_count = None

        # we do not need to re-calculate the spectrum for every train, since
        # there is only one more data for detectors like FastCCD.
        self._counter = 0
        self._update_frequency = 10

        self.reset()

    def process(self, proc_data, raw_data=None):
        """Override."""
        xgm = proc_data.xgm
        mono = proc_data.mono
        _, roi1_hist, _ = proc_data.roi.roi1_hist
        _, roi2_hist, _ = proc_data.roi.roi2_hist
        _, roi3_hist, _ = proc_data.roi.roi3_hist

        self._energies.append(mono.energy)
        self._xgm.append(xgm.intensity)
        self._I0.append(roi1_hist[-1])
        self._I1.append(roi2_hist[-1])
        self._I2.append(roi3_hist[-1])

        if self._counter == self._update_frequency:
            try:
                # re-calculate the spectra
                bin_center, absorptions, bin_count = compute_spectrum(
                    self._energies, self._I0, [self._I1, self._I2], self.n_bins)
            except ValueError as e:
                raise ProcessingError(repr(e))

            self._bin_center = bin_center
            self._absorptions = absorptions
            self._bin_count = bin_count

            self._counter = 0
        else:
            # use old values
            bin_center = self._bin_center
            absorptions = self._absorptions
            bin_count = self._bin_count

        self._counter += 1
        proc_data.xas.bin_center = bin_center
        proc_data.xas.absorptions = absorptions
        proc_data.xas.bin_count = bin_count

    def reset(self):
        self._energies.clear()
        self._xgm.clear()
        self._I0.clear()
        self._I1.clear()
        self._I2.clear()

        self._bin_center = np.array([])
        self._bin_count = np.array([])
        self._absorptions = [np.array([]), np.array([])]

        self._counter = 0
