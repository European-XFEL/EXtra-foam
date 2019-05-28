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

from .base_processor import LeafProcessor, SharedProperty
from ..exceptions import ProcessingError
from ...algorithms import compute_spectrum
from ...metadata import Metadata as mt
from ...helpers import profiler


class XasProcessor(LeafProcessor):
    """XasProcessor class.

    Attributes:
        mono_source_name (str): monochromator source name.
        n_bins (int): number of bins.
        bin_range (tuple): bin range.

    A processor which calculate absorption spectra based on different
    ROIs specified by the user.
    """

    mono_source_name = SharedProperty()
    n_bins = SharedProperty()
    bin_range = SharedProperty()

    def __init__(self):
        super().__init__()

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

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.XAS_PROC)
        if cfg is None:
            return

        self.mono_source_name = cfg["mono_source_name"]
        self.n_bins = int(cfg["n_bins"])
        self.bin_range = self.str2tuple(cfg['bin_range'])

    @profiler("XAS processor")
    def process(self, processed, raw=None):
        """Override."""

        # TODO: FIXME

        intensity = processed.xgm.intensity
        if not intensity:
            return
        energy = processed.mono.energy
        if not energy:
            return
        _, roi1_hist, _ = processed.roi.roi1_hist
        _, roi2_hist, _ = processed.roi.roi2_hist
        _, roi3_hist, _ = processed.roi.roi3_hist

        self._energies.append(energy)
        self._xgm.append(intensity)
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
        processed.xas.bin_center = bin_center
        processed.xas.absorptions = absorptions
        processed.xas.bin_count = bin_count

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
