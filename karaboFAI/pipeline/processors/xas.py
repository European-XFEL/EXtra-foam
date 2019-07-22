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

from .base_processor import CompositeProcessor
from ..exceptions import ProcessingError
from ...algorithms import compute_spectrum
from ...metadata import Metadata as mt
from ...utils import profiler


class XasProcessor(CompositeProcessor):
    """XasProcessor class.

    A processor which calculate absorption spectra based on different
    ROIs specified by the user.

    Attributes:
        _mono_src (str): monochromator source name.
        _n_bins (int): number of bins.
        _bin_range (tuple): bin range.
    """

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

        self._mono_src = None
        self._n_bins = None
        self._bin_range = None

        # we do not need to re-calculate the spectrum for every train, since
        # there is only one more data for detectors like FastCCD.
        self._counter = 0
        self._update_frequency = 10

        self._reset = False

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.XAS_PROC)

        self._mono_src = cfg["mono_source_name"]
        self._n_bins = int(cfg["n_bins"])
        self._bin_range = self.str2tuple(cfg['bin_range'])

        if 'reset' in cfg:
            self._meta.delete(mt.XAS_PROC, 'reset')
            # reset when commanded by the GUI
            self._reset = True

        if self._reset:
            self._energies.clear()
            self._xgm.clear()
            self._I0.clear()
            self._I1.clear()
            self._I2.clear()

            self._bin_center = np.array([])
            self._bin_count = np.array([])
            self._absorptions = [np.array([]), np.array([])]

            self._counter = 0

    @profiler("XAS Processor")
    def process(self, data):
        """Override."""

        processed = data['processed']

        processed.xas.reset = self._reset
        self._reset = False

        # TODO: FIXME

        intensity = processed.xgm.intensity
        if not intensity:
            return
        energy = processed.mono.energy
        if not energy:
            return
        _, fom1_hist, _ = processed.roi.fom1_hist
        _, fom2_hist, _ = processed.roi.fom2_hist
        _, fom3_hist, _ = processed.roi.fom3_hist

        self._energies.append(energy)
        self._xgm.append(intensity)
        self._I0.append(fom1_hist[-1])
        self._I1.append(fom2_hist[-1])
        self._I2.append(fom3_hist[-1])

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
