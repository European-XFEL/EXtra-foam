"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

XAS (X-ray absorption spectroscopy) algorithms

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from scipy.stats import binned_statistic


def compute_spectrum(energy, intensity_ref, intensities, n_bins):
    """Compute XAS spectrum.

    :param list energy: photon energy.
    :param list intensity_ref: reference intensity.
    :param list intensities: a list of signal intensities.
    :param int n_bins: number of energy bins.

    :return: (bin_center, absorptions, bin_count)
    :rtype: (numpy.ndarray, list of numpy.ndarray, numpy.ndarray)
    """
    if not energy:
        return np.array([]), [], np.array([])

    if n_bins < 0:
        raise ValueError("Negative number of bins!")

    try:
        if len(energy) != len(intensity_ref):
            raise ValueError("Input data have different lengths!")
        for intensity in intensities:
            if len(energy) != len(intensity):
                raise ValueError("Input data have different lengths!")
    except TypeError:
        raise TypeError("Not understandable inputs!")

    binned_mean = binned_statistic(
        energy, [intensity_ref] + intensities, 'mean', n_bins)
    bin_count = binned_statistic(energy, intensity_ref, 'count', n_bins)

    bin_edges = binned_mean[1]
    intensity_ref_avg = binned_mean[0][0]
    intensities_avg = binned_mean[0][1:]
    absorptions = [-np.log(intensity_avg/intensity_ref_avg)
                   for intensity_avg in intensities_avg]

    bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2.
    bin_count = bin_count[0]

    return bin_center, absorptions, bin_count
