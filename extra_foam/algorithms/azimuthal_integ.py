"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from scipy import constants

from .azimuthal_integrator import AzimuthalIntegrator, ConcentricRingFinder


# Plank-einstein relation (E=hv)
CONST_HC_E = constants.c * constants.h / constants.e


def energy2wavelength(energy):
    """Convert photon energy to wavelength.

    :param float energy: photon energy in eV.

    :return float wavelength: photon wavelength in m.
    """
    return CONST_HC_E / energy


def compute_q(dist, x, e):
    """Compute a single momentum transfer value.

    q = 4 * pi * sin(theta) / lambda

    :param float dist: distance from the sample to the detector plane
                       (orthogonal distance, not along the beam), in meter.
    :param float x: distance to the azimuthal integration center, in meter.
    :param float e: photon energy in eV.

    :return: momentum transfer in 1/m.
    """
    return 4 * np.pi * e / CONST_HC_E / np.sqrt(4 * dist ** 2 / x ** 2 + 1)
