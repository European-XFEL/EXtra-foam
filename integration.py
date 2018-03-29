from logging import log, INFO, ERROR, WARN
import math
import numpy as np
import pyFAI

from constants import (
    center_x, center_y, distance, pixel_size, qnorm_max, qnorm_min,
    wavelength_lambda
)


# setting the integrator
npt = 512
ai = pyFAI.AzimuthalIntegrator(dist=distance,
                               poni1=center_y*pixel_size,
                               poni2=center_x*pixel_size,
                               pixel1=pixel_size,
                               pixel2=pixel_size,
                               rot1=0, rot2=0, rot3=0,
                               wavelength=wavelength_lambda)


def integrate(images):
    Sq = None
    pulse_integ_result = []
    normalised_integ_result = []
    # hole_pixel_size is the size of the gap in the center
    # of the detector
    total_img = np.zeros((1024, 1024), dtype='float32')
    for pulse in range(images.shape[2]):
        combined_imgs = images[..., pulse]

        # Define mask data to subtract to reduce noise
        total_img = total_img + combined_imgs
        mask_data = np.zeros(combined_imgs.shape)
        mask_data[np.where(combined_imgs == 0)] = 1

        momentum, i_unc = ai.integrate1d(combined_imgs,
                                         npt,
                                         method="lut",
                                         mask=mask_data,
                                         radial_range=(0.1, 4),
                                         correctSolidAngle=True,
                                         polarization_factor=1,
                                         unit="q_A^-1")

        # Define or update the scattering signal
        if Sq is None:
            # Get the momentum transfer, known as q or scattering,
            # from the integration
            scattering = momentum * wavelength_lambda * 1e10 / (4 * math.pi)
            # 2-theta scattering angle

            Qnorm = momentum[np.where(np.logical_and(momentum >= qnorm_min,
                                                     momentum <= qnorm_max))]
            Sq = i_unc
        else:
            Sq = np.concatenate((Sq, i_unc))

        # Normalise
        to_integrate = i_unc[np.where(np.logical_and(momentum >= qnorm_min,
                                                     momentum <= qnorm_max))]
        N = np.trapz(to_integrate, x=Qnorm)

        i_unc = i_unc / N
        # Integration results are lists of tuples for each pulse, where the
        # content of the tuple is momentum, scattering
        pulse_integ_result.append(i_unc)
        normalised_integ_result.append(i_unc/N)

    return momentum, np.array(pulse_integ_result), np.array(normalised_integ_result)


means = {}  # dict of dicts where the key is an int, matching pulse_indices


def running_mean(normalised_data, pulse_indices):
    for idx in pulse_indices:
        scattering = normalised_data[idx]

        mean = means.get(idx, None)
        if not mean:
            means[idx] = {'scattering': scattering, 'count': 1}
            continue
 
        mean['scattering'] = np.array([((old * (mean['count']-1)) + new) / mean['count']
                                      for old, new in zip(mean['scattering'], scattering)])
        mean['count'] += 1

    return means


def differences(normalised_data, pulse_indices):
    diffs = {}
    for idx in pulse_indices:
        scattering = normalised_data[idx]
        mean = means.get(idx, None)
        if not mean:
            log(ERROR, "I was told to compute the difference for {}".format(idx))
            continue

        diff_scattering = np.array(scattering) - np.array(mean['scattering'])
        diffs[idx] = diff_scattering

    return diffs


def diff_integrals(differences, momentum, pulse_indices):
    integrals = {}
    for idx in pulse_indices:
        scattering = differences[idx]
        scattering = np.absolute(scattering)
    	# Normalise
        to_integrate = scattering[np.where(np.logical_and(momentum >= qnorm_min,
    	                                                  momentum <= qnorm_max))]

        Qnorm = momentum[np.where(np.logical_and(momentum >= qnorm_min,
    	                                         momentum <= qnorm_max))]
        val = np.trapz(to_integrate, x=Qnorm)
        integrals[idx] = val

    return integrals
