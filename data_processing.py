import logging
import sys
import time
import numpy as np
import pyFAI
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

import config as cfg


logging.basicConfig(level=logging.INFO,
                    format="PID %(process)5s: %(message)s",
                    stream=sys.stderr)
log = logging.getLogger()


def process_data(kb_data):
    """"""
    with File(cfg.GEOMETRY_FILE, 'r') as f:
        geom = LPDGeometry.from_h5_file_and_quad_positions(
            f, cfg.QUAD_POSITIONS)

    data, metadata = kb_data

    tid = next(iter(metadata.values()))["timestamp.tid"]

    modules_data = stack_detector_data(data, "image.data", only="LPD")

    if hasattr(modules_data, 'shape') is False \
            or modules_data.shape[-3:] != (16, 256, 256):
        log.info("Error in modules data of train {}".format(tid))
        return None

    # cell_data = stack_detector_data(train_data, "image.cellId",
    #                                 only="LPD")

    assembled, centre = geom.position_all_modules(modules_data)

    assembled = np.nan_to_num(assembled)
    data_mask = np.zeros(assembled.shape)  # 0 for valid pixel
    data_mask[(assembled <= 0) | (assembled > 1e4)] = 1

    # integrated.shape = (1243, 1145)
    integrated = np.zeros(assembled.shape[1:3])
    assembled[data_mask == 1] = 0
    integrated += np.sum(assembled, axis=0)

    ai = pyFAI.AzimuthalIntegrator(dist=cfg.DIST,
                                   poni1=cfg.CENTER_Y*cfg.PIXEL_SIZE,
                                   poni2=cfg.CENTER_X*cfg.PIXEL_SIZE,
                                   pixel1=cfg.PIXEL_SIZE,
                                   pixel2=cfg.PIXEL_SIZE,
                                   rot1=0,
                                   rot2=0,
                                   rot3=0,
                                   wavelength=cfg.LAMBDA_R)

    momentum = None
    intensities = []
    for i in range(np.minimum(modules_data.shape[0], cfg.PULSES_PER_TRAIN)):
        res = ai.integrate1d(assembled[i],
                             cfg.N_POINTS,
                             method=cfg.INTEGRATION_METHOD,
                             mask=data_mask[i],
                             radial_range=cfg.RADIAL_RANGE,
                             correctSolidAngle=True,
                             polarization_factor=1,
                             unit="q_A^-1")
        momentum = res.radial
        intensities.append(res.intensity)

    data = dict()
    data["tid"] = tid
    data["intensity"] = np.array(intensities)
    data["momentum"] = momentum

    return data


# def differences(self, normalised_data, pulse_indices):
#     means = {}  # dict of dicts where the key is an int, matching pulse_indices
#
#     diffs = {}
#     for idx in pulse_indices:
#         scattering = normalised_data[idx]
#         mean = means.get(idx, None)
#         if not mean:
#             log(ERROR, "I was told to compute the difference for {}".format(idx))
#             continue
#
#         diff_scattering = np.array(scattering) - np.array(mean['scattering'])
#         diffs[idx] = diff_scattering
#
#     return diffs
#
#
# def diff_integrals(self, differences, momentum, pulse_indices):
#     integrals = {}
#     for idx in pulse_indices:
#         scattering = differences[idx]
#         scattering = np.absolute(scattering)
#         # Normalise
#         to_integrate = scattering[np.where(np.logical_and(momentum >= qnorm_min,
#                                                           momentum <= qnorm_max))]
#
#         Qnorm = momentum[np.where(np.logical_and(momentum >= qnorm_min,
#                                                  momentum <= qnorm_max))]
#         val = np.trapz(to_integrate, x=Qnorm)
#         integrals[idx] = val
#
#     return integrals
