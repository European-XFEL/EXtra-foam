import numpy as np
import pyFAI
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

import config as cfg
from logger import log


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

    assembled_orig, centre = geom.position_all_modules(modules_data)

    assembled = np.nan_to_num(assembled_orig)
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
    data["images"] = assembled_orig

    return data
