import numpy as np
import pyFAI
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

import config as cfg
from logger import log


def _process_assembled_data(assembled_image, tid):
    """Process assembled image data.

    :param numpy.ndarray assembled_image: assembled image data.
    :param int tid: pulse id

    :return: results stored in a dictionary.
    """
    assembled = np.nan_to_num(assembled_image)
    data_mask = np.zeros(assembled.shape)  # 0 for valid pixel
    data_mask[(assembled <= cfg.MASK_RANGE[0])
              | (assembled > cfg.MASK_RANGE[1])] = 1

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
    for i in range(assembled.shape[0]):
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
    data["images"] = assembled_image

    return data


def process_data(kb_data):
    """Process data streaming by karabo_data from files."""
    if cfg.FROM_ASSEMBLED_DATA is False:
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
    else:
        # TODO: check the data format from the assembled data.
        tid = kb_data["header.trainId"]
        assembled_orig = kb_data["image.data"]

    n_pulses = np.minimum(assembled_orig.shape[0], cfg.PULSES_PER_TRAIN)
    return _process_assembled_data(assembled_orig[:n_pulses], tid)
