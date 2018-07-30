import time

import numpy as np
import pyFAI
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

from .config import Config as cfg
from .logging import logger


class DataProcessor(object):
    def __init__(self, geom_file=None):
        """"""
        self._geom = None
        if geom_file is not None:
            with File(geom_file, 'r') as f:
                self._geom = LPDGeometry.from_h5_file_and_quad_positions(
                    f, cfg.QUAD_POSITIONS)
            logger.info("Use geometry file: {}".format(geom_file))

    def process_assembled_data(self, assembled_data, tid):
        """Process assembled image data.

        :param numpy.ndarray assembled_data: assembled image data.
        :param int tid: pulse id

        :return: results stored in a dictionary.
        """
        t0 = time.perf_counter()

        ai = pyFAI.AzimuthalIntegrator(dist=cfg.DIST,
                                       poni1=cfg.CENTER_Y*cfg.PIXEL_SIZE,
                                       poni2=cfg.CENTER_X*cfg.PIXEL_SIZE,
                                       pixel1=cfg.PIXEL_SIZE,
                                       pixel2=cfg.PIXEL_SIZE,
                                       rot1=0,
                                       rot2=0,
                                       rot3=0,
                                       wavelength=cfg.LAMBDA_R)

        assembled = np.nan_to_num(assembled_data)

        momentum = None
        intensities = []
        for i in range(assembled.shape[0]):
            data_mask = np.zeros(assembled[i].shape)  # 0 for valid pixel
            data_mask[(assembled[i] <= cfg.MASK_RANGE[0])
                      | (assembled[i] > cfg.MASK_RANGE[1])] = 1
            res = ai.integrate1d(assembled[i],
                                 cfg.N_POINTS,
                                 method=cfg.INTEGRATION_METHOD,
                                 mask=data_mask,
                                 radial_range=cfg.RADIAL_RANGE,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")
            momentum = res.radial
            intensities.append(res.intensity)

        logger.debug("Time for azimuthal integration: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        data = dict()
        data["tid"] = tid
        data["intensity"] = np.array(intensities)
        data["momentum"] = momentum
        # trunc the data only for plot
        assembled[(assembled <= cfg.MASK_RANGE[0])
                  | (assembled > cfg.MASK_RANGE[1])] = 0
        data["image"] = assembled
        return data

    def process_calibrated_data(self, calibrated_data, *, from_file=False):
        """Process data streaming by karabo_data from files."""
        if self._geom is None:
            logger.info(
                "Geometry file is required to process calibrated data!")
        data, metadata = calibrated_data

        tid = next(iter(metadata.values()))["timestamp.tid"]

        t0 = time.perf_counter()

        if from_file is False:
            modules_data = next(iter(data.values()))["image.data"]
            modules_data = np.moveaxis(modules_data, 3, 0)
            logger.debug("Time for manipulating stacked data: {:.1f} ms"
                         .format(1000 * (time.perf_counter() - t0)))
        else:
            modules_data = stack_detector_data(data, "image.data", only="LPD")
            logger.debug("Time for stacking detector data: {:.1f} ms"
                         .format(1000 * (time.perf_counter() - t0)))

        if hasattr(modules_data, 'shape') is False \
                or modules_data.shape[-3:] != (16, 256, 256):
            logger.debug("Error in modules data of train {}".format(tid))
            return None

        # cell_data = stack_detector_data(train_data, "image.cellId",
        #                                 only="LPD")
        t0 = time.perf_counter()

        assembled_orig, centre = \
            self._geom.position_all_modules(modules_data)

        logger.debug("Time for assembling: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        n_pulses = np.minimum(assembled_orig.shape[0], cfg.PULSES_PER_TRAIN)
        return self.process_assembled_data(assembled_orig[:n_pulses], tid)
