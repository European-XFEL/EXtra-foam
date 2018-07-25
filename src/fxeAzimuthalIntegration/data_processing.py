import time

import numpy as np
import pyFAI
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

from .config import Config as cfg
from .logger import log


class DataProcessor(object):
    def __init__(self, geom_file=None):
        """"""
        self._geom = None
        if geom_file is not None:
            with File(geom_file, 'r') as f:
                self._geom = LPDGeometry.from_h5_file_and_quad_positions(
                    f, cfg.QUAD_POSITIONS)
            log.info("Use geometry file: {}".format(geom_file))

    def process_assembled_data(self, assembled_image, tid):
        """Process assembled image data.

        :param numpy.ndarray assembled_image: assembled image data.
        :param int tid: pulse id

        :return: results stored in a dictionary.
        """
        t0 = time.perf_counter()

        assembled = np.nan_to_num(assembled_image)
        data_mask = np.zeros(assembled.shape)  # 0 for valid pixel
        data_mask[(assembled <= cfg.MASK_RANGE[0])
                  | (assembled > cfg.MASK_RANGE[1])] = 1

        log.debug("Time for creating the mask: {:.1f} ms"
                  .format(1000 * (time.perf_counter() - t0)))

        ai = pyFAI.AzimuthalIntegrator(dist=cfg.DIST,
                                       poni1=cfg.CENTER_Y*cfg.PIXEL_SIZE,
                                       poni2=cfg.CENTER_X*cfg.PIXEL_SIZE,
                                       pixel1=cfg.PIXEL_SIZE,
                                       pixel2=cfg.PIXEL_SIZE,
                                       rot1=0,
                                       rot2=0,
                                       rot3=0,
                                       wavelength=cfg.LAMBDA_R)

        t0 = time.perf_counter()

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

        log.debug("Time for azimuthal integration: {:.1f} ms"
                  .format(1000 * (time.perf_counter() - t0)))

        data = dict()
        data["tid"] = tid
        data["intensity"] = np.array(intensities)
        data["momentum"] = momentum
        assembled[(assembled <= cfg.MASK_RANGE[0])
                  | (assembled > cfg.MASK_RANGE[1])] = 0
        data["image"] = assembled
        return data

    def process_calibrated_data(self, kb_data):
        """Process data streaming by karabo_data from files."""
        if self._geom is None:
            log.info("Geometry file is required to process calibrated data!")
        data, metadata = kb_data

        tid = next(iter(metadata.values()))["timestamp.tid"]

        t0 = time.perf_counter()

        modules_data = stack_detector_data(data, "image.data", only="LPD")
        log.debug("Time for stacking detector data: {:.1f} ms"
                  .format(1000 * (time.perf_counter() - t0)))

        if hasattr(modules_data, 'shape') is False \
                or modules_data.shape[-3:] != (16, 256, 256):
            log.debug("Error in modules data of train {}".format(tid))
            return None

        # cell_data = stack_detector_data(train_data, "image.cellId",
        #                                 only="LPD")
        t0 = time.perf_counter()

        assembled_orig, centre = \
            self._geom.position_all_modules(modules_data)

        log.debug("Time for assembling: {:.1f} ms"
                  .format(1000 * (time.perf_counter() - t0)))

        n_pulses = np.minimum(assembled_orig.shape[0], cfg.PULSES_PER_TRAIN)
        return self.process_assembled_data(assembled_orig[:n_pulses], tid)
