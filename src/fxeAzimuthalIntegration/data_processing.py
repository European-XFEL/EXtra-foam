import time

import numpy as np
import pyFAI
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

from .config import Config as cfg
from .logging import logger


class DataProcessor(object):
    def __init__(self, **kwargs):
        """"""
        self._geom = None

        for key in kwargs:
            if key == 'geom_file':
                with File(kwargs[key], 'r') as f:
                    self._geom = LPDGeometry.from_h5_file_and_quad_positions(
                        f, cfg.QUAD_POSITIONS)
                logger.info("Use geometry file: {}".format(kwargs[key]))
            elif key == 'photon_energy':
                # convert energy to wavelength
                self.wavelength = 1e-10 * 12.3984 / kwargs[key]
            elif key == 'sample_dist':
                self.sample_dist = kwargs[key]
            elif key == 'cx':
                self.cx = kwargs[key] * cfg.PIXEL_SIZE
            elif key == 'cy':
                self.cy = kwargs[key] * cfg.PIXEL_SIZE
            elif key == 'integration_method':
                self.integration_method = kwargs[key]
            elif key == 'integration_range':
                self.integration_range = kwargs[key]
            elif key == 'integration_points':
                self.integration_points = kwargs[key]

    def process_assembled_data(self, assembled_data, tid):
        """Process assembled image data.

        :param numpy.ndarray assembled_data: assembled image data.
        :param int tid: pulse id

        :return: results stored in a dictionary.
        """
        t0 = time.perf_counter()

        ai = pyFAI.AzimuthalIntegrator(dist=self.sample_dist,
                                       poni1=self.cy,
                                       poni2=self.cx,
                                       pixel1=cfg.PIXEL_SIZE,
                                       pixel2=cfg.PIXEL_SIZE,
                                       rot1=0,
                                       rot2=0,
                                       rot3=0,
                                       wavelength=self.wavelength)

        assembled = np.nan_to_num(assembled_data)

        momentum = None
        intensities = []
        for i in range(assembled.shape[0]):
            data_mask = np.zeros(assembled[i].shape)  # 0 for valid pixel
            data_mask[(assembled[i] <= cfg.MASK_RANGE[0])
                      | (assembled[i] > cfg.MASK_RANGE[1])] = 1
            res = ai.integrate1d(assembled[i],
                                 self.integration_points,
                                 method=self.integration_method,
                                 mask=data_mask,
                                 radial_range=self.integration_range,
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
