"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

Data processing module.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time

import numpy as np
from scipy import constants
import pyFAI
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

from .config import Config as cfg
from .logging import logger


def sub_array_with_range(y, x, range_=None):
    if range_ is None:
        return y, x
    indices = np.where(np.logical_and(x <= range_[1], x >= range_[0]))
    return y[indices], x[indices]


def integrate_curve(y, x, range_=None):
    itgt = np.trapz(*sub_array_with_range(y, x, range_))
    return itgt if itgt else 1.0


class ProcessedData:
    """A class which stores the processed data."""
    def __init__(self, tid):
        """Initialization."""
        if not isinstance(tid, int):
            raise ValueError("Train ID must be an integer!")
        # tid is not allowed to be modified once initialized.
        self._tid = tid
        self.intensity = None
        self.momentum = None
        self.image = None

    @property
    def tid(self):
        return self._tid

    def empty(self):
        """Check the goodness of the data."""
        if self.intensity is None or self.momentum is None \
                or self.image is None:
            return True
        return False


class DataProcessor(object):
    def __init__(self, **kwargs):
        """Initialization."""
        self._geom = None
        with File(kwargs['geom_file'], 'r') as f:
            self._geom = LPDGeometry.from_h5_file_and_quad_positions(
                f, kwargs['quad_positions'])
            logger.info("Loaded geometry file: {}".format(kwargs['geom_file']))

        self.wavelength = 1e-3 * constants.c * constants.h / constants.e\
            / kwargs['photon_energy']

        self.sample_dist = kwargs['sample_dist']
        self.cx = kwargs['cx'] * cfg.PIXEL_SIZE
        self.cy = kwargs['cy'] * cfg.PIXEL_SIZE
        self.integration_method = kwargs['integration_method']
        self.integration_range = kwargs['integration_range']
        self.integration_points = kwargs['integration_points']

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

        data = ProcessedData(tid)
        data.intensity = np.array(intensities)
        data.momentum = momentum
        # trunc the data only for plot
        assembled[(assembled <= cfg.MASK_RANGE[0])
                  | (assembled > cfg.MASK_RANGE[1])] = 0
        data.image = np.rot90(assembled, 3, axes=(1, 2))

        return data

    def process_calibrated_data(self, calibrated_data, *, from_file=False):
        """Process data streaming by karabo_data from files."""
        data, metadata = calibrated_data

        tid = next(iter(metadata.values()))["timestamp.tid"]

        t0 = time.perf_counter()

        if from_file is False:
            # TODO: should be able to specify the data source here.
            if len(metadata.items()) > 1:
                logger.warning(
                    "Received data from more than one data sources!")

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
            return ProcessedData(tid)

        # cell_data = stack_detector_data(train_data, "image.cellId",
        #                                 only="LPD")
        t0 = time.perf_counter()

        assembled_orig, centre = \
            self._geom.position_all_modules(modules_data)

        logger.debug("Time for assembling: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        n_pulses = np.minimum(assembled_orig.shape[0], cfg.PULSES_PER_TRAIN)
        return self.process_assembled_data(assembled_orig[:n_pulses], tid)
