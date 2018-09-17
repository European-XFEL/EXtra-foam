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
    """A class which stores the processed data.

    Attributes:
        tid (int): train ID.
        momentum (numpy.ndarray): x-axis of azimuthal integration result.
        intensity (numpy.ndarray): y-axis of azimuthal integration result.
        image (numpy.ndarray): assembled images for all the pulses.
        image_avg (numpy.ndarray): average of the assembled images over pulses.
    """
    def __init__(self, tid, *, momentum=None, intensity=None, assembled=None):
        """Initialization."""
        if not isinstance(tid, int):
            raise ValueError("Train ID must be an integer!")
        # tid is not allowed to be modified once initialized.
        self._tid = tid

        self.momentum = momentum
        self.intensity = intensity

        self.image = None
        self.image_avg = None
        # prefer data processing outside the GUI
        if assembled is not None:
            self.image = self.array2image(assembled)
            self.image_avg = self.array2image(np.mean(assembled, axis=0))

    @property
    def tid(self):
        return self._tid

    def empty(self):
        """Check the goodness of the data."""
        if self.intensity is None or self.momentum is None \
                or self.image is None:
            return True
        return False

    @staticmethod
    def array2image(x):
        """Convert array data to image data."""
        img = x / cfg.DISPLAY_RANGE[1]
        img *= 255.0
        return np.ma.filled(np.clip(img, 0, 255).astype(np.uint8), 0)


class DataProcessor(object):
    """Class for data processing.

    Attributes:
        pulse_range (tuple): (min. pulse ID, max. pulse ID) to be processed.
    """
    def __init__(self, **kwargs):
        """Initialization."""
        self.pulse_range = kwargs['pulse_range']

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

        self.mask_range = kwargs['mask_range']
        self.mask = kwargs['mask']

    def process_assembled_data(self, assembled, tid):
        """Process assembled image data.

        :param numpy.ndarray assembled: assembled image data.
        :param int tid: pulse id

        :return ProcessedData: data after processing.
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

        # apply inf, NaN and range mask to the assembled image
        # masked is a np.ma.MaskedArray object
        masked = np.ma.masked_outside(np.ma.masked_invalid(assembled),
                                      self.mask_range[0],
                                      self.mask_range[1])

        momentum = None
        intensities = []
        for i in range(assembled.shape[0]):
            if self.mask is not None:
                if self.mask.shape != assembled[i].shape:
                    raise ValueError(
                        "Mask and image have different shapes! {} and {}".
                        format(self.mask.shape, assembled[i].shape))
                masked[i].mask[self.mask == 255] = True

            # Here the assembled is still the original image data
            res = ai.integrate1d(masked[i],
                                 self.integration_points,
                                 method=self.integration_method,
                                 mask=masked[i].mask,
                                 radial_range=self.integration_range,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")
            momentum = res.radial
            intensities.append(res.intensity)

        logger.debug("Time for azimuthal integration: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        data = ProcessedData(tid,
                             momentum=momentum,
                             intensity=np.array(intensities),
                             assembled=masked)

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
            # (modules, x, y, memory cells) -> (memory cells, modules, y, x)
            modules_data = np.moveaxis(np.moveaxis(modules_data, 3, 0), 3, 2)
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

        assembled, centre = self._geom.position_all_modules(modules_data)
        # TODO: slice earlier to save computation time
        # assembled = np.rot90(
        #     assembled[self.pulse_range[0]:self.pulse_range[1] + 1],
        #     3,
        #     axes=(1, 2))
        assembled = assembled[self.pulse_range[0]:self.pulse_range[1] + 1]

        logger.debug("Time for assembling: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        return self.process_assembled_data(assembled, tid)
