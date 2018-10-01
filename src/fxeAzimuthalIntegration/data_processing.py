"""
Offline and online data analysis tool for Azimuthal integration at
FXE instrument, European XFEL.

Data processing module.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from threading import Thread
from queue import Empty, Full

import numpy as np
from scipy import constants
import pyFAI
from h5py import File

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

from .config import DataSource
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


def array2image(x):
    """Convert array data to image data."""
    np.nan_to_num(x, False)
    x /= cfg.DISPLAY_RANGE[1]
    x *= 255.0
    return x.astype(np.uint8)


class ProcessedData:
    """A class which stores the processed data.

    Attributes:
        tid (int): train ID.
        momentum (numpy.ndarray): x-axis of azimuthal integration result.
            Shape = (momentum)
        intensity (numpy.ndarray): y-axis of azimuthal integration result.
            Shape = (pulse_id, intensity)
        image (numpy.ndarray): assembled images for all the pulses.
            Shape = (pulse_id, y, x)
        image_mean (2D numpy.ndarray): average of the assembled images over
            pulses. Shape = (y, x)
    """
    def __init__(self, tid, *, momentum=None, intensity=None, assembled=None):
        """Initialization."""
        if not isinstance(tid, int):
            raise ValueError("Train ID must be an integer!")
        # tid is not allowed to be modified once initialized.
        self._tid = tid

        self.momentum = momentum
        self.intensity = intensity
        self.intensity_mean = np.mean(intensity, axis=0)

        t0 = time.perf_counter()

        self.image = None
        self.image_avg = None
        # prefer data processing outside the GUI
        if assembled is not None:
            # Visualize the individual image is optional. Therefore, we should
            # delay the array2image operation.
            self.image = assembled
            self.image_mean = array2image(np.nanmean(assembled, axis=0))

        logger.debug("Time for pre-processing: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

    @property
    def tid(self):
        return self._tid

    def empty(self):
        """Check the goodness of the data."""
        if self.intensity is None or self.momentum is None \
                or self.image is None:
            return True
        return False


class DataProcessor(Thread):
    """Class for data processing.

    Attributes:
        pulse_range (tuple): (min. pulse ID, max. pulse ID) to be processed.
    """
    def __init__(self, in_queue, out_queue, **kwargs):
        """Initialization."""
        super().__init__()

        self._in_queue = in_queue
        self._out_queue = out_queue

        self.source = kwargs['source']
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
        self.img_mask = kwargs['mask']

        self._running = False

    def run(self):
        logger.debug("Start data processing...")
        self._running = True
        while self._running:
            try:
                data = self._in_queue.get(timeout=0.01)
            except Empty:
                continue

            t0 = time.perf_counter()

            if self.source == DataSource.CALIBRATED_FILE:
                processed_data = self.process_calibrated_data(data, from_file=True)
            elif self.source == DataSource.CALIBRATED:
                processed_data = self.process_calibrated_data(data)
            elif self.source == DataSource.ASSEMBLED:
                processed_data = self.process_assembled_data(data)
            elif self.source == DataSource.PROCESSED:
                processed_data = data[0]
            else:
                raise ValueError("Unknown data source!")

            logger.debug("Time for data processing: {:.1f} ms in total!\n"
                         .format(1000 * (time.perf_counter() - t0)))

            try:
                self._out_queue.put(processed_data, timeout=cfg.TIMEOUT)
            except Full:
                pass

            logger.debug("Size of in and out queues: {}, {}".
                         format(self._in_queue.qsize(), self._out_queue.qsize()))

    def terminate(self):
        self._running = False

    def process_assembled_data(self, assembled, tid):
        """Process assembled image data.

        :param numpy.ndarray assembled: assembled image data.
        :param int tid: pulse id

        :return ProcessedData: data after processing.
        """
        ai = pyFAI.AzimuthalIntegrator(dist=self.sample_dist,
                                       poni1=self.cy,
                                       poni2=self.cx,
                                       pixel1=cfg.PIXEL_SIZE,
                                       pixel2=cfg.PIXEL_SIZE,
                                       rot1=0,
                                       rot2=0,
                                       rot3=0,
                                       wavelength=self.wavelength)

        t0 = time.perf_counter()

        mask = np.zeros_like(assembled, dtype=bool)
        assembled[np.isnan(assembled)] = np.inf
        mask[(assembled < self.mask_range[0]) |
             (assembled > self.mask_range[1])] = True

        if self.img_mask is not None:
            for i in range(assembled.shape[0]):
                if self.img_mask.shape != assembled[0].shape:
                    raise ValueError(
                        "Mask and image have different shapes! {} and {}".
                        format(self.img_mask.shape, assembled[i].shape))
                mask[i][self.img_mask == 255] = True

        assembled[mask] = np.nan

        logger.debug("Time for masking: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        t0 = time.perf_counter()

        momentum = None
        intensities = []
        for i in range(assembled.shape[0]):
            res = ai.integrate1d(assembled[i],
                                 self.integration_points,
                                 method=self.integration_method,
                                 mask=mask[i],
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
                             assembled=assembled)

        return data

    def process_calibrated_data(self, calibrated_data, *, from_file=False):
        """Process data streaming by karabo_data from files."""
        data, metadata = calibrated_data

        t0 = time.perf_counter()

        if from_file is False:
            if len(metadata.items()) > 1:
                logger.warning("Found multiple data sources!")

            tid = metadata[cfg.SOURCE]["timestamp.tid"]
            modules_data = data[cfg.SOURCE]["image.data"]

            # (modules, x, y, memory cells) -> (memory cells, modules, y, x)
            modules_data = np.moveaxis(np.moveaxis(modules_data, 3, 0), 3, 2)
        else:
            tid = next(iter(metadata.values()))["timestamp.tid"]
            modules_data = stack_detector_data(data, "image.data", only="LPD")

        logger.debug("Time for moveaxis/stacking: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        if hasattr(modules_data, 'shape') is False \
                or modules_data.shape[-3:] != (16, 256, 256):
            logger.debug("Error in modules data of train {}".format(tid))
            return ProcessedData(tid)

        t0 = time.perf_counter()

        assembled, centre = self._geom.position_all_modules(modules_data)
        # TODO: slice earlier to save computation time
        assembled = assembled[self.pulse_range[0]:self.pulse_range[1] + 1]

        logger.debug("Time for assembling: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        return self.process_assembled_data(assembled, tid)
