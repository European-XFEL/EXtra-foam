"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data processor.

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

from .data_model import DataSource, ProcessedData
from ..config import Config as cfg
from ..logger import logger


class DataProcessor(Thread):
    """Class for data processing.

    Attributes:
        source (DataSource): data source.
        pulse_range (tuple): the min. and max. pulse ID to be processed.
            (int, int)
        _geom (LPDGeometry): geometry.
        wavelength (float): photon wavelength in meter.
        sample_dist (float): distance from the sample to the detector
            plan (orthogonal distance, not along the beam), in meter.
        cx (int): coordinate of the point of normal incidence along the
            detector's first dimension, in pixels.
        cy (int): coordinate of the point of normal incidence along the
            detector's second dimension, in pixels
        integration_method (string): the azimuthal integration method
            supported by pyFAI.
        integration_range (tuple): the lower and upper range of
            the integration radial unit. (float, float)
        integration_points (int): number of points in the integration
            output pattern.
        mask_range (tuple):
        image_mask (numpy.ndarray):
    """
    def __init__(self, in_queue, out_queue, **kwargs):
        """Initialization.

        :param Queue in_queue: a queue of data from the ZMQ bridge.
        :param Queue out_queue: a queue of processed data
        """
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
        self.image_mask = kwargs['mask']

        self._running = False

    def run(self):
        """Run the data processor."""
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
        """Terminate the data processor."""
        self._running = False

    def process_assembled_data(self, assembled, tid):
        """Process assembled image data.

        :param numpy.ndarray assembled: assembled image data.
        :param int tid: train ID

        :return ProcessedData: processed data.
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

        # pre-processing

        t0 = time.perf_counter()

        # original data contains 'nan', 'inf' and '-inf' pixels

        assembled_mean = np.nanmean(assembled, axis=0)

        # Convert 'nan' to '-inf' and it will later be converted to 0.
        # We do not convert 'nan' to 0 because: if the lower range of
        # mask is a negative value, 0 will be converted to a value
        # between 0 and 255 later.
        assembled_mean[np.isnan(assembled_mean)] = -np.inf
        assembled[np.isnan(assembled)] = -np.inf

        logger.debug("Time for pre-processing: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        # azimuthal integration

        t0 = time.perf_counter()

        if self.image_mask is None:
            base_mask = np.zeros_like(assembled[0], dtype=np.uint8)
        else:
            if self.image_mask.shape != assembled[0].shape:
                raise ValueError(
                    "Mask and image have different shapes! {} and {}".
                    format(self.image_mask.shape, assembled[0].shape))
            base_mask = self.image_mask

        momentum = None
        intensities = np.zeros([assembled.shape[0], self.integration_points])
        for i in range(assembled.shape[0]):
            mask = np.copy(base_mask)
            # apply threshold mask
            mask[(assembled[i] < self.mask_range[0]) |
                 (assembled[i] > self.mask_range[1])] = 1
            res = ai.integrate1d(assembled[i],
                                 self.integration_points,
                                 method=self.integration_method,
                                 mask=mask,
                                 radial_range=self.integration_range,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")
            if i == 0:
                momentum = res.radial
            intensities[i] = res.intensity

        logger.debug("Time for azimuthal integration: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        # clip the value in the array
        np.clip(assembled_mean,
                self.mask_range[0], self.mask_range[1], out=assembled_mean)
        # now 'assembled_mean' contains only numerical values within
        # the mask range

        # Note: 'assembled' still contains 'inf' and '-inf', we only do
        #       the clip later when necessary in order not to waste
        #       computing power.

        data = ProcessedData(tid,
                             momentum=momentum,
                             intensity=np.array(intensities),
                             intensity_mean=np.mean(intensities, axis=0),
                             image=assembled,
                             image_mean=assembled_mean,
                             image_mask=self.image_mask)

        return data

    def process_calibrated_data(self, calibrated_data, *, from_file=False):
        """Process calibrated data.

        :param tuple calibrated_data: (data, metadata). See return of
            KaraboBridge.Client.next().
        :param bool from_file: True for data streamed from files and False
            for data from the online ZMQ bridge.

        :return ProcessedData: processed data.
        """
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
