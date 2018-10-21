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
from concurrent.futures import ProcessPoolExecutor
import queue
import warnings

import numpy as np
from scipy import constants
import pyFAI
from h5py import File
import fabio

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

from ..widgets.pyqtgraph import QtCore
from .data_model import DataSource, ProcessedData
from ..config import config
from ..logger import logger
from .proc_utils import down_sample, up_sample


class DataProcessor(QtCore.QThread):
    """Class for data processing.

    Attributes:
        source (DataSource): data source.
        pulse_range (tuple): the min. and max. pulse ID to be processed.
            (int, int)
        _geom (LPDGeometry): geometry.
        wavelength (float): photon wavelength in meter.
        sample_distance (float): distance from the sample to the detector
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

    # post message in the main GUI
    messager = QtCore.pyqtSignal(str)

    def __init__(self, parent, in_queue, out_queue):
        """Initialization.

        :param Queue in_queue: a queue of data from the ZMQ bridge.
        :param Queue out_queue: a queue of processed data
        """
        super().__init__(parent=parent)

        self.messager.connect(parent.onMessageReceived)
        self.messager.emit("Data processor started!")

        self._in_queue = in_queue
        self._out_queue = out_queue

        self.image_mask = None
        self.image_mask_initialized = False

        # -------------------------------------------------------------
        # define shared parameters
        # -------------------------------------------------------------

        self.source_sp = None
        self.geom_sp = None
        self.sample_distance_sp = None
        self.center_coordinate_sp = None
        self.integration_method_sp = None
        self.integration_range_sp = None
        self.integration_points_sp = None
        self.mask_range_sp = None
        self.wavelength_sp = None
        self.pulse_range_sp = None

        # -------------------------------------------------------------
        # define slots' behaviors
        # -------------------------------------------------------------

        parent.data_source_sp.connect(self.onSourceChanged)
        parent.geometry_sp.connect(self.onGeometryChanged)
        parent.sample_distance_sp.connect(self.onSampleDistanceChanged)
        parent.center_coordinate_sp.connect(self.onCenterCoordinateChanged)
        parent.integration_method_sp.connect(self.onIntegrationMethodChanged)
        parent.integration_range_sp.connect(self.onIntegrationRangeChanged)
        parent.integration_points_sp.connect(self.onIntegrationPointsChanged)
        parent.mask_range_sp.connect(self.onMaskRangeChanged)
        parent.photon_energy_sp.connect(self.onPhotonEnergyChanged)
        parent.pulse_range_sp.connect(self.onPulseRangeChanged)

        parent.image_mask_sgn.connect(self.onImageMaskChanged)

        self._running = False

    @QtCore.pyqtSlot(str)
    def onImageMaskChanged(self, filename):
        try:
            self.image_mask = fabio.open(filename).data
            msg = "Image mask {} loaded!".format(filename)
        except (IOError, OSError) as e:
            msg = str(e)
        finally:
            self.messager.emit(msg)

    @QtCore.pyqtSlot(object)
    def onSourceChanged(self, value):
        self.source_sp = value

    @QtCore.pyqtSlot(str, list)
    def onGeometryChanged(self, filename, quad_positions):
        with File(filename, 'r') as f:
            self.geom_sp = LPDGeometry.from_h5_file_and_quad_positions(
                f, quad_positions)

    @QtCore.pyqtSlot(float)
    def onSampleDistanceChanged(self, value):
        self.sample_distance_sp = value

    @QtCore.pyqtSlot(int, int)
    def onCenterCoordinateChanged(self, cx, cy):
        self.center_coordinate_sp = (cx * config["PIXEL_SIZE"],
                                     cy * config["PIXEL_SIZE"])

    @QtCore.pyqtSlot(str)
    def onIntegrationMethodChanged(self, value):
        self.integration_method_sp = value

    @QtCore.pyqtSlot(float, float)
    def onIntegrationRangeChanged(self, lb, ub):
        self.integration_range_sp = (lb, ub)

    @QtCore.pyqtSlot(int)
    def onIntegrationPointsChanged(self, value):
        self.integration_points_sp = value

    @QtCore.pyqtSlot(float, float)
    def onMaskRangeChanged(self, lb, ub):
        self.mask_range_sp = (lb, ub)

    @QtCore.pyqtSlot(float)
    def onPhotonEnergyChanged(self, value):
        constant = 1e-3 * constants.c * constants.h / constants.e
        self.wavelength_sp = constant / value

    @QtCore.pyqtSlot(int, int)
    def onPulseRangeChanged(self, lb, ub):
        self.pulse_range_sp = (lb, ub)

    def run(self):
        """Run the data processor."""
        self._running = True
        logger.debug("Start data processing...")
        while self._running:
            try:
                data = self._in_queue.get(timeout=config['TIMEOUT'])
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            if self.source_sp == DataSource.CALIBRATED_FILE:
                processed_data = self.process_calibrated_data(
                    data, from_file=True)
            elif self.source_sp == DataSource.CALIBRATED:
                processed_data = self.process_calibrated_data(data)
            elif self.source_sp == DataSource.ASSEMBLED:
                processed_data = self.process_assembled_data(data)
            elif self.source_sp == DataSource.PROCESSED:
                processed_data = data[0]
            else:
                raise ValueError("Unknown data source!")

            logger.debug("Time for data processing: {:.1f} ms in total!\n"
                         .format(1000 * (time.perf_counter() - t0)))

            while self._running:
                try:
                    self._out_queue.put(processed_data,
                                        timeout=config['TIMEOUT'])
                    break
                except queue.Full:
                    continue

            logger.debug("Size of in and out queues: {}, {}".format(
                self._in_queue.qsize(), self._out_queue.qsize()))

    def terminate(self):
        self._running = False

    def process_assembled_data(self, assembled, tid):
        """Process assembled image data.

        :param numpy.ndarray assembled: assembled image data.
        :param int tid: train ID

        :return ProcessedData: processed data.
        """
        ai = pyFAI.AzimuthalIntegrator(dist=self.sample_distance_sp,
                                       poni1=self.center_coordinate_sp[1],
                                       poni2=self.center_coordinate_sp[0],
                                       pixel1=config["PIXEL_SIZE"],
                                       pixel2=config["PIXEL_SIZE"],
                                       rot1=0,
                                       rot2=0,
                                       rot3=0,
                                       wavelength=self.wavelength_sp)

        # pre-processing

        t0 = time.perf_counter()

        # original data contains 'nan', 'inf' and '-inf' pixels

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if config["DOWN_SAMPLE_IMAGE_MEAN"]:
                # Down-sampling the average image by a factor of two will
                # reduce the data processing time considerably, while the
                # azimuthal integration will not be affected.
                assembled_mean = np.nanmean(down_sample(assembled), axis=0)
            else:
                assembled_mean = np.nanmean(assembled, axis=0)

        # Convert 'nan' to '-inf' and it will later be converted to 0.
        # We do not convert 'nan' to 0 because: if the lower range of
        # mask is a negative value, 0 will be converted to a value
        # between 0 and 255 later.
        assembled_mean[np.isnan(assembled_mean)] = -np.inf

        logger.debug("Time for pre-processing: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        # azimuthal integration

        t0 = time.perf_counter()

        if self.image_mask is None:
            base_mask = np.zeros_like(assembled[0], dtype=np.uint8)
        else:
            if self.image_mask.shape != assembled[0].shape:
                try:
                    # do up-sample only once
                    if not self.image_mask_initialized:
                        old_shape = self.image_mask.shape
                        self.image_mask = up_sample(
                            self.image_mask, assembled[0].shape)
                        self.image_mask_initialized = True
                        logger.debug("Up-sample mask with shape {} to {}".
                                     format(old_shape, self.image_mask.shape))
                    else:
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(
                        "Invalid mask shape {} for image with shape {}".
                        format(self.image_mask.shape, assembled[0].shape))
            base_mask = self.image_mask

        global _integrate1d_imp

        def _integrate1d_imp(i):
            """Use for multiprocessing."""
            # convert 'nan' to '-inf', as explained above
            assembled[i][np.isnan(assembled[i])] = -np.inf

            # add threshold mask
            mask = np.copy(base_mask)
            mask[(assembled[i] < self.mask_range_sp[0]) |
                 (assembled[i] > self.mask_range_sp[1])] = 1

            # do integration
            ret = ai.integrate1d(assembled[i],
                                 self.integration_points_sp,
                                 method=self.integration_method_sp,
                                 mask=mask,
                                 radial_range=self.integration_range_sp,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")

            return ret.radial, ret.intensity

        workers = config["WORKERS"]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                chunksize = int(np.ceil(assembled.shape[0] / workers))
                rets = executor.map(_integrate1d_imp, range(assembled.shape[0]),
                                    chunksize=chunksize)
            momentums, intensities = zip(*rets)
        else:
            momentums = []
            intensities = []
            for i in range(assembled.shape[0]):
                momentum, intensity = _integrate1d_imp(i)
                momentums.append(momentum)
                intensities.append(intensity)

        momentum = momentums[0]

        logger.debug("Time for azimuthal integration: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        # clip the value in the array
        np.clip(assembled_mean,
                self.mask_range_sp[0], self.mask_range_sp[1], out=assembled_mean)
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
                logger.debug("Found multiple data sources!")

            tid = metadata[config["SOURCE_NAME"]]["timestamp.tid"]
            modules_data = data[config["SOURCE_NAME"]]["image.data"]

            # (modules, x, y, memory cells) -> (memory cells, modules, y, x)
            modules_data = np.moveaxis(np.moveaxis(modules_data, 3, 0), 3, 2)
        else:
            tid = next(iter(metadata.values()))["timestamp.tid"]

            try:
                if config["TOPIC"] == "FXE":
                    dev = 'LPD'
                else:
                    dev = ''

                modules_data = stack_detector_data(
                    data, "image.data", only=dev)
            # To handle a bug when using the recent karabo_data on the
            # old data set:
            # 1. Missing "image.data" will raise KeyError!
            # 2. Different modules could have different shapes, e.g.
            #    a train with 32 pulses could has a module with shape
            #    (4, 256, 256), which means the data for some pulses
            #    were lost. It will raise ValueError!
            #
            # Note: we log the information in 'debug' since otherwise it
            #       will go to the log window and cause problems like
            #       segmentation fault.
            except (KeyError, ValueError) as e:
                logger.debug("Error in stacking detector data: " + str(e))
                return ProcessedData(tid)

        logger.debug("Time for moveaxis/stacking: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        if hasattr(modules_data, 'shape') is False \
                or modules_data.shape[-3:] != (16, 256, 256):
            logger.debug("Error in modules data of train {}".format(tid))
            return ProcessedData(tid)

        t0 = time.perf_counter()

        assembled, centre = self.geom_sp.position_all_modules(modules_data)
        # This is a bug in old version of karabo_data. The above function
        # could return a numpy.ndarray with shape (0, x, x)
        if assembled.shape[0] == 0:
            logger.debug("Bad shape {} in assembled image of train {}".
                         format(assembled.shape, tid))
            return ProcessedData(tid)
        # TODO: slice earlier to save computation time
        assembled = assembled[self.pulse_range_sp[0]:self.pulse_range_sp[1] + 1]

        logger.debug("Time for assembling: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        return self.process_assembled_data(assembled, tid)
