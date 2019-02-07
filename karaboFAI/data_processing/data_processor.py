"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Fast azimuthal integration data processor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from concurrent.futures import ThreadPoolExecutor
import queue

import numpy as np
from scipy import constants
import pyFAI
from h5py import File
import fabio

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

from ..widgets.pyqtgraph import QtCore
from .data_model import DataSource, ProcessedData, RoiData, LaserOnOffData
from ..config import config
from ..logger import logger
from .proc_utils import nanmean_axis0_para
from ..worker import Worker
from .laser_on_off_processor import LaserOnOffProcessor


class DataProcessor(Worker):
    """Processing data received from the bridge.

    Attributes:
        source_sp (DataSource): data source.
        pulse_range_sp (tuple): (min, max) pulse ID to be processed.
            (int, int)
        geom_sp (LPDGeometry): geometry.
        wavelength_sp (float): photon wavelength in meter.
        sample_distance_sp (float): distance from the sample to the
            detector plan (orthogonal distance, not along the beam),
            in meter.
        center_coordinate_sp (tuple): (Cx, Cy), where Cx is the
            coordinate of the point of normal incidence along the
            detector's second dimension, in pixels, and Cy is the
            coordinate of the point of normal incidence along the
            detector's first dimension, in pixels. (int, int)
        integration_method_sp (string): the azimuthal integration
            method supported by pyFAI.
        integration_range_sp (tuple): the lower and upper range of
            the integration radial unit. (float, float)
        integration_points_sp (int): number of points in the
            integration output pattern.
        threshold_mask_sp (tuple): (min, max), the pixel value outside
            the range will be clipped to the corresponding edge.
        image_mask (numpy.ndarray): a 2D mask.
    """
    def __init__(self, in_queue, out_queue):
        """Initialization.

        :param Queue in_queue: a queue of data from the ZMQ bridge.
        :param Queue out_queue: a queue of processed data
        """
        super().__init__()

        self._in_queue = in_queue
        self._out_queue = out_queue

        # whether to turn azimuthal integration on
        self._enable_ai = True
        # whether to turn laser on-off on
        self._enable_on_off = True
        # whether to turn correlation analysis on
        self._enable_correlation = True

        self.image_mask = None

        # shared parameters are updated by signal-slot
        # Note: shared parameters should end with '_sp'

        self.source_type_sp = None
        self.source_name_sp = None  # detector source name
        self.pulse_range_sp = None
        self.geom_sp = None
        self.wavelength_sp = None
        self.sample_distance_sp = None
        self.center_coordinate_sp = None
        self.integration_method_sp = None
        self.integration_range_sp = None
        self.integration_points_sp = None
        self.threshold_mask_sp = (config["MASK_RANGE"][0],
                                  config["MASK_RANGE"][1])
        self.roi1 = None
        self.roi2 = None

        self.correlation_param1 = None
        self.correlation_param2 = None

        self._laser_on_off_processor = LaserOnOffProcessor()

    @QtCore.pyqtSlot(str)
    def onImageMaskChanged(self, filename):
        try:
            self.image_mask = fabio.open(filename).data
            msg = "Image mask {} loaded!".format(filename)
        except (IOError, OSError) as e:
            msg = str(e)
        finally:
            self.log(msg)

    @QtCore.pyqtSlot(object)
    def onSourceTypeChange(self, value):
        self.source_type_sp = value

    @QtCore.pyqtSlot(str)
    def onSourceNameChange(self, value):
        self.source_name_sp = value

    @QtCore.pyqtSlot(str, list)
    def onGeometryChanged(self, filename, quad_positions):
        if config['DETECTOR'] == 'LPD':
            with File(filename, 'r') as f:
                self.geom_sp = LPDGeometry.from_h5_file_and_quad_positions(
                    f, quad_positions)
        elif config['DETECTOR'] == 'AGIPD':
            try:
                from karabo_data.geometry2 import AGIPD_1MGeometry
            except (ImportError, ModuleNotFoundError):
                logger.debug(
                    "You are not in the correct branch for SPB experiment!")
                raise

            self.geom_sp = AGIPD_1MGeometry.from_crystfel_geom(filename)

    @QtCore.pyqtSlot(object, list, list)
    def onOffPulseStateChange(self, mode, on_pulse_ids, off_pulse_ids):
        self._laser_on_off_processor.laser_mode = mode
        self._laser_on_off_processor.on_pulse_ids = on_pulse_ids
        self._laser_on_off_processor.off_pulse_ids = off_pulse_ids

    @QtCore.pyqtSlot(int)
    def onMovingAverageWindowChange(self, value):
        self._laser_on_off_processor.moving_average_window = value

    @QtCore.pyqtSlot(float, float)
    def onNormalizationRangeChange(self, lb, ub):
        self._laser_on_off_processor.normalization_range = (lb, ub)

    @QtCore.pyqtSlot(float, float)
    def onOnOffIntegrationRangeChange(self, lb, ub):
        self._laser_on_off_processor.integration_range = (lb, ub)

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
    def onThresholdMaskChange(self, lb, ub):
        self.threshold_mask_sp = (lb, ub)

    @QtCore.pyqtSlot(float)
    def onPhotonEnergyChanged(self, photon_energy):
        """Compute photon wavelength (m) from photon energy (keV)."""
        # Plank-einstein relation (E=hv)
        HC_E = 1e-3 * constants.c * constants.h / constants.e
        self.wavelength_sp = HC_E / photon_energy

    @QtCore.pyqtSlot(int, int)
    def onPulseRangeChanged(self, lb, ub):
        self.pulse_range_sp = (lb, ub)

    @QtCore.pyqtSlot(str, str)
    def onCorrelationParam1Change(self, src, key):
        self.correlation_param1 = (src, key)

    @QtCore.pyqtSlot(str, str)
    def onCorrelationParam2Change(self, src, key):
        self.correlation_param2 = (src, key)

    @QtCore.pyqtSlot(bool, int, int, int, int)
    def onRoi1Changed(self, activated, w, h, cx, cy):
        if activated:
            self.roi1 = (w, h, cx, cy)
        else:
            self.roi1 = None

    @QtCore.pyqtSlot(bool, int, int, int, int)
    def onRoi2Changed(self, activated, w, h, cx, cy):
        if activated:
            self.roi2 = (w, h, cx, cy)
        else:
            self.roi2 = None

    @QtCore.pyqtSlot()
    def onRoiHistClear(self):
        RoiData.clear()

    @QtCore.pyqtSlot(int)
    def onEnableAiStateChange(self, state):
        if state == QtCore.Qt.Checked:
            self._enable_ai = True
        else:
            self._enable_ai = False

    def run(self):
        """Run the data processor."""
        self._running = True
        self.log("Data processor started!")
        while self._running:
            try:
                data = self._in_queue.get(timeout=config['TIMEOUT'])
            except queue.Empty:
                continue

            t0 = time.perf_counter()

            if self.source_type_sp == DataSource.CALIBRATED_FILE:
                processed_data = self.process_calibrated_data(
                    data, from_file=True)
            elif self.source_type_sp == DataSource.CALIBRATED:
                processed_data = self.process_calibrated_data(data)
            elif self.source_type_sp == DataSource.PROCESSED:
                processed_data = data[0]
            else:
                self.log("Unknown data source!")

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

        self.log("Data processor stopped!")

    def preprocess_data(self, assembled, tid):
        """Data pre-processing.

        The original data contains 'nan', 'inf' and '-inf' pixels

        :param numpy.ndarray assembled: assembled image data,
            (pulse_id, y, x) for pulse-resolved data and (y, x)
            for train-resolved data.
        :param int tid: train ID.

        :return ProcessedData: pre-processed data.
        """
        mask_min, mask_max = self.threshold_mask_sp

        t0 = time.perf_counter()

        if assembled.ndim == 3:
            # pulse resolved

            assembled = assembled[self.pulse_range_sp[0]:self.pulse_range_sp[1]]
            assembled_mean = nanmean_axis0_para(assembled,
                                                max_workers=8, chunk_size=20)
        else:
            # train resolved

            # 'assembled' is a reference to the array data received from the
            # pyzmq. The array data is only readable since the data is owned
            # by a pointer in the zmq message (it is not copied). However,
            # other data like data['metadata'] is writeable.
            assembled = np.copy(assembled)

            # we want assembled to be untouched
            assembled_mean = np.copy(assembled)

        # Convert 'nan' to '-inf' and it will later be converted to the
        # lower range of mask, which is usually 0.
        # We do not convert 'nan' to 0 because: if the lower range of
        # mask is a negative value, 0 will be converted to a value
        # between 0 and 255 later.
        assembled_mean[np.isnan(assembled_mean)] = -np.inf
        # clip the array, which now will contain only numerical values
        # within the mask range
        np.clip(assembled_mean, mask_min, mask_max, out=assembled_mean)

        if self.image_mask is None:
            image_mask = np.zeros_like(assembled_mean, dtype=np.uint8)
        else:
            if self.image_mask.shape != assembled_mean.shape:
                self.log("Invalid mask shape {} for image with shape {}".
                         format(self.image_mask.shape, assembled_mean.shape))

            image_mask = self.image_mask

        logger.debug("Time for pre-processing: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        # Note: 'assembled' still contains 'inf' and '-inf', we only do
        #       the clip later when necessary in order not to waste
        #       computing power.
        data = ProcessedData(tid)
        data.images = assembled
        data.image_mean = assembled_mean
        data.threshold_mask = (mask_min, mask_max)
        data.image_mask = image_mask

        return data

    def perform_azimuthal_integration(self, data):
        """Perform azimuthal integration.

        :param ProcessedData data: data after pre-processing.

        :return ProcessedData: processed data containing azimuthal
            integration result.
        """
        assembled = data.images
        image_mask = data.image_mask
        mask_min, mask_max = data.threshold_mask

        ai = pyFAI.AzimuthalIntegrator(dist=self.sample_distance_sp,
                                       poni1=self.center_coordinate_sp[1],
                                       poni2=self.center_coordinate_sp[0],
                                       pixel1=config["PIXEL_SIZE"],
                                       pixel2=config["PIXEL_SIZE"],
                                       rot1=0,
                                       rot2=0,
                                       rot3=0,
                                       wavelength=self.wavelength_sp)

        t0 = time.perf_counter()

        if assembled.ndim == 3:
            # pulse-resolved

            def _integrate1d_imp(i):
                """Use for multiprocessing."""
                # convert 'nan' to '-inf', as explained above
                assembled[i][np.isnan(assembled[i])] = -np.inf

                mask = np.copy(image_mask)
                # merge image mask and threshold mask
                mask[(assembled[i] < mask_min) | (assembled[i] > mask_max)] = 1

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

            with ThreadPoolExecutor(max_workers=4) as executor:
                rets = executor.map(_integrate1d_imp,
                                    range(assembled.shape[0]))

            momentums, intensities = zip(*rets)
            momentum = momentums[0]
            intensities = np.array(intensities)
            intensities_mean = np.mean(intensities, axis=0)

        else:
            # train-resolved

            mask = np.copy(image_mask)
            # merge image mask and threshold mask
            mask[(assembled < mask_min) | (assembled > mask_max)] = 1

            ret = ai.integrate1d(assembled,
                                 self.integration_points_sp,
                                 method=self.integration_method_sp,
                                 mask=mask,
                                 radial_range=self.integration_range_sp,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")

            momentum = ret.radial
            # There is only one intensity data, but we use plural here to
            # be consistent with pulse-resolved data.
            intensities_mean = ret.intensity
            # for the convenience of data processing later
            intensities = np.expand_dims(intensities_mean, axis=0)

        logger.debug("Time for azimuthal integration: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        data.momentum = momentum
        data.intensities = intensities
        data.intensity_mean = intensities_mean

    def postprocess_data(self, data):
        """Data post-processing after azimuthal integration

        :param ProcessedData data: data includes azimuthal integration
            information.
        """
        # Add ROI information

        # Note: We need to put some data in the history, even if ROI is not
        # activated. This is required for the case that ROI1 and ROI2 were
        # activate at different times.
        if data.tid > 0:
            img = data.image_mean

            # it should be valid to set ROI intensity to zero if the data
            # is not available
            roi1_intensity = 0
            if self.roi1 is not None:
                if not self._validate_roi(*self.roi1, *img.shape):
                    self.roi1 = None
                else:
                    data.roi1 = self.roi1
                    w, h, cx, cy = self.roi1
                    roi1_intensity = np.sum(img[cy:cy+h, cx:cx+w])

            roi2_intensity = 0
            if self.roi2 is not None:
                if not self._validate_roi(*self.roi2, *img.shape):
                    self.roi2 = None
                else:
                    data.roi2 = self.roi2
                    w, h, cx, cy = self.roi2
                    roi2_intensity = np.sum(data.image_mean[cy:cy+h, cx:cx+w])

            data.update_roi_hist(data.tid, roi1_intensity, roi2_intensity)

        return data

    def _validate_roi(self, w, h, cx, cy, img_h, img_w):
        """Check whether the ROI is within the image."""
        if w < 0 or h < 0 or cx + w > img_w or cy + h > img_h:
            return False
        return True

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
            tid = metadata[self.source_name_sp]["timestamp.tid"]

            # Data coming from bridge in case of JungFrau will have
            # different key. To be included
            modules_data = data[self.source_name_sp]["image.data"]

            if config["DETECTOR"] == "LPD":
                # (modules, x, y, memory cells) -> (memory cells, modules, y, x)
                modules_data = np.moveaxis(np.moveaxis(modules_data, 3, 0), 3, 2)
        else:
            # get the train ID of the first metadata
            tid = next(iter(metadata.values()))["timestamp.tid"]

            expected_shape = config["EXPECTED_SHAPE"]

            if config["DETECTOR"] in ("LPD", "AGIPD"):
                try:
                    modules_data = stack_detector_data(
                        data, "image.data", only=config["DETECTOR"])

                    if not hasattr(modules_data, 'shape') \
                            or modules_data.shape[-len(expected_shape):] != expected_shape:
                        raise ValueError("Error in the shape of modules data")

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

            elif config["DETECTOR"] == 'JungFrau':
                try:
                    # (modules, y, x)
                    modules_data = data[self.source_name_sp]['data.adc']
                except KeyError:
                    logger.debug(f"Source [{self.source_name_sp}] is not in "
                                 f"the received data!")
                    return ProcessedData(tid)
            elif config["DETECTOR"] == "FastCCD":
                try:
                    # (y, x)
                    modules_data = data[self.source_name_sp]["data.image.pixels"]
                except KeyError:
                    logger.debug(f"Source [{self.source_name_sp}] is not in "
                                 f"the received data!")
                    return ProcessedData(tid)

        logger.debug("Time for moveaxis/stacking: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        t0 = time.perf_counter()

        # AGIPD, LPD
        if modules_data.ndim == 4:
            assembled, centre = self.geom_sp.position_all_modules(modules_data)
        # JungFrau
        elif modules_data.ndim == 3:
            # In the future, we may need a position_all_modules for JungFrau
            # or we can simply stack the image.
            assembled = modules_data.squeeze(axis=0)
        # FastCCD
        else:
            assembled = modules_data

        logger.debug("Time for assembling: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        # This is a bug in old version of karabo_data. The above function
        # could return a numpy.ndarray with shape (0, x, x)
        if assembled.shape[0] == 0:
            logger.debug("Bad shape {} in assembled image of train {}".
                         format(assembled.shape, tid))
            return ProcessedData(tid)

        # data processing work flow

        proc_data = self.preprocess_data(assembled, tid)

        if self._enable_ai:
            self.perform_azimuthal_integration(proc_data)
            self._laser_on_off_processor.process(proc_data)

        return self.postprocess_data(proc_data)
