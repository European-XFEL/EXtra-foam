"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data processors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import queue

import numpy as np
from scipy import constants
from h5py import File

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from karabo_data import stack_detector_data
from karabo_data.geometry import LPDGeometry

from .data_model import ProcessedData
from .worker import Worker
from ..algorithms import normalize_curve, slice_curve
from ..config import (
    AiNormalizer, config, DataSource, FomName, OpLaserMode, RoiValueType
)
from ..gui import QtCore
from ..logger import logger


class AbstractProcessor(QtCore.QObject):
    """Base class for specific data processor."""

    message_sgn = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        if parent is not None:
            parent.register_processor(self)

    def log(self, msg):
        """Log information in the main GUI."""
        self.message_sgn.emit(msg)

    def process(self, data):
        raise NotImplementedError


class CorrelationProcessor(AbstractProcessor):
    """CorrelationProcessor class.

    Add correlation information into processed data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fom_name = None

        self.normalizer = None
        self.auc_x_range = None
        self.fom_itgt_range = None

    def process(self, data):
        """Process the data

        :param tuple data: (ProcessedData, dict) where the former should
            contain azimuthal integration information while the later is
            a reference to the original data received from the bridge.
        """
        proc_data, orig_data = data

        if self.fom_name is None:
            return

        if self.fom_name == FomName.AI_MEAN:
            momentum = proc_data.momentum
            if momentum is None:
                self.log("Azimuthal integration result is not available!")
                return
            intensity = proc_data.intensity_mean

            if self.normalizer == AiNormalizer.AUC:
                normalized_intensity = normalize_curve(
                    intensity, momentum, *self.auc_x_range)
            elif self.normalizer == AiNormalizer.ROI:
                _, roi1_hist, _ = proc_data.roi.roi1_hist
                _, roi2_hist, _ = proc_data.roi.roi2_hist

                try:
                    denominator = (roi1_hist[-1] + roi2_hist[-1])/2.
                except IndexError:
                    # this could happen if the history is clear just now
                    # TODO: we may need to improve here
                    return

                if denominator == 0:
                    self.log("ROI value is zero!")
                    return
                normalized_intensity = intensity / denominator

            else:
                self.log("Unexpected normalizer!")
                return

            # calculate figure-of-merit
            fom = slice_curve(
                normalized_intensity, momentum, *self.fom_itgt_range)[0]
            fom = np.sum(np.abs(fom))

        elif self.fom_name == FomName.AI_ON_OFF:
            _, foms, _ = proc_data.on_off.foms
            if not foms:
                self.log("Laser on-off result is not available!")
                return
            fom = foms[-1]

        elif self.fom_name == FomName.ROI1:
            _, roi1_hist, _ = proc_data.roi.roi1_hist
            fom = roi1_hist[-1]

        elif self.fom_name == FomName.ROI2:
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            fom = roi2_hist[-1]

        elif self.fom_name == FomName.ROI_SUM:
            _, roi1_hist, _ = proc_data.roi.roi1_hist
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            fom = roi1_hist[-1] + roi2_hist[-1]

        elif self.fom_name == FomName.ROI_SUB:
            _, roi1_hist, _ = proc_data.roi.roi1_hist
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            fom = roi1_hist[-1] - roi2_hist[-1]

        else:
            self.log("Unexpected FOM name!")
            return

        for param in ProcessedData.get_correlators():
            _, _, info = getattr(proc_data.correlation, param)
            if info['device_id'] == "Any":
                # orig_data cannot be empty here
                setattr(proc_data.correlation, param, (fom, proc_data.tid))
            else:
                try:
                    device_data = orig_data[info['device_id']]
                except KeyError:
                    self.log(f"Device '{info['device_id']}' is not in the data!")
                    continue

                try:
                    setattr(proc_data.correlation, param,
                            (fom, device_data[info['property']]))
                except KeyError:
                    self.log(f"{info['device_id']} does not have property "
                             f"'{info['property']}'")


class SampleDegradationProcessor(AbstractProcessor):
    """Process the data.

    Note: only for pulse-resolved detectors.

    :param ProcessedData proc_data: data after the assembling and
        azimuthal integration for individual pulses.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.auc_x_range = None
        self.fom_itgt_range = None

    def process(self, proc_data):
        """Process the proc_data.

        :param ProcessedData proc_data: data after the assembling and
            azimuthal integration for individual pulses.
        """
        if proc_data.n_pulses == 1:
            # train-resolved
            return

        momentum = proc_data.momentum
        intensities = proc_data.intensities

        # normalize azimuthal integration curves for each pulse
        normalized_pulse_intensities = []
        for pulse_intensity in intensities:
            normalized = normalize_curve(
                pulse_intensity, momentum, *self.auc_x_range)
            normalized_pulse_intensities.append(normalized)

        # calculate the different between each pulse and the first one
        diffs = [p - normalized_pulse_intensities[0]
                 for p in normalized_pulse_intensities]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self.fom_itgt_range)[0]
            foms.append(np.sum(np.abs(fom)))

        proc_data.sample_degradation_foms = foms


class LaserOnOffProcessor(AbstractProcessor):
    """LaserOnOffProcessor class.

    A processor which calculated the moving average of the average of the
    azimuthal integration of all laser-on and laser-off pulses, as well
    as their difference. It also calculates the the figure of merit (FOM),
    which is integration of the absolute aforementioned difference.
    """

    message_sgn = QtCore.pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.laser_mode = None
        self.on_pulse_ids = None
        self.off_pulse_ids = None

        self.abs_difference = True

        self.moving_avg_window = 1

        self.normalizer = None
        self.auc_x_range = None
        self.fom_itgt_range = None

        self._on_train_received = False
        self._off_train_received = False

        # if an on-pulse is followed by an on-pulse, drop the previous one
        self._drop_last_on_pulse = False

        # moving average
        self._on_pulses_ma = None
        self._off_pulses_ma = None
        # The histories of on/off pulses by train, which are used in
        # calculating moving average (MA)
        self._on_pulses_hist = deque()
        self._off_pulses_hist = deque()

    def process(self, proc_data):
        """Process the data.

        :param ProcessedData proc_data: data after the assembling and
            azimuthal integration for individual pulses.
        """
        if self.laser_mode == OpLaserMode.INACTIVE:
            return

        momentum = proc_data.momentum
        intensities = proc_data.intensities

        n_pulses = intensities.shape[0]
        max_on_pulse_id = max(self.on_pulse_ids)
        if max_on_pulse_id >= n_pulses:
            self.log(f"On-pulse ID {max_on_pulse_id} out of range "
                     f"(0 - {n_pulses - 1})")
            return

        max_off_pulse_id = max(self.off_pulse_ids)
        if max_off_pulse_id >= n_pulses:
            self.log(f"Off-pulse ID {max_off_pulse_id} out of range "
                     f"(0 - {n_pulses - 1})")
            return

        if self.laser_mode == OpLaserMode.NORMAL:
            # compare laser-on/off pulses in the same train
            self._on_train_received = True
            self._off_train_received = True
        else:
            # compare laser-on/off pulses in different trains
            if self.laser_mode == OpLaserMode.NORMAL.EVEN_ON:
                flag = 0  # on-train has even train ID
            elif self.laser_mode == OpLaserMode.ODD_ON:
                flag = 1  # on-train has odd train ID
            else:
                self.log(f"Unexpected laser mode! {self.laser_mode}")
                return

            # Off-train will only be acknowledged when an on-train
            # was received! This ensures that in the visualization
            # it always shows the on-train plot alone first, which
            # is followed by a combined plots if the next train is
            # an off-train pulse.
            #
            # Note: if this logic changes, one also need to modify
            #       the visualization part.
            if self._on_train_received:
                if proc_data.tid % 2 == 1 ^ flag:
                    # an on-pulse is followed by an off-pulse
                    self._off_train_received = True
                else:
                    # an on-pulse is followed by an on-pulse
                    self._drop_last_on_pulse = True
            else:
                # an off-pulse is followed by an on-pulse
                if proc_data.tid % 2 == flag:
                    self._on_train_received = True

        # update and plot

        normalized_on_pulse = None
        normalized_off_pulse = None

        if self._on_train_received:
            # update on-pulse
            if self.laser_mode == OpLaserMode.NORMAL or \
                    not self._off_train_received:

                this_on_pulses = intensities[self.on_pulse_ids].mean(axis=0)

                if self._drop_last_on_pulse:
                    length = len(self._on_pulses_hist)
                    self._on_pulses_ma += \
                        (this_on_pulses - self._on_pulses_hist.pop()) / length
                    self._drop_last_on_pulse = False
                else:
                    if self._on_pulses_ma is None:
                        self._on_pulses_ma = np.copy(this_on_pulses)
                    elif len(self._on_pulses_hist) < self.moving_avg_window:
                        self._on_pulses_ma += \
                                (this_on_pulses - self._on_pulses_ma) \
                                / (len(self._on_pulses_hist) + 1)
                    elif len(self._on_pulses_hist) == self.moving_avg_window:
                        self._on_pulses_ma += \
                            (this_on_pulses - self._on_pulses_hist.popleft()) \
                            / self.moving_avg_window
                    else:
                        raise ValueError  # should never reach here

                self._on_pulses_hist.append(this_on_pulses)

            normalized_on_pulse = normalize_curve(
                self._on_pulses_ma, momentum, *self.auc_x_range)

        diff = None
        fom = None
        if self._off_train_received:
            # update off-pulse

            this_off_pulses = intensities[self.off_pulse_ids].mean(axis=0)

            self._off_pulses_hist.append(this_off_pulses)

            if self._off_pulses_ma is None:
                self._off_pulses_ma = np.copy(this_off_pulses)
            elif len(self._off_pulses_hist) <= self.moving_avg_window:
                self._off_pulses_ma += \
                        (this_off_pulses - self._off_pulses_ma) \
                        / len(self._off_pulses_hist)
            elif len(self._off_pulses_hist) == self.moving_avg_window + 1:
                self._off_pulses_ma += \
                    (this_off_pulses - self._off_pulses_hist.popleft()) \
                    / self.moving_avg_window
            else:
                raise ValueError  # should never reach here

            normalized_off_pulse = normalize_curve(
                self._off_pulses_ma, momentum, *self.auc_x_range)

            diff = normalized_on_pulse - normalized_off_pulse

            # calculate figure-of-merit and update history
            fom = slice_curve(diff, momentum, *self.fom_itgt_range)[0]
            if self.abs_difference:
                fom = np.sum(np.abs(fom))
            else:
                fom = np.sum(fom)

            # an extra check
            # TODO: check whether it is necessary
            if len(self._on_pulses_hist) != len(self._off_pulses_hist):
                raise ValueError("Length of on-pulse history {} != length "
                                 "of off-pulse history {}".
                                 format(len(self._on_pulses_hist),
                                        len(self._off_pulses_hist)))

            # reset flags
            self._on_train_received = False
            self._off_train_received = False

        proc_data.on_off.on_pulse = normalized_on_pulse
        proc_data.on_off.off_pulse = normalized_off_pulse
        proc_data.on_off.diff = diff
        proc_data.on_off.foms = (proc_data.tid, fom)

    def reset(self):
        """Override."""
        self._on_train_received = False
        self._off_train_received = False
        self._drop_last_on_pulse = False
        self._on_pulses_ma = None
        self._off_pulses_ma = None
        self._on_pulses_hist.clear()
        self._off_pulses_hist.clear()


class DataProcessor(Worker):
    """Processing data received from the bridge.

    Attributes:
        source_type_sp (DataSource): type of the data source.
        source_name_sp (str): source name of the detector
        pulse_id_range_sp (tuple): (min, max) pulse ID to be processed.
            (int, int)
        geom_sp (LPDGeometry): geometry.
        wavelength_sp (float): photon wavelength in meter.
        sample_distance_sp (float): distance from the sample to the
            detector plan (orthogonal distance, not along the beam),
            in meter.
        poni_sp (tuple): (Cx, Cy), where Cx is the
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
        roi1 (tuple): (w, h, px, py) of the current ROI1.
        roi2 (tuple): (w, h, px, py) of the current ROI2.
    """
    def __init__(self, in_queue, out_queue):
        """Initialization.

        :param Queue in_queue: a queue of data from the ZMQ bridge.
        :param Queue out_queue: a queue of processed data
        """
        super().__init__()

        self._in_queue = in_queue
        self._out_queue = out_queue

        self._detector = config['DETECTOR']

        # whether to turn azimuthal integration on
        self._enable_ai = False

        # shared parameters are updated by signal-slot
        # Note: shared parameters should end with '_sp'

        self.source_type_sp = None
        self.source_name_sp = None
        self.pulse_id_range_sp = None
        self.geom_sp = None
        self.wavelength_sp = None
        self.sample_distance_sp = None
        self.poni_sp = None
        self.integration_method_sp = None
        self.integration_range_sp = None
        self.integration_points_sp = None

        self.roi1 = None
        self.roi2 = None
        self._roi_value_type = None

        self._laser_on_off_proc = LaserOnOffProcessor(parent=self)
        self._sample_degradation_proc = SampleDegradationProcessor(parent=self)
        self._correlation_proc = CorrelationProcessor(parent=self)

    @QtCore.pyqtSlot(str)
    def onMessageReceived(self, msg):
        self.log(msg)

    @QtCore.pyqtSlot(object)
    def onSourceTypeChange(self, value):
        self.source_type_sp = value

    @QtCore.pyqtSlot(str)
    def onSourceNameChange(self, value):
        self.source_name_sp = value

    @QtCore.pyqtSlot(str, list)
    def onGeometryChange(self, filename, quad_positions):
        if self._detector == 'LPD':
            with File(filename, 'r') as f:
                self.geom_sp = LPDGeometry.from_h5_file_and_quad_positions(
                    f, quad_positions)
        elif self._detector == 'AGIPD':
            try:
                from karabo_data.geometry2 import AGIPD_1MGeometry
            except (ImportError, ModuleNotFoundError):
                self.log(
                    "You are not in the correct branch for SPB experiment!")
                raise

            self.geom_sp = AGIPD_1MGeometry.from_crystfel_geom(filename)

    @QtCore.pyqtSlot(object, list, list)
    def onOffPulseStateChange(self, mode, on_pulse_ids, off_pulse_ids):
        if mode != self._laser_on_off_proc.laser_mode:
            self._laser_on_off_proc.laser_mode = mode
            self._laser_on_off_proc.reset()
            ProcessedData.clear_onoff_hist()

        self._laser_on_off_proc.on_pulse_ids = on_pulse_ids
        self._laser_on_off_proc.off_pulse_ids = off_pulse_ids

    @QtCore.pyqtSlot(int)
    def onAbsDifferenceStateChange(self, state):
        self._laser_on_off_proc.abs_difference = state == QtCore.Qt.Checked

    @QtCore.pyqtSlot(int)
    def onMovingAverageWindowChange(self, value):
        self._laser_on_off_proc.moving_avg_window = value

    @QtCore.pyqtSlot(float, float)
    def onAucXRangeChange(self, lb, ub):
        self._laser_on_off_proc.auc_x_range = (lb, ub)
        self._sample_degradation_proc.auc_x_range = (lb, ub)
        self._correlation_proc.auc_x_range = (lb, ub)

    @QtCore.pyqtSlot(float, float)
    def onFomIntegrationRangeChange(self, lb, ub):
        self._laser_on_off_proc.fom_itgt_range = (lb, ub)
        self._sample_degradation_proc.fom_itgt_range = (lb, ub)
        self._correlation_proc.fom_itgt_range = (lb, ub)

    @QtCore.pyqtSlot(object)
    def onAiNormalizeChange(self, normalizer):
        self._laser_on_off_proc.normalizer = normalizer
        self._correlation_proc.normalizer = normalizer

    @QtCore.pyqtSlot(float)
    def onSampleDistanceChange(self, value):
        self.sample_distance_sp = value

    @QtCore.pyqtSlot(int, int)
    def onPoniChange(self, cx, cy):
        self.poni_sp = (cy, cx)  # be careful with the sequence

    @QtCore.pyqtSlot(str)
    def onIntegrationMethodChange(self, value):
        self.integration_method_sp = value

    @QtCore.pyqtSlot(float, float)
    def onIntegrationRangeChange(self, lb, ub):
        self.integration_range_sp = (lb, ub)

    @QtCore.pyqtSlot(int)
    def onIntegrationPointsChange(self, value):
        self.integration_points_sp = value

    @QtCore.pyqtSlot(float)
    def onPhotonEnergyChange(self, photon_energy):
        """Compute photon wavelength (m) from photon energy (keV)."""
        # Plank-einstein relation (E=hv)
        HC_E = 1e-3 * constants.c * constants.h / constants.e
        self.wavelength_sp = HC_E / photon_energy

    @QtCore.pyqtSlot(int, int)
    def onPulseIdRangeChange(self, lb, ub):
        self.pulse_id_range_sp = (lb, ub)

    @QtCore.pyqtSlot()
    def onLaserOnOffClear(self):
        ProcessedData.clear_onoff_hist()
        self._laser_on_off_proc.reset()

    @QtCore.pyqtSlot(int)
    def onEnableAiStateChange(self, state):
        self._enable_ai = state == QtCore.Qt.Checked

    @QtCore.pyqtSlot()
    def onCorrelationClear(self):
        ProcessedData.clear_correlation_hist()

    @QtCore.pyqtSlot(int, str, str)
    def onCorrelationParamChange(self, idx, device_id, ppt):
        ProcessedData.add_correlator(idx, device_id, ppt)

    @QtCore.pyqtSlot(object)
    def onCorrelationFomChange(self, fom):
        self._correlation_proc.fom_name = fom
        # clear all correlation data
        ProcessedData.clear_correlation_hist()

    def update_roi1_region(self, activated, w, h, px, py):
        if activated:
            self.roi1 = (w, h, px, py)
        else:
            self.roi1 = None

    def update_roi2_region(self, activated, w, h, px, py):
        if activated:
            self.roi2 = (w, h, px, py)
        else:
            self.roi2 = None

    def clear_roi_hist(self):
        ProcessedData.clear_roi_hist()

    def update_roi_value_type(self, value):
        self._roi_value_type = value
        ProcessedData.clear_roi_hist()

    def register_processor(self, processor):
        processor.message_sgn.connect(self.onMessageReceived)

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

    def perform_azimuthal_integration(self, data):
        """Perform azimuthal integration.

        :param ProcessedData data: data after pre-processing.
        """
        assembled = data.image.images
        try:
            image_mask = data.image.image_mask
        except ValueError as e:
            self.log(str(e) + "\nInvalid image mask!")
            raise
        pixel_size = data.image.pixel_size
        poni = data.image.poni
        mask_min, mask_max = data.image.threshold_mask

        ai = AzimuthalIntegrator(dist=self.sample_distance_sp,
                                 poni1=poni[0] * pixel_size,
                                 poni2=poni[1] * pixel_size,
                                 pixel1=pixel_size,
                                 pixel2=pixel_size,
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

            mask = image_mask != 0
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

    def process_roi(self, data):
        """Add ROI information into the processed data.

        Note: We need to put some data in the history, even if ROI is not
        activated. This is required for the case that ROI1 and ROI2 were
        activate at different times.

        :param ProcessedData data: data after preprocessing.
        """
        def get_roi_value(roi_param, full_image):
            w, h, px, py = roi_param
            roi_img = full_image[py:py + h, px:px + w]
            if self._roi_value_type == RoiValueType.SUM:
                ret = np.sum(roi_img)
            elif self._roi_value_type == RoiValueType.MEAN:
                ret = np.mean(roi_img)
            else:
                ret = 0

            return ret

        def validate_roi(w, h, px, py, img_h, img_w):
            """Check whether the ROI is within the image."""
            if px < 0 or py < 0 or px + w > img_w or py + h > img_h:
                return False
            return True

        tid = data.tid
        if tid > 0:
            img = data.image.masked_mean

            # it should be valid to set ROI intensity to zero if the data
            # is not available
            roi1_value = 0
            if self.roi1 is not None:
                if not validate_roi(*self.roi1, *img.shape):
                    self.roi1 = None
                else:
                    data.roi.roi1 = self.roi1
                    roi1_value = get_roi_value(self.roi1, img)

            roi2_value = 0
            if self.roi2 is not None:
                if not validate_roi(*self.roi2, *img.shape):
                    self.roi2 = None
                else:
                    data.roi.roi2 = self.roi2
                    roi2_value = get_roi_value(self.roi2, img)

            data.roi.roi1_hist = (tid, roi1_value)
            data.roi.roi2_hist = (tid, roi2_value)

    def process_calibrated_data(self, calibrated_data, *, from_file=False):
        """Process the calibrated data.

        :param tuple calibrated_data: (data, metadata). See return of
            KaraboBridge.Client.next().
        :param bool from_file: True for data streamed from files and False
            for data from the online ZMQ bridge.

        :return ProcessedData: processed data which is ready for
            visualization.
        """
        data, metadata = calibrated_data

        detector = self._detector
        src_name = self.source_name_sp

        t0 = time.perf_counter()

        if not from_file:
            tid = metadata[src_name]["timestamp.tid"]

            det_data = data[src_name]

            if detector in ('AGIPD', 'LPD'):
                modules_data = det_data["image.data"]
                if detector == 'LPD':
                    # (modules, x, y, memory cells)
                    # -> (memory cells, modules, y, x)
                    modules_data = np.moveaxis(
                        np.moveaxis(modules_data, 3, 0), 3, 2)

            elif detector == "JungFrau":
                # (y, x, modules)
                modules_data = det_data["data.adc"]
                if modules_data.shape[-1] == 1:
                    modules_data = modules_data.squeeze(axis=-1)
                else:
                    raise NotImplementedError

            elif detector == "FastCCD":
                # (y, x, 1)
                modules_data = det_data["data.image"]
                modules_data = modules_data.squeeze(axis=-1)

            else:
                raise NotImplementedError

        else:
            # get the train ID of the first metadata
            tid = next(iter(metadata.values()))["timestamp.tid"]

            expected_shape = config["EXPECTED_SHAPE"]

            if detector in ('AGIPD', 'LPD'):
                try:
                    modules_data = stack_detector_data(
                        data, "image.data", only=detector)

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
                    self.log("Error in stacking detector data: " + str(e))
                    return ProcessedData(tid)

            elif detector == 'JungFrau':
                if src_name not in data:
                    self.log(f"Source [{src_name}] is not in "
                             f"the received data!")
                    return ProcessedData(tid)
                else:
                    try:
                        # (modules, y, x)
                        modules_data = data[src_name]['data.adc']
                        if modules_data.shape[0] == 1:
                            modules_data = modules_data.squeeze(axis=0)
                        else:
                            raise NotImplementedError
                    except KeyError:
                        raise

            elif detector == "FastCCD":
                if src_name not in data:
                    self.log(f"Source [{src_name}] is not in "
                             f"the received data!")
                    return ProcessedData(tid)
                else:
                    try:
                        # (y, x)
                        modules_data = data[src_name][
                            'data.image.pixels']
                    except KeyError:
                        raise
            else:
                raise NotImplementedError

        logger.debug("Time for moveaxis/stacking: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        t0 = time.perf_counter()

        # Note:
        #
        # For train-resolved detector, assembled is a reference
        # to the array data received from the pyzmq. This array data
        # is only readable since the data is owned by a pointer in
        # the zmq message (it is not copied). However, other data
        # like data['metadata'] is writeable.

        # AGIPD, LPD
        if detector in ('AGIPD', 'LPD'):
            assembled, centre = self.geom_sp.position_all_modules(modules_data)

        # JungFrau
        elif detector == 'JungFrau':
            assembled = np.copy(modules_data)

        # FastCCD
        elif detector == 'FastCCD':
            assembled = np.copy(modules_data)

        else:
            raise NotImplementedError

        logger.debug("Time for assembling: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        # This is a bug in old version of karabo_data. The above function
        # could return a numpy.ndarray with shape (0, x, x)
        if assembled.shape[0] == 0:
            self.log("Bad shape {} in assembled image of train {}".
                     format(assembled.shape, tid))
            return ProcessedData(tid)

        if assembled.ndim == 3:
            assembled = assembled[slice(*self.pulse_id_range_sp)]

        # data processing work flow:
        #
        # pre-processing -> retrieve ROI information -> perform azimuthal
        # integration -> perform laser on-off analysis -> add correlation
        # information

        proc_data = ProcessedData(tid, assembled, poni=self.poni_sp)

        self.process_roi(proc_data)

        if self._enable_ai:
            self.perform_azimuthal_integration(proc_data)
            self._sample_degradation_proc.process(proc_data)
            self._laser_on_off_proc.process(proc_data)

        # Process correlation after laser-on-off since correlation may
        # requires the laser-on-off result
        self._correlation_proc.process((proc_data, data))

        return proc_data
