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
import numpy as np

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from .data_model import ProcessedData
from ..algorithms import normalize_curve, slice_curve
from ..config import AiNormalizer, FomName, OpLaserMode, RoiValueType
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
            if not roi1_hist:
                return
            fom = roi1_hist[-1]

        elif self.fom_name == FomName.ROI2:
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            if not roi2_hist:
                return
            fom = roi2_hist[-1]

        elif self.fom_name == FomName.ROI_SUM:
            _, roi1_hist, _ = proc_data.roi.roi1_hist
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            if not roi1_hist:
                return
            fom = roi1_hist[-1] + roi2_hist[-1]

        elif self.fom_name == FomName.ROI_SUB:
            _, roi1_hist, _ = proc_data.roi.roi1_hist
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            if not roi1_hist:
                return
            fom = roi1_hist[-1] - roi2_hist[-1]

        else:
            self.log("Unexpected FOM name!")
            return

        for param in ProcessedData.get_correlators():
            _, _, info = getattr(proc_data.correlation, param)
            if info['device_id'] == "Any":
                # orig_data cannot be empty here
                setattr(proc_data.correlation, param, (proc_data.tid, fom))
            else:
                try:
                    device_data = orig_data[info['device_id']]
                except KeyError:
                    self.log(f"Device '{info['device_id']}' is not in the data!")
                    continue

                try:
                    if info['property'] in device_data:
                        ppt = info['property']
                    else:
                        # From the file
                        ppt = info['property'] + '.value'

                    setattr(proc_data.correlation, param,
                            (device_data[ppt], fom))

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


class RegionOfInterestProcessor(AbstractProcessor):
    """Process region of interest.

    Attributes:
        roi1 (tuple): (w, h, px, py) of the current ROI1.
        roi2 (tuple): (w, h, px, py) of the current ROI2.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.roi1 = None
        self.roi2 = None
        self.roi_value_type = None

    def process(self, data):
        """Add ROI information into the processed data.

        Note: We need to put some data in the history, even if ROI is not
        activated. This is required for the case that ROI1 and ROI2 were
        activate at different times.

        :param ProcessedData data: data after preprocessing.
        """
        roi1 = self.roi1
        roi2 = self.roi2
        roi_value_type = self.roi_value_type

        tid = data.tid
        if tid > 0:
            img = data.image.masked_mean

            # it should be valid to set ROI intensity to zero if the data
            # is not available
            roi1_value = 0
            if roi1 is not None:
                if not self._validate_roi(*roi1, *img.shape):
                    self.roi1 = None
                else:
                    data.roi.roi1 = roi1
                    roi1_value = self._get_roi_value(roi1, roi_value_type, img)

            roi2_value = 0
            if roi2 is not None:
                if not self._validate_roi(*roi2, *img.shape):
                    self.roi2 = None
                else:
                    data.roi.roi2 = roi2
                    roi2_value = self._get_roi_value(roi2, roi_value_type, img)

            data.roi.roi1_hist = (tid, roi1_value)
            data.roi.roi2_hist = (tid, roi2_value)

    @staticmethod
    def _get_roi_value(roi_param, roi_value_type, full_image):
        w, h, px, py = roi_param
        roi_img = full_image[py:py + h, px:px + w]
        if roi_value_type == RoiValueType.SUM:
            ret = np.sum(roi_img)
        elif roi_value_type == RoiValueType.MEAN:
            ret = np.mean(roi_img)
        else:
            ret = 0

        return ret

    @staticmethod
    def _validate_roi(w, h, px, py, img_h, img_w):
        """Check whether the ROI is within the image."""
        if px < 0 or py < 0 or px + w > img_w or py + h > img_h:
            return False
        return True


class AzimuthalIntegrationProcessor(AbstractProcessor):
    """Perform azimuthal integration.

    Attributes:
        wavelength (float): photon wavelength in meter.
        sample_distance (float): distance from the sample to the
            detector plan (orthogonal distance, not along the beam),
            in meter.
        poni (tuple): (Cy, Cx), where Cy is the coordinate of the
            point of normal incidence along the detector's first
            dimension, in pixels, and Cx is the coordinate of the
            point of normal incidence along the detector's second
            dimension, in pixels. (int, int)
        integration_method (string): the azimuthal integration
            method supported by pyFAI.
        integration_range (tuple): the lower and upper range of
            the integration radial unit. (float, float)
        integration_points (int): number of points in the
            integration output pattern.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sample_distance = None
        self.wavelength = None
        self.poni = None
        self.integration_method = None
        self.integration_range = None
        self.integration_points = None

        self.auc_x_range = None
        self.fom_itgt_range = None

    def process(self, data):
        sample_distance = self.sample_distance
        wavelength = self.wavelength
        poni1, poni2 = self.poni
        integration_points = self.integration_points
        integration_method = self.integration_method
        integration_range = self.integration_range

        assembled = data.image.images
        try:
            image_mask = data.image.image_mask
        except ValueError as e:
            self.log(str(e) + "\nInvalid image mask!")
            raise
        pixel_size = data.image.pixel_size
        poni1, poni2 = data.image.poni(poni1, poni2)
        mask_min, mask_max = data.image.threshold_mask

        ai = AzimuthalIntegrator(dist=sample_distance,
                                 poni1=poni1 * pixel_size,
                                 poni2=poni2 * pixel_size,
                                 pixel1=pixel_size,
                                 pixel2=pixel_size,
                                 rot1=0,
                                 rot2=0,
                                 rot3=0,
                                 wavelength=wavelength)

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
                                     integration_points,
                                     method=integration_method,
                                     mask=mask,
                                     radial_range=integration_range,
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
                                 integration_points,
                                 method=integration_method,
                                 mask=mask,
                                 radial_range=integration_range,
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
