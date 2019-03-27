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
import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from .data_model import ProcessedData
from ..algorithms import normalize_curve, slice_curve
from ..config import config, AiNormalizer, FomName, OpLaserMode, RoiValueType
from ..logger import logger


class AbstractProcessor:
    """Base class for specific data processor."""

    def __init__(self):
        self.__enabled = True

        self.next = None  # next processor in the pipeline

    def setEnabled(self, state):
        self.__enabled = state

    def isEnabled(self):
        return self.__enabled

    def process(self, proc_data, raw_data=None):
        """Process data.

        :param ProcessedData proc_data: processed data.
        :param dict raw_data: raw data received from the bridge.

        :return str: error message.
        """
        raise NotImplementedError


class HeadProcessor(AbstractProcessor):
    """Head node of a processor graph."""
    def process(self, proc_data, raw_data=None):
        return ''


class CorrelationProcessor(AbstractProcessor):
    """Add correlation information into processed data.

    Attributes:
        normalizer (int): normalizer type for calculating FOM from
            azimuthal integration result.
        auc_x_range (tuple): x range for calculating AUC, which is used as
            a normalizer of the azimuthal integration.
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """
    def __init__(self):
        super().__init__()

        self.fom_name = None

        self.normalizer = None
        self.auc_x_range = None
        self.fom_itgt_range = None

    def process(self, proc_data, raw_data=None):
        """Override."""
        if self.fom_name is None:
            return

        if self.fom_name == FomName.AI_MEAN:
            momentum = proc_data.momentum
            if momentum is None:
                return "Azimuthal integration result is not available!"
            intensity = proc_data.intensity_mean

            if self.normalizer == AiNormalizer.AUC:
                normalized_intensity = normalize_curve(
                    intensity, momentum, *self.auc_x_range)
            elif self.normalizer == AiNormalizer.ROI:
                _, roi1_hist, _ = proc_data.roi.roi1_hist
                _, roi2_hist, _ = proc_data.roi.roi2_hist

                try:
                    denominator = (roi1_hist[-1] + roi2_hist[-1])/2.
                except IndexError as e:
                    # this could happen if the history is clear just now
                    return repr(e)

                if denominator == 0:
                    return "ROI value is zero!"
                normalized_intensity = intensity / denominator

            else:
                return f"Unknown normalizer: {self.normalizer}!"

            # calculate figure-of-merit
            fom = slice_curve(
                normalized_intensity, momentum, *self.fom_itgt_range)[0]
            fom = np.sum(np.abs(fom))

        elif self.fom_name == FomName.AI_ON_OFF:
            _, foms, _ = proc_data.on_off.foms
            if not foms:
                return "Laser on-off result is not available!"
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
            return f"Unknown FOM name: {self.fom_name}!"

        for param in ProcessedData.get_correlators():
            _, _, info = getattr(proc_data.correlation, param)
            if info['device_id'] == "Any":
                # orig_data cannot be empty here
                setattr(proc_data.correlation, param, (proc_data.tid, fom))
            else:
                try:
                    device_data = raw_data[info['device_id']]
                except KeyError:
                    return f"Device '{info['device_id']}' is not in the data!"

                try:
                    if info['property'] in device_data:
                        ppt = info['property']
                    else:
                        # From the file
                        ppt = info['property'] + '.value'

                    setattr(proc_data.correlation, param,
                            (device_data[ppt], fom))

                except KeyError:
                    return f"'{info['device_id']}'' does not have property " \
                           f"'{info['property']}'"


class SampleDegradationProcessor(AbstractProcessor):
    """SampleDegradationProcessor.

    Only for pulse-resolved detectors.

    Attributes:
        auc_x_range (tuple): x range for calculating AUC, which is used as
            a normalizer of the azimuthal integration.
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """
    def __init__(self):
        super().__init__()

        self.auc_x_range = None
        self.fom_itgt_range = None

    def process(self, proc_data, raw_data=None):
        """Override."""
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
        roi_value_type (int): type of ROI value.
    """
    def __init__(self):
        super().__init__()

        self._rois = [None] * len(config["ROI_COLORS"])

        self.roi_value_type = None

    def get_roi(self, rank):
        return self._rois[rank-1]

    def set_roi(self, rank, value):
        self._rois[rank-1] = value

    def process(self, proc_data, raw_data=None):
        """Override.

        Note: We need to put some data in the history, even if ROI is not
        activated. This is required for the case that ROI1 and ROI2 were
        activate at different times.
        """
        rois = copy.copy(self._rois)
        roi_value_type = self.roi_value_type

        tid = proc_data.tid
        if tid > 0:
            img = proc_data.image.masked_mean

            for i, roi in enumerate(rois):
                # it should be valid to set ROI intensity to zero if the data
                # is not available
                value = 0
                if roi is not None:
                    if not self._validate_roi(*roi, *img.shape):
                        self._rois[i] = None
                    else:
                        setattr(proc_data.roi, f"roi{i+1}", roi)
                        value = self._get_roi_value(roi, roi_value_type, img)

                setattr(proc_data.roi, f"roi{i+1}_hist", (tid, value))

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
        integration_center (tuple): (Cx, Cy) in pixels. (int, int)
        integration_method (string): the azimuthal integration
            method supported by pyFAI.
        integration_range (tuple): the lower and upper range of
            the integration radial unit. (float, float)
        integration_points (int): number of points in the
            integration output pattern.
    """
    def __init__(self):
        super().__init__()

        self.sample_distance = None
        self.wavelength = None
        self.integration_center = None
        self.integration_method = None
        self.integration_range = None
        self.integration_points = None

    def process(self, proc_data, raw_data=None):
        sample_distance = self.sample_distance
        wavelength = self.wavelength
        cx, cy = self.integration_center
        integration_points = self.integration_points
        integration_method = self.integration_method
        integration_range = self.integration_range

        assembled = proc_data.image.images
        try:
            image_mask = proc_data.image.image_mask
        # TODO: check! why ValueError?
        except ValueError as e:
            return repr(e) + ": Invalid image mask!"

        pixel_size = proc_data.image.pixel_size
        poni2, poni1 = proc_data.image.pos_inv(cx, cy)
        mask_min, mask_max = proc_data.image.threshold_mask

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

        proc_data.momentum = momentum
        proc_data.intensities = intensities
        proc_data.intensity_mean = intensities_mean


class LaserOnOffProcessor(AbstractProcessor):
    """LaserOnOffProcessor class.

    A processor which calculated the moving average of the average of the
    azimuthal integration of all laser-on and laser-off pulses, as well
    as their difference. It also calculates the the figure of merit (FOM),
    which is integration of the absolute aforementioned difference.

    Attributes:
        laser_mode (int): Laser on/off mode.
        on_pulse_ids (list): a list of laser-on pulse IDs.
        off_pulse_ids (list): a list of laser-off pulse IDs.
        abs_difference (bool): True for calculating the absolute value of
            difference between laser-on and laser-off.
        moving_avg_window (int): moving average window size.
        auc_x_range (tuple): x range for calculating AUC, which is used as
            a normalizer of the azimuthal integration.
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """

    def __init__(self):
        super().__init__()

        self.laser_mode = None
        self.on_pulse_ids = None
        self.off_pulse_ids = None

        self.abs_difference = True

        self.moving_avg_window = 1

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

    def process(self, proc_data, raw_data=None):
        """Override."""
        if self.laser_mode == OpLaserMode.PRE_DEFINED_OFF:
            return

        momentum = proc_data.momentum
        intensities = proc_data.intensities

        n_pulses = intensities.shape[0]
        max_on_pulse_id = max(self.on_pulse_ids)
        if max_on_pulse_id >= n_pulses:
            return f"On-pulse ID {max_on_pulse_id} out of range " \
                   f"(0 - {n_pulses - 1})"

        max_off_pulse_id = max(self.off_pulse_ids)
        if max_off_pulse_id >= n_pulses:
            return f"Off-pulse ID {max_off_pulse_id} out of range " \
                   f"(0 - {n_pulses - 1})"

        if self.laser_mode == OpLaserMode.SAME_TRAIN:
            # compare laser-on/off pulses in the same train
            self._on_train_received = True
            self._off_train_received = True
        else:
            # compare laser-on/off pulses in different trains
            if self.laser_mode == OpLaserMode.EVEN_TRAIN_ON:
                flag = 0  # on-train has even train ID
            elif self.laser_mode == OpLaserMode.ODD_TRAIN_ON:
                flag = 1  # on-train has odd train ID
            else:
                return f"Unknown laser mode: {self.laser_mode}"

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
            if self.laser_mode == OpLaserMode.SAME_TRAIN or \
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
                        raise ValueError("Unexpected code reached!")

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
                raise ValueError("Unexpected code reached!")

            normalized_off_pulse = normalize_curve(
                self._off_pulses_ma, momentum, *self.auc_x_range)

            diff = normalized_on_pulse - normalized_off_pulse

            # calculate figure-of-merit and update history
            fom = slice_curve(diff, momentum, *self.fom_itgt_range)[0]
            if self.abs_difference:
                fom = np.sum(np.abs(fom))
            else:
                fom = np.sum(fom)

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
