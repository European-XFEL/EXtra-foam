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
from .exceptions import ProcessingError
from ..algorithms import (
    compute_spectrum, intersection, normalize_curve, slice_curve
)
from ..config import config, AiNormalizer, FomName, PumpProbeMode, RoiFom
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
        pass


class CorrelationProcessor(AbstractProcessor):
    """Add correlation information into processed data.

    Attributes:
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """
    def __init__(self):
        super().__init__()

        self.fom_name = None
        self.fom_itgt_range = None

    def process(self, proc_data, raw_data=None):
        """Override."""
        if self.fom_name is None:
            return

        if self.fom_name == FomName.AI_MEAN:
            momentum = proc_data.momentum
            if momentum is None:
                raise ProcessingError(
                    "Azimuthal integration result is not available!")
            intensity = proc_data.intensity_mean

            # calculate figure-of-merit
            fom = slice_curve(intensity, momentum, *self.fom_itgt_range)[0]
            fom = np.sum(np.abs(fom))

        elif self.fom_name == FomName.AI_PUMP_PROBE:
            _, foms, _ = proc_data.pp.foms
            if not foms:
                raise ProcessingError("Laser on-off result is not available!")
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
            raise ProcessingError(f"Unknown FOM name: {self.fom_name}!")

        for param in ProcessedData.get_correlators():
            _, _, info = getattr(proc_data.correlation, param)
            if info['device_id'] == "Any":
                # orig_data cannot be empty here
                setattr(proc_data.correlation, param, (proc_data.tid, fom))
            else:
                try:
                    device_data = raw_data[info['device_id']]
                except KeyError:
                    raise ProcessingError(
                        f"Device '{info['device_id']}' is not in the data!")

                try:
                    if info['property'] in device_data:
                        ppt = info['property']
                    else:
                        # From the file
                        ppt = info['property'] + '.value'

                    setattr(proc_data.correlation, param,
                            (device_data[ppt], fom))

                except KeyError:
                    raise ProcessingError(
                        f"'{info['device_id']}'' does not have property "
                        f"'{info['property']}'")


class SampleDegradationProcessor(AbstractProcessor):
    """SampleDegradationProcessor.

    Only for pulse-resolved detectors.

    Attributes:
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """
    def __init__(self):
        super().__init__()

        self.fom_itgt_range = None

    def process(self, proc_data, raw_data=None):
        """Override."""
        if proc_data.n_pulses == 1:
            # train-resolved
            return

        momentum = proc_data.momentum
        intensities = proc_data.intensities

        # calculate the different between each pulse and the first one
        diffs = [p - intensities[0] for p in intensities]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self.fom_itgt_range)[0]
            foms.append(np.sum(np.abs(fom)))

        proc_data.sample_degradation_foms = foms


class RoiProcessor(AbstractProcessor):
    """Process region of interest.

    Attributes:
        roi_fom (int): type of ROI FOM.
    """
    def __init__(self):
        super().__init__()

        self._rois = [None] * len(config["ROI_COLORS"])

        self.roi_fom = None

    def set(self, rank, value):
        self._rois[rank-1] = value

    def process(self, proc_data, raw_data=None):
        """Override.

        Note: We need to put some data in the history, even if ROI is not
        activated. This is required for the case that ROI1 and ROI2 were
        activate at different times.
        """
        roi_fom = self.roi_fom

        tid = proc_data.tid
        if tid > 0:
            img = proc_data.image.masked_mean
            img_ref = proc_data.image.masked_ref

            rois = copy.copy(self._rois)
            for i, roi in enumerate(rois):
                # it should be valid to set ROI intensity to zero if the data
                # is not available
                value = 0
                value_ref = 0
                if roi is not None:
                    roi = intersection(*roi, *img.shape[::-1], 0, 0)
                    if roi[0] < 0 or roi[1] < 0:
                        self._rois[i] = None
                    else:
                        setattr(proc_data.roi, f"roi{i+1}", roi)
                        value = self._get_roi_fom(roi, roi_fom, img)
                        value_ref = self._get_roi_fom(roi, roi_fom, img_ref)
                setattr(proc_data.roi, f"roi{i+1}_hist", (tid, value))
                setattr(proc_data.roi, f"roi{i + 1}_hist_ref", (tid, value_ref))

    @staticmethod
    def _get_roi_fom(roi_param, roi_fom, img):
        if roi_fom is None or img is None:
            return 0

        w, h, x, y = roi_param
        roi_img = img[y:y + h, x:x + w]
        if roi_fom == RoiFom.SUM:
            ret = np.sum(roi_img)
        elif roi_fom == RoiFom.MEAN:
            ret = np.mean(roi_img)
        else:
            ret = 0

        return ret


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
        normalizer (int): normalizer type for calculating FOM from
            azimuthal integration result.
        auc_x_range (tuple): x range for calculating AUC, which is used as
            a normalizer of the azimuthal integration.
    """
    def __init__(self):
        super().__init__()

        self.sample_distance = None
        self.wavelength = None
        self.integration_center = None
        self.integration_method = None
        self.integration_range = None
        self.integration_points = None
        self.normalizer = None
        self.auc_x_range = None

    def process(self, proc_data, raw_data=None):
        sample_distance = self.sample_distance
        wavelength = self.wavelength
        cx, cy = self.integration_center
        integration_points = self.integration_points
        integration_method = self.integration_method
        integration_range = self.integration_range

        assembled = proc_data.image.images
        reference = proc_data.image.ref
        image_mask = proc_data.image.image_mask

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
                rets = executor.map(_integrate1d_imp, range(assembled.shape[0]))

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
            # for the convenience of data processing later; use copy() here
            # to avoid to be normalized twice
            intensities = np.expand_dims(intensities_mean.copy(), axis=0)

        logger.debug("Time for azimuthal integration: {:.1f} ms"
                     .format(1000 * (time.perf_counter() - t0)))

        if reference is not None:
            mask = image_mask != 0
            # merge image mask and threshold mask
            mask[(reference < mask_min) | (reference > mask_max)] = 1

            ret = ai.integrate1d(reference,
                                 integration_points,
                                 method=integration_method,
                                 mask=mask,
                                 radial_range=integration_range,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")

            ref_intensity = ret.intensity
        else:
            ref_intensity = None

        self._normalize(
            momentum, intensities, intensities_mean, ref_intensity, proc_data)

    def _normalize(self, momentum, intensities, intensities_mean,
                   ref_intensity, proc_data):
        if self.normalizer == AiNormalizer.AUC:
            try:
                intensities_mean = normalize_curve(
                    intensities_mean, momentum, *self.auc_x_range)
                if ref_intensity is not None:
                    ref_intensity = normalize_curve(
                        ref_intensity, momentum, *self.auc_x_range)
                else:
                    ref_intensity = np.zeros_like(intensities_mean)

                # normalize azimuthal integration curves for each pulse
                for i, intensity in enumerate(intensities):
                    intensities[i][:] = normalize_curve(
                        intensity, momentum, *self.auc_x_range)

            except ValueError as e:
                raise ProcessingError(e)

        else:
            _, roi1_hist, _ = proc_data.roi.roi1_hist
            _, roi2_hist, _ = proc_data.roi.roi2_hist
            _, roi1_hist_ref, _ = proc_data.roi.roi1_hist_ref
            _, roi2_hist_ref, _ = proc_data.roi.roi2_hist_ref

            denominator = 0
            denominator_ref = 0
            try:
                if self.normalizer == AiNormalizer.ROI1:
                    denominator = roi1_hist[-1]
                    denominator_ref = roi1_hist_ref[-1]
                elif self.normalizer == AiNormalizer.ROI2:
                    denominator = roi2_hist[-1]
                    denominator_ref = roi2_hist_ref[-1]
                elif self.normalizer == AiNormalizer.ROI_SUM:
                    denominator = roi1_hist[-1] + roi2_hist[-1]
                    denominator_ref = roi1_hist_ref[-1] + roi2_hist_ref[-1]
                elif self.normalizer == AiNormalizer.ROI_SUB:
                    denominator = roi1_hist[-1] - roi2_hist[-1]
                    denominator_ref = roi1_hist_ref[-1] - roi2_hist_ref[-1]

            except IndexError as e:
                # this could happen if the history is clear just now
                raise ProcessingError(e)

            if denominator == 0:
                raise ProcessingError(
                    "Invalid normalizer: sum of ROI(s) is zero!")
            intensities_mean /= denominator
            intensities /= denominator

            if ref_intensity is not None:
                if denominator_ref == 0:
                    raise ProcessingError(
                        "Invalid reference normalizer: sum of ROI(s) is zero!")
                ref_intensity /= denominator_ref
            else:
                ref_intensity = np.zeros_like(intensities_mean)

        proc_data.momentum = momentum
        proc_data.intensities = intensities
        proc_data.intensity_mean = intensities_mean
        proc_data.reference_intensity = ref_intensity


class PumpProbeProcessor(AbstractProcessor):
    """PumpProbeProcessor class.

    A processor which calculated the average of the azimuthal integration
    of all pump/probe (on/off) pulses, as well as their difference.
    It also calculates the the figure of merit (FOM), which is integration
    of the absolute aforementioned difference.

    Attributes:
        mode (int): Pump-probe mode.
        on_pulse_ids (list): a list of laser-on pulse IDs.
        off_pulse_ids (list): a list of laser-off pulse IDs.
        abs_difference (bool): True for calculating the absolute value of
            difference between laser-on and laser-off.
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """

    def __init__(self):
        super().__init__()

        self.mode = None
        self.on_pulse_ids = None
        self.off_pulse_ids = None
        self.abs_difference = True
        self.fom_itgt_range = None

        self._prev_on = None

    def process(self, proc_data, raw_data=None):
        """Override."""
        on_pulse_ids = self.on_pulse_ids
        off_pulse_ids = self.off_pulse_ids
        momentum = proc_data.momentum
        intensities = proc_data.intensities
        ref_intensity = proc_data.reference_intensity

        n_pulses = intensities.shape[0]
        max_on_pulse_id = max(on_pulse_ids)
        if max_on_pulse_id >= n_pulses:
            raise ProcessingError(f"On-pulse ID {max_on_pulse_id} out of range "
                                  f"(0 - {n_pulses - 1})")

        if self.mode != PumpProbeMode.PRE_DEFINED_OFF:
            max_off_pulse_id = max(off_pulse_ids)
            if max_off_pulse_id >= n_pulses:
                raise ProcessingError(f"Off-pulse ID {max_off_pulse_id} out of "
                                      f"range (0 - {n_pulses - 1})")

        on_train_received = False
        off_train_received = False
        if self.mode in (PumpProbeMode.PRE_DEFINED_OFF, PumpProbeMode.SAME_TRAIN):
            # compare laser-on/off pulses in the same train
            on_train_received = True
            off_train_received = True
        else:
            # compare laser-on/off pulses in different trains
            if self.mode == PumpProbeMode.EVEN_TRAIN_ON:
                flag = 0  # on-train has even train ID
            elif self.mode == PumpProbeMode.ODD_TRAIN_ON:
                flag = 1  # on-train has odd train ID
            else:
                raise ProcessingError(f"Unknown laser mode: {self.mode}")

            if proc_data.tid % 2 == 1 ^ flag:
                off_train_received = True
            else:
                on_train_received = True

        # Off-train will only be acknowledged when an on-train
        # was received! This ensures that in the visualization
        # it always shows the on-train plot alone first, which
        # is followed by a combined plots if the next train is
        # an off-train pulse.

        diff = None
        fom = None
        if on_train_received:
            self._prev_on = intensities[on_pulse_ids].mean(axis=0)

        on_pulse = self._prev_on
        off_pulse = None

        if off_train_received and on_pulse is not None:
            if self.mode == PumpProbeMode.PRE_DEFINED_OFF:
                off_pulse = ref_intensity
            else:
                off_pulse = intensities[off_pulse_ids].mean(axis=0)

            diff = on_pulse - off_pulse

            # calculate figure-of-merit and update history
            fom = slice_curve(diff, momentum, *self.fom_itgt_range)[0]
            if self.abs_difference:
                fom = np.sum(np.abs(fom))
            else:
                fom = np.sum(fom)

            # reset flags
            self._prev_on = None

        proc_data.pp.on_pulse = on_pulse
        proc_data.pp.off_pulse = off_pulse
        proc_data.pp.diff = diff
        proc_data.pp.foms = (proc_data.tid, fom)

    def reset(self):
        """Override."""
        self._prev_on = None


class XasProcessor(AbstractProcessor):
    """XasProcessor class.

    A processor which calculate absorption spectra based on different
    ROIs specified by the user.
    """

    def __init__(self):
        super().__init__()

        self.n_bins = 10

        self._energies = []
        self._xgm = []
        self._I0 = []
        self._I1 = []
        self._I2 = []

        self._bin_center = None
        self._absorptions = None
        self._bin_count = None

        # we do not need to re-calculate the spectrum for every train, since
        # there is only one more data for detectors like FastCCD.
        self._counter = 0
        self._update_frequency = 10

        self.reset()

    def process(self, proc_data, raw_data=None):
        """Override."""
        xgm = proc_data.xgm
        mono = proc_data.mono
        _, roi1_hist, _ = proc_data.roi.roi1_hist
        _, roi2_hist, _ = proc_data.roi.roi2_hist
        _, roi3_hist, _ = proc_data.roi.roi3_hist

        self._energies.append(mono.energy)
        self._xgm.append(xgm.intensity)
        self._I0.append(roi1_hist[-1])
        self._I1.append(roi2_hist[-1])
        self._I2.append(roi3_hist[-1])

        if self._counter == self._update_frequency:
            # re-calculate the spectra
            bin_center, absorptions, bin_count = compute_spectrum(
                self._energies, self._I0, [self._I1, self._I2], self.n_bins)

            self._bin_center = bin_center
            self._absorptions = absorptions
            self._bin_count = bin_count

            self._counter = 0
        else:
            # use old values
            bin_center = self._bin_center
            absorptions = self._absorptions
            bin_count = self._bin_count

        self._counter += 1

        proc_data.xas.bin_center = bin_center
        proc_data.xas.absorptions = absorptions
        proc_data.xas_bin_count = bin_count

    def reset(self):
        self._energies.clear()
        self._xgm.clear()
        self._I0.clear()
        self._I1.clear()
        self._I2.clear()

        self._bin_center = np.array([])
        self._bin_count = np.array([])
        self._absorptions = [np.array([]), np.array([])]

        self._counter = 0
