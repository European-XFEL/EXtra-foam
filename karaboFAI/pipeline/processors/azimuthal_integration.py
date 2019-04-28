"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

AzimuthalIntegrationProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from .base_processor import LeafProcessor, CompositeProcessor, SharedProperty
from ..exceptions import ProcessingError
from ...algorithms import normalize_curve, slice_curve
from ...config import AiNormalizer
from ...helpers import profiler


class AzimuthalIntegrationProcessor(CompositeProcessor):
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
        fom_itgt_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """
    sample_distance = SharedProperty()
    wavelength = SharedProperty()
    integration_center = SharedProperty()
    integration_method = SharedProperty()
    integration_range = SharedProperty()
    integration_points = SharedProperty()
    normalizer = SharedProperty()
    auc_x_range = SharedProperty()
    fom_itgt_range = SharedProperty()

    def __init__(self):
        super().__init__()

        self.add(PulseAiProcessor())

        next = CompositeProcessor()
        next.add(PulseResolvedAiFomProcessor())
        self.add(next)


class PulseAiProcessor(LeafProcessor):
    @profiler("Azimuthal integration processor")
    def run(self, processed, raw=None):
        sample_distance = self.sample_distance
        wavelength = self.wavelength
        cx, cy = self.integration_center
        integration_points = self.integration_points
        integration_method = self.integration_method
        integration_range = self.integration_range

        assembled = processed.image.images
        reference = processed.image.masked_ref
        image_mask = processed.image.image_mask

        pixel_size = processed.image.pixel_size
        poni2, poni1 = processed.image.pos_inv(cx, cy)
        mask_min, mask_max = processed.image.threshold_mask

        ai = AzimuthalIntegrator(dist=sample_distance,
                                 poni1=poni1 * pixel_size,
                                 poni2=poni2 * pixel_size,
                                 pixel1=pixel_size,
                                 pixel2=pixel_size,
                                 rot1=0,
                                 rot2=0,
                                 rot3=0,
                                 wavelength=wavelength)

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

        if reference is not None:
            mask = image_mask != 0
            # merge image mask and threshold mask
            mask[(reference <= mask_min) | (reference >= mask_max)] = 1
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
            momentum, intensities, intensities_mean, ref_intensity, processed)

    def _normalize(self, momentum, intensities, intensities_mean,
                   ref_intensity, processed):
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
            _, roi1_hist, _ = processed.roi.roi1_hist
            _, roi2_hist, _ = processed.roi.roi2_hist
            _, roi1_hist_ref, _ = processed.roi.roi1_hist_ref
            _, roi2_hist_ref, _ = processed.roi.roi2_hist_ref

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

        processed.ai.momentum = momentum
        processed.ai.intensities = intensities
        processed.ai.intensity_mean = intensities_mean
        processed.ai.reference_intensity = ref_intensity


class PulseResolvedAiFomProcessor(LeafProcessor):
    """PulseResolvedAiFomProcessor.

    Only for pulse-resolved detectors.
    """
    @profiler("Pulse-resolved azimuthal integration FOM processor")
    def run(self, processed, raw=None):
        """Override."""
        if processed.n_pulses == 1:
            # train-resolved
            return

        momentum = processed.ai.momentum
        intensities = processed.ai.intensities

        # calculate the different between each pulse and the first one
        diffs = [p - intensities[0] for p in intensities]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self.fom_itgt_range)[0]
            foms.append(np.sum(np.abs(fom)))

        processed.ai.pulse_fom = foms
