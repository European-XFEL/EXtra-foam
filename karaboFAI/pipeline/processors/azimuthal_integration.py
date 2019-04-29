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

from .base_processor import (
    LeafProcessor, CompositeProcessor, SharedProperty,
    StopCompositionProcessing
)
from ..exceptions import ProcessingError
from ...algorithms import normalize_curve, slice_curve
from ...config import AiNormalizer, PumpProbeType
from ...helpers import profiler


class AzimuthalIntegrationProcessor(CompositeProcessor):
    """AzimuthalIntegrationProcessor class.

    Perform azimuthal integration.

    Attributes:
        pulsed_ai (bool): True for performing azimuthal integration for
            individual pulses. It only affect the performance with the
            pulse-resolved detectors.
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
    pulsed_ai = SharedProperty()

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

        self.add(AiPulsedProcessor())
        self.add(AiPumpProbeProcessor())

    def process(self, processed, raw):
        pass


class AiPulsedProcessor(CompositeProcessor):
    """AiPulsedProcessor class.

    Calculate azimuthal integration for individual pulses in a train.
    """
    def __init__(self):
        super().__init__()

        self.add(AiPulsedNormProcessor())
        self.add(AiPulseFomProcessor())

    @profiler("Azimuthal integration pulsed processor")
    def process(self, processed, raw=None):
        if not self.pulsed_ai:
            return

        cx, cy = self.integration_center
        integration_points = self.integration_points
        integration_method = self.integration_method
        integration_range = self.integration_range

        assembled = processed.image.images
        image_mask = processed.image.image_mask

        pixel_size = processed.image.pixel_size
        poni2, poni1 = processed.image.pos_inv(cx, cy)
        mask_min, mask_max = processed.image.threshold_mask

        ai = AzimuthalIntegrator(dist=self.sample_distance,
                                 poni1=poni1 * pixel_size,
                                 poni2=poni2 * pixel_size,
                                 pixel1=pixel_size,
                                 pixel2=pixel_size,
                                 rot1=0,
                                 rot2=0,
                                 rot3=0,
                                 wavelength=self.wavelength)

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

        processed.ai.momentum = momentum
        processed.ai.intensities = intensities
        processed.ai.intensity_mean = intensities_mean


class AiPulsedNormProcessor(LeafProcessor):
    """AiPulsedNormProcessor class.

    Normalize the azimuthal integration result from AiPulsedProcessor.
    """
    @profiler("Azimuthal integration normalization processor")
    def process(self, processed, raw=None):
        momentum = processed.ai.momentum
        intensity_mean = processed.ai.intensity_mean
        intensities = processed.ai.intensities
        auc_x_range = self.auc_x_range

        if momentum is None:
            raise StopCompositionProcessing

        if self.normalizer == AiNormalizer.AUC:
            intensity_mean = normalize_curve(
                intensity_mean, momentum, *auc_x_range)

            # normalize azimuthal integration curves for each pulse
            for i, intensity in enumerate(intensities):
                intensities[i][:] = normalize_curve(
                    intensity, momentum, *auc_x_range)
        else:
            _, roi1_hist, _ = processed.roi.roi1_hist
            _, roi2_hist, _ = processed.roi.roi2_hist

            denominator = 0
            try:
                if self.normalizer == AiNormalizer.ROI1:
                    denominator = roi1_hist[-1]
                elif self.normalizer == AiNormalizer.ROI2:
                    denominator = roi2_hist[-1]
                elif self.normalizer == AiNormalizer.ROI_SUM:
                    denominator = roi1_hist[-1] + roi2_hist[-1]
                elif self.normalizer == AiNormalizer.ROI_SUB:
                    denominator = roi1_hist[-1] - roi2_hist[-1]

            except IndexError as e:
                # this could happen if the history is clear just now
                raise ProcessingError(e)

            if denominator == 0:
                raise ProcessingError(
                    "Invalid normalizer: sum of ROI(s) is zero!")
            intensity_mean /= denominator
            intensities /= denominator

        processed.ai.momentum = momentum
        processed.ai.intensities = intensities
        processed.ai.intensity_mean = intensity_mean


class AiPulseFomProcessor(LeafProcessor):
    """AiPulseFomProcessor class.

    Calculate azimuthal integration FOM for pulse-resolved detectors.
    """
    @profiler("Azimuthal integration pulsed FOM processor")
    def process(self, processed, raw=None):
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


class AiPumpProbeProcessor(CompositeProcessor):
    """AiPumpProbeProcessor class.

    Calculate azimuthal integration FOM for a pair of on and off images.
    """
    def __init__(self):
        super().__init__()

        self.add(AiPumpProbeNormProcessor())
        self.add(AiPumpProbeFomProcessor())

    @profiler("Azimuthal integration pump-probe processor")
    def process(self, processed, raw=None):
        if processed.pp.analysis_type != PumpProbeType.AZIMUTHAL_INTEGRATION:
            raise StopCompositionProcessing

        on_image = processed.pp.on_image_mean
        off_image = processed.pp.off_image_mean
        if on_image is None or off_image is None:
            raise StopCompositionProcessing

        cx, cy = self.integration_center
        integration_points = self.integration_points
        integration_method = self.integration_method
        integration_range = self.integration_range

        image_mask = processed.image.image_mask

        pixel_size = processed.image.pixel_size
        poni2, poni1 = processed.image.pos_inv(cx, cy)
        mask_min, mask_max = processed.image.threshold_mask

        ai = AzimuthalIntegrator(dist=self.sample_distance,
                                 poni1=poni1 * pixel_size,
                                 poni2=poni2 * pixel_size,
                                 pixel1=pixel_size,
                                 pixel2=pixel_size,
                                 rot1=0,
                                 rot2=0,
                                 rot3=0,
                                 wavelength=self.wavelength)

        def _integrate1d_on_off_imp(img):
            """Use for multiprocessing."""
            mask = np.copy(image_mask)
            # merge image mask and threshold mask
            mask[(img <= mask_min) | (img >= mask_max)] = 1

            # do integration
            ret = ai.integrate1d(img,
                                 integration_points,
                                 method=integration_method,
                                 mask=mask,
                                 radial_range=integration_range,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")

            return ret.radial, ret.intensity

        with ThreadPoolExecutor(max_workers=2) as executor:
            rets = executor.map(_integrate1d_on_off_imp, [on_image, off_image])

        momentums, intensities = zip(*rets)
        processed.pp.x_data = momentums[0]
        processed.pp.on_data = intensities[0]
        processed.pp.off_data = intensities[1]


class AiPumpProbeNormProcessor(LeafProcessor):
    """AiPumpProbeNormProcessor class.

    Normalize the pump-probe azimuthal integration result.
    """
    @profiler("Azimuthal integration pump-probe normalization processor")
    def process(self, processed, raw=None):
        x_data = processed.pp.x_data
        on_data = processed.pp.on_data
        off_data = processed.pp.off_data
        auc_x_range = self.auc_x_range

        if self.normalizer == AiNormalizer.AUC:
            on_data = normalize_curve(on_data, x_data, *auc_x_range)
            off_data = normalize_curve(off_data, x_data, *auc_x_range)
        else:
            _, roi1_hist, _ = processed.roi.roi1_hist
            _, roi2_hist, _ = processed.roi.roi2_hist

            denominator = 0
            try:
                if self.normalizer == AiNormalizer.ROI1:
                    denominator = roi1_hist[-1]
                elif self.normalizer == AiNormalizer.ROI2:
                    denominator = roi2_hist[-1]
                elif self.normalizer == AiNormalizer.ROI_SUM:
                    denominator = roi1_hist[-1] + roi2_hist[-1]
                elif self.normalizer == AiNormalizer.ROI_SUB:
                    denominator = roi1_hist[-1] - roi2_hist[-1]

            except IndexError as e:
                # this could happen if the history is clear just now
                raise ProcessingError(e)

            if denominator == 0:
                raise ProcessingError(
                    "Invalid normalizer: sum of ROI(s) is zero!")
            on_data /= denominator
            off_data /= denominator

        processed.pp.on_data = on_data
        processed.pp.off_data = off_data


class AiPumpProbeFomProcessor(LeafProcessor):
    """AiPumpProbeFomProcessor class.

    Calculate the pump-probe FOM.
    """
    @profiler("Azimuthal integration pump-probe FOM processor")
    def process(self, processed, raw=None):
        x_data = processed.pp.x_data
        on_data = processed.pp.on_data
        off_data = processed.pp.off_data

        on_off_data = on_data - off_data
        fom = slice_curve(on_off_data, x_data, *self.fom_itgt_range)[0]
        fom = np.sum(np.abs(fom))

        processed.pp.on_off_data = on_off_data
        processed.pp.fom = (processed.tid, fom)
