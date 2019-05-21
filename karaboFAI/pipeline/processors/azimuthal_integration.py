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
from scipy import constants

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from .base_processor import (
    LeafProcessor, CompositeProcessor, SharedProperty,
    StopCompositionProcessing
)
from ..exceptions import ProcessingError
from ...algorithms import normalize_auc, slice_curve
from ...config import AiNormalizer, PumpProbeType
from ...metadata import Metadata as mt
from ...helpers import profiler


def energy2wavelength(energy):
    # Plank-einstein relation (E=hv)
    HC_E = 1e-3 * constants.c * constants.h / constants.e
    return HC_E / energy


class AzimuthalIntegrationProcessor(CompositeProcessor):
    """AzimuthalIntegrationProcessor class.

    Perform azimuthal integration.

    Attributes:
        enable_pulsed_ai (bool): True for performing azimuthal integration for
            individual pulses. It only affect the performance with the
            pulse-resolved detectors.
        photon_energy (float): photon energy in keV.
        sample_distance (float): distance from the sample to the
            detector plan (orthogonal distance, not along the beam),
            in meter.
        integ_center_x (int): Cx in pixel.
        integ_center_y (int): Cy in pixel.
        integ_method (string): the azimuthal integration
            method supported by pyFAI.
        integ_range (tuple): the lower and upper range of
            the integration radial unit. (float, float)
        integ_points (int): number of points in the
            integration output pattern.
        normalizer (int): normalizer type for calculating FOM from
            azimuthal integration result.
        auc_range (tuple): x range for calculating AUC, which is used as
            a normalizer of the azimuthal integration.
        fom_integ_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
    """

    sample_distance = SharedProperty()
    photon_energy = SharedProperty()
    integ_center_x = SharedProperty()
    integ_center_y = SharedProperty()
    integ_method = SharedProperty()
    integ_range = SharedProperty()
    integ_points = SharedProperty()
    normalizer = SharedProperty()
    auc_range = SharedProperty()
    fom_integ_range = SharedProperty()

    enable_pulsed_ai = SharedProperty()

    def __init__(self):
        super().__init__()

        self.add(AiPulsedProcessor())
        self.add(AiPumpProbeProcessor())

    def process(self, processed, raw):
        pass

    def update(self):
        cfg = self._db.hgetall(mt.AZIMUTHAL_INTEG_PROC)
        gp_cfg = self._db.hgetall(mt.GENERAL_PROC)

        self.sample_distance = float(gp_cfg['sample_distance'])
        self.photon_energy = float(gp_cfg['photon_energy'])
        self.integ_center_x = int(cfg['integ_center_x'])
        self.integ_center_y = int(cfg['integ_center_y'])
        self.integ_method = cfg['integ_method']
        self.integ_range = self.str2tuple(cfg['integ_range'])
        self.integ_points = int(cfg['integ_points'])
        self.normalizer = AiNormalizer(int(cfg['normalizer']))
        self.auc_range = self.str2tuple(cfg['auc_range'])
        self.fom_integ_range = self.str2tuple(cfg['fom_integ_range'])
        self.enable_pulsed_ai = cfg['enable_pulsed_ai'] == 'True'


class AiPulsedProcessor(CompositeProcessor):
    """AiPulsedProcessor class.

    Calculate azimuthal integration for individual pulses in a train.
    """
    def __init__(self):
        super().__init__()

        self.add(AiPulsedFomProcessor())

    @profiler("Azimuthal integration pulsed processor")
    def process(self, processed, raw=None):
        if not self.enable_pulsed_ai:
            return

        cx = self.integ_center_x
        cy = self.integ_center_y
        integ_points = self.integ_points
        integ_method = self.integ_method
        integ_range = self.integ_range

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
                                 wavelength=energy2wavelength(self.photon_energy))

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
                                     integ_points,
                                     method=integ_method,
                                     mask=mask,
                                     radial_range=integ_range,
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
                                 integ_points,
                                 method=integ_method,
                                 mask=mask,
                                 radial_range=integ_range,
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

        normalized_intensity_mean, normalized_intensities = self._normalize(
            processed, momentum, intensities_mean, intensities)

        processed.ai.momentum = momentum
        processed.ai.intensities = normalized_intensities
        processed.ai.intensity_mean = normalized_intensity_mean

    def _normalize(self, processed, momentum, intensities_mean, intensities):
        auc_range = self.auc_range

        if self.normalizer == AiNormalizer.AUC:
            intensities_mean = normalize_auc(
                intensities_mean, momentum, *auc_range)

            # normalize azimuthal integration curves for each pulse
            for i, intensity in enumerate(intensities):
                intensities[i][:] = normalize_auc(
                    intensity, momentum, *auc_range)
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
            intensities_mean /= denominator
            intensities /= denominator

        return intensities_mean, intensities


class AiPulsedFomProcessor(LeafProcessor):
    """AiPulsedFomProcessor class.

    Calculate azimuthal integration FOM for pulse-resolved detectors.
    """
    @profiler("Azimuthal integration pulsed FOM processor")
    def process(self, processed, raw=None):
        """Override."""
        if processed.n_pulses == 1:
            # train-resolved
            return

        momentum = processed.ai.momentum
        if momentum is None:
            return
        intensities = processed.ai.intensities

        # calculate the difference between each pulse and the first one
        diffs = [p - intensities[0] for p in intensities]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self.fom_integ_range)[0]
            foms.append(np.sum(np.abs(fom)))

        processed.ai.pulse_fom = foms


class AiPumpProbeProcessor(CompositeProcessor):
    """AiPumpProbeProcessor class.

    Calculate azimuthal integration FOM for a pair of on and off images.
    """
    def __init__(self):
        super().__init__()

        self.add(AiPumpProbeFomProcessor())

    @profiler("Azimuthal integration pump-probe processor")
    def process(self, processed, raw=None):
        if processed.pp.analysis_type != PumpProbeType.AZIMUTHAL_INTEG:
            raise StopCompositionProcessing

        on_image = processed.pp.on_image_mean
        off_image = processed.pp.off_image_mean
        if on_image is None or off_image is None:
            raise StopCompositionProcessing

        cx = self.integ_center_x
        cy = self.integ_center_y
        integ_points = self.integ_points
        integ_method = self.integ_method
        integ_range = self.integ_range

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
                                 wavelength=energy2wavelength(self.photon_energy))

        def _integrate1d_on_off_imp(img):
            """Use for multiprocessing."""
            mask = np.copy(image_mask)
            # merge image mask and threshold mask
            mask[(img <= mask_min) | (img >= mask_max)] = 1

            # do integration
            ret = ai.integrate1d(img,
                                 integ_points,
                                 method=integ_method,
                                 mask=mask,
                                 radial_range=integ_range,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")

            return ret.radial, ret.intensity

        with ThreadPoolExecutor(max_workers=2) as executor:
            rets = executor.map(_integrate1d_on_off_imp, [on_image, off_image])

        momentums, intensities = zip(*rets)
        momentum = momentums[0]
        on_intensity = intensities[0]
        off_intensity = intensities[1]

        processed.pp.data = (momentum, on_intensity, off_intensity)


class AiPumpProbeFomProcessor(LeafProcessor):
    """AiPumpProbeFomProcessor class.

    Calculate the pump-probe FOM.
    """
    @profiler("Azimuthal integration pump-probe FOM processor")
    def process(self, processed, raw=None):
        momentum, on_ma, off_ma = processed.pp.data

        norm_on_ma, norm_off_ma = self._normalize(
            processed, momentum, on_ma, off_ma)
        norm_on_off_ma = norm_on_ma - norm_off_ma

        fom = slice_curve(norm_on_off_ma, momentum, *self.fom_integ_range)[0]
        if processed.pp.abs_difference:
            fom = np.sum(np.abs(fom))
        else:
            fom = np.sum(fom)

        processed.pp.norm_on_ma = norm_on_ma
        processed.pp.norm_off_ma = norm_off_ma
        processed.pp.norm_on_off_ma = norm_on_off_ma
        processed.pp.fom = fom

    def _normalize(self, processed, momentum, on, off):
        auc_range = self.auc_range

        if self.normalizer == AiNormalizer.AUC:
            try:
                on = normalize_auc(on, momentum, *auc_range)
                off = normalize_auc(off, momentum, *auc_range)
            except ValueError as e:
                raise ProcessingError(str(e))

        else:
            on_roi = processed.pp.on_roi
            off_roi = processed.pp.off_roi
            if on_roi is None or off_roi is None:
                raise ProcessingError("ROI information is not available")

            if self.normalizer in (AiNormalizer.ROI1, AiNormalizer.ROI_SUB):
                on_denominator = np.sum(on_roi)
                off_denominator = np.sum(off_roi)
            else:
                raise ProcessingError(
                    f"Normalizer is not supported in pump-probe analysis")

            if on_denominator == 0 or off_denominator == 0:
                raise ProcessingError(
                    "Invalid normalizer: sum of ROI(s) is zero!")

            on /= on_denominator
            off /= off_denominator

        return on, off
