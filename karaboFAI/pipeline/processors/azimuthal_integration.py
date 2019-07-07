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
import functools

import numpy as np
from scipy import constants

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from .base_processor import (
    LeafProcessor, CompositeProcessor, SharedProperty,
)
from ..exceptions import ProcessingError
from ...algorithms import mask_image, normalize_auc, slice_curve
from ...config import VectorNormalizer, AnalysisType, config
from ...metadata import Metadata as mt
from ...utils import profiler


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
        wavelength (float): photon wavelength in meter.
        sample_dist (float): distance from the sample to the
            detector plan (orthogonal distance, not along the beam),
            in meter.
        poni1 (float): poni1 in meter.
        poni2 (float): poni2 in meter.
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

    sample_dist = SharedProperty()
    poni1 = SharedProperty()
    poni2 = SharedProperty()
    wavelength = SharedProperty()

    integ_method = SharedProperty()
    integ_range = SharedProperty()
    integ_points = SharedProperty()

    normalizer = SharedProperty()
    auc_range = SharedProperty()
    fom_integ_range = SharedProperty()

    analysis_type = SharedProperty()

    def __init__(self):
        super().__init__()

        self.add(AiProcessor())

    def update(self):
        """Override."""
        gp_cfg = self._meta.get_all(mt.GENERAL_PROC)
        self.sample_dist = float(gp_cfg['sample_distance'])
        self.wavelength = energy2wavelength(float(gp_cfg['photon_energy']))

        cfg = self._meta.get_all(mt.AZIMUTHAL_INTEG_PROC)
        pixel_size = config['PIXEL_SIZE']
        self.poni1 = int(cfg['integ_center_y']) * pixel_size
        self.poni2 = int(cfg['integ_center_x']) * pixel_size

        self.integ_method = cfg['integ_method']
        self.integ_range = self.str2tuple(cfg['integ_range'])
        self.integ_points = int(cfg['integ_points'])
        self.normalizer = VectorNormalizer(int(cfg['normalizer']))
        self.auc_range = self.str2tuple(cfg['auc_range'])
        self.fom_integ_range = self.str2tuple(cfg['fom_integ_range'])

        if cfg['enable_pulsed_ai'] == 'True':
            self._update_analysis(AnalysisType.PULSE_AZIMUTHAL_INTEG)
        else:
            self._update_analysis(AnalysisType.UNDEFINED)


class AiProcessor(LeafProcessor):
    """AiProcessor class.

    Calculate azimuthal integration in a train.
    """
    def __init__(self):
        super().__init__()

        self._integrator = None

    def _update_integrator(self):
        if self._integrator is None:
            self._integrator = AzimuthalIntegrator(
                dist=self.sample_dist,
                pixel1=config['PIXEL_SIZE'],
                pixel2=config['PIXEL_SIZE'],
                poni1=self.poni1,
                poni2=self.poni2,
                rot1=0,
                rot2=0,
                rot3=0,
                wavelength=self.wavelength)
        else:
            if self._integrator.dist != self.sample_dist \
                    or self._integrator.wavelength != self.wavelength \
                    or self._integrator.poni1 != self.poni1 \
                    or self._integrator.poni2 != self.poni2:
                # dist, poni1, poni2, rot1, rot2, rot3, wavelength
                self._integrator.set_param((self.sample_dist,
                                            self.poni1,
                                            self.poni2,
                                            0,
                                            0,
                                            0,
                                            self.wavelength))

        return self._integrator

    @profiler("Azimuthal Integration Processor")
    def process(self, data):
        processed = data['processed']

        integrator = self._update_integrator()
        itgt1d = functools.partial(integrator.integrate1d,
                                   method=self.integ_method,
                                   radial_range=self.integ_range,
                                   correctSolidAngle=True,
                                   polarization_factor=1,
                                   unit="q_A^-1")
        integ_points = self.integ_points

        # pulse-resolved azimuthal integration
        if self._has_analysis(AnalysisType.PULSE_AZIMUTHAL_INTEG):

            image_mask = processed.image.image_mask
            threshold_mask = processed.image.threshold_mask

            pulse_images = processed.image.images
            # pulse_images could also be a list in the process of analysis
            # type switching
            if isinstance(pulse_images, np.ndarray):

                def _integrate1d_imp(i):
                    masked = mask_image(pulse_images[i],
                                        threshold_mask=threshold_mask,
                                        image_mask=image_mask)
                    return itgt1d(masked, integ_points)

                intensities = []  # pulsed A.I.
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for i, ret in zip(range(len(pulse_images)),
                                      executor.map(_integrate1d_imp,
                                                   range(len(pulse_images)))):
                        if i == 0:
                            momentum = ret.radial
                        intensities.append(self._normalize(
                            processed, momentum, ret.intensity))

                # calculate the difference between each pulse and the
                # first one
                diffs = [p - intensities[0] for p in intensities]

                # calculate the figure of merit for each pulse
                foms = []
                for diff in diffs:
                    fom = slice_curve(diff, momentum, *self.fom_integ_range)[0]
                    foms.append(np.sum(np.abs(fom)))

                processed.ai.momentum = momentum
                processed.ai.intensities = intensities
                processed.ai.intensities_foms = foms
                # It is not correct to calculate the mean of intensities since
                # the result is equivalent to setting all nan to zero instead
                # of nanmean.

        # train-resolved azimuthal integration
        if self._has_any_analysis([AnalysisType.TRAIN_AZIMUTHAL_INTEG,
                                   AnalysisType.PULSE_AZIMUTHAL_INTEG]):
            mean_ret = itgt1d(processed.image.masked_mean, integ_points)

            momentum = mean_ret.radial
            intensity = mean_ret.intensity
            fom = slice_curve(intensity, momentum, *self.fom_integ_range)[0]
            fom = np.sum(np.abs(fom))

            processed.ai.momentum = momentum
            processed.ai.intensity = self._normalize(
                processed, momentum, intensity)
            processed.ai.intensity_fom = fom

        # pump-probe azimuthal integration
        if self._has_analysis(AnalysisType.PP_AZIMUTHAL_INTEG):
            on_image = processed.pp.on_image_mean
            off_image = processed.pp.off_image_mean

            if on_image is not None and off_image is not None:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    on_off_rets = executor.map(
                        lambda x: itgt1d(*x), ((on_image, integ_points),
                                               (off_image, integ_points)))
                on_ret, off_ret = on_off_rets

                processed.pp.data = (on_ret.radial,
                                     on_ret.intensity,
                                     off_ret.intensity)
                momentum, on_ma, off_ma = processed.pp.data

                norm_on_ma = self._normalize(processed, on_ret.radial, on_ma)
                norm_off_ma = self._normalize(processed,  on_ret.radial, off_ma)
                norm_on_off_ma = norm_on_ma - norm_off_ma
                fom = slice_curve(norm_on_off_ma,
                                  momentum,
                                  *self.fom_integ_range)[0]
                if processed.pp.abs_difference:
                    fom = np.sum(np.abs(fom))
                else:
                    fom = np.sum(fom)

                processed.ai.momentum = momentum
                processed.pp.norm_on_ma = norm_on_ma
                processed.pp.norm_off_ma = norm_off_ma
                processed.pp.norm_on_off_ma = norm_on_off_ma
                processed.pp.x = momentum
                processed.pp.fom = fom

    def _normalize(self, processed, momentum, intensity):
        """Normalize the azimuthal integration result.

        :param ProcessedData processed: processed data.
        :param numpy.ndarray momentum: momentum (q value).
        :param numpy.ndarray intensity: intensity.
        """
        auc_range = self.auc_range

        if self.normalizer == VectorNormalizer.AUC:
            # normalized by area under curve (AUC)
            intensity = normalize_auc(intensity, momentum, *auc_range)

        else:
            # normalized by ROI

            roi1_fom = processed.roi.roi1_fom
            roi2_fom = processed.roi.roi2_fom

            if self.normalizer == VectorNormalizer.ROI1:
                if roi1_fom is None:
                    raise ProcessingError("ROI1 is not activated!")
                denominator = roi1_fom
            elif self.normalizer == VectorNormalizer.ROI2:
                if roi2_fom is None:
                    raise ProcessingError("ROI2 is not activated!")
                denominator = roi2_fom
            else:
                if roi1_fom is None:
                    raise ProcessingError("ROI1 is not activated!")
                if roi2_fom is None:
                    raise ProcessingError("ROI2 is not activated!")

                if self.normalizer == VectorNormalizer.ROI_SUM:
                    denominator = roi1_fom + roi2_fom
                elif self.normalizer == VectorNormalizer.ROI_SUB:
                    denominator = roi1_fom - roi2_fom
                else:
                    raise ProcessingError(f"Unknown normalizer: {repr(self.normalizer)}")

            if denominator == 0:
                raise ProcessingError("Normalizer (ROI) is zero!")

            intensity /= denominator

        return intensity
