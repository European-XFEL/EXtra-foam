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
from ...config import AiNormalizer, AnalysisType, config
from ...ipc import redis_subscribe
from ...metadata import Metadata as mt
from ...utils import profiler


def energy2wavelength(energy):
    # Plank-einstein relation (E=hv)
    HC_E = 1e-3 * constants.c * constants.h / constants.e
    return HC_E / energy


def _check_image_mask(mask_shape, image_shape):
    if mask_shape != image_shape:
        raise ProcessingError(f"Mask shape {mask_shape} differs "
                              f"from the image shape {image_shape}")


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
        image_mask (numpy.ndarray): image mask. Shape = (y, x).
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

    image_mask = SharedProperty()

    analysis_type = SharedProperty()

    def __init__(self):
        super().__init__()

        self.add(AiProcessor())
        self.add(AiProcessorPulsedFom())
        self.add(AiProcessorPumpProbe())
        self.add(AiProcessorPumpProbeFom())

        self.image_mask = None
        self._mask_command = redis_subscribe("command:image_mask",
                                             decode_responses=False)

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.AZIMUTHAL_INTEG_PROC)
        gp_cfg = self._meta.get_all(mt.GENERAL_PROC)
        if cfg is None or gp_cfg is None:
            return

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

        if cfg['enable_pulsed_ai'] == 'True':
            self._update_analysis(AnalysisType.PULSE_AZIMUTHAL_INTEG)
        else:
            self._update_analysis(AnalysisType.UNDEFINED)

        while True:
            msg = self._mask_command.get_message()
            if msg is None:
                break
            self._image_mask_handler(msg)

    def _image_mask_handler(self, msg):
        if isinstance(msg['data'], int):
            # TODO: why there is an additional message?
            return

        mode, x, y, w, h, w_img, h_img = self.str2list(
            msg['data'].decode("utf-8"), handler=int)

        if mode in (0, 1):
            if self.image_mask is None:
                self.image_mask = np.zeros((h_img, w_img), dtype=np.bool)
            # 1 for masking and 0 for unmasking
            self.image_mask[y:y+h, x:x+w] = bool(mode)
        elif mode == -1:
            # clear mask
            self.image_mask = np.zeros((h_img, w_img), dtype=np.bool)
        else:
            # the next message contains the bytes for a whole mask
            msg = self._mask_command.get_message()
            packed_bits = np.frombuffer(msg['data'], dtype=np.uint8)
            self.image_mask = np.unpackbits(packed_bits).reshape(h, w).astype(
                np.bool, casting='unsafe')


class AiProcessor(LeafProcessor):
    """AiProcessor class.

    Calculate azimuthal integration in a train.
    """
    @profiler("Azimuthal Integration Processor")
    def process(self, data):
        cx = self.integ_center_x
        cy = self.integ_center_y
        integ_points = self.integ_points
        integ_method = self.integ_method
        integ_range = self.integ_range

        processed = data['processed']

        assembled = processed.image.images

        pixel_size = config['PIXEL_SIZE']
        poni2, poni1 = cx, cy
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

        if self.image_mask is None:
            mask = np.zeros(processed.image.shape, dtype=np.bool)
        else:
            mask = np.copy(self.image_mask)
            # image shape could change due to quadrant movement for
            # pulse-resolved detectors
            _check_image_mask(mask.shape, assembled[-2:].shape)

        # -------------------------------------------------------------
        # pulsed azimuthal integration is only applied to
        # pulsed-resolved detectors
        # -------------------------------------------------------------

        if self._has_analysis(AnalysisType.PULSE_AZIMUTHAL_INTEG):

            def _integrate1d_imp(i):
                """Use for multiprocessing."""
                # convert 'nan' to '-inf', as explained above
                assembled[i][np.isnan(assembled[i])] = -np.inf

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

            # intensities is a tuple
            momentums, intensities = zip(*rets)
            momentum = momentums[0]
            intensities = self._normalize(processed, momentum, np.array(intensities))

            processed.ai.momentum = momentum
            processed.ai.intensities = intensities

        # -------------------------------------------------------------
        # perform azimuthal integration on the masked mean
        # -------------------------------------------------------------

        masked_mean = processed.image.masked_mean

        if self._has_any_analysis([AnalysisType.TRAIN_AZIMUTHAL_INTEG,
                                   AnalysisType.PULSE_AZIMUTHAL_INTEG]):
            # merge image mask and threshold mask
            mask[(masked_mean < mask_min) | (masked_mean > mask_max)] = 1
            ret = ai.integrate1d(masked_mean,
                                 integ_points,
                                 method=integ_method,
                                 mask=mask,
                                 radial_range=integ_range,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")

            momentum = ret.radial
            intensity_mean = self._normalize(
                processed, momentum, ret.intensity)

            processed.ai.momentum = momentum
            processed.ai.intensity_mean = intensity_mean

    def _normalize(self, processed, momentum, intensity):
        auc_range = self.auc_range

        if self.normalizer == AiNormalizer.AUC:

            if intensity.ndim == 2:
                # normalize azimuthal integration curves for each pulse
                for i, item in enumerate(intensity):
                    intensity[i][:] = normalize_auc(item, momentum, *auc_range)
            else:
                intensity = normalize_auc(intensity, momentum, *auc_range)

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
            intensity /= denominator

        return intensity


class AiProcessorPulsedFom(LeafProcessor):
    """AiProcessorPulsedFom class.

    Calculate azimuthal integration FOM for pulse-resolved detectors.
    """
    @profiler("Azimuthal Integration Pulsed FOM Processor")
    def process(self, data):
        """Override."""
        if not self._has_analysis(AnalysisType.PULSE_AZIMUTHAL_INTEG):
            return

        processed = data['processed']

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


class AiProcessorPumpProbe(LeafProcessor):
    """AiProcessorPumpProbe class.

    Calculate azimuthal integration FOM for a pair of on and off images.
    """
    def __init__(self):
        super().__init__()

    @profiler("Azimuthal Integration Pump-probe Processor")
    def process(self, data):
        if not self._has_analysis(AnalysisType.PP_AZIMUTHAL_INTEG):
            raise StopCompositionProcessing

        processed = data['processed']

        on_image = processed.pp.on_image_mean
        off_image = processed.pp.off_image_mean
        if on_image is None or off_image is None:
            raise StopCompositionProcessing

        cx = self.integ_center_x
        cy = self.integ_center_y
        integ_points = self.integ_points
        integ_method = self.integ_method
        integ_range = self.integ_range

        pixel_size = config['PIXEL_SIZE']
        poni2, poni1 = cx, cy
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
            if self.image_mask is None:
                mask = np.zeros_like(img, dtype=np.bool)
            else:
                mask = np.copy(self.image_mask)
                # image shape could change due to quadrant movement for
                # pulse-resolved detectors
                _check_image_mask(mask.shape, img.shape)

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


class AiProcessorPumpProbeFom(LeafProcessor):
    """AiProcessorPumpProbeFom class.

    Calculate the pump-probe FOM.
    """
    @profiler("Azimuthal integration pump-probe FOM processor")
    def process(self, data):

        processed = data['processed']

        momentum, on_ma, off_ma = processed.pp.data

        norm_on_ma, norm_off_ma = self._normalize(
            processed, momentum, on_ma, off_ma)
        norm_on_off_ma = norm_on_ma - norm_off_ma

        fom = slice_curve(norm_on_off_ma, momentum, *self.fom_integ_range)[0]
        if processed.pp.abs_difference:
            fom = np.sum(np.abs(fom))
        else:
            fom = np.sum(fom)

        processed.pp.x = momentum
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
