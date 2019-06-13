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
            if mask.shape != processed.image.shape:
                raise ProcessingError(f"Mask shape {mask.shape} differs "
                                      f"from the image shape {processed.image.shape}")

        # collect images
        assembled = dict()

        if self._has_analysis(AnalysisType.PULSE_AZIMUTHAL_INTEG):
            for i, img in enumerate(processed.image.images):
                assembled[i] = img

        # TODO: for train-resolved detectors, if TRAIN_AZIMUTHAL_INTEG
        #       and PP_AZIMUTHAL_INTEG are activated at the same time,
        #       the same image will be integrated twice!

        if self._has_any_analysis([AnalysisType.TRAIN_AZIMUTHAL_INTEG,
                                   AnalysisType.PULSE_AZIMUTHAL_INTEG]):
            assembled['masked_mean'] = processed.image.masked_mean

        if self._has_analysis(AnalysisType.PP_AZIMUTHAL_INTEG):
            on_image = processed.pp.on_image_mean
            off_image = processed.pp.off_image_mean
            if on_image is not None and off_image is not None:
                assembled['pp_on'] = on_image
                assembled['pp_off'] = off_image

        if not assembled:
            return

        def _integrate1d_imp(key):
            """Use for multiprocessing."""
            # convert 'nan' to '-inf', as explained above
            assembled[key][np.isnan(assembled[key])] = -np.inf

            # merge image mask and threshold mask
            mask[(assembled[key] < mask_min) | (assembled[key] > mask_max)] = 1

            # do integration
            ret = ai.integrate1d(assembled[key],
                                 integ_points,
                                 method=integ_method,
                                 mask=mask,
                                 radial_range=integ_range,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 unit="q_A^-1")
            return ret

        intensities = dict()  # for pulsed A.I.
        with ThreadPoolExecutor(max_workers=4) as executor:
            for key, ret in zip(assembled.keys(),
                                executor.map(_integrate1d_imp, assembled.keys())):

                momentum = ret.radial
                if processed.ai.momentum is None:
                    processed.ai.momentum = momentum

                if isinstance(key, int):
                    # pulsed A.I.
                    intensities[key] = self._normalize(
                        processed, momentum, ret.intensity)
                elif key == "masked_mean":
                    # average image over a train
                    processed.ai.intensity_mean = self._normalize(
                        processed, momentum, ret.intensity)
                elif key == "pp_on":
                    # average on-pulse image
                    on_intensity = ret.intensity
                elif key == "pp_off":
                    # average off-pulse image
                    off_intensity = ret.intensity

        if intensities:
            processed.ai.intensities = intensities

        if "pp_on" in assembled and "pp_off" in assembled:
            processed.pp.data = (momentum, on_intensity, off_intensity)
            momentum, on_ma, off_ma = processed.pp.data

            norm_on_ma = self._normalize(
                processed, momentum, on_ma)
            norm_off_ma = self._normalize(
                processed, momentum, off_ma)
            norm_on_off_ma = norm_on_ma - norm_off_ma

            processed.pp.norm_on_ma = norm_on_ma
            processed.pp.norm_off_ma = norm_off_ma
            processed.pp.norm_on_off_ma = norm_on_off_ma

    def _normalize(self, processed, momentum, intensity):
        """Normalize the azimuthal integration result.

        :param ProcessedData processed: processed data.
        :param numpy.ndarray momentum: momentum (q value).
        :param numpy.ndarray intensity: intensity.
        """
        auc_range = self.auc_range

        if self.normalizer == AiNormalizer.AUC:
            # normalized by area under curve (AUC)
            intensity = normalize_auc(intensity, momentum, *auc_range)

        else:
            # normalized by ROI

            roi1_fom = processed.roi.roi1_fom
            roi2_fom = processed.roi.roi2_fom

            if self.normalizer == AiNormalizer.ROI1:
                if roi1_fom is None:
                    raise ProcessingError("ROI1 is not activated!")
                denominator = roi1_fom
            elif self.normalizer == AiNormalizer.ROI2:
                if roi2_fom is None:
                    raise ProcessingError("ROI2 is not activated!")
                denominator = roi2_fom
            else:
                if roi1_fom is None:
                    raise ProcessingError("ROI1 is not activated!")
                if roi2_fom is None:
                    raise ProcessingError("ROI2 is not activated!")

                if self.normalizer == AiNormalizer.ROI_SUM:
                    denominator = roi1_fom + roi2_fom
                elif self.normalizer == AiNormalizer.ROI_SUB:
                    denominator = roi1_fom - roi2_fom
                else:
                    raise ProcessingError(f"Unknown normalizer: {repr(self.normalizer)}")

            if denominator == 0:
                raise ProcessingError("Normalizer (ROI) is zero!")

            intensity /= denominator

        return intensity


class AiProcessorPulsedFom(LeafProcessor):
    """AiProcessorPulsedFom class.

    Calculate azimuthal integration FOM for pulse-resolved detectors.
    """
    @profiler("Azimuthal Integration Pulsed FOM Processor")
    def process(self, data):
        """Override."""
        processed = data['processed']
        momentum = processed.ai.momentum
        intensities = processed.ai.intensities
        if intensities is None:
            return

        # calculate the difference between each pulse and the first one
        diffs = [p - intensities[0] for p in intensities.values()]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self.fom_integ_range)[0]
            foms.append(np.sum(np.abs(fom)))

        processed.ai.pulse_fom = foms


class AiProcessorPumpProbeFom(LeafProcessor):
    """AiProcessorPumpProbeFom class.

    Calculate the pump-probe FOM.
    """
    @profiler("Azimuthal integration pump-probe FOM processor")
    def process(self, data):
        processed = data['processed']
        momentum = processed.ai.momentum
        norm_on_off_ma = processed.pp.norm_on_off_ma
        if not self._has_analysis(AnalysisType.PP_AZIMUTHAL_INTEG):
            return

        fom = slice_curve(norm_on_off_ma, momentum, *self.fom_integ_range)[0]
        if processed.pp.abs_difference:
            fom = np.sum(np.abs(fom))
        else:
            fom = np.sum(fom)

        processed.pp.x = momentum
        processed.pp.fom = fom
