"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from concurrent.futures import ThreadPoolExecutor
import functools

import numpy as np
from scipy import ndimage

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from ..exceptions import ProcessingError
from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray, PumpProbeData
from ...algorithms import slice_curve
from ...config import AnalysisType, Normalizer
from ...database import Metadata as mt
from ...utils import profiler
from ...logger import logger


from extra_foam.algorithms import (
    energy2wavelength, find_peaks_1d, mask_image_data
)


class _AzimuthalIntegProcessorBase(_BaseProcessor):
    """Base class for AzimuthalIntegProcessors.

    Attributes:
        _sample_dist (float): distance from the sample to the
            detector plan (orthogonal distance, not along the beam),
            in meter.
        _pixel1 (float): pixel size along axis 1 in meter.
        _pixel2 (float): pixel size along axis 2 in meter.
        _poni1 (float): poni1 in meter.
        _poni2 (float): poni2 in meter.
        _wavelength (float): photon wavelength in meter.
        _integ_method (string): the azimuthal integration
            method supported by pyFAI.
        _integ_range (tuple): the lower and upper range of
            the integration radial unit. (float, float)
        _integ_points (int): number of points in the
            integration output pattern.
        _normalizer (int): normalizer type for calculating FOM from
            azimuthal integration result.
        _auc_range (tuple): x range for calculating AUC, which is used as
            a normalizer of the azimuthal integration.
        _fom_integ_range (tuple): integration range for calculating FOM from
            the normalized azimuthal integration.
        _integrator (AzimuthalIntegrator): AzimuthalIntegrator instance.
        _q_map (numpy.ndarray): momentum transfer of map of the detector image.
            q = 4 * pi * sin(theta) / lambda
        _ma_window (int): moving average window size.
        _find_peaks (bool): whether to apply peak finding.
    """

    # maximum number of peaks expected
    _MAX_N_PEAKS = 10

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

        self._sample_dist = None
        self._pixel1 = None
        self._pixel2 = None
        self._poni1 = None
        self._poni2 = None
        self._wavelength = None

        self._integ_method = None
        self._integ_range = None
        self._integ_points = None

        self._normalizer = Normalizer.UNDEFINED
        self._auc_range = (-np.inf, np.inf)
        self._fom_integ_range = (-np.inf, np.inf)

        self._integrator = None
        self._q_map = None

        self._find_peaks = True
        self._peak_prominence = None
        self._peak_slicer = slice(None, None)

        self._use_reference = False

    def update(self):
        """Override."""
        g_cfg, cfg, rfg = self._meta.hget_all_multi(
            [mt.GLOBAL_PROC, mt.AZIMUTHAL_INTEG_PROC, mt.REFERENCE_IMAGE_PROC])

        self._sample_dist = float(g_cfg['sample_distance'])
        self._wavelength = energy2wavelength(1e3 * float(g_cfg['photon_energy']))
        self._update_moving_average(g_cfg)

        self._pixel1 = float(cfg['pixel_size_y'])
        self._pixel2 = float(cfg['pixel_size_x'])
        self._poni1 = float(cfg['integ_center_y']) * self._pixel1
        self._poni2 = float(cfg['integ_center_x']) * self._pixel2

        self._integ_method = cfg['integ_method']
        self._integ_range = self.str2tuple(cfg['integ_range'])
        self._integ_points = int(cfg['integ_points'])
        self._normalizer = Normalizer(int(cfg['normalizer']))
        self._auc_range = self.str2tuple(cfg['auc_range'])
        self._fom_integ_range = self.str2tuple(cfg['fom_integ_range'])

        self._find_peaks = cfg['peak_finding'] == 'True'
        self._peak_prominence = float(cfg['peak_prominence'])
        self._peak_slicer = self.str2slice(cfg['peak_slicer'])
        self._use_reference = rfg['reference_used']

    def _update_integrator(self):
        if self._integrator is None:
            self._integrator = AzimuthalIntegrator(
                dist=self._sample_dist,
                pixel1=self._pixel1,
                pixel2=self._pixel2,
                poni1=self._poni1,
                poni2=self._poni2,
                rot1=0,
                rot2=0,
                rot3=0,
                wavelength=self._wavelength)
        else:
            if self._integrator.dist != self._sample_dist \
                    or self._integrator.wavelength != self._wavelength \
                    or self._integrator.poni1 != self._poni1 \
                    or self._integrator.poni2 != self._poni2:
                # dist, poni1, poni2, rot1, rot2, rot3, wavelength
                self._integrator.set_param((self._sample_dist,
                                            self._poni1,
                                            self._poni2,
                                            0,
                                            0,
                                            0,
                                            self._wavelength))

        try:
            # 1/nm -> 1/A
            self._q_map = 0.1 * self._integrator._cached_array["q_center"]
        except KeyError:
            pass

        return self._integrator

    def _update_moving_average(self, v):
        pass


class AzimuthalIntegProcessorPulse(_AzimuthalIntegProcessorBase):
    """Pulse-resolved azimuthal integration processor."""

    @profiler("Azimuthal Integration Processor (Pulse)")
    def process(self, data):
        if not self._meta.has_analysis(AnalysisType.AZIMUTHAL_INTEG_PULSE):
            return

        processed = data['processed']
        assembled = data['assembled']['sliced']

        integrator = self._update_integrator()
        integ1d = functools.partial(integrator.integrate1d,
                                    method=self._integ_method,
                                    radial_range=self._integ_range,
                                    correctSolidAngle=True,
                                    polarization_factor=1,
                                    unit="q_A^-1")
        integ_points = self._integ_points

        threshold_mask = processed.image.threshold_mask
        image_mask = processed.image.image_mask
        
        def _integrate1d_imp(i):
            masked = assembled[i].copy()
            mask = np.zeros_like(image_mask)
            mask_image_data(masked,
                            image_mask=image_mask,
                            threshold_mask=threshold_mask,
                            out=mask)
            return integ1d(masked, integ_points, mask=mask)

        intensities = []  # pulsed A.I.
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, ret in zip(range(len(assembled)),
                              executor.map(_integrate1d_imp,
                                           range(len(assembled)))):
                if i == 0:
                    momentum = ret.radial
                intensities.append(ret.intensity)

        # intensities = self._normalize_fom(
        #     processed, np.array(intensities), self._normalizer,
        #     x=momentum, auc_range=self._auc_range)

        # calculate the difference between each pulse and the
        # first one
        diffs = [p - intensities[0] for p in intensities]

        # calculate the figure of merit for each pulse
        foms = []
        for diff in diffs:
            fom = slice_curve(diff, momentum, *self._fom_integ_range)[0]
            foms.append(np.sum(np.abs(fom)))

        ai = processed.pulse.ai
        ai.x = momentum
        ai.y = intensities
        ai.fom = foms

        # Note: It is not correct to calculate the mean of intensities
        #       since the result is equivalent to setting all nan to zero
        #       instead of nanmean.


class AzimuthalIntegProcessorTrain(_AzimuthalIntegProcessorBase):
    """Train-resolved azimuthal integration processor."""

    _intensity_ma = MovingAverageArray()
    _intensity_on_ma = MovingAverageArray()
    _intensity_off_ma = MovingAverageArray()

    _analysis_types = [AnalysisType.AZIMUTHAL_INTEG,
                       AnalysisType.AZIMUTHAL_INTEG_PEAK,
                       AnalysisType.AZIMUTHAL_INTEG_PEAK_Q,
                       AnalysisType.AZIMUTHAL_INTEG_COM]

    def __init__(self):
        super().__init__()

        self._set_ma_window(1)

    def _set_ma_window(self, v):
        self._ma_window = v
        self.__class__._intensity_ma.window = v
        self.__class__._intensity_on_ma.window = v
        self.__class__._intensity_off_ma.window = v

    def _reset_ma(self):
        del self._intensity_ma
        del self._intensity_on_ma
        del self._intensity_off_ma

    def _update_moving_average(self, cfg):
        """Override."""
        if 'reset_ma' in cfg:
            self._reset_ma()

        v = int(cfg['ma_window'])
        if self._ma_window != v:
            self._set_ma_window(v)

    def _process_fom(self, ai):
        """Helper function to compute the FOM

        This is used for both the main train data and pump-probe data.

        :param AzimuthalIntegrationData ai: The data object to work on. Note
                                            that PumpProbeData (a subclass) is
                                            supported too.
        """
        if self._find_peaks:
            peaks, _ = find_peaks_1d(ai.y,
                                     prominence=self._peak_prominence)
            peaks = peaks[self._peak_slicer]

            if len(peaks) > self._MAX_N_PEAKS:
                peaks = None
            ai.peaks = peaks

        intensity, momentum = slice_curve(ai.y, ai.x, *self._fom_integ_range)

        # If this is for pump-probe data, ensure we respect the abs_difference
        # setting.
        is_pump_probe = type(ai) == PumpProbeData
        if is_pump_probe and ai.abs_difference:
            intensity = np.abs(intensity)

        def has_analysis(analysis_type):
            if type(ai) == PumpProbeData:
                return ai.analysis_type == analysis_type
            else:
                return self._meta.has_analysis(analysis_type)

        if has_analysis(AnalysisType.AZIMUTHAL_INTEG):
            # If we are in pump-probe mode, then the intensity has already been
            # absolute-valued (if requested).
            ai.fom = np.sum(intensity if is_pump_probe else np.abs(intensity))

        if has_analysis(AnalysisType.AZIMUTHAL_INTEG_COM):
            com = ndimage.center_of_mass(intensity)[0]
            com_index = int(round(com)) if not np.isnan(com) else -1

            # If a dark image has been subtracted, it's possible that many of
            # the values in the image and thus the I(q) will be negative. This
            # could cause the calculated CoM to be out-of-bounds, in which case
            # we simply discard the results. This typically only happens when
            # the X-ray beam is not in view of the detector, in which case we
            # don't care about the I(q) anyway.
            if com_index >= 0 and com_index < len(intensity):
                ai.center_of_mass = (momentum[com_index], intensity[com_index])
            else:
                ai.center_of_mass = (np.nan, np.nan)

        want_peak_fom = has_analysis(AnalysisType.AZIMUTHAL_INTEG_PEAK)
        want_peak_q_fom = has_analysis(AnalysisType.AZIMUTHAL_INTEG_PEAK_Q)
        if (want_peak_fom or want_peak_q_fom) and not self._find_peaks:
            raise ProcessingError("Peak finding must be enabled to use a peak FOM")

        have_peaks = peaks is not None and len(peaks) > 0

        if want_peak_fom:
            ai.max_peak = max([intensity[peak] for peak in peaks]) if have_peaks else np.nan

        if want_peak_q_fom:
            if have_peaks:
                peak_index = max(peaks, key=lambda idx: intensity[idx])
                ai.max_peak_q = momentum[peak_index]
            else:
                ai.max_peak_q = np.nan

    @profiler("Azimuthal Integration Processor (Train)")
    def process(self, data):
        processed = data['processed']

        integrator = self._update_integrator()
        integ1d = functools.partial(integrator.integrate1d,
                                    method=self._integ_method,
                                    radial_range=self._integ_range,
                                    correctSolidAngle=True,
                                    polarization_factor=1,
                                    unit="q_A^-1")
        integ_points = self._integ_points

        if self._meta.has_any_analysis(self._analysis_types):
            mask = processed.image.mask
            mean_ret = integ1d(processed.image.masked_mean, integ_points, mask=mask)
            momentum = mean_ret.radial
            intensity = self._normalize_fom(
                processed, mean_ret.intensity, self._normalizer,
                x=momentum, auc_range=self._auc_range)
            self._intensity_ma = intensity

            ai = processed.ai
            ai.x = momentum
            ai.y = self._intensity_ma
            ai.q_map = self._q_map

            if self._use_reference:
                reference_image = processed.image.reference
                if reference_image is None or not reference_image.any():
                    logger.error(f"No reference has been set. Please set Reference in Reference Tab")
                    
                else:
                    ai_reference = integ1d(reference_image, integ_points, mask=mask)
                    ai.reference = ai_reference

            self._process_fom(ai)

        # ------------------------------------
        # pump-probe azimuthal integration
        # ------------------------------------

        if processed.pp.analysis_type in self._analysis_types:
            pp = processed.pp

            image_on = pp.image_on
            image_off = pp.image_off
            mask_on = pp.on.mask
            mask_off = pp.off.mask

            if image_on is not None and image_off is not None:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    on_off_rets = executor.map(
                        lambda img, npts, msk: integ1d(img, npts, mask=msk),
                        (image_on, image_off),
                        (integ_points, integ_points),
                        (mask_on, mask_off)
                    )
                on_ret, off_ret = on_off_rets

                momentum = on_ret.radial

                self._intensity_on_ma = on_ret.intensity
                self._intensity_off_ma = off_ret.intensity

                y_on, y_off = self._normalize_fom_pp(
                    processed, self._intensity_on_ma, self._intensity_off_ma,
                    self._normalizer, x=on_ret.radial, auc_range=self._auc_range)
                vfom = y_on - y_off

                pp.y_on = y_on
                pp.y_off = y_off
                pp.x = momentum
                pp.y = vfom

                self._process_fom(pp)
