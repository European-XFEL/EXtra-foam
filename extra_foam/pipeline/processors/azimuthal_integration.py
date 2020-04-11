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

from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray
from ...algorithms import slice_curve
from ...config import AnalysisType, Normalizer, list_azimuthal_integ_methods
from ...database import Metadata as mt
from ...utils import profiler

from extra_foam.algorithms import energy2wavelength, mask_image_data


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
    """

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

        self._reset_ma = False

    def update(self):
        """Override."""
        g_cfg = self._meta.hget_all(mt.GLOBAL_PROC)
        self._sample_dist = float(g_cfg['sample_distance'])
        self._wavelength = energy2wavelength(1e3 * float(g_cfg['photon_energy']))
        self._update_moving_average(g_cfg)

        cfg = self._meta.hget_all(mt.AZIMUTHAL_INTEG_PROC)
        self._pixel1 = float(cfg['pixel_size_y'])
        self._pixel2 = float(cfg['pixel_size_x'])
        self._poni1 = int(cfg['integ_center_y']) * self._pixel1
        self._poni2 = int(cfg['integ_center_x']) * self._pixel2

        self._integ_method = cfg['integ_method']
        self._integ_range = self.str2tuple(cfg['integ_range'])
        self._integ_points = int(cfg['integ_points'])
        self._normalizer = Normalizer(int(cfg['normalizer']))
        self._auc_range = self.str2tuple(cfg['auc_range'])
        self._fom_integ_range = self.str2tuple(cfg['fom_integ_range'])

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

    def __init__(self):
        super().__init__()

        self._set_ma_window(1)

    def _set_ma_window(self, v):
        self._ma_window = v
        self.__class__._intensity_ma.window = v
        self.__class__._intensity_on_ma.window = v
        self.__class__._intensity_off_ma.window = v

    def _update_moving_average(self, cfg):
        if 'reset_ma_ai' in cfg:
            # reset moving average
            del self._intensity_ma
            del self._intensity_on_ma
            del self._intensity_off_ma
            self._meta.hdel(mt.GLOBAL_PROC, 'reset_ma_ai')

        v = int(cfg['ma_window'])
        if self._ma_window != v:
            self._set_ma_window(v)

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

        if self._meta.has_analysis(AnalysisType.AZIMUTHAL_INTEG):
            mask = processed.image.mask
            mean_ret = integ1d(processed.image.masked_mean, integ_points, mask=mask)

            momentum = mean_ret.radial
            intensity = self._normalize_fom(
                processed, mean_ret.intensity, self._normalizer,
                x=momentum, auc_range=self._auc_range)
            self._intensity_ma = intensity

            fom = slice_curve(self._intensity_ma, momentum, *self._fom_integ_range)[0]
            fom = np.sum(np.abs(fom))

            ai = processed.ai
            ai.x = momentum
            ai.y = self._intensity_ma
            ai.fom = fom
            ai.q_map = self._q_map

        # ------------------------------------
        # pump-probe azimuthal integration
        # ------------------------------------

        if processed.pp.analysis_type == AnalysisType.AZIMUTHAL_INTEG:
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
                sliced = slice_curve(vfom, momentum, *self._fom_integ_range)[0]

                if pp.abs_difference:
                    fom = np.sum(np.abs(sliced))
                else:
                    fom = np.sum(sliced)

                pp.y_on = y_on
                pp.y_off = y_off
                pp.x = momentum
                pp.y = vfom
                pp.fom = fom
