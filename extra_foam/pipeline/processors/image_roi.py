"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import math

import numpy as np

from ...algorithms import nanhist_with_stats, slice_curve
from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray, RectRoiGeom
from ..exceptions import ProcessingError, UnknownParameterError
from ...ipc import process_logger as logger
from ...database import Metadata as mt
from ...utils import profiler
from ...config import AnalysisType, Normalizer, RoiCombo, RoiFom, RoiProjType

from extra_foam.algorithms import (
    intersection, mask_image_data
)


class _RoiProcessorBase(_BaseProcessor):
    """Base ROI processor.

    Attributes:
        _fom_combo (RoiCombo): ROI combination when calculating FOM.
        _fom_type (RoiFom): ROI FOM type.
        _fom_norm (Normalizer): ROI FOM normalizer.
        _hist_combo (RoiCombo): ROI combination when calculating histogram.
        _hist_n_bins (int): number of bins for ROI histogram.
        _hist_bin_range (tuple): bin range for ROI histogram.
        _norm_combo (RoiCombo): ROI combination when calculating ROI
            normalizer.
        _norm_type (RoiFom): ROI normalizer type.
    """

    _fom_handlers = {
        RoiFom.SUM: np.nansum,
        RoiFom.MEAN: np.nanmean,
        RoiFom.MEDIAN: np.nanmedian,
        RoiFom.MAX: np.nanmax,
        RoiFom.MIN: np.nanmin
    }

    def __init__(self):
        super().__init__()

        self._fom_combo = RoiCombo.ROI1
        self._fom_type = RoiFom.SUM
        self._fom_norm = Normalizer.UNDEFINED

        self._hist_combo = RoiCombo.ROI1
        self._hist_n_bins = 10
        self._hist_bin_range = (-math.inf, math.inf)

        self._norm_combo = RoiCombo.ROI3
        self._norm_type = RoiFom.SUM

    def update(self):
        cfg = self._meta.hget_all(mt.ROI_PROC)

        self._fom_combo = RoiCombo(int(cfg['fom:combo']))
        self._fom_type = RoiFom(int(cfg['fom:type']))
        self._fom_norm = Normalizer(int(cfg['fom:norm']))

        self._hist_combo = RoiCombo(int(cfg['hist:combo']))
        self._hist_n_bins = int(cfg['hist:n_bins'])
        self._hist_bin_range = self.str2tuple(cfg["hist:bin_range"])

        self._norm_combo = RoiCombo(int(cfg['norm:combo']))
        self._norm_type = RoiFom(int(cfg['norm:type']))

        return cfg

    @staticmethod
    def _get_roi_combo(roi1, roi2, combo, topic):
        """Calculate the combination of two ROIs.

        The two ROIs must have the same shape.
        """
        if combo == RoiCombo.ROI1:
            roi_combo = roi1
        elif combo == RoiCombo.ROI2:
            roi_combo = roi2
        else:
            if roi1 is None or roi2 is None:
                return

            if roi1.shape != roi2.shape:
                raise ProcessingError(
                    f"[ROI][{topic}] ROI1 and ROI2 must have the same shape")

            if combo == RoiCombo.ROI1_SUB_ROI2:
                roi_combo = roi1 - roi2
            elif combo == RoiCombo.ROI1_ADD_ROI2:
                roi_combo = roi1 + roi2
            else:
                raise UnknownParameterError(
                    f"[ROI][histogram] Unknown ROI histogram combo: "
                    f"{combo}")

        return roi_combo


class ImageRoiPulse(_RoiProcessorBase):
    """Pulse-resolved ROI processor.

    Attributes:
        _geom1, _geom2, _geom3, _geom4 (list): ROI geometries.
    """

    def __init__(self):
        super().__init__()

        self._geom1 = RectRoiGeom.INVALID
        self._geom2 = RectRoiGeom.INVALID
        self._geom3 = RectRoiGeom.INVALID
        self._geom4 = RectRoiGeom.INVALID

    def update(self):
        """Override."""
        cfg = super().update()

        self._geom1 = self.str2list(cfg[f'geom1'], handler=int)
        self._geom2 = self.str2list(cfg[f'geom2'], handler=int)
        self._geom3 = self.str2list(cfg[f'geom3'], handler=int)
        self._geom4 = self.str2list(cfg[f'geom4'], handler=int)

    @profiler("ROI Processor (pulse)")
    def process(self, data):
        processed = data['processed']
        assembled = data['assembled']['sliced']

        roi = processed.roi
        img_shape = assembled.shape[-2:]
        img_geom = [0, 0, img_shape[1], img_shape[0]]
        roi.geom1.geometry = intersection(self._geom1, img_geom)
        roi.geom2.geometry = intersection(self._geom2, img_geom)
        roi.geom3.geometry = intersection(self._geom3, img_geom)
        roi.geom4.geometry = intersection(self._geom4, img_geom)

        if self._pulse_resolved:
            self._process_norm(assembled, processed)
            self._process_fom(assembled, processed)
            self._process_hist(processed)

    def _compute_fom(self, roi, fom_type, image_mask, threshold_mask):
        if roi is None:
            return

        try:
            handler = self._fom_handlers[fom_type]
        except KeyError:
            raise UnknownParameterError(
                f"[ROI][FOM] Unknown FOM type: {fom_type}")

        mask_image_data(roi,
                        image_mask=image_mask,
                        threshold_mask=threshold_mask,
                        keep_nan=True)
        return handler(roi, axis=(-1, -2))

    def _process_norm(self, assembled, processed):
        """Calculate pulse-resolved ROI normalizers.

        Always calculate.
        """
        if not self._meta.has_analysis(AnalysisType.ROI_NORM_PULSE):
            return

        threshold_mask = processed.image.threshold_mask
        image_mask = processed.image.image_mask

        roi = processed.roi

        roi3 = roi.geom3.rect(assembled)
        mask3 = None if image_mask is None else roi.geom3.rect(image_mask)
        roi4 = roi.geom4.rect(assembled)
        mask4 = None if image_mask is None else roi.geom4.rect(image_mask)

        if self._norm_combo == RoiCombo.ROI3:
            processed.pulse.roi.norm = self._compute_fom(
                roi3, self._norm_type, mask3, threshold_mask)
        elif self._norm_combo == RoiCombo.ROI4:
            processed.pulse.roi.norm = self._compute_fom(
                roi4, self._norm_type, mask4, threshold_mask)
        else:
            norm3 = self._compute_fom(
                roi3, self._norm_type, mask3, threshold_mask)
            norm4 = self._compute_fom(
                roi4, self._norm_type, mask4, threshold_mask)
            if norm3 is not None and norm4 is not None:
                if self._norm_combo == RoiCombo.ROI3_SUB_ROI4:
                    processed.pulse.roi.norm = norm3 - norm4
                elif self._norm_combo == RoiCombo.ROI3_ADD_ROI4:
                    processed.pulse.roi.norm = norm3 + norm4
                else:
                    raise UnknownParameterError(
                        f"[ROI][normalizer] Unknown ROI combo: "
                        f"{self._norm_combo}")

        # Note: Exception will not be raised if norm is None due to ROI3
        #       or/and ROI4 are not available. Users are responsible to
        #       check whether they have activated and set a valid ROI region
        #       when they need ROI information in their analysis.

    def _process_fom(self, assembled, processed):
        """Calculate pulse-resolved ROI FOMs.

        Always calculate.
        """
        if not self._meta.has_analysis(AnalysisType.ROI_FOM_PULSE):
            return

        threshold_mask = processed.image.threshold_mask
        image_mask = processed.image.image_mask

        roi = processed.roi

        roi1 = roi.geom1.rect(assembled)
        mask1 = None if image_mask is None else roi.geom1.rect(image_mask)
        roi2 = roi.geom2.rect(assembled)
        mask2 = None if image_mask is None else roi.geom2.rect(image_mask)

        if self._fom_combo == RoiCombo.ROI1:
            processed.pulse.roi.fom = self._compute_fom(
                roi1, self._fom_type, mask1, threshold_mask)
        elif self._fom_combo == RoiCombo.ROI2:
            processed.pulse.roi.fom = self._compute_fom(
                roi2, self._fom_type, mask2, threshold_mask)
        else:
            fom1 = self._compute_fom(
                roi1, self._fom_type, mask1, threshold_mask)
            fom2 = self._compute_fom(
                roi2, self._fom_type, mask2, threshold_mask)
            if fom1 is not None and fom2 is not None:
                if self._fom_combo == RoiCombo.ROI1_SUB_ROI2:
                    processed.pulse.roi.fom = fom1 - fom2
                elif self._fom_combo == RoiCombo.ROI1_ADD_ROI2:
                    processed.pulse.roi.fom = fom1 + fom2
                else:
                    raise UnknownParameterError(
                        f"[ROI][FOM] Unknown ROI combo: {self._fom_combo}")

        # TODO: normalize

    def _process_hist(self, processed):
        """Calculate ROI histogram for POI pulses."""
        if self._hist_combo == RoiCombo.UNDEFINED:
            return

        roi_data = processed.roi
        image_data = processed.image
        for idx in image_data.poi_indices:
            img = image_data.images[idx]
            roi1 = roi_data.geom1.rect(img)
            roi2 = roi_data.geom2.rect(img)
            try:
                roi = self._get_roi_combo(
                    roi1, roi2, self._hist_combo, "histogram")
            except ProcessingError as e:
                logger.error(str(e))
                break

            if roi is None:
                continue

            processed.pulse.roi.hist[idx] = nanhist_with_stats(
                roi, self._hist_bin_range, self._hist_n_bins)

            if image_data.poi_indices[1] == image_data.poi_indices[0]:
                # skip the second one if two POIs have the same index
                break


class ImageRoiTrain(_RoiProcessorBase):
    """Train-resolved ROI processor.

    Attributes:
        _proj_combo (RoiCombo): ROI combination when calculating ROI
            projection.
        _proj_type (RoiProjType): ROI projection type.
        _proj_direct (str): ROI projection direction.
        _proj_norm (Normalizer): normalizer type for calculating
            FOM from 1D projection result.
        _proj_auc_range (tuple): x range for calculating AUC.
        _proj_fom_integ_range (tuple): integration range for calculating
            FOM from the normalized projection.
        _ma_window (int): moving average window size.
    """

    _proj_handlers = {
        RoiProjType.SUM: np.nansum,
        RoiProjType.MEAN: np.nanmean,
    }

    _roi1 = MovingAverageArray()
    _roi2 = MovingAverageArray()
    _roi3 = MovingAverageArray()
    _roi4 = MovingAverageArray()

    _roi1_on = MovingAverageArray()
    _roi2_on = MovingAverageArray()
    _roi3_on = MovingAverageArray()
    _roi4_on = MovingAverageArray()

    _roi1_off = MovingAverageArray()
    _roi2_off = MovingAverageArray()
    _roi3_off = MovingAverageArray()
    _roi4_off = MovingAverageArray()

    def __init__(self):
        super().__init__()

        self._proj_combo = RoiCombo.ROI1
        self._proj_type = RoiProjType.SUM
        self._proj_direct = 'x'
        self._proj_norm = Normalizer.UNDEFINED
        self._proj_auc_range = (0, math.inf)
        self._proj_fom_integ_range = (0, math.inf)

        self._ma_window = 1

    def update(self):
        """Override."""
        g_cfg = self._meta.hget_all(mt.GLOBAL_PROC)
        self._update_moving_average(g_cfg)

        cfg = super().update()

        self._proj_combo = RoiCombo(int(cfg['proj:combo']))
        self._proj_type = RoiProjType(int(cfg['proj:type']))
        self._proj_direct = cfg['proj:direct']
        self._proj_norm = Normalizer(int(cfg['proj:norm']))
        self._proj_auc_range = self.str2tuple((cfg['proj:auc_range']))
        self._proj_fom_integ_range = self.str2tuple((cfg['proj:fom_integ_range']))

    def _reset_roi_moving_average(self):
        del self._roi1
        del self._roi2
        del self._roi3
        del self._roi4

        del self._roi1_on
        del self._roi2_on
        del self._roi3_on
        del self._roi4_on

        del self._roi1_off
        del self._roi2_off
        del self._roi3_off
        del self._roi4_off

    def _set_roi_moving_average_window(self, v):
        self.__class__._roi1.window = v
        self.__class__._roi2.window = v
        self.__class__._roi3.window = v
        self.__class__._roi4.window = v

        self.__class__._roi1_on.window = v
        self.__class__._roi2_on.window = v
        self.__class__._roi3_on.window = v
        self.__class__._roi4_on.window = v

        self.__class__._roi1_off.window = v
        self.__class__._roi2_off.window = v
        self.__class__._roi3_off.window = v
        self.__class__._roi4_off.window = v

    def _update_moving_average(self, cfg):
        """Overload."""
        if 'reset_ma_roi' in cfg:
            self._reset_roi_moving_average()
            self._meta.hdel(mt.GLOBAL_PROC, 'reset_ma_roi')

        v = int(cfg['ma_window'])
        if self._ma_window != v:
            self._set_roi_moving_average_window(v)
        self._ma_window = v

    @profiler("ROI Processor (train)")
    def process(self, data):
        processed = data['processed']
        roi = processed.roi

        masked_mean = processed.image.masked_mean

        # update moving average
        self._roi1 = roi.geom1.rect(masked_mean)
        self._roi2 = roi.geom2.rect(masked_mean)
        self._roi3 = roi.geom3.rect(masked_mean)
        self._roi4 = roi.geom4.rect(masked_mean)

        self._process_hist(processed)
        self._process_norm(processed)
        self._process_fom(processed)
        self._process_proj(processed)

        # update pump-probe moving average

        masked_on = processed.pp.image_on
        if masked_on is None:
            return
        masked_off = processed.pp.image_off

        self._roi1_on = roi.geom1.rect(masked_on)
        self._roi1_off = roi.geom1.rect(masked_off)
        self._roi2_on = roi.geom2.rect(masked_on)
        self._roi2_off = roi.geom2.rect(masked_off)
        self._roi3_on = roi.geom3.rect(masked_on)
        self._roi3_off = roi.geom3.rect(masked_off)
        self._roi4_on = roi.geom4.rect(masked_on)
        self._roi4_off = roi.geom4.rect(masked_off)

        if processed.pp.analysis_type != AnalysisType.UNDEFINED:
            self._process_norm_pump_probe(processed)
            if processed.pp.analysis_type == AnalysisType.ROI_FOM:
                self._process_fom_pump_probe(processed)
            elif processed.pp.analysis_type == AnalysisType.ROI_PROJ:
                self._process_proj_pump_probe(processed)

    def _compute_fom(self, roi, fom_type):
        if roi is None:
            return

        try:
            handler = self._fom_handlers[fom_type]
            return handler(roi)
        except KeyError:
            raise UnknownParameterError(
                f"[ROI][FOM] Unknown FOM type: {fom_type}")

    def _process_hist(self, processed):
        """Calculate ROI histogram."""
        if self._hist_combo == RoiCombo.UNDEFINED:
            return

        try:
            roi = self._get_roi_combo(
                self._roi1, self._roi2, self._hist_combo, "histogram")
        except ProcessingError as e:
            logger.error(str(e))
            return

        if roi is not None:
            hist = processed.roi.hist
            hist.hist, hist.bin_centers, hist.mean, hist.median, hist.std = \
                nanhist_with_stats(roi, self._hist_bin_range, self._hist_n_bins)

    def _process_norm(self, processed):
        """Calculate train-resolved ROI normalizer."""
        roi = processed.roi

        norm3 = self._compute_fom(self._roi3, self._norm_type)
        norm4 = self._compute_fom(self._roi4, self._norm_type)

        if self._norm_combo == RoiCombo.ROI3:
            roi.norm = norm3
        elif self._norm_combo == RoiCombo.ROI4:
            roi.norm = norm4
        else:
            if norm3 is None or norm4 is None:
                return

            if self._norm_combo == RoiCombo.ROI3_SUB_ROI4:
                roi.norm = norm3 - norm4
            elif self._norm_combo == RoiCombo.ROI3_ADD_ROI4:
                roi.norm = norm3 + norm4
            else:
                raise UnknownParameterError(
                    f"[ROI][normalizer] Unknown ROI combo: {self._norm_combo}")

    def _process_fom(self, processed):
        """Calculate train-resolved ROI FOM."""
        roi = processed.roi

        fom1 = self._compute_fom(self._roi1, self._fom_type)
        fom2 = self._compute_fom(self._roi2, self._fom_type)

        if self._fom_combo == RoiCombo.ROI1:
            roi.fom = fom1
        elif self._fom_combo == RoiCombo.ROI2:
            roi.fom = fom2
        else:
            if fom1 is None or fom2 is None:
                return

            if self._fom_combo == RoiCombo.ROI1_SUB_ROI2:
                roi.fom = fom1 - fom2
            elif self._fom_combo == RoiCombo.ROI1_ADD_ROI2:
                roi.fom = fom1 + fom2
            else:
                raise UnknownParameterError(
                    f"[ROI][FOM] Unknown ROI combo: {self._fom_combo}")

        # TODO: normalize

    def _process_norm_pump_probe(self, processed):
        """Calculate train-resolved pump-probe ROI normalizers."""
        pp = processed.pp

        norm3_on = self._compute_fom(self._roi3_on, self._norm_type)
        norm3_off = self._compute_fom(self._roi3_off, self._norm_type)

        norm4_on = self._compute_fom(self._roi4_on, self._norm_type)
        norm4_off = self._compute_fom(self._roi4_off, self._norm_type)

        if self._norm_combo == RoiCombo.ROI3:
            pp.on.roi_norm = norm3_on
            pp.off.roi_norm = norm3_off
        elif self._norm_combo == RoiCombo.ROI4:
            pp.on.roi_norm = norm4_on
            pp.off.roi_norm = norm4_off
        else:
            if norm3_on is None or norm4_on is None:
                return

            if self._norm_combo == RoiCombo.ROI3_SUB_ROI4:
                pp.on.roi_norm = norm3_on - norm4_on
                pp.off.roi_norm = norm3_off - norm4_off
            elif self._norm_combo == RoiCombo.ROI3_ADD_ROI4:
                pp.on.roi_norm = norm3_on + norm4_on
                pp.off.roi_norm = norm3_off + norm4_off
            else:
                raise UnknownParameterError(
                    f"[ROI][normalizer] Unknown ROI combo: {self._norm_combo}")

    def _process_fom_pump_probe(self, processed):
        """Calculate train-resolved pump-probe ROI FOMs."""
        pp = processed.pp

        fom1_on = self._compute_fom(self._roi1_on, self._fom_type)
        fom1_off = self._compute_fom(self._roi1_off, self._fom_type)

        fom2_on = self._compute_fom(self._roi2_on, self._fom_type)
        fom2_off = self._compute_fom(self._roi2_off, self._fom_type)

        if self._fom_combo == RoiCombo.ROI1:
            fom_on = fom1_on
            fom_off = fom1_off
        elif self._fom_combo == RoiCombo.ROI2:
            fom_on = fom2_on
            fom_off = fom2_off
        else:
            if fom1_on is None or fom2_on is None:
                return

            if self._fom_combo == RoiCombo.ROI1_SUB_ROI2:
                fom_on = fom1_on - fom2_on
                fom_off = fom1_off - fom2_off
            elif self._fom_combo == RoiCombo.ROI1_ADD_ROI2:
                fom_on = fom1_on + fom2_on
                fom_off = fom1_off + fom2_off
            else:
                raise UnknownParameterError(
                    f"[ROI][FOM] Unknown ROI combo: {self._fom_combo}")

        if fom_on is None:
            return

        # TODO: normalize

        pp.fom = fom_on - fom_off

    def _compute_proj(self, roi):
        if roi is None:
            return

        handler = self._proj_handlers[self._proj_type]
        if self._proj_direct == "x":
            return handler(roi, axis=-2)
        elif self._proj_direct == "y":
            return handler(roi, axis=-1)
        else:
            raise UnknownParameterError(
                f"[ROI][projection] Unknown projection direction: "
                f"{self._proj_direct}")

    def _process_proj(self, processed):
        """Calculate train-resolved ROI projection."""
        try:
            roi_combo = self._get_roi_combo(
                self._roi1, self._roi2, self._proj_combo, "projection")
        except ProcessingError as e:
            logger.error(str(e))
            return

        if roi_combo is None:
            return
        proj = self._compute_proj(roi_combo)

        x = np.arange(len(proj))

        normalized_proj = self._normalize_fom(
            processed, proj, self._proj_norm, x=x, auc_range=self._proj_auc_range)
        fom = np.sum(normalized_proj)

        roi = processed.roi
        roi.proj.x = x
        roi.proj.y = normalized_proj
        roi.proj.fom = fom

    def _process_proj_pump_probe(self, processed):
        """Calculate train-resolved pump-probe ROI projections."""
        pp = processed.pp

        try:
            roi_combo_on = self._get_roi_combo(
                self._roi1_on, self._roi2_on, self._proj_combo, "projection")
            roi_combo_off = self._get_roi_combo(
                self._roi1_off, self._roi2_off, self._proj_combo, "projection")
        except ProcessingError as e:
            logger.error(str(e))
            return

        if roi_combo_on is None:
            return

        y_on = self._compute_proj(roi_combo_on)
        y_off = self._compute_proj(roi_combo_off)

        x = np.arange(len(y_on))

        normalized_y_on, normalized_y_off = self._normalize_fom_pp(
            processed, y_on, y_off, self._proj_norm,
            x=x, auc_range=self._proj_auc_range)

        normalized_y = normalized_y_on - normalized_y_off

        sliced = slice_curve(normalized_y, x, *self._proj_fom_integ_range)[0]
        if pp.abs_difference:
            fom = np.sum(np.abs(sliced))
        else:
            fom = np.sum(sliced)

        pp.y_on = normalized_y_on
        pp.y_off = normalized_y_off
        pp.x = x
        pp.y = normalized_y
        pp.fom = fom
