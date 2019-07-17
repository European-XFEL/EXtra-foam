"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

RoiProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import CompositeProcessor

from ..exceptions import ProcessingError
from ...algorithms import normalize_auc, slice_curve
from ...metadata import Metadata as mt
from ...config import config, RoiFom, AnalysisType, VFomNormalizer
from ...utils import profiler

from karaboFAI.cpp import intersection


class RoiProcessor(CompositeProcessor):
    """RoiProcessor class.

    Process region of interest.

    Attributes:
        _regions (list): a list of ROI regions (x, y, w, h).
        _visibilities (list): a list of boolean which indicates whether the
            corresponding ROI is visible.
        _roi_fom_handler (callable): hanlder used to calculate the FOM of a
            given ROI.
        _proj_normalizer (VFomNormalizer): normalizer type for calculating
            FOM from 1D projection result.
        _proj_auc_range (tuple): x range for calculating AUC, which is
            used as a normalizer of 1D projection.
        _proj_fom_integ_range (tuple): integration range for calculating
            FOM from the normalized 1D projection.
    """
    def __init__(self):
        super().__init__()

        n_rois = len(config["ROI_COLORS"])

        self._regions = [(0, 0, -1, -1)] * n_rois
        self._visibilities = [False] * n_rois
        self._roi_fom_handler = None

        self._proj_normalizer = None
        self._proj_auc_range = None
        self._proj_fom_integ_range = None

        self._reset = False

    def update(self):
        """Override."""
        cfg = self._meta.get_all(mt.ROI_PROC)

        # Do not reset since it might be useful to compare the result
        # from different handler and normalizer.
        if RoiFom(int(cfg['fom_type'])) == RoiFom.SUM:
            self._roi_fom_handler = np.sum
        else:  # fom_type == RoiFom.MEAN:
            self._roi_fom_handler = np.mean

        for i, _ in enumerate(self._regions, 1):
            self._visibilities[i-1] = cfg[f'visibility{i}'] == 'True'
            self._regions[i-1] = self.str2list(cfg[f'region{i}'], handler=int)

        self._proj_normalizer = VFomNormalizer(
            int(cfg['proj:normalizer']))
        self._proj_auc_range = self.str2tuple(cfg['proj:auc_range'])
        self._proj_fom_integ_range = self.str2tuple(
            cfg['proj:fom_integ_range'])

        if 'reset' in cfg:
            self._meta.delete(mt.ROI_PROC, 'reset')
            # reset when commanded by the GUI
            self._reset = True

    @staticmethod
    def get_roi_image(roi_region, img, copy=True):
        x, y, w, h = roi_region
        return np.array(img[y:y + h, x:x + w], copy=copy)

    @profiler("ROI Processor")
    def process(self, data):
        # Note: We need to put some data in the history, even if ROI is not
        # activated. This is required for the case that different ROIs are
        # activated at different times.
        processed = data['processed']

        processed.roi.reset = self._reset
        self._reset = False

        img = processed.image.masked_mean

        for i, roi in enumerate(self._regions):
            if self._visibilities[i]:
                # find the intersection between the ROI and the image
                roi = intersection([*roi], [0, 0, *img.shape[::-1]])
                # get the new ROI parameters
                x, y, w, h = roi
                if w > 0 and h > 0:
                    # set the corrected roi
                    setattr(processed.roi, f"roi{i+1}", roi)

                    roi_img = RoiProcessor.get_roi_image(
                        roi, img, copy=False)

                    # calculate the 1D projection no matter it is needed
                    # or not since the calculation is cheap
                    proj_x = np.sum(roi_img, axis=-2)
                    proj_y = np.sum(roi_img, axis=-1)
                    setattr(processed.roi, f"roi{i+1}_proj_x", proj_x)
                    setattr(processed.roi, f"roi{i+1}_proj_y", proj_y)
                    # calculate the FOM of the ROI
                    fom = self._roi_fom_handler(roi_img)
                    setattr(processed.roi, f"roi{i+1}_fom", fom)

        # pump-probe

        # use ROI1 for signal
        roi = processed.roi.roi1
        if roi is None:
            return

        # use ROI2 for background
        roi_bkg = processed.roi.roi2
        if processed.pp.analysis_type != AnalysisType.ROI1_DIV_ROI2:
            if roi_bkg is not None and roi_bkg[-2:] != roi[-2:]:
                raise ProcessingError("Shapes of ROI1 and ROI2 are different")

        on_image = processed.pp.on_image_mean
        off_image = processed.pp.off_image_mean

        if on_image is None or off_image is None:
            return

        on_roi = RoiProcessor.get_roi_image(roi, on_image)
        off_roi = RoiProcessor.get_roi_image(roi, off_image)
        # ROI background subtraction, which is also a kind of normalization
        if roi_bkg is not None:
            on_roi_bkg = RoiProcessor.get_roi_image(
                roi_bkg, on_image, copy=False)
            off_roi_bkg = RoiProcessor.get_roi_image(
                roi_bkg, off_image, copy=False)
            if processed.pp.analysis_type != AnalysisType.ROI1_DIV_ROI2:
                on_roi -= on_roi_bkg
                off_roi -= off_roi_bkg
            else:
                denominator_on = np.sum(on_roi_bkg)
                denominator_off = np.sum(off_roi_bkg)
                if denominator_on == 0 or denominator_off == 0:
                    raise ProcessingError(
                        "Invalid denominator: Total intenstity in ROI2 is 0")
                on_roi /= denominator_on
                off_roi /= denominator_off

        if processed.pp.analysis_type in (AnalysisType.ROI1_SUB_ROI2,
                                          AnalysisType.ROI1_DIV_ROI2):
            processed.pp.data = (None, on_roi, off_roi)
            x, on_ma, off_ma = processed.pp.data  # get the moving average

            # for now, no normalization is applied
            norm_on_ma = np.copy(on_ma)
            norm_off_ma = np.copy(off_ma)
            norm_on_off_ma = norm_on_ma - norm_off_ma

            if processed.pp.abs_difference:
                fom = self._roi_fom_handler(np.abs(norm_on_off_ma))
            else:
                fom = self._roi_fom_handler(norm_on_off_ma)

        else:
            if processed.pp.analysis_type == AnalysisType.ROI1_SUB_ROI2_PROJECTION_X:
                axis = -2
            elif processed.pp.analysis_type == AnalysisType.ROI1_SUB_ROI2_PROJECTION_Y:
                axis = -1
            else:
                return

            x_data = np.arange(on_roi.shape[::-1][axis])
            # 1D projection
            on_data = np.sum(on_roi, axis=axis)
            off_data = np.sum(off_roi, axis=axis)

            # set data and calculate moving average
            processed.pp.data = (x_data, on_data, off_data)
            x, on_ma, off_ma = processed.pp.data

            try:
                norm_on_ma = normalize_auc(
                    on_ma, x_data, *self._proj_auc_range)
                norm_off_ma = normalize_auc(
                    off_ma, x_data, *self._proj_auc_range)
            except ValueError as e:
                raise ProcessingError(str(e))

            norm_on_off_ma = norm_on_ma - norm_off_ma

            sliced = slice_curve(norm_on_off_ma, x_data,
                                 *self._proj_fom_integ_range)[0]
            if processed.pp.abs_difference:
                fom = np.sum(np.abs(sliced))
            else:
                fom = np.sum(sliced)

        processed.pp.on_roi = on_roi
        processed.pp.off_roi = off_roi
        processed.pp.x = x
        processed.pp.norm_on_ma = norm_on_ma
        processed.pp.norm_off_ma = norm_off_ma
        processed.pp.norm_on_off_ma = norm_on_off_ma
        processed.pp.fom = fom
