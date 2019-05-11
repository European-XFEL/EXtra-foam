"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

RoiProcessor.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import copy

import numpy as np

from .base_processor import (
    CompositeProcessor, LeafProcessor, SharedProperty,
    StopCompositionProcessing
)
from ..exceptions import ProcessingError
from ...algorithms import intersection, normalize_auc, slice_curve
from ...config import config, RoiFom, PumpProbeType, Projection1dNormalizer
from ...helpers import profiler


class RoiProcessor(CompositeProcessor):
    """RoiProcessor class.

    Process region of interest.

    Attributes:
        regions (list): a list of ROI regions (x, y, w, h) or None if the
            corresponding ROI is not activated.
        fom_type (RoiFom): type of ROI FOM.
    """
    regions = SharedProperty()
    visibilities = SharedProperty()
    roi_fom_handler = SharedProperty()
    fom_type = SharedProperty()
    proj1d_normalizer = SharedProperty()
    proj1d_auc_range = SharedProperty()
    proj1d_fom_integ_range = SharedProperty()

    def __init__(self):
        super().__init__()

        # initialization
        self.regions = [None] * len(config["ROI_COLORS"])
        self.visibilities = copy.copy(self.regions)

        self.roi_fom_handler = None

        self.add(RoiFomProcessor())
        self.add(RoiPumpProbeRoiProcessor())

    def update(self):
        cfg = self._meta.roi_getall()

        self.fom_type = RoiFom(int(cfg['fom_type']))

        if self.fom_type == RoiFom.SUM:
            self.roi_fom_handler = np.sum
        elif self.fom_type == RoiFom.MEAN:
            self.roi_fom_handler = np.mean
        else:
            self.fom_type.fom_type = None
            self.roi_fom_handler = None

        for i, _ in enumerate(self.regions, 1):
            self.visibilities[i-1] = cfg[f'visibility{i}'] == 'True'
            self.regions[i-1] = self.str2list(cfg[f'region{i}'], handler=int)

        self.proj1d_normalizer = Projection1dNormalizer(int(cfg['proj1d:normalizer']))
        self.proj1d_auc_range = self.str2tuple(cfg['proj1d:auc_range'])
        self.proj1d_fom_integ_range = self.str2tuple(cfg['proj1d:fom_integ_range'])

    @staticmethod
    def get_roi_image(roi_region, img, copy=True):
        x, y, w, h = roi_region
        return np.array(img[y:y + h, x:x + w], copy=copy)


class RoiFomProcessor(LeafProcessor):
    """RoiFomProcessor class.

    Take the on and off images calculated by the PumpProbeProcessor
    and extract the ROIs of both on and off images using ROI1. The
    figure-of-merit is sum or mean of the difference between these
    two ROI images.
    """
    @profiler("ROI processor")
    def process(self, processed, raw=None):
        """Override.

        Note: We need to put some data in the history, even if ROI is not
        activated. This is required for the case that different ROIs are
        activated at different times.
        """
        tid = processed.tid
        if tid > 0:
            img = processed.image.masked_mean

            rois = copy.copy(self.regions)
            for i, roi in enumerate(rois):
                # it should be valid to set ROI intensity to zero if the data
                # is not available
                fom = 0
                if self.visibilities[i]:
                    x, y, w, h = roi
                    roi = intersection([x, y, w, h], [0, 0, *img.shape[::-1]])
                    x, y, w, h = roi
                    if w > 0 and h > 0:
                        # set the corrected roi
                        setattr(processed.roi, f"roi{i+1}", roi)

                        roi_img = RoiProcessor.get_roi_image(
                            roi, img, copy=False)

                        proj_x = np.sum(roi_img, axis=-2)
                        proj_y = np.sum(roi_img, axis=-1)
                        setattr(processed.roi, f"roi{i+1}_proj_x", proj_x)
                        setattr(processed.roi, f"roi{i+1}_proj_y", proj_y)

                        fom = self.roi_fom_handler(roi_img)
                        setattr(processed.roi, f"roi{i+1}_fom", fom)


class RoiPumpProbeRoiProcessor(CompositeProcessor):
    """RoiPumpProbeRoiProcessor class.

    Extract the ROI image for on/off pulses respectively.
    """
    def __init__(self):
        super().__init__()

        self.add(RoiPumpProbeProj1dProcessor())

    @profiler("ROI processor")
    def process(self, processed, raw=None):
        # use ROI1 for signal
        roi = processed.roi.roi1
        if roi is None:
            raise StopCompositionProcessing

        # use ROI2 for background
        roi_bkg = processed.roi.roi2
        if processed.pp.analysis_type != PumpProbeType.ROI1_BY_ROI2:
            if roi_bkg is not None and roi_bkg[:2] != roi[:2]:
                raise ProcessingError("Shapes of ROI1 and ROI2 are different")

        on_image = processed.pp.on_image_mean
        off_image = processed.pp.off_image_mean
        if on_image is None or off_image is None:
            return StopCompositionProcessing

        on_roi = RoiProcessor.get_roi_image(roi, on_image)
        off_roi = RoiProcessor.get_roi_image(roi, off_image)
        # ROI background subtraction, which is also a kind of normalization
        if roi_bkg is not None:
            on_roi_bkg = RoiProcessor.get_roi_image(
                roi_bkg, on_image, copy=False)
            off_roi_bkg = RoiProcessor.get_roi_image(
                roi_bkg, off_image, copy=False)
            if processed.pp.analysis_type != PumpProbeType.ROI1_BY_ROI2:
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

        # set the current on/off ROIs
        processed.pp.on_roi = on_roi
        processed.pp.off_roi = off_roi

        if processed.pp.analysis_type in (PumpProbeType.ROI,
                                          PumpProbeType.ROI1_BY_ROI2):
            processed.pp.data = (None, on_roi, off_roi)
            _, on_ma, off_ma = processed.pp.data  # get the moving average

            # for now, no normalization is applied
            norm_on_ma = np.copy(on_ma)
            norm_off_ma = np.copy(off_ma)
            norm_on_off_ma = norm_on_ma - norm_off_ma

            if processed.pp.abs_difference:
                fom = self.roi_fom_handler(np.abs(norm_on_off_ma))
            else:
                fom = self.roi_fom_handler(norm_on_off_ma)

            processed.pp.norm_on_ma = norm_on_ma
            processed.pp.norm_off_ma = norm_off_ma
            processed.pp.norm_on_off_ma = norm_on_off_ma
            processed.pp.fom = fom


class RoiPumpProbeProj1dProcessor(LeafProcessor):
    """RoiPumpProbeProj1dProcessor class.

    Calculate the 1D projection for on/off ROIs.
    """
    def __init__(self):
        super().__init__()

    def process(self, processed, raw=None):
        if processed.pp.analysis_type == PumpProbeType.ROI_PROJECTION_X:
            axis = -2
        elif processed.pp.analysis_type == PumpProbeType.ROI_PROJECTION_Y:
            axis = -1
        else:
            return

        on_roi = processed.pp.on_roi
        off_roi = processed.pp.off_roi

        x_data = np.arange(on_roi.shape[::-1][axis])
        # 1D projection
        on_data = np.sum(on_roi, axis=axis)
        off_data = np.sum(off_roi, axis=axis)

        # set data and calculate moving average
        processed.pp.data = (x_data, on_data, off_data)
        _, on_ma, off_ma = processed.pp.data

        try:
            norm_on_ma = normalize_auc(
                on_ma, x_data, *self.proj1d_auc_range)
            norm_off_ma = normalize_auc(
                off_ma, x_data, *self.proj1d_auc_range)
        except ValueError as e:
            raise ProcessingError(str(e))

        norm_on_off_ma = norm_on_ma - norm_off_ma

        sliced = slice_curve(norm_on_off_ma, x_data,
                             *self.proj1d_fom_integ_range)[0]
        if processed.pp.abs_difference:
            fom = np.sum(np.abs(sliced))
        else:
            fom = np.sum(sliced)

        processed.pp.norm_on_ma = norm_on_ma
        processed.pp.norm_off_ma = norm_off_ma
        processed.pp.norm_on_off_ma = norm_on_off_ma
        processed.pp.fom = fom
