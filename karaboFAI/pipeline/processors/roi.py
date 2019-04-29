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

from .base_processor import LeafProcessor
from ...algorithms import intersection
from ...config import config, RoiFom
from ...helpers import profiler


class RoiProcessor(LeafProcessor):
    """Process region of interest.

    Attributes:
        roi_fom (int): type of ROI FOM.
    """
    def __init__(self):
        super().__init__()

        self._rois = [None] * len(config["ROI_COLORS"])

        self.roi_fom = None

    def set(self, rank, value):
        self._rois[rank-1] = value

    @profiler("ROI processor")
    def process(self, processed, raw=None):
        """Override.

        Note: We need to put some data in the history, even if ROI is not
        activated. This is required for the case that ROI1 and ROI2 were
        activate at different times.
        """
        roi_fom = self.roi_fom

        tid = processed.tid
        if tid > 0:
            img = processed.image.masked_mean
            img_ref = processed.image.masked_ref

            rois = copy.copy(self._rois)
            for i, roi in enumerate(rois):
                # it should be valid to set ROI intensity to zero if the data
                # is not available
                value = 0
                value_ref = 0
                if roi is not None:
                    roi = intersection(*roi, *img.shape[::-1], 0, 0)
                    if roi[0] < 0 or roi[1] < 0:
                        self._rois[i] = None
                    else:
                        setattr(processed.roi, f"roi{i+1}", roi)
                        value = self._get_roi_fom(roi, roi_fom, img)
                        value_ref = self._get_roi_fom(roi, roi_fom, img_ref)
                setattr(processed.roi, f"roi{i+1}_hist", (tid, value))
                setattr(processed.roi, f"roi{i+1}_hist_ref", (tid, value_ref))

    @staticmethod
    def _get_roi_fom(roi_param, roi_fom, img):
        if roi_fom is None or img is None:
            return 0

        w, h, x, y = roi_param
        roi_img = img[y:y + h, x:x + w]
        if roi_fom == RoiFom.SUM:
            ret = np.sum(roi_img)
        elif roi_fom == RoiFom.MEAN:
            ret = np.mean(roi_img)
        else:
            ret = 0

        return ret


