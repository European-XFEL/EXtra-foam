"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import _BaseProcessor
from ..data_model import MovingAverageArray
from ..exceptions import ProcessingError
from ...algorithms import slice_curve, mask_image
from ...database import Metadata as mt
from ...config import AnalysisType, VFomNormalizer
from ...utils import profiler

from karaboFAI.cpp import intersection


class _RectROI:
    """RectROI used in RoiProcessor.

    Note: there is a class RectROI on the GUI part.
    """
    def __init__(self):
        self._x = 0
        self._y = 0
        self._w = -1
        self._h = -1

        self._activated = False

    def get_image(self, img, copy=False):
        """Get the ROI from a given image.

        :return: numpy.ndarray if there is intersection between the ROI
            and the image and None if not.
        """
        x, y, w, h = self.intersect(img)
        if w > 0 and h > 0:
            return np.array(img[y:y + h, x:x + w], copy=copy)

    def get_images(self, imgs, copy=False):
        """Get the ROI from an array of images.

        :return: numpy.ndarray if there is intersection between the ROI
            and the image and None if not.
        """
        x, y, w, h = self.intersect(imgs[0])

        if w > 0 and h > 0:
            return np.array(imgs[:, y:y + h, x:x + w], copy=copy)

    def get_images_pp(self, img_on, img_off, copy=False):
        """Convenient function to get on/off images together.

        on/off images ought to have the same shape.
        """
        x, y, w, h = self.intersect(img_on)
        if w > 0 and h > 0:
            return np.array(img_on[y:y + h, x:x + w], copy=copy), \
                   np.array(img_off[y:y + h, x:x + w], copy=copy)
        return None, None

    def intersect(self, img):
        """Get the intersection region between the ROI and an image.

        :return: it returns [0, 0, -1, -1] if the ROI is not activated.
            Otherwise [x, y, w, h] of the intersection region.
        """
        if not self._activated:
            return [0, 0, -1, -1]
        return intersection([self._x, self._y, self._w, self._h],
                            [0, 0, *img.shape[::-1]])

    @property
    def rect(self):
        return [self._x, self._y, self._w, self._h]

    @rect.setter
    def rect(self, v):
        self._x, self._y, self._w, self._h = v

    @property
    def activated(self):
        return self._activated

    @activated.setter
    def activated(self, v):
        self._activated = bool(v)


def project_x(img):
    """Return projection of an image in the x direction."""
    if img is None:
        return None
    return np.sum(img, axis=-2)


def project_y(img):
    """Return projection of an image in the y direction."""
    if img is None:
        return None
    return np.sum(img, axis=-1)


class _RoiProcessBase(_BaseProcessor):
    """Base class for RoiProcessors.

    Attributes:
        _normalizer (VFomNormalizer): normalizer type for calculating
            FOM from 1D projection result.
        _auc_range (tuple): x range for calculating AUC.
        _fom_integ_range (tuple): integration range for calculating
            FOM from the normalized 1D projection.
        _ma_window (int): moving average window size.
    """
    def __init__(self):
        super().__init__()

        self._roi1 = _RectROI()
        self._roi2 = _RectROI()
        self._roi3 = _RectROI()
        self._roi4 = _RectROI()

        self._has_img1 = False
        self._has_img2 = False
        self._has_img12 = False
        self._has_img3 = False
        self._has_img4 = False
        self._has_img34 = False

        self._direction = 'x'
        self._normalizer = VFomNormalizer.AUC

        self._auc_range = None
        self._fom_integ_range = None

        self._ma_window = 1

    def update(self):
        """Override."""
        g_cfg = self._meta.get_all(mt.GLOBAL_PROC)
        self._update_moving_average(g_cfg)

        cfg = self._meta.get_all(mt.ROI_PROC)

        self._roi1.activated = cfg[f'visibility1'] == 'True'
        self._roi1.rect = self.str2list(cfg[f'region1'], handler=int)
        self._roi2.activated = cfg[f'visibility2'] == 'True'
        self._roi2.rect = self.str2list(cfg[f'region2'], handler=int)
        self._roi3.activated = cfg[f'visibility3'] == 'True'
        self._roi3.rect = self.str2list(cfg[f'region3'], handler=int)
        self._roi4.activated = cfg[f'visibility4'] == 'True'
        self._roi4.rect = self.str2list(cfg[f'region4'], handler=int)

        self._direction = cfg['proj:direction']
        self._normalizer = VFomNormalizer(int(cfg['proj:normalizer']))
        self._auc_range = self.str2tuple(cfg['proj:auc_range'])
        self._fom_integ_range = self.str2tuple(cfg['proj:fom_integ_range'])

    def _update_moving_average(self, v):
        pass


class RoiProcessorPulse(_RoiProcessBase):
    """Pulse-resolved RoiProcessor."""
    @profiler("ROI Processor (pulse)")
    def process(self, data):
        processed = data['processed']
        assembled = data['detector']['assembled']

        self._process_roi1(processed, assembled)
        self._process_roi2(processed, assembled)

    def _process_roi1(self, processed, assembled):
        """Process pulse-resolved ROI1 FOM a train."""
        if not self._has_analysis(AnalysisType.ROI1_PULSE):
            return

        threshold_mask = processed.image.threshold_mask
        image_mask = processed.image.image_mask

        # for speed
        roi = processed.pulse.roi

        roi.rect1 = self._roi1.intersect(assembled[0])

        # get the current ROI images
        img1 = self._roi1.get_images(assembled)

        # set up the flags
        self._has_img1 = img1 is not None

        if self._has_img1:
            roi_img_mask = None
            if image_mask is not None:
                x, y, w, h = self._roi1.intersect(image_mask)
                roi_img_mask = image_mask[y:y + h, x:x + w]

            foms = []
            for i in range(len(img1)):
                foms.append(np.sum(mask_image(img1[i],
                                              image_mask=roi_img_mask,
                                              threshold_mask=threshold_mask)))
            roi.roi1.fom = foms

    def _process_roi2(self, processed, assembled):
        """Process pulse-resolved ROI2 FOM a train."""
        if not self._has_analysis(AnalysisType.ROI2_PULSE):
            return

        threshold_mask = processed.image.threshold_mask
        image_mask = processed.image.image_mask

        # for speed
        roi = processed.pulse.roi

        roi.rect2 = self._roi2.intersect(assembled[0])

        # get the current ROI images
        img2 = self._roi2.get_images(assembled)

        # set up the flags
        self._has_img2 = img2 is not None

        if self._has_img2:
            roi_img_mask = None
            if image_mask is not None:
                x, y, w, h = self._roi2.intersect(image_mask)
                roi_img_mask = image_mask[y:y + h, x:x + w]

            foms = []
            for i in range(len(img2)):
                foms.append(np.sum(mask_image(img2[i],
                                              image_mask=roi_img_mask,
                                              threshold_mask=threshold_mask)))
            roi.roi2.fom = foms


class RoiProcessorTrain(_RoiProcessBase):
    """Train-resolved RoiProcessor."""

    _img1 = MovingAverageArray()
    _img2 = MovingAverageArray()
    _img3 = MovingAverageArray()
    _img4 = MovingAverageArray()

    _img1_on = MovingAverageArray()
    _img2_on = MovingAverageArray()
    _img3_on = MovingAverageArray()
    _img4_on = MovingAverageArray()

    _img1_off = MovingAverageArray()
    _img2_off = MovingAverageArray()
    _img3_off = MovingAverageArray()
    _img4_off = MovingAverageArray()

    def _update_moving_average(self, cfg):
        if 'reset_roi' in cfg:
            # reset moving average
            del self._img1
            del self._img2
            del self._img3
            del self._img4

            del self._img1_on
            del self._img2_on
            del self._img3_on
            del self._img4_on

            del self._img1_off
            del self._img2_off
            del self._img3_off
            del self._img4_off

            self._meta.delete(mt.GLOBAL_PROC, 'reset_roi')

        v = int(cfg['ma_window'])
        if self._ma_window != v:
            self.__class__._img1.window = v
            self.__class__._img2.window = v
            self.__class__._img3.window = v
            self.__class__._img4.window = v

            self.__class__._img1_on.window = v
            self.__class__._img2_on.window = v
            self.__class__._img3_on.window = v
            self.__class__._img4_on.window = v

            self.__class__._img1_off.window = v
            self.__class__._img2_off.window = v
            self.__class__._img3_off.window = v
            self.__class__._img4_off.window = v

        self._ma_window = v

    @profiler("ROI Processor (train)")
    def process(self, data):
        processed = data['processed']

        error_messages = []

        self._process_rois(processed)
        self._process_rois_pp(processed)

        try:
            self._process_proj(processed)
        except ProcessingError as e:
            error_messages.append(repr(e))

        try:
            self._process_proj_pp(processed)
        except ProcessingError as e:
            error_messages.append(repr(e))

        if error_messages:
            raise ProcessingError("; ".join(error_messages))

    def _process_rois(self, processed):
        """Process averaged image in a train."""
        # for speed
        roi = processed.roi

        masked_mean = processed.image.masked_mean

        roi.rect1 = self._roi1.intersect(masked_mean)
        roi.rect2 = self._roi2.intersect(masked_mean)
        roi.rect3 = self._roi3.intersect(masked_mean)
        roi.rect4 = self._roi4.intersect(masked_mean)

        # get the current ROI images
        img1 = self._roi1.get_image(masked_mean)
        img2 = self._roi2.get_image(masked_mean)
        img3 = self._roi3.get_image(masked_mean)
        img4 = self._roi4.get_image(masked_mean)

        # set up the flags
        self._has_img1 = img1 is not None
        self._has_img2 = img2 is not None
        self._has_img12 = (img1 is not None and img2 is not None
                           and img1.shape == img2.shape)
        self._has_img3 = img3 is not None
        self._has_img4 = img4 is not None
        self._has_img34 = (img3 is not None and img4 is not None
                           and img3.shape == img4.shape)

        self._img1 = img1
        self._img2 = img2
        self._img3 = img3
        self._img4 = img4

        if self._has_img1:
            roi.roi1.fom = np.sum(self._img1)
        if self._has_img2:
            roi.roi2.fom = np.sum(self._img2)
        if self._has_img3:
            roi.norm3 = np.sum(self._img3)
        if self._has_img4:
            roi.norm4 = np.sum(self._img4)
        if self._has_img12:
            fom1 = roi.roi1.fom
            fom2 = roi.roi2.fom
            roi.roi1_sub_roi2.fom = fom1 - fom2
            roi.roi1_add_roi2.fom = fom1 + fom2
        if self._has_img34:
            norm3 = roi.norm3
            norm4 = roi.norm4
            roi.norm3_sub_norm4 = norm3 - norm4
            roi.norm3_add_norm4 = norm3 + norm4

    def _process_rois_pp(self, processed):
        """Process on/off ROI images if PUMP_PROBE is registered."""
        if processed.pp.analysis_type == AnalysisType.UNDEFINED:
            return

        # on/off images can only be non-None at the same time
        img_on_full = processed.pp.image_on
        img_off_full = processed.pp.image_off
        if img_on_full is None:
            return

        self._img1_on, self._img1_off = self._roi1.get_images_pp(
            img_on_full, img_off_full)
        self._img2_on, self._img2_off = self._roi2.get_images_pp(
            img_on_full, img_off_full)
        self._img3_on, self._img3_off = self._roi3.get_images_pp(
            img_on_full, img_off_full)
        self._img4_on, self._img4_off = self._roi4.get_images_pp(
            img_on_full, img_off_full)

        # for speed
        on = processed.roi.on
        off = processed.roi.off

        if self._has_img1:
            on.fom1, off.fom1 = np.sum(self._img1_on), np.sum(self._img1_off)
        if self._has_img2:
            on.fom2, off.fom2 = np.sum(self._img2_on), np.sum(self._img2_off)
        # We cannot tell which normalization is used from the analysis type,
        # therefore, we calculate all of them.
        # TODO: consider to register normalizer type
        if self._has_img3:
            on.norm3, off.norm3 = np.sum(self._img3_on), np.sum(self._img3_off)
        if self._has_img4:
            on.norm4, off.norm4 = np.sum(self._img4_on), np.sum(self._img4_off)

        if self._has_img12:
            fom1_on, fom1_off = on.fom1, off.fom1
            fom2_on, fom2_off = on.fom2, off.fom2
            on.fom1_sub_fom2 = fom1_on - fom2_on
            off.fom1_sub_fom2 = fom1_off - fom2_off
            on.fom1_add_fom2 = fom1_on + fom2_on
            off.fom1_add_fom2 = fom1_off + fom2_off

        if self._has_img34:
            fom3_on, fom3_off = on.norm3, off.norm3
            fom4_on, fom4_off = on.norm4, off.norm4
            on.norm3_sub_norm4 = fom3_on - fom4_on
            off.norm3_sub_norm4 = fom3_off - fom4_off
            on.norm3_add_norm4 = fom3_on + fom4_on
            off.norm3_add_norm4 = fom3_off + fom4_off

    def _process_proj(self, processed):
        # calculate 1D projection

        # We calculate the 1D projections for ROI1, ROI2, ROI1+ROI2, ROI1-ROI2
        # no matter whether the analysis types are registered.
        if self._direction == 'x':
            handler = project_x
        else:
            handler = project_y

        # for speed
        roi = processed.roi

        require_roi1 = False
        require_roi2 = False
        if self._has_any_analysis([AnalysisType.PROJ_ROI1_SUB_ROI2,
                                   AnalysisType.PROJ_ROI1_ADD_ROI2]):
            require_roi1 = True
            require_roi2 = True
        elif self._has_analysis(AnalysisType.PROJ_ROI1):
            require_roi1 = True
        elif self._has_analysis(AnalysisType.PROJ_ROI2):
            require_roi2 = True

        if require_roi1 and self._has_img1:
            proj1 = handler(self._img1)
            x = np.arange(len(proj1))

            normalized = self._normalize_vfom(processed, proj1, self._normalizer,
                                              x=x, auc_range=self._auc_range)

            sliced = slice_curve(normalized, x, *self._fom_integ_range)[0]

            fom = np.sum(np.abs(sliced))

            roi.proj1.x = x
            roi.proj1.vfom = normalized
            roi.proj1.fom = fom

        if require_roi2 and self._has_img2:
            proj2 = handler(self._img2)
            x = np.arange(len(proj2))

            normalized = self._normalize_vfom(processed, proj2, self._normalizer,
                                              x=x, auc_range=self._auc_range)

            sliced = slice_curve(normalized, x, *self._fom_integ_range)[0]

            fom = np.sum(np.abs(sliced))

            roi.proj2.x = x
            roi.proj2.vfom = normalized
            roi.proj2.fom = fom

        if self._has_img12 and require_roi1 and require_roi2:
            vfom1 = roi.proj1.vfom
            vfom2 = roi.proj2.vfom

            vfom1_sub_vfom2 = vfom1 - vfom2
            vfom1_add_vfom2 = vfom1 + vfom2

            x = np.arange(len(vfom1_sub_vfom2))
            sliced1_sub_sliced2 = slice_curve(
                vfom1_sub_vfom2, x, *self._fom_integ_range)[0]
            sliced1_add_sliced2 = slice_curve(
                vfom1_add_vfom2, x, *self._fom_integ_range)[0]

            fom1_sub_fom2 = np.sum(np.abs(sliced1_sub_sliced2))
            fom1_add_fom2 = np.sum(np.abs(sliced1_add_sliced2))

            roi.proj1_sub_proj2.x = x
            roi.proj1_sub_proj2.vfom = vfom1_sub_vfom2
            roi.proj1_sub_proj2.fom = fom1_sub_fom2

            roi.proj1_add_proj2.x = x
            roi.proj1_add_proj2.vfom = vfom1_add_vfom2
            roi.proj1_add_proj2.fom = fom1_add_fom2

    def _process_proj_pp(self, processed):
        """ROI pump-probe 1D projection analysis."""
        # on/off images can only be non-None at the same time
        # Note: this is the only flag that we should use to check whether we
        #       have on/off images for the current train.
        img_on_full = processed.pp.image_on
        if img_on_full is None:
            return

        img_on, img_off = None, None
        analysis_type = processed.pp.analysis_type
        if analysis_type == AnalysisType.PROJ_ROI1 and self._has_img1:
            img_on = self._img1_on
            img_off = self._img1_off
            proj = processed.roi.proj1
        elif analysis_type == AnalysisType.PROJ_ROI2 and self._has_img2:
            img_on = self._img2_on
            img_off = self._img2_off
            proj = processed.roi.proj2
        elif analysis_type == AnalysisType.PROJ_ROI1_SUB_ROI2 and self._has_img12:
            img_on = self._img1_on - self._img2_on
            img_off = self._img1_off - self._img2_off
            proj = processed.roi.proj1_sub_proj2
        elif analysis_type == AnalysisType.PROJ_ROI1_ADD_ROI2 and self._has_img12:
            img_on = self._img1_on + self._img2_on
            img_off = self._img1_off + self._img2_off
            proj = processed.roi.proj1_add_proj2

        if img_on is None:
            return

        # calculate 1D projection
        if self._direction == 'x':
            handler = project_x
        else:
            handler = project_y

        proj_on, proj_off = handler(img_on), handler(img_off)

        x = np.arange(len(proj_on))

        vfom_on, vfom_off = self._normalize_vfom_pp(
            processed, proj_on, proj_off, self._normalizer,
            x=x, auc_range=self._auc_range)

        vfom = vfom_on - vfom_off

        sliced = slice_curve(vfom, x, *self._fom_integ_range)[0]
        if processed.pp.abs_difference:
            fom = np.sum(np.abs(sliced))
        else:
            fom = np.sum(sliced)

        pp = processed.pp
        pp.x = x
        pp.vfom_on = vfom_on
        pp.vfom_off = vfom_off
        pp.vfom = vfom
        pp.fom = fom
        pp.x_label = proj.x_label
        pp.vfom_label = f"[pump-probe] {proj.vfom_label}"
