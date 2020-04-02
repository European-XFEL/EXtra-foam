"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import _BaseProcessor
from ..data_model import RawImageData
from ..exceptions import ImageProcessingError, ProcessingError
from ...database import Metadata as mt
from ...ipc import (
    CalConstantsSub, ImageMaskSub, ReferenceSub
)
from ...ipc import process_logger as logger
from ...utils import profiler
from ...config import config

from extra_foam.algorithms import (
    correct_image_data, mask_image_data, nanmean_image_data
)


_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']


class ImageProcessor(_BaseProcessor):
    """ImageProcessor class.

    Attributes:
        _dark (RawImageData): store the moving average of dark
            images in a train. Shape = (indices, y, x) for pulse-resolved
            and shape = (y, x) for train-resolved
        _correct_gain (bool): whether to apply gain correction.
        _correct_offset (bool): whether to apply offset correction.
        _gain (numpy.ndarray): gain constants. Shape = (memory cell, y, x)
        _offset (numpy.ndarray): offset constants. Shape = (memory cell, y, x)
        _gain_slicer (slice): a slice object used to slice the right memory
            cells from the gain constants.
        _offset_slicer (slice): a slice object used to slice the right memory
            cells from the offset constants.
        _gain_mean (numpy.ndarray): average of gain constants over memory
            cell. Shape = (y, x)
        _offset_mean (numpy.ndarray): average of offset constants over memory
            cell. Shape = (y, x)
        _dark_as_offset (bool): True for using recorded dark trains as offset.
        _recording_dark (bool): whether a dark run is being recorded.
        _dark_mean (bool): average of recorded dark trains over memory
            cell. Shape = (y, x)
        _image_mask (numpy.ndarray): image mask. For pulse-resolved detectors,
            this image mask is shared by all the pulses in a train. However,
            their overall mask could still be different after applying the
            threshold mask. Shape = (y, x), dtype = np.bool
        _threshold_mask (tuple):  (lower, upper) of the threshold.
            It should be noted that a pixel with value outside of the boundary
            will be masked as Nan/0, depending on the masking policy.
        _reference (numpy.ndarray): reference image.
        _poi_indices (list): indices of POI pulses.
    """

    # give it a huge window for now since I don't want to touch the
    # implementation of the base class for now.
    _dark = RawImageData(999999)

    def __init__(self):
        super().__init__()

        self._correct_gain = True
        self._correct_offset = True
        self._gain = None
        self._offset = None
        self._gain_slicer = slice(None, None)
        self._offset_slicer = slice(None, None)
        self._compute_gain_mean = False
        self._compute_offset_mean = False
        self._gain_mean = None
        self._offset_mean = None

        self._dark_as_offset = True
        self._recording_dark = False
        self._dark_mean = None

        self._image_mask = None
        self._threshold_mask = None
        self._reference = None

        self._poi_indices = None

        self._ref_sub = ReferenceSub()
        self._mask_sub = ImageMaskSub()
        self._cal_sub = CalConstantsSub()

    def update(self):
        # image
        cfg = self._meta.hget_all(mt.IMAGE_PROC)

        self._correct_gain = cfg['correct_gain'] == 'True'
        self._correct_offset = cfg['correct_offset'] == 'True'

        gain_slicer = self.str2slice(cfg['gain_slicer'])
        if gain_slicer != self._gain_slicer:
            self._compute_gain_mean = True
            self._gain_slicer = gain_slicer
        offset_slicer = self.str2slice(cfg['offset_slicer'])
        if offset_slicer != self._offset_slicer:
            self._compute_offset_mean = True
            self._offset_slicer = offset_slicer

        dark_as_offset = cfg['dark_as_offset'] == 'True'
        if dark_as_offset != self._dark_as_offset:
            self._compute_offset_mean = True
            self._dark_as_offset = dark_as_offset

        self._recording_dark = cfg['recording_dark'] == 'True'
        if 'remove_dark' in cfg:
            self._meta.hdel(mt.IMAGE_PROC, 'remove_dark')
            del self._dark
            self._dark_mean = None

        self._threshold_mask = self.str2tuple(
            cfg['threshold_mask'], handler=float)

        # global
        gp_cfg = self._meta.hget_all(mt.GLOBAL_PROC)
        self._poi_indices = [
            int(gp_cfg['poi1_index']), int(gp_cfg['poi2_index'])]

    @profiler("Image Processor (pulse)")
    def process(self, data):
        image_data = data['processed'].image
        assembled = data['assembled']['data']
        catalog = data['catalog']
        det = catalog.main_detector
        pulse_slicer = catalog.get_slicer(det)

        if assembled.ndim == 3:
            sliced_assembled = assembled[pulse_slicer]
            sliced_indices = list(range(
                *(pulse_slicer.indices(assembled.shape[0]))))
            n_sliced = len(sliced_indices)
        else:
            sliced_assembled = assembled
            sliced_indices = [0]
            n_sliced = 1

        if self._recording_dark:
            self._record_dark(assembled)

        sliced_gain, sliced_offset = self._update_gain_offset(assembled.shape)
        correct_image_data(sliced_assembled,
                           gain=sliced_gain,
                           offset=sliced_offset,
                           slicer=pulse_slicer)

        # Note: This will be needed by the pump_probe_processor to calculate
        #       the mean of assembled images. Also, the on/off indices are
        #       based on the sliced data.
        data['assembled']['sliced'] = sliced_assembled

        image_shape = sliced_assembled.shape[-2:]
        self._update_image_mask(image_shape)
        self._update_reference()

        # Avoid sending all images around
        image_data.images = [None] * n_sliced
        image_data.poi_indices = self._poi_indices
        self._update_pois(image_data, sliced_assembled)
        image_data.gain_mean = self._gain_mean
        image_data.offset_mean = self._offset_mean
        if self._dark is not None:
            # default is 0
            image_data.n_dark_pulses = 1 if self._dark.ndim == 2 \
                                         else len(self._dark)
        image_data.dark_count = self.__class__._dark.count
        image_data.image_mask = self._image_mask
        image_data.threshold_mask = self._threshold_mask
        image_data.reference = self._reference
        image_data.sliced_indices = sliced_indices

    def _record_dark(self, assembled):
        if self._dark is None:
            # _dark should not share the memory with
            # data[src] since the latter will be dark subtracted.
            self._dark = assembled.copy()
        else:
            # moving average (it reset the current moving average if the
            # new dark has a different shape)
            self._dark = assembled

        # For visualization of the dark:
        # FIXME: it would be better to calculate sliced dark mean.
        self._dark_mean = nanmean_image_data(self._dark)

    def _update_image_mask(self, image_shape):
        image_mask = self._mask_sub.update(self._image_mask, image_shape)
        if image_mask is not None and image_mask.shape != image_shape:
            if np.sum(image_mask) == 0:
                # reset the empty image mask automatically
                image_mask = None
            else:
                # This could only happen when the mask is loaded from the files
                # and the image shapes in the ImageTool is different from the
                # shape of the live images.
                # The original image mask remains the same.
                raise ImageProcessingError(
                    f"[Image processor] The shape of the image mask "
                    f"{image_mask.shape} is different from the shape of the image "
                    f"{image_shape}!")

        if image_mask is None:
            image_mask = np.zeros(image_shape, dtype=np.bool)

        self._image_mask = image_mask

    def _update_reference(self):
        try:
            updated, ref = self._ref_sub.update()
        except Exception as e:
            raise ImageProcessingError(str(e))

        if not updated:
            return

        self._reference = ref
        if ref is not None:
            if ref.dtype != _IMAGE_DTYPE:
                self._reference = ref.astype(_IMAGE_DTYPE)

            logger.info(f"[Image processor] Loaded reference image with "
                        f"shape = {ref.shape}")
        else:
            logger.info(f"[Image processor] Reference image removed")

    def _update_gain_offset(self, expected_shape):
        try:
            new_gain, gain, new_offset, offset = self._cal_sub.update(
                self._gain, self._offset)
        except Exception as e:
            raise ImageProcessingError(str(e))

        self._compute_gain_mean |= new_gain
        self._compute_offset_mean |= new_offset

        # Set new value before checking shape. It avoids loading data once
        # again due to the wrong slicer.
        self._gain = gain
        self._offset = offset

        sliced_gain = None
        if self._correct_gain:
            if gain is not None:
                sliced_gain = gain[self._gain_slicer]
                if self._compute_gain_mean:
                    # For visualization of the gain
                    self._gain_mean = nanmean_image_data(sliced_gain)
                    self._compute_gain_mean = False
            else:
                if self._compute_gain_mean:
                    self._gain_mean = None
                    self._compute_gain_mean = False

            if sliced_gain is not None and \
                    sliced_gain.shape != expected_shape:
                raise ImageProcessingError(
                    f"[Image processor] Shape of the gain constant "
                    f"{sliced_gain.shape} is different from the data "
                    f"{expected_shape}")

        sliced_offset = None
        if self._correct_offset:
            if self._dark_as_offset:
                sliced_offset = self._dark
                self._offset_mean = self._dark_mean
            elif offset is not None:
                sliced_offset = offset[self._offset_slicer]
                if self._compute_offset_mean:
                    # For visualization of the offset
                    self._offset_mean = nanmean_image_data(sliced_offset)
                    self._compute_offset_mean = False
            else:
                if self._compute_offset_mean:
                    self._offset_mean = None
                    self._compute_offset_mean = False

            if sliced_offset is not None and \
                    sliced_offset.shape != expected_shape:
                raise ImageProcessingError(
                    f"[Image processor] Shape of the offset constant "
                    f"{sliced_offset.shape} is different from the data "
                    f"{expected_shape}")

        return sliced_gain, sliced_offset

    def _update_pois(self, image_data, assembled):
        if assembled.ndim == 2 or image_data.poi_indices is None:
            return

        n_images = image_data.n_images
        out_of_bound_indices = []
        # only keep POI in 'images'
        for i in image_data.poi_indices:
            if i < n_images:
                image_data.images[i] = assembled[i].copy()
                mask_image_data(image_data.images[i],
                                image_mask=self._image_mask,
                                threshold_mask=self._threshold_mask,
                                keep_nan=True)
            else:
                out_of_bound_indices.append(i)

            if image_data.poi_indices[1] == image_data.poi_indices[0]:
                # skip the second one if two POIs have the same index
                break

        if out_of_bound_indices:
            # This is still ProcessingError since it is not fatal and should
            # not stop the pipeline.
            raise ProcessingError(
                f"[Image processor] POI indices {out_of_bound_indices[0]} "
                f"is out of bound (0 - {n_images-1}")
