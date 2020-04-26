"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import _BaseProcessor
from .image_assembler import ImageAssemblerFactory
from ..data_model import RawImageData
from ..exceptions import ImageProcessingError, ProcessingError
from ...database import Metadata as mt
from ...ipc import (
    CalConstantsSub, ImageMaskSub, ReferenceSub
)
from ...ipc import process_logger as logger
from ...utils import profiler
from ...config import config, _MAX_INT32

from extra_foam.algorithms import (
    correct_image_data, mask_image_data, nanmean_image_data
)


_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']


class ImageProcessor(_BaseProcessor):
    """ImageProcessor class.

    Attributes:
        _require_geom (bool): whether a Geometry is required to assemble
            the detector modules.
        _dark (RawImageData): store the moving average of dark
            images in a train. Shape = (indices, y, x) for pulse-resolved
            and shape = (y, x) for train-resolved
        _correct_gain (bool): whether to apply gain correction.
        _correct_offset (bool): whether to apply offset correction.
        _full_gain (numpy.ndarray): gain constants loaded from the
            file/database. Shape = (memory cell, y, x)
        _full_offset (numpy.ndarray): offset constants loaded from the
            file/database. Shape = (memory cell, y, x)
        _gain_cells (slice): a slice object used to slice the right memory
            cells from the gain constants.
        _offset_cells (slice): a slice object used to slice the right memory
            cells from the offset constants.
        _gain (numpy.ndarray): gain constants of the selected memory
            cells. Shape = (memory cell, y, x)
        _offset (numpy.ndarray): offset constants of the selected memory
            cells. Shape = (memory cell, y, x)
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

    _dark = RawImageData(_MAX_INT32)

    def __init__(self):
        super().__init__()

        self._assembler = ImageAssemblerFactory.create(config['DETECTOR'])
        self._require_geom = config['REQUIRE_GEOMETRY']

        self._correct_gain = True
        self._correct_offset = True
        self._full_gain = None
        self._full_offset = None
        self._gain_cells = None
        self._offset_cells = None
        self._gain_cells_updated = False
        self._offset_cells_updated = False
        self._gain = None
        self._offset = None
        self._gain_mean = None
        self._offset_mean = None

        self._dark_as_offset = True
        self._recording_dark = False
        del self._dark
        self._dark_mean = None

        self._image_mask = None
        self._image_mask_in_modules = None
        self._threshold_mask = None
        self._reference = None

        self._poi_indices = None

        self._ref_sub = ReferenceSub()
        self._mask_sub = ImageMaskSub()
        self._cal_sub = CalConstantsSub()

    def update(self):
        cfg = self._meta.hget_all(mt.IMAGE_PROC)
        geom_cfg = self._meta.hget_all(mt.GEOMETRY_PROC)
        global_cfg = self._meta.hget_all(mt.GLOBAL_PROC)

        self._assembler.update(
            geom_cfg,
            mask_tile=cfg["mask_tile"] == 'True',
            mask_asic=cfg["mask_asic"] == 'True',
        )

        self._correct_gain = cfg['correct_gain'] == 'True'
        self._correct_offset = cfg['correct_offset'] == 'True'

        gain_cells = self.str2slice(cfg['gain_cells'])
        if gain_cells != self._gain_cells:
            self._gain_cells_updated = True
            self._gain_cells = gain_cells
        else:
            self._gain_cells_updated = False

        offset_cells = self.str2slice(cfg['offset_cells'])
        if offset_cells != self._offset_cells:
            self._offset_cells_updated = True
            self._offset_cells = offset_cells
        else:
            self._offset_cells_updated = False

        dark_as_offset = cfg['dark_as_offset'] == 'True'
        if dark_as_offset != self._dark_as_offset:
            self._dark_as_offset = dark_as_offset

        self._recording_dark = cfg['recording_dark'] == 'True'
        if 'remove_dark' in cfg:
            self._meta.hdel(mt.IMAGE_PROC, 'remove_dark')
            del self._dark
            self._dark_mean = None

        self._threshold_mask = self.str2tuple(
            cfg['threshold_mask'], handler=float)

        self._poi_indices = [
            int(global_cfg['poi1_index']), int(global_cfg['poi2_index'])]

    @profiler("Image processor")
    def process(self, data):

        self._assembler.process(data)

        image_data = data['processed'].image
        assembled = data['assembled']['data']
        catalog = data['catalog']
        det = catalog.main_detector
        pulse_slicer = catalog.get_slicer(det)

        if assembled.ndim == 3:
            sliced_assembled = assembled[pulse_slicer]
            image_data.sliced_indices = list(range(
                *(pulse_slicer.indices(assembled.shape[0]))))
            n_sliced = len(image_data.sliced_indices)
        else:
            sliced_assembled = assembled
            image_data.sliced_indices = [0]
            n_sliced = 1

        if self._recording_dark:
            self._record_dark(assembled)
        if self._dark is not None:
            # default is 0
            image_data.n_dark_pulses = 1 if self._dark.ndim == 2 \
                else len(self._dark)
        image_data.dark_mean = self._dark_mean
        image_data.dark_count = self.__class__._dark.count

        self._update_gain_offset()
        image_data.gain_mean = self._gain_mean
        image_data.offset_mean = self._offset_mean
        self._correct_image_data(sliced_assembled, pulse_slicer)

        # Note: This will be needed by the pump_probe_processor to calculate
        #       the mean of assembled images. Also, the on/off indices are
        #       based on the sliced data.
        data['assembled']['sliced'] = sliced_assembled

        self._update_image_mask(sliced_assembled.shape[-2:])
        image_data.image_mask = self._image_mask
        image_data.image_mask_in_modules = self._image_mask_in_modules
        image_data.threshold_mask = self._threshold_mask

        self._update_reference()
        image_data.reference = self._reference

        # Avoid sending all images around
        image_data.images = [None] * n_sliced
        image_data.poi_indices = self._poi_indices
        self._update_pois(image_data, sliced_assembled)

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
        try:
            updated, image_mask = self._mask_sub.update(
                self._image_mask, image_shape)
        except Exception as e:
            raise ImageProcessingError(str(e))

        if updated and self._require_geom:
            # keep a mask in modules for assembling later
            geom = self._assembler.geometry

            if self._image_mask_in_modules is None:
                self._image_mask_in_modules = geom.output_array_for_dismantle_fast(
                    dtype=np.bool)

            geom.dismantle_all_modules(
                image_mask, out=self._image_mask_in_modules)

        if image_mask.shape != image_shape:
            if np.sum(image_mask) == 0:
                # reset the empty image mask automatically
                image_mask = np.zeros(image_shape, dtype=np.bool)
            else:
                if self._require_geom:
                    # reassemble a mask
                    geom = self._assembler.geometry
                    image_mask = geom.output_array_for_position_fast(dtype=np.bool)
                    geom.position_all_modules(
                        self._image_mask_in_modules, image_mask)
                else:
                    # This can happen if the image shapes in the ImageTool is
                    # different from the shape of in the pipeline, i.e. the
                    # shape of the image just changed.
                    raise ImageProcessingError(
                        f"[Image processor] The shape of the image mask "
                        f"{image_mask.shape} is different from the shape of "
                        f"the image {image_shape}!")

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

    def _update_gain_offset(self):
        try:
            gain_updated, gain, offset_updated, offset = self._cal_sub.update()
        except Exception as e:
            raise ImageProcessingError(str(e))

        if gain_updated:
            if gain is not None:
                if gain.dtype != _IMAGE_DTYPE:
                    gain = gain.astype(_IMAGE_DTYPE)
                self._full_gain = gain

                logger.info(f"[Image processor] Loaded gain constants with "
                            f"shape = {gain.shape}")

                if gain.ndim == 3:
                    self._gain = gain[self._gain_cells]
                    self._gain_mean = nanmean_image_data(self._gain)
                else:
                    # no need to copy
                    self._gain = gain
                    self._gain_mean = gain

            else:
                self._full_gain = None
                logger.info(f"[Image processor] Gain constants removed")
                self._gain = None
                self._gain_mean = None
        else:
            if self._gain_cells_updated and self._full_gain is not None:
                self._gain = self._full_gain[self._gain_cells]
                self._gain_mean = nanmean_image_data(self._gain)

        if offset_updated:
            if offset is not None:
                if offset.dtype != _IMAGE_DTYPE:
                    offset = offset.astype(_IMAGE_DTYPE)
                self._full_offset = offset

                logger.info(f"[Image processor] Loaded offset constants with "
                            f"shape = {offset.shape}")

                if offset.ndim == 3:
                    self._offset = offset[self._offset_cells]
                    self._offset_mean = nanmean_image_data(self._offset)
                else:
                    # no need to copy
                    self._offset = offset
                    self._offset_mean = offset

            else:
                self._full_offset = None
                logger.info(f"[Image processor] Offset constants removed")
                self._offset = None
                self._offset_mean = None

        else:
            if self._offset_cells_updated:
                if self._full_offset is not None:
                    self._offset = self._full_offset[self._offset_cells]
                    self._offset_mean = nanmean_image_data(self._offset)
                else:
                    self._offset = None
                    self._offset_mean = None

    def _correct_image_data(self, sliced_assembled, slicer):
        gain = self._gain if self._correct_gain else None

        if self._correct_offset:
            offset = self._dark if self._dark_as_offset else self._offset
        else:
            offset = None

        if sliced_assembled.ndim == 3:
            if gain is not None:
                gain = gain[slicer]
            if offset is not None:
                offset = offset[slicer]

        if gain is not None and sliced_assembled.shape != gain.shape:
            raise ImageProcessingError(
                f"Assembled shape {sliced_assembled.shape} and "
                f"gain shape {gain.shape} are different!")

        if offset is not None and sliced_assembled.shape != offset.shape:
            raise ImageProcessingError(
                f"Assembled shape {sliced_assembled.shape} and "
                f"offset shape {offset.shape} are different!")

        correct_image_data(sliced_assembled, gain=gain, offset=offset)

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
                                threshold_mask=self._threshold_mask)
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
