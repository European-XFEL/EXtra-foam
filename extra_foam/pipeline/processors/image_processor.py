"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from .base_processor import _BaseProcessor
from ..data_model import RawImageData
from ..exceptions import ImageProcessingError, ProcessingError
from ...algorithms import mask_image
from ...database import Metadata as mt
from ...ipc import ImageMaskSub, ReferenceSub
from ...utils import profiler
from ...config import config

from extra_foam.cpp import nanmeanImageArray, nanmeanTwoImages


class ImageProcessor(_BaseProcessor):
    """ImageProcessor class.

    Attributes:
        _background (float): a uniform background value.
        _recording (bool): whether a dark run is being recorded.
        _dark_run (RawImageData): store the moving average of dark
            images in a train. Shape = (indices, y, x) for pulse-resolved
            and shape = (y, x) for train-resolved
        _dark_mean (numpy.ndarray): average of all the dark images in
            the dark run. Shape = (y, x)
        _image_mask (numpy.ndarray): image mask array. Shape = (y, x),
            dtype=np.bool
        _threshold_mask (tuple): threshold mask.
        _reference (numpy.ndarray): reference image.
        _pulse_slicer (slice): a slice object which will be used to slice
            images for pulse-resolved analysis. The slicing is applied
            before applying any pulse filters to select less pulses.
        _poi_indices (list): indices of POI pulses.
    """

    # give it a huge window for now since I don't want to touch the
    # implementation of the base class for now.
    _dark_run = RawImageData(config['MAX_DARK_TRAIN_COUNT'])

    def __init__(self):
        super().__init__()

        self._dark_subtraction = True
        self._background = 0.0

        self._recording = False
        self._dark_mean = None

        self._image_mask = None
        self._threshold_mask = None
        self._reference = None

        self._pulse_slicer = slice(None, None)
        self._poi_indices = None

        self._ref_sub = ReferenceSub()
        self._mask_sub = ImageMaskSub()

    def update(self):
        # image
        cfg = self._meta.hget_all(mt.IMAGE_PROC)

        self._dark_subtraction = cfg['dark_subtraction'] == 'True'
        self._background = float(cfg['background'])
        self._threshold_mask = self.str2tuple(cfg['threshold_mask'],
                                              handler=float)

        # global
        gp_cfg = self._meta.hget_all(mt.GLOBAL_PROC)

        try:
            self._poi_indices = [int(gp_cfg['poi1_index']),
                                 int(gp_cfg['poi2_index'])]
        except KeyError:
            # Train-resolved detector or poiWindow has not been opened yet.
            pass

        if 'remove_dark' in gp_cfg:
            self._meta.hdel(mt.GLOBAL_PROC, 'remove_dark')
            del self._dark_run
            self._dark_mean = None

        self._recording = gp_cfg['recording_dark'] == 'True'

    @profiler("Image Processor (pulse)")
    def process(self, data):
        image_data = data['processed'].image
        assembled = data['detector']['assembled']

        if self._recording:
            if self._dark_run is None:
                # _dark_run should not share the memory with
                # data['detector']['assembled'] since the latter will
                # be dark subtracted.
                self._dark_run = assembled.copy()
            else:
                # moving average (it reset the current moving average if the
                # new dark has a different shape)
                self._dark_run = assembled

            # For visualization of the dark_mean:
            #
            # This is also a relatively expensive operation. But, in principle,
            # users should not trigger many other analysis when recording dark.
            if self._dark_run.ndim == 3:
                self._dark_mean = nanmeanImageArray(self._dark_run)
            else:
                self._dark_mean = self._dark_run.copy()

        n_total = assembled.shape[0] if assembled.ndim == 3 else 1
        pulse_slicer = data['detector']['pulse_slicer']
        sliced_assembled = assembled[pulse_slicer]
        sliced_indices = list(range(*(pulse_slicer.indices(n_total))))
        n_sliced = len(sliced_indices)

        dark_run = self._dark_run
        if self._dark_subtraction and dark_run is not None:
            sliced_dark = dark_run[pulse_slicer]
            try:
                # subtract the dark_run from assembled if any
                sliced_assembled -= sliced_dark
            except ValueError:
                raise ImageProcessingError(
                    f"[Image processor] Shape of the dark train {sliced_dark.shape} "
                    f"is different from the data {sliced_assembled.shape}")

        # Note: This will be needed by the pump_probe_processor to calculate
        #       the mean of assembled images. Also, the on/off indices are
        #       based on the sliced data.
        data['detector']['assembled'] = sliced_assembled

        image_shape = sliced_assembled.shape[-2:]
        self._update_image_mask(image_shape)
        self._update_reference(image_shape)

        # Avoid sending all images around
        # TODO: consider to use the 'virtual stack' in karabo_data, then
        #       for train-resolved data, set image_data.images == assembled
        #       https://github.com/European-XFEL/karabo_data/pull/196
        image_data.images = [None] * n_sliced
        image_data.poi_indices = self._poi_indices
        self._update_pois(image_data, sliced_assembled)
        image_data.background = self._background
        image_data.dark_mean = self._dark_mean
        if dark_run is not None:
            # default is 0
            image_data.n_dark_pulses = 1 if dark_run.ndim == 2 \
                                         else len(dark_run)
        image_data.dark_count = self.__class__._dark_run.count
        image_data.image_mask = self._image_mask
        image_data.threshold_mask = self._threshold_mask
        image_data.reference = self._reference
        image_data.sliced_indices = sliced_indices

    def _update_image_mask(self, image_shape):
        image_mask = self._mask_sub.update(self._image_mask, image_shape)
        if image_mask is not None and image_mask.shape != image_shape:
            # This could only happen when the mask is loaded from the files
            # and the image shapes in the ImageTool is different from the
            # shape of the live images.
            # The original image mask remains the same.
            raise ImageProcessingError(
                f"[Image processor] The shape of the image mask "
                f"{image_mask.shape} is different from the shape of the image "
                f"{image_shape}!")

        self._image_mask = image_mask

    def _update_reference(self, image_shape):
        ref = self._ref_sub.update(self._reference)

        if ref is not None and ref.shape != image_shape:
            # The original reference remains the same. It ensures the error
            # message if the shape of the image changes (e.g. quadrant
            # positions change on the fly).
            raise ImageProcessingError(
                f"[Image processor] The shape of the reference {ref.shape} is "
                f"different from the shape of the image {image_shape}!")

        self._reference = ref

    def _update_pois(self, image_data, assembled):
        if assembled.ndim == 2 or image_data.poi_indices is None:
            return

        n_images = image_data.n_images
        out_of_bound_poi_indices = []
        # only keep POI in 'images'
        for i in image_data.poi_indices:
            if i < n_images:
                image_data.images[i] = mask_image(
                    assembled[i],
                    threshold_mask=self._threshold_mask,
                    image_mask=self._image_mask
                )
            else:
                out_of_bound_poi_indices.append(i)

        if out_of_bound_poi_indices:
            # This is still ProcessingError since it is not fatal and should
            # not stop the pipeline.
            raise ProcessingError(
                f"[Image processor] POI indices {out_of_bound_poi_indices[0]} "
                f"is out of bound (0 - {n_images-1}")
