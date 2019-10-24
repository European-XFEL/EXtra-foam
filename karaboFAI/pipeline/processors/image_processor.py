"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

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

from karaboFAI.cpp import nanmeanImageArray, nanmeanTwoImages


class ImageProcessor(_BaseProcessor):
    """ImageProcessor class.

    Attributes:
        _background (float): a uniform background value.
        _recording (bool): whether a dark run is being recorded.
        _process_dark (bool): whether process the dark run while recording.
            used only when _recording = True.
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
        self._process_dark = False
        self._dark_mean = None

        self._image_mask = None
        self._threshold_mask = None
        self._reference = None

        self._pulse_slicer = slice(None, None)
        self._poi_indices = [0, 0]

        self._ref_sub = ReferenceSub()
        self._mask_sub = ImageMaskSub()

    def update(self):
        # image
        cfg = self._meta.get_all(mt.IMAGE_PROC)

        self._dark_subtraction = cfg['dark_subtraction'] == 'True'
        self._background = float(cfg['background'])
        self._threshold_mask = self.str2tuple(cfg['threshold_mask'],
                                              handler=float)

        # global
        gp_cfg = self._meta.get_all(mt.GLOBAL_PROC)

        self._poi_indices = [int(gp_cfg['poi1_index']),
                             int(gp_cfg['poi2_index'])]

        if 'reset_dark' in gp_cfg:
            self._meta.delete(mt.GLOBAL_PROC, 'reset_dark')
            del self._dark_run
            self._dark_mean = None

        try:
            self._recording = gp_cfg['recording_dark'] == 'True'
        except KeyError:
            # TODO: we need a solution for widget that is not opened at
            #       start-up but is responsible for updating metadata.
            self._recording = False

        try:
            self._process_dark = gp_cfg['process_dark'] == 'True'
        except KeyError:
            self._process_dark = False

    @profiler("Image Processor (pulse)")
    def process(self, data):
        image_data = data['processed'].image
        assembled = data['detector']['assembled']
        pulse_slicer = data['detector']['pulse_slicer']
        n_total = assembled.shape[0] if assembled.ndim == 3 else 1

        data['detector']['assembled'] = assembled[pulse_slicer]
        sliced_indices = list(range(*(pulse_slicer.indices(n_total))))
        n_images = len(sliced_indices)

        if self._recording:
            if self._dark_run is None:
                # Dark_run should not share memory with
                # data['detector']['assembled'].
                # after pulse slicing
                self._dark_run = data['detector']['assembled'].copy()
            else:
                # moving average
                self._dark_run = data['detector']['assembled']

            # for visualizing the dark_mean
            # This is also a relatively expensive operation. But, in principle,
            # users should not trigger many other analysis when recording dark.
            if self._dark_run.ndim == 3:
                self._dark_mean = nanmeanImageArray(self._dark_run)
            else:
                self._dark_mean = self._dark_run.copy()

        assembled = data['detector']['assembled']

        if self._dark_subtraction:
            # subtract the dark_run from assembled if any
            if (not self._recording and self._dark_run is not None) \
                    or (self._recording and self._process_dark):
                dt_shape = assembled.shape
                dk_shape = self._dark_run.shape

                if dt_shape != dk_shape:
                    raise ImageProcessingError(
                        f"[Image processor] Shape of the dark train {dk_shape} "
                        f"is different from the data {dt_shape}")
                assembled -= self._dark_run

        image_shape = assembled.shape[-2:]
        self._update_image_mask(image_shape)
        self._update_reference(image_shape)

        # Avoid sending all images around
        # TODO: consider to use the 'virtual stack' in karabo_data, then
        #       for train-resolved data, set image_data.images == assembled
        #       https://github.com/European-XFEL/karabo_data/pull/196
        image_data.images = [None] * n_images
        image_data.poi_indices = self._poi_indices
        self._update_pois(image_data, assembled)
        image_data.background = self._background
        image_data.dark_mean = self._dark_mean
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
        if assembled.ndim == 2:
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
