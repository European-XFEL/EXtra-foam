"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeProcessors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import time

import numpy as np

from .base_processor import _BaseProcessor
from ..data_model import RawImageData
from ..exceptions import (
    ProcessingError, PumpProbeIndexError, DropAllPulsesError
)
from ...algorithms import mask_image
from ...metadata import Metadata as mt
from ...ipc import ImageMaskSub, ReferenceSub
from ...utils import profiler
from ...config import config, PumpProbeMode

from karaboFAI.cpp import (
    nanmeanImages, nanmeanTwoImages, xtMovingAverage
)


class ImageProcessorPulse(_BaseProcessor):
    """ImageProcessorPulse class.

    Attributes:
        _raw_data (RawImageData): store the moving average of the
            raw images in a train.
        _background (float): a uniform background value.
        _recording (bool): whether a dark run is being recorded.
        _process_dark (bool): whether process the dark run while recording.
            used only when _recording = True.
        _dark_run (numpy.ndarray): store the moving average of dark
            images in a train. Shape = (indices, y, x)
        _dark_mean (numpy.ndarray): average of all the dark images in
            the dark run. Shape = (y, x)
        _image_mask (numpy.ndarray): image mask array. Shape = (y, x),
            dtype=np.bool
        _threshold_mask (tuple): threshold mask.
        _reference (numpy.ndarray): reference image.
        _pulse_slicer (slice): a slice object which will be used to slice
            images for pulse-resolved analysis. The slicing is applied
            before applying any data reduction algorithm to select less
            pulses.
        _poi_indices (list): indices of POI pulses.
    """

    # TODO: in the future, the data should be store at a shared
    #       memory space.
    _raw_data = RawImageData()

    # give it a huge window for now since I don't want to touch the
    # implementation of the base class for now.
    _dark_run = RawImageData(config['MAX_DARK_TRAIN_COUNT'])

    def __init__(self):
        super().__init__()

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

        self.__class__._raw_data.window = int(cfg['ma_window'])
        self._background = float(cfg['background'])
        self._threshold_mask = self.str2tuple(cfg['threshold_mask'],
                                              handler=float)

        # global
        gp_cfg = self._meta.get_all(mt.GLOBAL_PROC)

        self._pulse_slicer = self.str2slice(gp_cfg['selected_pulse_indices'])

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
        assembled = data['assembled']
        if assembled.ndim == 3:
            n_total = assembled.shape[0]
        else:
            n_total = 1

        pulse_slicer = self._pulse_slicer
        data['assembled'] = assembled[pulse_slicer]
        sliced_indices = list(range(*(pulse_slicer.indices(n_total))))
        n_images = len(sliced_indices)

        if self._recording:
            # TODO: calculating the moving average for a train of images
            #       is very expensive!!!
            if self._dark_run is None:
                # dark_run should not share memory with data['assembled']
                self._dark_run = data['assembled'].copy()  # after pulse slicing
            else:
                self._dark_run = data['assembled']
            # for visualizing the dark_mean
            # This is also a relatively expensive operation. But, in principle,
            # users should not trigger many other analysis when recording dark.
            if self._dark_run.ndim == 3:
                self._dark_mean = nanmeanImages(self._dark_run)
            else:
                self._dark_mean = self._dark_run.copy()
        # Make it the moving average for train resolved detectors
        # It is worth noting that the moving average here does not use
        # nanmean!!!
        self._raw_data = data['assembled']
        # Be careful! data['assembled'] and self._raw_data share memory
        data['assembled'] = self._raw_data
        assembled = data['assembled']
        # subtract the dark_run from assembled if any
        if (not self._recording and self._dark_run is not None) \
                or (self._recording and self._process_dark):
            if assembled.ndim == 3:
                assembled -= self._dark_run
            else:
                # TODO: remove the moving average of images for train-resolved
                #       detectors.
                data['assembled'] = assembled - self._dark_run
                assembled = data['assembled']

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
        image_data.ma_count = self.__class__._raw_data.count
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
            raise ProcessingError(
                f"The shape of the image mask {image_mask.shape} is "
                f"different from the shape of the image {image_shape}!")

        self._image_mask = image_mask

    def _update_reference(self, image_shape):
        ref = self._ref_sub.update(self._reference)

        if ref is not None and ref.shape != image_shape:
            # The original reference remains the same. It ensures the error
            # message if the shape of the image changes (e.g. quadrant
            # positions change on the fly).
            raise ProcessingError(
                f"The shape of the reference {ref.shape} is different "
                f"from the shape of the image {image_shape}!")

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
            raise ProcessingError(
                f"[Image processor] POI indices {out_of_bound_poi_indices[0]} "
                f"is out of bound (0 - {n_images-1}")


class ImageProcessorTrain(_BaseProcessor):
    """ImageProcessorTrain class.

    Attributes:
        _pp_mode (PumpProbeMode): pump-probe analysis mode.
        _on_indices (list): a list of laser-on pulse indices.
        _off_indices (list): a list of laser-off pulse indices.
        _prev_unmasked_on (numpy.ndarray): the most recent on-pulse image.
    """

    def __init__(self):
        super().__init__()

        self._pp_mode = PumpProbeMode.UNDEFINED
        self._on_indices = []
        self._off_indices = []
        self._prev_unmasked_on = None

    def update(self):
        """Override."""
        # pump-probe
        pp_cfg = self._meta.get_all(mt.PUMP_PROBE_PROC)

        self._pp_mode = PumpProbeMode(int(pp_cfg['mode']))
        self._on_indices = self.str2list(
            pp_cfg['on_pulse_indices'], handler=int)
        self._off_indices = self.str2list(
            pp_cfg['off_pulse_indices'], handler=int)

    @profiler("Image Processor (train)")
    def process(self, data):
        processed = data['processed']
        assembled = data['assembled']

        tid = processed.tid
        image_data = processed.image
        image_mask = image_data.image_mask
        threshold_mask = image_data.threshold_mask
        reference = image_data.reference
        n_images = image_data.n_images
        dropped_indices = image_data.dropped_indices

        # pump-probe means
        on_image, off_image, curr_indices, curr_means = \
            self._compute_on_off_images(tid, assembled, dropped_indices,
                                        reference=reference)

        # avoid calculating nanmean more than once
        if curr_indices == list(range(n_images)):
            if len(curr_means) == 1:
                images_mean = curr_means[0].copy()
            else:
                images_mean = nanmeanTwoImages(on_image, off_image)
        else:
            if assembled.ndim == 3:
                if dropped_indices:
                    indices = list(set(range(n_images)) - set(dropped_indices))
                    if not indices:
                        raise DropAllPulsesError(
                            f"{tid}: all pulses were dropped")
                    images_mean = nanmeanImages(assembled, indices)
                else:
                    # for performance
                    images_mean = nanmeanImages(assembled)
            else:
                # Note: _image is _mean for train-resolved detectors
                images_mean = assembled

        # apply mask to the averaged images of the train
        masked_mean = mask_image(images_mean,
                                 threshold_mask=threshold_mask,
                                 image_mask=image_mask)

        processed.image.mean = images_mean
        processed.image.masked_mean = masked_mean

        # apply mask to the averaged on/off images
        # Note: due to the in-place masking, the pump-probe code the the
        #       rest code are interleaved.
        if on_image is not None:
            mask_image(on_image,
                       threshold_mask=threshold_mask,
                       image_mask=image_mask,
                       inplace=True)

            mask_image(off_image,
                       threshold_mask=threshold_mask,
                       image_mask=image_mask,
                       inplace=True)

            processed.pp.image_on = on_image
            processed.pp.image_off = off_image

    def _compute_on_off_images(self, tid, assembled, dropped_indices,
                               *, reference=None):
        curr_indices = []
        curr_means = []
        on_image = None
        off_image = None

        mode = self._pp_mode
        if mode != PumpProbeMode.UNDEFINED:

            self._parse_on_off_indices(assembled.shape)

            if assembled.ndim == 3:
                self._validate_on_off_indices(assembled.shape[0])

            on_indices = list(set(self._on_indices) - set(dropped_indices))
            off_indices = list(set(self._off_indices) - set(dropped_indices))

            # on and off are not from different trains
            if mode in (PumpProbeMode.PRE_DEFINED_OFF,
                        PumpProbeMode.SAME_TRAIN):
                if assembled.ndim == 3:
                    # pulse resolved
                    if not on_indices:
                        raise DropAllPulsesError(
                            f"{tid}: all on pulses were dropped")
                    on_image = nanmeanImages(assembled, on_indices)

                    curr_indices.extend(on_indices)
                    curr_means.append(on_image)
                else:
                    on_image = assembled.copy()

                if mode == PumpProbeMode.PRE_DEFINED_OFF:
                    if reference is None:
                        off_image = np.zeros_like(on_image)
                    else:
                        # do not operate on the original reference image
                        off_image = reference.copy()
                else:
                    # train-resolved data does not have the mode 'SAME_TRAIN'
                    if not off_indices:
                        raise DropAllPulsesError(
                            f"{tid}: all off pulses were dropped")
                    off_image = nanmeanImages(assembled, off_indices)
                    curr_indices.extend(off_indices)
                    curr_means.append(off_image)

            if mode in (PumpProbeMode.EVEN_TRAIN_ON,
                        PumpProbeMode.ODD_TRAIN_ON):
                # on and off are from different trains

                if mode == PumpProbeMode.EVEN_TRAIN_ON:
                    flag = 1
                else:  # mode == PumpProbeMode.ODD_TRAIN_ON:
                    flag = 0

                if tid % 2 == 1 ^ flag:
                    if assembled.ndim == 3:
                        if not on_indices:
                            raise DropAllPulsesError(
                                f"{tid}: all on pulses were dropped")
                        self._prev_unmasked_on = nanmeanImages(
                            assembled, on_indices)
                        curr_indices.extend(on_indices)
                        curr_means.append(self._prev_unmasked_on)
                    else:
                        self._prev_unmasked_on = assembled.copy()

                else:
                    if self._prev_unmasked_on is not None:
                        on_image = self._prev_unmasked_on
                        self._prev_unmasked_on = None
                        # acknowledge off image only if on image
                        # has been received
                        if assembled.ndim == 3:
                            if not off_indices:
                                raise DropAllPulsesError(
                                    f"{tid}: all off pulses were dropped")
                            off_image = nanmeanImages(assembled, off_indices)
                            curr_indices.extend(off_indices)
                            curr_means.append(off_image)
                        else:
                            off_image = assembled.copy()

        return on_image, off_image, sorted(curr_indices), curr_means

    def _parse_on_off_indices(self, shape):
        if len(shape) == 3:
            # pulse-resolved
            all_indices = list(range(shape[0]))
        else:
            # train-resolved (indeed not used)
            all_indices = [0]

        # convert [-1] to a list of indices
        if self._on_indices[0] == -1:
            self._on_indices = all_indices
        if self._off_indices[0] == -1:
            self._off_indices = all_indices

    def _validate_on_off_indices(self, n_pulses):
        """Check pulse index when on/off pulses in the same train.

        Note: We can not check it in the GUI side since we do not know
              how many pulses are there in the train.
        """
        mode = self._pp_mode

        # check index range
        if mode == PumpProbeMode.PRE_DEFINED_OFF:
            max_index = max(self._on_indices)
        else:
            max_index = max(max(self._on_indices), max(self._off_indices))

        if max_index >= n_pulses:
            raise PumpProbeIndexError(f"Index {max_index} is out of range for"
                                      f" a train with {n_pulses} pulses!")

        if mode == PumpProbeMode.SAME_TRAIN:
            # check pulse index overlap in on- and off- indices
            common = set(self._on_indices).intersection(self._off_indices)
            if common:
                raise PumpProbeIndexError(
                    "Pulse indices {} are found in both on- and off- pulses.".
                    format(','.join([str(v) for v in common])))
