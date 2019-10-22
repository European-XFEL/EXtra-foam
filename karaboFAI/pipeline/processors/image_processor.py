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
from ..data_model import RawImageData
from ..exceptions import (
    DropAllPulsesError, ImageProcessingError, ProcessingError,
    PumpProbeIndexError,
)
from ...algorithms import mask_image
from ...database import Metadata as mt
from ...ipc import ImageMaskSub, ReferenceSub
from ...utils import profiler
from ...config import config, PumpProbeMode

from karaboFAI.cpp import nanmeanTrain, nanmeanTwo


class ImageProcessorPulse(_BaseProcessor):
    """ImageProcessorPulse class.

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
                # dark_run should not share memory with data['detector']['assembled']
                self._dark_run = data['detector']['assembled'].copy()  # after pulse slicing
            else:
                # moving average
                self._dark_run = data['detector']['assembled']

            # for visualizing the dark_mean
            # This is also a relatively expensive operation. But, in principle,
            # users should not trigger many other analysis when recording dark.
            if self._dark_run.ndim == 3:
                self._dark_mean = nanmeanTrain(self._dark_run)
            else:
                self._dark_mean = self._dark_run.copy()

        assembled = data['detector']['assembled']
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


class ImageProcessorTrain(_BaseProcessor):
    """ImageProcessorTrain class.

    Attributes:
        _pp_mode (PumpProbeMode): pump-probe analysis mode.
        _on_indices (list): a list of laser-on pulse indices.
        _off_indices (list): a list of laser-off pulse indices.
        _prev_unmasked_on (numpy.ndarray): the most recent on-pulse image.
        _prev_xgm_on (double): the most recent xgm intensity
    """

    def __init__(self):
        super().__init__()

        self._pp_mode = PumpProbeMode.UNDEFINED
        self._on_indices = []
        self._off_indices = []
        self._prev_unmasked_on = None
        self._prev_xgm_on = None

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
        assembled = data['detector']['assembled']

        tid = processed.tid
        image_data = processed.image
        image_mask = image_data.image_mask
        threshold_mask = image_data.threshold_mask
        reference = image_data.reference
        n_images = image_data.n_images
        dropped_indices = image_data.dropped_indices

        intensity = processed.pulse.xgm.intensity

        # pump-probe means
        on_image, off_image, on_intensity, off_intensity, curr_indices, curr_means = \
            self._compute_on_off_images(tid, assembled, intensity, dropped_indices,
                                        reference=reference)

        # avoid calculating nanmean more than once
        if curr_indices == list(range(n_images)):
            if len(curr_means) == 1:
                images_mean = curr_means[0].copy()
            else:
                images_mean = nanmeanTwo(on_image, off_image)
        else:
            if assembled.ndim == 3:
                if dropped_indices:
                    indices = list(set(range(n_images)) - set(dropped_indices))
                    if not indices:
                        raise DropAllPulsesError(
                            f"{tid}: all pulses were dropped")
                    images_mean = nanmeanTrain(assembled, indices)
                else:
                    # for performance
                    images_mean = nanmeanTrain(assembled)
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

            processed.xgm.on.intensity = on_intensity
            processed.xgm.off.intensity = off_intensity

    def _compute_on_off_images(self, tid, assembled, intensity, dropped_indices,
                               *, reference=None):
        curr_indices = []
        curr_means = []
        on_image = None
        off_image = None
        on_intensity = None
        off_intensity = None

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
                    on_image = nanmeanTrain(assembled, on_indices)

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
                    off_image = nanmeanTrain(assembled, off_indices)
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
                        self._prev_unmasked_on = nanmeanTrain(
                            assembled, on_indices)
                        curr_indices.extend(on_indices)
                        curr_means.append(self._prev_unmasked_on)
                    else:
                        self._prev_unmasked_on = assembled.copy()

                    if intensity is not None:
                        self._prev_xgm_on = np.mean(intensity[on_indices])
                else:
                    if self._prev_unmasked_on is not None:
                        on_image = self._prev_unmasked_on
                        on_intensity = self._prev_xgm_on
                        self._prev_unmasked_on = None
                        # acknowledge off image only if on image
                        # has been received
                        if assembled.ndim == 3:
                            if not off_indices:
                                raise DropAllPulsesError(
                                    f"{tid}: all off pulses were dropped")
                            off_image = nanmeanTrain(assembled, off_indices)
                            curr_indices.extend(off_indices)
                            curr_means.append(off_image)
                        else:
                            off_image = assembled.copy()

                        if intensity is not None:
                            off_intensity = np.mean(intensity[off_indices])

        return on_image, off_image, on_intensity, off_intensity, sorted(curr_indices), curr_means

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
