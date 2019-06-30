"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

PumpProbeProcessors.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import CompositeProcessor
from ..data_model import ProcessedData
from ..exceptions import ProcessingError
from ...algorithms import nanmean_image, mask_image
from ...metadata import Metadata as mt
from ...command import CommandProxy
from ...utils import profiler
from ...config import AnalysisType, PumpProbeMode


class _RawImageData:
    """Stores moving average of raw images."""
    def __init__(self, images=None):
        self._images = None  # moving average (original data)
        self._ma_window = 1
        self._ma_count = 0

        self._images = None
        if images is not None:
            self.images = images

    @property
    def n_images(self):
        if self._images is None:
            return 0

        if self._images.ndim == 3:
            return self._images.shape[0]
        return 1

    @property
    def pulse_resolved(self):
        return self._images.ndim == 3

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, data):
        """Set new image data."""
        if not isinstance(data, np.ndarray):
            raise TypeError(r"Image data must be numpy.ndarray!")

        if data.ndim <= 1 or data.ndim > 3:
            raise ValueError(
                f"The shape of images must be (y, x) or (n_pulses, y, x)!")

        if self._images is not None and self._ma_window > 1:
            if data.shape != self._images.shape:
                # Note: this can happen, for example, when the quadrant
                #       positions of the LPD detectors changes.
                self._images = data
                self._ma_count = 1
                return

            if self._ma_count < self._ma_window:
                self._ma_count += 1
                self._images += (data - self._images) / self._ma_count
            else:  # self._ma_count == self._ma_window
                # here is an approximation
                self._images += (data - self._images) / self._ma_window

        else:  # self._images is None or self._ma_window == 1
            self._images = data
            self._ma_count = 1

    @property
    def ma_window(self):
        return self._ma_window

    @ma_window.setter
    def ma_window(self, v):
        if not isinstance(v, int) or v <= 0:
            v = 1

        if v < self._ma_window:
            # if the new window size is smaller than the current one,
            # we reset the original image sum and count
            self._ma_window = v
            self._ma_count = 0
            self._images = None

        self._ma_window = v

    @property
    def ma_count(self):
        return self._ma_count

    def clear(self):
        self._images = None
        self._ma_window = 1
        self._ma_count = 0


class ImageProcessor(CompositeProcessor):
    """ImageProcessor class.

    A group of image processors. ProcessedData is constructed here.

    Attributes:
        _raw_data (_RawImageData): store the moving average of the
            raw images in a train.
        _ma_window (int): moving average window size.
        _background (float): a uniform background value.
        _threshold_mask (tuple): threshold mask.
        _image_mask
        _reference

        _pulse_index_filter (list): a list of pulse indices.
        _vip_pulse_indices (list): indices of VIP pulses.

        _pp_mode (PumpProbeMode): pump-probe analysis mode.
        _on_indices (list): a list of laser-on pulse indices.
        _off_indices (list): a list of laser-off pulse indices.
        _prev_unmasked_on:

    """

    def __init__(self):
        super().__init__()

        self._raw_data = _RawImageData()
        self._ma_window = 1
        self._background = 0.0
        self._threshold_mask = None
        self._reference = None
        self._image_mask = None

        self._cmd_proxy = CommandProxy()

        self._pulse_index_filter = None
        self._vip_pulse_indices = None

        self._pp_mode = None
        self._on_indices = None
        self._off_indices = None
        # the most recent on-pulse image
        self._prev_unmasked_on = None

    def update(self):
        """Override."""
        # image analysis
        cfg = self._meta.get_all(mt.IMAGE_PROC)

        self._ma_window = int(cfg['ma_window'])
        self._background = float(cfg['background'])
        self._threshold_mask = self.str2tuple(cfg['threshold_mask'],
                                              handler=float)

        # general analysis
        gp_cfg = self._meta.get_all(mt.GENERAL_PROC)

        self._pulse_index_filter = self.str2list(
            gp_cfg['selected_pulse_indices'], handler=int)

        self._vip_pulse_indices = [int(gp_cfg['vip_pulse1_index']),
                                   int(gp_cfg['vip_pulse2_index'])]

        # pump-probe
        pp_cfg = self._meta.get_all(mt.PUMP_PROBE_PROC)

        self._pp_mode = PumpProbeMode(int(pp_cfg['mode']))
        self._on_indices = self.str2list(
            pp_cfg['on_pulse_indices'], handler=int)
        self._off_indices = self.str2list(
            pp_cfg['off_pulse_indices'], handler=int)

    @profiler("Image Processor")
    def process(self, data):
        tid = data['tid']
        assembled = data['assembled']
        if assembled.ndim == 3 and self._pulse_index_filter[0] != -1:
            # apply the pulse index filter
            assembled = assembled[self._pulse_index_filter]

        self._raw_data.ma_window = self._ma_window
        self._raw_data.images = assembled
        # make it the moving average
        # be careful, data['assembled'] and self._raw_data share memory
        data['assembled'] = self._raw_data.images

        keep = None  # keep all
        # 'keep' is only required by pulsed-resolved data
        if assembled.ndim == 3 and \
                not self._has_analysis(AnalysisType.PULSE_AZIMUTHAL_INTEG):
            # keep only the VIPs
            keep = self._vip_pulse_indices

        # update the reference image
        ref = self._cmd_proxy.get_ref_image()
        if ref is not None:
            if isinstance(ref, np.ndarray):
                self._reference = ref
            else:
                self._reference = None

        image_shape = self._raw_data.images.shape[-2:]
        if self._image_mask is None or self._image_mask.shape != image_shape:
            # initialize or check the existing image mask
            self._image_mask = np.zeros(image_shape, dtype=np.bool)

        # update image mask
        self._image_mask = self._cmd_proxy.update_mask(self._image_mask)
        if self._image_mask is not None \
                and self._image_mask.shape != image_shape:
            # This could only happen when the mask is loaded from the files
            # and the image shapes in the ImageTool is different from the
            # shape of the live images.
            raise ProcessingError("The shape of the image mask {} is "
                                  "different from the shape of the image {}!")

        # pump-probe images
        on_image, off_image = self._compute_on_off_images(tid, assembled)

        # apply mask
        self._mask_on_off_images(on_image, off_image)

        processed = ProcessedData(
            data['tid'], self._raw_data.images,
            reference=self._reference,
            background=self._background,
            image_mask=self._image_mask,
            threshold_mask=self._threshold_mask,
            ma_window=self._raw_data.ma_window,
            ma_count=self._raw_data.ma_count,
            keep=keep
        )

        processed.pp.on_image_mean = on_image
        processed.pp.off_image_mean = off_image

        data['processed'] = processed
        del data['assembled']

    def _compute_on_off_images(self, tid, assembled):
        on_image = None
        off_image = None

        mode = self._pp_mode
        # on and off are not from different trains
        if mode in (PumpProbeMode.PRE_DEFINED_OFF, PumpProbeMode.SAME_TRAIN):
            if assembled.ndim == 3:
                # pulse resolved
                on_image = nanmean_image(assembled[self._on_indices])
            else:
                on_image = assembled.copy()

            if mode == PumpProbeMode.PRE_DEFINED_OFF:
                if self._reference is None:
                    off_image = np.zeros_like(on_image)
                else:
                    # do not operate on the original reference image
                    off_image = self._reference.copy()
            else:
                # train-resolved data does not have the mode 'SAME_TRAIN'
                off_image = nanmean_image(assembled[self._off_indices])

        if mode in (PumpProbeMode.EVEN_TRAIN_ON, PumpProbeMode.ODD_TRAIN_ON):
            # on and off are from different trains

            if mode == PumpProbeMode.EVEN_TRAIN_ON:
                flag = 1
            else:  # mode == PumpProbeMode.ODD_TRAIN_ON:
                flag = 0

            if tid % 2 == 1 ^ flag:
                if assembled.ndim == 3:
                    self._prev_unmasked_on = nanmean_image(
                        assembled[self._on_indices])
                else:
                    self._prev_unmasked_on = assembled.copy()

            else:
                if self._prev_unmasked_on is not None:
                    on_image = self._prev_unmasked_on
                    self._prev_unmasked_on = None
                    # acknowledge off image only if on image has been received
                    if assembled.ndim == 3:
                        off_image = nanmean_image(
                            assembled[self._off_indices])
                    else:
                        off_image = assembled.copy()

        return on_image, off_image

    def _mask_on_off_images(self, on_image, off_image):
        if on_image is not None:
            mask_image(on_image,
                       threshold_mask=self._threshold_mask,
                       image_mask=self._image_mask,
                       inplace=True)

        if off_image is not None:
            mask_image(off_image,
                       threshold_mask=self._threshold_mask,
                       image_mask=self._image_mask,
                       inplace=True)
