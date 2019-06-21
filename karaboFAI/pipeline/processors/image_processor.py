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

from .base_processor import LeafProcessor, CompositeProcessor, SharedProperty
from ..data_model import ProcessedData
from ..exceptions import ProcessingError
from ...algorithms import nanmean_axis0_para, mask_by_threshold
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
        ma_window (int): moving average window size.
        background (float): a uniform background value.
        threshold_mask (tuple): threshold mask.
        pulse_index_filter (list): a list of pulse indices.
        vip_pulse_indices (list): indices of VIP pulses.
        pp_mode (PumpProbeMode): pump-probe analysis mode.
        on_indices (list): a list of laser-on pulse indices.
        off_indices (list): a list of laser-off pulse indices.

        _pulse_indices (list): selected pulse indices.
        _raw_data (_RawImageData): store the moving average of the
            raw images in a train.
    """
    ma_window = SharedProperty()
    background = SharedProperty()
    threshold_mask = SharedProperty()

    pulse_index_filter = SharedProperty()
    vip_pulse_indices = SharedProperty()

    pp_mode = SharedProperty()
    on_indices = SharedProperty()
    off_indices = SharedProperty()

    def __init__(self):
        super().__init__()

        self.add(GeneralImageProcessor())
        self.add(PumpProbeImageProcessor())

    def update(self):
        """Override."""
        # image analysis
        cfg = self._meta.get_all(mt.IMAGE_PROC)

        self.ma_window = int(cfg['ma_window'])
        self.background = float(cfg['background'])
        self.threshold_mask = self.str2tuple(cfg['threshold_mask'],
                                             handler=float)

        # general analysis
        gp_cfg = self._meta.get_all(mt.GENERAL_PROC)

        self.pulse_index_filter = self.str2list(
            gp_cfg['selected_pulse_indices'], handler=int)

        self.vip_pulse_indices = [int(gp_cfg['vip_pulse1_index']),
                                  int(gp_cfg['vip_pulse2_index'])]

        # pump-probe
        pp_cfg = self._meta.get_all(mt.PUMP_PROBE_PROC)

        self.pp_mode = PumpProbeMode(int(pp_cfg['mode']))
        self.on_indices = self.str2list(pp_cfg['on_pulse_indices'],
                                        handler=int)
        self.off_indices = self.str2list(pp_cfg['off_pulse_indices'],
                                         handler=int)


class GeneralImageProcessor(LeafProcessor):
    """GeneralImageProcessor class.

    The GeneralImageProcessor manages the lifetime of raw image data.
    Construct the processed data with index filter applied.
    """

    def __init__(self):
        super().__init__()

        self._raw_data = _RawImageData()
        self._reference = None

        self._cmd_proxy = CommandProxy()

    @profiler("Image Processor")
    def process(self, data):
        assembled = data['assembled']
        if assembled.ndim == 3 and self.pulse_index_filter[0] != -1:
            # apply the pulse index filter
            assembled = assembled[self.pulse_index_filter]

        self._raw_data.ma_window = self.ma_window
        self._raw_data.images = assembled
        # make it the moving average
        # be careful, data['assembled'] and self._raw_data share memory
        data['assembled'] = self._raw_data.images

        # 'keep' is only required by pulsed-resolved data
        if assembled.ndim == 3:
            if self._has_analysis(AnalysisType.PULSE_AZIMUTHAL_INTEG):
                # keep all
                keep = list(range(len(assembled)))
            else:
                # keep only the VIPs
                keep = self.vip_pulse_indices
        else:
            keep = None

        # update the reference image
        ref = self._cmd_proxy.get_ref_image()
        if ref is not None:
            if isinstance(ref, np.ndarray):
                self._reference = ref
            else:
                self._reference = None

        data['processed'] = ProcessedData(
            data['tid'], self._raw_data.images,
            reference=self._reference,
            background=self.background,
            threshold_mask=self.threshold_mask,
            ma_window=self._raw_data.ma_window,
            ma_count=self._raw_data.ma_count,
            keep=keep
        )


class PumpProbeImageProcessor(LeafProcessor):
    """Retrieve the on/off images.

    It sets both on_image_mean and off_image_mean in PRE_DEFINED_OFF or
    SAME_TRAIN modes as well as when an on-pulse is followed by an off
    pulse in EVEN_TRAIN_ON or ODD_TRAIN_ON modes.

    Attributes:
        _prev_on (None/numpy.ndarray): the most recent on-pulse image.
    """

    def __init__(self):
        super().__init__()

        self._prev_on = None

    @profiler("Pump-probe Image Processor")
    def process(self, data):
        mode = self.pp_mode
        if mode == PumpProbeMode.UNDEFINED:
            return

        assembled = data['assembled']
        processed = data['processed']
        threshold_mask = self.threshold_mask

        # on and off are not from different trains
        if mode in (PumpProbeMode.PRE_DEFINED_OFF, PumpProbeMode.SAME_TRAIN):
            if assembled.ndim == 3:
                # pulse resolved
                on_image = mask_by_threshold(nanmean_axis0_para(
                    assembled[self.on_indices]), *threshold_mask)
            else:
                on_image = processed.image.masked_mean

            if mode == PumpProbeMode.PRE_DEFINED_OFF:
                off_image = processed.image.reference
                if off_image is None:
                    off_image = np.zeros_like(on_image)
            else:
                # train-resolved data does not have the mode 'SAME_TRAIN'
                off_image = mask_by_threshold(nanmean_axis0_para(
                    assembled[self.off_indices]), *threshold_mask)

            processed.pp.on_image_mean = on_image
            processed.pp.off_image_mean = off_image

            return

        # on and off are from different trains

        if mode == PumpProbeMode.EVEN_TRAIN_ON:
            flag = 1
        else:  # mode == PumpProbeMode.ODD_TRAIN_ON:
            flag = 0

        if processed.tid % 2 == 1 ^ flag:
            if processed.pulse_resolved:
                self._prev_on = mask_by_threshold(nanmean_axis0_para(
                    assembled[self.on_indices]), *threshold_mask)
            else:
                self._prev_on = processed.image.masked_mean
        else:
            if self._prev_on is not None:
                processed.pp.on_image_mean = self._prev_on
                self._prev_on = None

                # acknowledge off image only if on image has been received
                if processed.pulse_resolved:
                    processed.pp.off_image_mean = mask_by_threshold(
                        nanmean_axis0_para(
                            assembled[self.off_indices]), *threshold_mask)
                else:
                    processed.pp.off_image_mean = processed.image.masked_mean

        del data['assembled']
