"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from .base_processor import _BaseProcessor
from ..exceptions import (
    DropAllPulsesError, ProcessingError, PumpProbeIndexError
)
from ...config import AnalysisType, PumpProbeMode
from ...database import Metadata as mt
from ...utils import profiler

from extra_foam.algorithms import mask_image, nanmeanImageArray, nanmeanTwoImages


class PumpProbeProcessor(_BaseProcessor):
    """PumpProbeProcessor class.

    Attributes:
        analysis_type (AnalysisType): pump-probe analysis type.
        _mode (PumpProbeMode): pump-probe analysis mode.
        _indices_on (list): a list of laser-on pulse indices.
        _indices_off (list): a list of laser-off pulse indices.
        _prev_unmasked_on (numpy.ndarray): the most recent on-pulse image.
        _prev_xgm_on (double): the most recent xgm intensity.
        _abs_difference (bool): True for calculating absolute different
            between on/off pulses.
    """

    def __init__(self):
        super().__init__()

        self.analysis_type = AnalysisType.UNDEFINED

        self._mode = PumpProbeMode.UNDEFINED
        self._indices_on = []
        self._indices_off = []

        self._reset = False
        self._abs_difference = False

        self._prev_unmasked_on = None
        self._prev_xgm_on = None

    def update(self):
        """Override."""
        cfg = self._meta.hget_all(mt.PUMP_PROBE_PROC)

        if self._update_analysis(AnalysisType(int(cfg['analysis_type'])),
                                 register=False):
            self._reset = True

        mode = PumpProbeMode(int(cfg['mode']))
        if mode != self._mode:
            self._reset = True
            self._mode = mode

        abs_difference = cfg['abs_difference'] == 'True'
        if abs_difference != self._abs_difference:
            self._reset = True
            self._abs_difference = abs_difference

        if 'reset' in cfg:
            self._meta.hdel(mt.PUMP_PROBE_PROC, 'reset')
            # reset when commanded by the GUI
            self._reset = True

        self._indices_on = self.str2list(
            cfg['on_pulse_indices'], handler=int)
        self._indices_off = self.str2list(
            cfg['off_pulse_indices'], handler=int)

    @profiler("Pump-probe processor")
    def process(self, data):
        processed = data['processed']
        assembled = data['detector']['assembled']

        pp = processed.pp
        pp.reset = self._reset
        self._reset = False
        pp.mode = self._mode
        pp.indices_on = self._indices_on
        pp.indices_off = self._indices_off
        pp.analysis_type = self.analysis_type
        pp.abs_difference = self._abs_difference

        tid = processed.tid

        # parameters used for processing pump-probe images
        image_data = processed.image
        image_mask = image_data.image_mask
        threshold_mask = image_data.threshold_mask
        reference = image_data.reference
        n_images = image_data.n_images
        xgm_intensity = processed.pulse.xgm.intensity

        dropped_indices = processed.pidx.dropped_indices(n_images).tolist()

        # pump-probe means
        image_on, image_off, on_intensity, off_intensity, curr_indices, curr_means = \
            self._compute_on_off_data(tid,
                                      assembled,
                                      xgm_intensity,
                                      dropped_indices,
                                      reference=reference)

        # avoid calculating nanmean more than once
        if curr_indices == list(range(n_images)):
            if len(curr_means) == 1:
                images_mean = curr_means[0].copy()
            else:
                images_mean = nanmeanTwoImages(image_on, image_off)
        else:
            if assembled.ndim == 3:
                if dropped_indices:
                    indices = list(set(range(n_images)) - set(dropped_indices))
                    if not indices:
                        raise DropAllPulsesError(
                            f"{tid}: all pulses were dropped")
                    images_mean = nanmeanImageArray(assembled, indices)
                else:
                    # for performance
                    images_mean = nanmeanImageArray(assembled)
            else:
                # Note: _image is _mean for train-resolved detectors
                images_mean = assembled

        # apply mask to the averaged images of the train
        masked_mean = images_mean.copy()
        mask_image(masked_mean,
                   image_mask=image_mask,
                   threshold_mask=threshold_mask)

        processed.image.mean = images_mean
        processed.image.masked_mean = masked_mean

        # apply mask to the averaged on/off images
        # Note: due to the in-place masking, the pump-probe code the the
        #       rest code are interleaved.
        if image_on is not None:
            mask_image(image_on,
                       image_mask=image_mask,
                       threshold_mask=threshold_mask)

            mask_image(image_off,
                       image_mask=image_mask,
                       threshold_mask=threshold_mask)

            processed.pp.image_on = image_on
            processed.pp.image_off = image_off

            processed.xgm.on.intensity = on_intensity
            processed.xgm.off.intensity = off_intensity

    def _compute_on_off_data(self,
                             tid,
                             assembled,
                             xgm_intensity,
                             dropped_indices, *,
                             reference=None):
        curr_indices = []
        curr_means = []
        image_on = None
        image_off = None
        on_intensity = None
        off_intensity = None

        mode = self._mode
        if mode != PumpProbeMode.UNDEFINED:

            self._parse_on_off_indices(assembled.shape)

            if assembled.ndim == 3:
                self._validate_on_off_indices(assembled.shape[0])

            indices_on = list(set(self._indices_on) - set(dropped_indices))
            indices_off = list(set(self._indices_off) - set(dropped_indices))

            # on and off are not from different trains
            if mode in (PumpProbeMode.PRE_DEFINED_OFF,
                        PumpProbeMode.SAME_TRAIN):
                if assembled.ndim == 3:
                    # pulse resolved
                    if not indices_on:
                        raise DropAllPulsesError(
                            f"{tid}: all on pulses were dropped")
                    image_on = nanmeanImageArray(assembled, indices_on)

                    curr_indices.extend(indices_on)
                    curr_means.append(image_on)
                else:
                    image_on = assembled.copy()

                if mode == PumpProbeMode.PRE_DEFINED_OFF:
                    if reference is None:
                        image_off = np.zeros_like(image_on)
                    else:
                        # do not operate on the original reference image
                        image_off = reference.copy()
                else:
                    # train-resolved data does not have the mode 'SAME_TRAIN'
                    if not indices_off:
                        raise DropAllPulsesError(
                            f"{tid}: all off pulses were dropped")
                    image_off = nanmeanImageArray(assembled, indices_off)
                    curr_indices.extend(indices_off)
                    curr_means.append(image_off)

                if xgm_intensity is not None:
                    try:
                        on_intensity = np.mean(xgm_intensity[indices_on])
                        off_intensity = np.mean(xgm_intensity[indices_off])
                    except IndexError as e:
                        raise ProcessingError(f"XGM intensity: {repr(e)}")

            if mode in (PumpProbeMode.EVEN_TRAIN_ON,
                        PumpProbeMode.ODD_TRAIN_ON):
                # on and off are from different trains

                if mode == PumpProbeMode.EVEN_TRAIN_ON:
                    flag = 1
                else:  # mode == PumpProbeMode.ODD_TRAIN_ON:
                    flag = 0

                if tid % 2 == 1 ^ flag:
                    if assembled.ndim == 3:
                        if not indices_on:
                            raise DropAllPulsesError(
                                f"{tid}: all on pulses were dropped")
                        self._prev_unmasked_on = nanmeanImageArray(
                            assembled, indices_on)
                        curr_indices.extend(indices_on)
                        curr_means.append(self._prev_unmasked_on)
                    else:
                        self._prev_unmasked_on = assembled.copy()

                    if xgm_intensity is not None:
                        try:
                            self._prev_xgm_on = np.mean(xgm_intensity[indices_on])
                        except IndexError as e:
                            raise ProcessingError(f"XGM intensity: {repr(e)}")
                else:
                    if self._prev_unmasked_on is not None:
                        image_on = self._prev_unmasked_on
                        on_intensity = self._prev_xgm_on
                        self._prev_unmasked_on = None
                        # acknowledge off image only if on image has been received
                        if assembled.ndim == 3:
                            if not indices_off:
                                raise DropAllPulsesError(
                                    f"{tid}: all off pulses were dropped")
                            image_off = nanmeanImageArray(assembled, indices_off)
                            curr_indices.extend(indices_off)
                            curr_means.append(image_off)
                        else:
                            image_off = assembled.copy()

                        if xgm_intensity is not None:
                            try:
                                off_intensity = np.mean(xgm_intensity[indices_off])
                            except IndexError as e:
                                raise ProcessingError(f"XGM intensity: {repr(e)}")

        return (image_on, image_off, on_intensity, off_intensity,
                sorted(curr_indices), curr_means)

    def _parse_on_off_indices(self, shape):
        if len(shape) == 3:
            # pulse-resolved
            all_indices = list(range(shape[0]))
        else:
            # train-resolved (indeed not used)
            all_indices = [0]

        # convert [-1] to a list of indices
        if self._indices_on[0] == -1:
            self._indices_on = all_indices
        if self._indices_off[0] == -1:
            self._indices_off = all_indices

    def _validate_on_off_indices(self, n_pulses):
        """Check pulse index when on/off pulses in the same train.

        Note: We can not check it in the GUI side since we do not know
              how many pulses are there in the train.
        """
        # check index range
        if self._mode == PumpProbeMode.PRE_DEFINED_OFF:
            max_index = max(self._indices_on)
        else:
            max_index = max(max(self._indices_on), max(self._indices_off))

        if max_index >= n_pulses:
            raise PumpProbeIndexError(f"Index {max_index} is out of range for"
                                      f" a train with {n_pulses} pulses!")

        if self._mode == PumpProbeMode.SAME_TRAIN:
            # check pulse index overlap in on- and off- indices
            common = set(self._indices_on).intersection(self._indices_off)
            if common:
                raise PumpProbeIndexError(
                    "Pulse indices {} are found in both on- and off- pulses.".
                    format(','.join([str(v) for v in common])))
