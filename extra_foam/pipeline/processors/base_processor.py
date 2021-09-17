"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod
import math

import numpy as np

from ..exceptions import (
    ProcessingError, SkipTrainError, UnknownParameterError
)
from ...database import MetaProxy
from ...algorithms import normalize_auc
from ...config import AnalysisType, config, Normalizer


class State(ABC):
    """Base class of processor state."""
    @abstractmethod
    def on_enter(self, proc):
        pass

    @abstractmethod
    def next(self):
        """Return the next state."""
        pass

    def update(self, proc):
        self.on_enter(proc)


class StateOn(State):
    """State on.

    The state when the processor is ready to start.
    """
    def on_enter(self, proc):
        handler = proc.on_handler
        if handler is not None:
            handler(proc)

    def next(self):
        return StateProcessing()


class StateProcessing(State):
    """State processing.

    The state when the processor is processing data.
    """
    def on_enter(self, proc):
        handler = proc.processing_handler
        if handler is not None:
            handler(proc)

    def next(self):
        return StateOn()


class SharedProperty:
    """Property shared among Processors.

    Define a property which is shared by the current processor and its
    children processors.
    """
    def __init__(self):
        self.name = None

    def __get__(self, instance, instance_type):
        if instance is None:
            return self

        if self.name not in instance._params:
            # initialization
            instance._params[self.name] = None

        return instance._params[self.name]

    def __set__(self, instance, value):
        instance._params[self.name] = value


class _RedisParserMixin:
    """_RedisParserMixin class.

    Due to the performance concern, methods in this class are not suppose
    to cover all the corner cases, passing an arbitrary input may result
    in undefined behavior.
    """
    @staticmethod
    def str2tuple(text, delimiter=",", handler=float):
        """Convert a string to a tuple.

        The string is expected to be the result of str(tp), where tp is a
        tuple.

        For example:
            str2tuple('(1, 2)') -> (1.0, 2.0)
        """
        splitted = text[1:-1].split(delimiter)
        return handler(splitted[0]), handler(splitted[1])

    @staticmethod
    def str2list(text, delimiter=",", handler=float):
        """Convert a string to a list.

        The string is expected to be the result of str(lt), where lt is a
        list.

        For example:
            str2list('[1, 2, 3]') -> [1.0, 2.0, 3.0]
        """
        if not text[1:-1]:
            return []
        return [handler(v) for v in text[1:-1].split(delimiter)]

    @staticmethod
    def str2slice(text):
        """Convert a string to a slice object.

        The string is expected to the result of str(lt), where lt can be
        converted to a slice object by slice(*lt).

        For example:
            str2slice('[None, 2]' -> slice(None, 2)
        """
        return slice(*[None if v.strip() == 'None' else int(v)
                       for v in text[1:-1].split(',')])


class MetaProcessor(type):
    def __new__(mcs, name, bases, class_dict):
        for key, value in class_dict.items():
            if isinstance(value, SharedProperty):
                value.name = key

        cls = type.__new__(mcs, name, bases, class_dict)
        return cls


class _BaseProcessorMixin:
    @staticmethod
    def _normalize_fom(processed, y, normalizer, *, x=None, auc_range=None):
        """Normalize FOM/VFOM.

        :param ProcessedData processed: processed data.
        :param numpy.ndarray y: y values.
        :param Normalizer normalizer: normalizer type.
        :param numpy.ndarray x: x values used with AUC normalizer..
        :param tuple auc_range: normalization range with AUC normalizer.
        """
        if normalizer == Normalizer.UNDEFINED:
            return y

        if normalizer == Normalizer.AUC:
            # normalized by area under curve (AUC)
            normalized = normalize_auc(y, x, auc_range)
        elif normalizer == Normalizer.XGM:
            # normalized by XGM
            intensity = processed.pulse.xgm.intensity
            if intensity is None:
                raise ProcessingError("XGM normalizer is not available!")
            denominator = np.mean(intensity)

            if denominator == 0:
                raise ProcessingError("XGM normalizer is zero!")

            normalized = y / denominator
        elif normalizer == Normalizer.DIGITIZER:
            # normalized by DIGITIZER
            channel = processed.pulse.digitizer.ch_normalizer
            pulse_integral = processed.pulse.digitizer[channel].pulse_integral
            if pulse_integral is None:
                raise ProcessingError("Digitizer normalizer is not available!")
            denominator = np.mean(pulse_integral)

            if denominator == 0:
                raise ProcessingError("Digitizer normalizer is zero!")

            normalized = y / denominator
        elif normalizer == Normalizer.ROI:
            # normalized by ROI
            denominator = processed.roi.norm

            if denominator is None:
                raise ProcessingError("ROI normalizer is not available!")

            if denominator == 0:
                raise ProcessingError("ROI normalizer is zero!")

            normalized = y / denominator

        else:
            raise UnknownParameterError(
                f"Unknown normalizer: {repr(normalizer)}")

        return normalized

    @staticmethod
    def _normalize_fom_pp(processed, y_on, y_off, normalizer, *,
                          x=None, auc_range=None):
        """Normalize pump-probe FOM/VFOM.

        :param ProcessedData processed: processed data.
        :param numpy.ndarray y_on: pump y values.
        :param numpy.ndarray y_off: probe y values.
        :param Normalizer normalizer: normalizer type.
        :param numpy.ndarray x: x values used with AUC normalizer..
        :param tuple auc_range: normalization range with AUC normalizer.
        """
        if normalizer == Normalizer.UNDEFINED:
            return y_on, y_off

        if normalizer == Normalizer.AUC:
            # normalized by AUC
            normalized_on = normalize_auc(y_on, x, auc_range)
            normalized_off = normalize_auc(y_off, x, auc_range)

        elif normalizer == Normalizer.XGM:
            # normalized by XGM
            denominator_on = processed.pp.on.xgm_intensity
            denominator_off = processed.pp.off.xgm_intensity

            if denominator_on is None or denominator_off is None:
                raise ProcessingError("XGM normalizer is not available!")

            if denominator_on == 0:
                raise ProcessingError("XGM normalizer (on) is zero!")

            if denominator_off == 0:
                raise ProcessingError("XGM normalizer (off) is zero!")

            normalized_on = y_on / denominator_on
            normalized_off = y_off / denominator_off

        elif normalizer == Normalizer.DIGITIZER:
            # normalized by Digitizer
            denominator_on = processed.pp.on.digitizer_pulse_integral
            denominator_off = processed.pp.off.digitizer_pulse_integral

            if denominator_on is None or denominator_off is None:
                raise ProcessingError("Digitizer normalizer is not available!")

            if denominator_on == 0:
                raise ProcessingError("Digitizer normalizer (on) is zero!")

            if denominator_off == 0:
                raise ProcessingError("Digitizer normalizer (off) is zero!")

            normalized_on = y_on / denominator_on
            normalized_off = y_off / denominator_off

        elif normalizer == Normalizer.ROI:
            # normalized by ROI
            denominator_on = processed.pp.on.roi_norm
            denominator_off = processed.pp.off.roi_norm

            if denominator_on is None:
                raise ProcessingError("ROI normalizer (on) is not available!")

            if denominator_off is None:
                raise ProcessingError("ROI normalizer (off) is not available!")

            if denominator_on == 0:
                raise ProcessingError("ROI normalizer (on) is zero!")

            if denominator_off == 0:
                raise ProcessingError("ROI normalizer (off) is zero!")

            normalized_on = y_on / denominator_on
            normalized_off = y_off / denominator_off

        else:
            raise UnknownParameterError(
                f"Unknown normalizer: {repr(normalizer)}")

        return normalized_on, normalized_off

    @staticmethod
    def _fetch_property_data(tid, raw, src):
        """Fetch property data from raw data.

        :param int tid: train ID.
        :param dict raw: raw data.
        :param str src: source.

        :returns (value, error str)
        """
        if not src:
            # not activated is not an error
            return None, ""

        try:
            return raw[src], ""
        except KeyError:
            return None, f"[{tid}] '{src}' not found!"

    @staticmethod
    def filter_train_by_vrange(v, vrange, src):
        """Filter a train by train-resolved value.

        :param float v: value of a control data.
        :param tuple vrange: value range.
        :param str src: data source.
        """
        if vrange is not None:
            lb, ub = vrange
            if v > ub or v < lb:
                raise SkipTrainError(f"<{src}> value {v:.4e} is "
                                     f"out of range [{lb}, {ub}]")

    @staticmethod
    def filter_pulse_by_vrange(arr, vrange, index_mask):
        """Filter pulses in a train by pulse-resolved value.

        :param numpy.array arr: pulse-resolved values of data
            in a train.
        :param tuple vrange: value range.
        :param PulseIndexMask index_mask: pulse index msk
        """
        if vrange is not None:
            lb, ub = vrange

            if not math.isinf(lb) and not math.isinf(ub):
                index_mask.mask_by_array((arr > ub) | (arr < lb))
            elif not math.isinf(lb):
                index_mask.mask_by_array(arr < lb)
            elif not math.isinf(ub):
                index_mask.mask_by_array(arr > ub)


class _BaseProcessor(_BaseProcessorMixin, _RedisParserMixin,
                     metaclass=MetaProcessor):
    """Data processor interface."""

    def __init__(self):
        self._pulse_resolved = config["PULSE_RESOLVED"]

        self._meta = MetaProxy()

    def _update_analysis(self, analysis_type, *, register=True):
        """Update analysis type.

        :param AnalysisType analysis_type: analysis type.
        :param bool register: True for (un)register the analysis type.

        :return: True if the analysis type has changed and False for not.
        """
        if not isinstance(analysis_type, AnalysisType):
            raise UnknownParameterError(
                f"Unknown analysis type: {str(analysis_type)}")

        if analysis_type != self.analysis_type:
            if register:
                # unregister the old
                if self.analysis_type is not None:
                    self._meta.unregister_analysis(self.analysis_type)

                # register the new one
                if analysis_type != AnalysisType.UNDEFINED:
                    self._meta.register_analysis(analysis_type)

            self.analysis_type = analysis_type
            return True

        return False

    def run_once(self, data):
        """Composition interface.

        :param dict data: data which contains raw and processed data, etc.
        """
        self.update()
        self.process(data)

    def update(self):
        """Update metadata."""
        raise NotImplementedError

    def process(self, data):
        """Process data.

        :param dict data: data which contains raw and processed data, etc.
        """
        raise NotImplementedError


class _FomProcessor(_BaseProcessor):
    """Wrapper over _BaseProcessor for processors that need to use FOMs"""

    _ai_analysis_types = [AnalysisType.AZIMUTHAL_INTEG,
                          AnalysisType.AZIMUTHAL_INTEG_PEAK,
                          AnalysisType.AZIMUTHAL_INTEG_PEAK_Q,
                          AnalysisType.AZIMUTHAL_INTEG_COM]

    def __init__(self, name):
        """Create a _FomProcessor

        :param str name: Name of the processor, to use in logs.
        """
        super().__init__()

        self._name = name
        # used to check whether pump-probe FOM is available
        self._pp_fail_flag = 0
        self._pp_analysis_type = AnalysisType.UNDEFINED

    def _extract_ai_fom(self, ai, analysis_type):
        """Extract a azimuthal integration FOM

        :param AzimuthalIntegrationData ai: The data object to get the FOM from.
        :param AnalysisType analysis_type: The type of analysis.

        :return: A numeric value (i.e. float)
        """
        if analysis_type == AnalysisType.AZIMUTHAL_INTEG:
            return ai.fom
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG_PEAK:
            return ai.max_peak
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG_PEAK_Q:
            return ai.max_peak_q
        elif analysis_type == AnalysisType.AZIMUTHAL_INTEG_COM:
            # For the center-of-mass, we use the q value as the FOM
            return ai.center_of_mass[0] if ai.center_of_mass is not None else None
        else:
            raise ProcessingError("Unrecognized azimuthal integration FOM")

    def _extract_fom(self, processed):
        """Extract a figure-of-merit from the current train

        :param ProcessedData processed: the processed data with an FOM.

        :return: A tuple of (ret, fom, fom_slave), where `ret` is a DataItem,
                 `fom` is its FOM (e.g. float), and fom_slave is a boolean.
        """
        analysis_type = self.analysis_type
        fom_slave = None
        ret = None

        if analysis_type == AnalysisType.PUMP_PROBE:
            ret = processed.pp
            fom = ret.fom
            if fom is None:
                self._pp_fail_flag += 1
                # if on/off pulses are in different trains, pump-probe FOM is
                # only calculated every other train.
                if self._pp_fail_flag == 2:
                    self._pp_fail_flag = 0
                    raise ProcessingError("Pump-probe FOM is not available")
                return None, None, None
            else:
                self._pp_fail_flag = 0
        elif analysis_type == AnalysisType.ROI_FOM:
            ret = processed.roi
            fom = ret.fom
            fom_slave = ret.fom_slave
            if fom is None:
                raise ProcessingError("ROI FOM is not available")
        elif analysis_type == AnalysisType.ROI_PROJ:
            ret = processed.roi.proj
            fom = ret.fom
            if fom is None:
                raise ProcessingError("ROI projection FOM is not available")
        elif analysis_type in self._ai_analysis_types:
            ret = processed.ai
            fom = self._extract_ai_fom(ret, analysis_type)

            if fom is None:
                raise ProcessingError(
                    "Azimuthal integration FOM is not available")
        else:
            raise UnknownParameterError(
                f"[{self._name}] Unknown analysis type: {self.analysis_type}")

        return ret, fom, fom_slave
