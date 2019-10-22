"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Data processor interface.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod

import numpy as np

from ..exceptions import ProcessingError
from ...database import MetaProxy
from ...database import Metadata as mt
from ...algorithms import normalize_auc
from ...config import AnalysisType, VFomNormalizer


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


class _BaseProcessor(_RedisParserMixin, metaclass=MetaProcessor):
    """Data processor interface."""

    def __init__(self):

        self._parent = None

        self._state = StateOn()

        self._meta = MetaProxy()

        self.on_handler = None
        self.processing_handler = None

    def _has_analysis(self, analysis_type):
        count = self._meta.get(mt.ANALYSIS_TYPE, analysis_type)
        return bool(count) and int(count) > 0

    def _has_any_analysis(self, analysis_type_list):
        if not isinstance(analysis_type_list, (tuple, list)):
            raise TypeError("Input must be a tuple or list!")

        for analysis_type in analysis_type_list:
            count = self._meta.get(mt.ANALYSIS_TYPE, analysis_type)
            if bool(count) and int(count) > 0:
                return True
        return False

    def _has_all_analysis(self, analysis_type_list):
        if not isinstance(analysis_type_list, (tuple, list)):
            raise TypeError("Input must be a tuple or list!")

        for analysis_type in analysis_type_list:
            count = self._meta.get(mt.ANALYSIS_TYPE, analysis_type)
            if not (bool(count) and int(count) > 0):
                return False
        return True

    def _update_analysis(self, analysis_type, *, register=True):
        """Update analysis type.

        :param AnalysisType analysis_type: analysis type.
        :param bool register: True for (un)register the analysis type.

        :return: True if the analysis type has changed and False for not.
        """
        if not isinstance(analysis_type, AnalysisType):
            raise ProcessingError(
                f"Unknown analysis type: {str(analysis_type)}")

        if analysis_type != self.analysis_type:
            if register:
                # unregister the old
                if self.analysis_type is not None:
                    self._meta.increase_by(
                        mt.ANALYSIS_TYPE, self.analysis_type, -1)

                # register the new one
                if analysis_type != AnalysisType.UNDEFINED:
                    if self._meta.get(mt.ANALYSIS_TYPE, analysis_type) is None:
                        # set analysis type if it does not exist
                        self._meta.set(mt.ANALYSIS_TYPE, analysis_type, 0)
                    self._meta.increase_by(mt.ANALYSIS_TYPE, analysis_type, 1)

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

    @staticmethod
    def _normalize_vfom(processed, y, normalizer, *, x=None, auc_range=None):
        """Normalize VFOM.

        :param ProcessedData processed: processed data.
        :param numpy.ndarray y: y values.
        :param VFomNormalizer normalizer: normalizer type.
        :param numpy.ndarray x: x values used with AUC normalizer..
        :param tuple auc_range: normalization range with AUC normalizer.
        """
        if normalizer == VFomNormalizer.AUC:
            # normalized by area under curve (AUC)
            normalized = normalize_auc(y, x, auc_range)
        elif normalizer == VFomNormalizer.XGM:
            # normalized by XGM
            intensity = processed.pulse.xgm.intensity
            if intensity is None:
                raise ProcessingError("XGM normalizer is not available!")
            denominator = np.mean(intensity)

            if denominator == 0:
                raise ProcessingError("XGM normalizer is zero!")

            normalized = y / denominator
        else:
            # normalized by ROI
            if normalizer == VFomNormalizer.ROI3:
                denominator = processed.roi.norm3
            elif normalizer == VFomNormalizer.ROI4:
                denominator = processed.roi.norm4
            elif normalizer == VFomNormalizer.ROI3_SUB_ROI4:
                denominator = processed.roi.norm3_sub_norm4
            elif normalizer == VFomNormalizer.ROI3_ADD_ROI4:
                denominator = processed.roi.norm3_add_norm4
            else:
                raise ProcessingError(f"Unknown normalizer: {repr(normalizer)}")

            if denominator is None:
                raise ProcessingError("ROI normalizer is not available!")

            if denominator == 0:
                raise ProcessingError("ROI normalizer is zero!")

            normalized = y / denominator

        return normalized

    @staticmethod
    def _normalize_vfom_pp(processed, y_on, y_off, normalizer, *,
                           x=None, auc_range=None):
        """Normalize the azimuthal integration result.

        :param ProcessedData processed: processed data.
        :param numpy.ndarray y_on: pump y values.
        :param numpy.ndarray y_off: probe y values.
        :param VFomNormalizer normalizer: normalizer type.
        :param numpy.ndarray x: x values used with AUC normalizer..
        :param tuple auc_range: normalization range with AUC normalizer.
        """
        if normalizer == VFomNormalizer.AUC:
            # normalized by area under curve (AUC)
            normalized_on = normalize_auc(y_on, x, auc_range)
            normalized_off = normalize_auc(y_off, x, auc_range)
        elif normalizer == VFomNormalizer.XGM:
            # normalized by XGM
            denominator_on = processed.xgm.on.intensity
            denominator_off = processed.xgm.off.intensity

            if denominator_on is None or denominator_off is None:
                raise ProcessingError("XGM normalizer is not available!")

            if denominator_on == 0:
                raise ProcessingError("XGM normalizer (on) is zero!")

            if denominator_off == 0:
                raise ProcessingError("XGM normalizer (off) is zero!")

            normalized_on = y_on / denominator_on
            normalized_off = y_off / denominator_off

        else:
            # normalized by ROI
            on = processed.roi.on
            off = processed.roi.off

            if normalizer == VFomNormalizer.ROI3:
                denominator_on = on.norm3
                denominator_off = off.norm3
            elif normalizer == VFomNormalizer.ROI4:
                denominator_on = on.norm4
                denominator_off = off.norm4
            elif normalizer == VFomNormalizer.ROI3_SUB_ROI4:
                denominator_on = on.norm3_sub_norm4
                denominator_off = off.norm3_sub_norm4
            elif normalizer == VFomNormalizer.ROI3_ADD_ROI4:
                denominator_on = on.norm3_add_norm4
                denominator_off = off.norm3_add_norm4
            else:
                raise ProcessingError(f"Unknown normalizer: {repr(normalizer)}")

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

        return normalized_on, normalized_off

    @staticmethod
    def _fetch_property_data(tid, raw, src_name, ppt):
        """Fetch property data from raw data.

        :param int tid: train ID.
        :param dict raw: raw data.
        :param str src_name: device ID.
        :param str ppt: property name.

        :returns (value, error str)
        """
        if not src_name or not ppt:
            # not activated is not an error
            return None, ""

        if src_name == "Any":
            return tid, ""
        else:
            try:
                device_data = raw[src_name]
            except KeyError:
                return None, f"[{tid}] source '{src_name}' is not in the data!"

            ppt_orig = ppt
            try:
                if ppt not in device_data:
                    # instrument data from file
                    ppt += '.value'
                return device_data[ppt], ""
            except KeyError:
                return None, f"[{tid}] '{src_name}' does not contain " \
                             f"property '{ppt_orig}'"
