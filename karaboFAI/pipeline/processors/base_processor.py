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
import copy

from ..exceptions import ProcessingError
from ...metadata import MetaProxy
from ...metadata import Metadata as mt
from ...algorithms import normalize_auc
from ...config import AnalysisType, VFomNormalizer


def _get_slow_data(tid, raw, device_id, ppt):
    """Get slow data.

    :param int tid: train ID.
    :param dict raw: raw data.
    :param str device_id: device ID.
    :param str ppt: property name.

    :returns (value, error str)
    """
    if not device_id or not ppt:
        # not activated is not an error
        return None, ""

    if device_id == "Any":
        return tid, ""
    else:
        try:
            device_data = raw[device_id]
        except KeyError:
            return None, f"Device '{device_id}' is not in the data!"

        try:
            if ppt not in device_data:
                # from file
                ppt += '.value'
            return device_data[ppt], ""

        except KeyError:
            return None, f"'{device_id}'' does not have property '{ppt}'"


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
        normalized = normalize_auc(y, x, *auc_range)
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
        normalized_on = normalize_auc(y_on, x, *auc_range)
        normalized_off = normalize_auc(y_off, x, *auc_range)
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


class MovingAverageData:
    """Moving average data descriptor."""
    def __init__(self, n=1):
        """Initialization.

        :param int n: number of moving average data.
        """
        self._data = [None] * n

        self._ma_window = 1
        self._ma_count = 0

    def __get__(self, instance, instance_type):
        return self._data

    def __set__(self, instance, data):
        if not isinstance(data, (tuple, list)):
            data = [data]

        if self._data[0] is not None \
                and hasattr(self._data[0], 'shape') \
                and self._data[0].shape != data[0].shape:
            # reset moving average if data shape changes
            self._ma_count = 0
            self._data = [None] * len(self._data)

        if self._ma_window > 1 and self._ma_count > 0:
            if self._ma_count < self._ma_window:
                self._ma_count += 1
                denominator = self._ma_count
            else:   # self._ma_count == self._ma_window
                # here is an approximation
                denominator = self._ma_window

            for i in range(len(self._data)):
                self._data[i] += (data[i] - self._data[i]) / denominator

        else:  # self._ma_window == 1
            for i in range(len(self._data)):
                self._data[i] = data[i]
            if self._ma_window > 1:
                self._ma_count = 1  # 0 -> 1

    @property
    def moving_average_window(self):
        return self._ma_window

    @moving_average_window.setter
    def moving_average_window(self, v):
        if not isinstance(v, int) or v <= 0:
            v = 1

        if v < self._ma_window:
            # if the new window size is smaller than the current one,
            # we reset everything
            self._ma_window = v
            self._ma_count = 0
            self._data = [None] * len(self._data)

        self._ma_window = v

    @property
    def moving_average_count(self):
        return self._ma_count

    def clear(self):
        self._ma_window = 1
        self._ma_count = 0
        self._data = [None] * len(self._data)


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


class StopCompositionProcessing(Exception):
    """StopCompositionProcessing

    Exception raised to stop the process train of a Composite processor.
    """
    pass


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

        self._params = dict()

        self.on_handler = None
        self.processing_handler = None

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            if item in self._params:
                return self._params[item]
            else:
                raise

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

    @abstractmethod
    def run_once(self, processed):
        """Composition interface.

        :param ProcessedData processed: processed data.
        """
        pass

    def process(self, processed):
        """Process data."""
        pass

    @abstractmethod
    def reset_all(self):
        pass

    def reset(self):
        pass


class LeafProcessor(_BaseProcessor):

    def run_once(self, processed):
        # self._state = self._state.next()
        # self._state.update(self)
        self.process(processed)
        # self._state = self._state.next()

    def reset_all(self):
        self.reset()


class CompositeProcessor(_BaseProcessor):
    def __init__(self):
        super().__init__()

        self._children = []

    def add(self, child):
        if not isinstance(child, LeafProcessor):
            raise TypeError("Child processors must be LeafProcessor!")

        self._children.append(child)
        child._parent = self

    def remove(self, child):
        self._children.remove(child)

    def pop(self):
        """Remove and return the last child."""
        return self._children.pop(-1)

    def update(self):
        """Update metadata."""
        pass

    def run_once(self, processed):
        try:
            self.update()

            # froze all the shared properties
            params = copy.deepcopy(self._params)

            self.process(processed)

            for child in self._children:
                child._params.update(params)
                child.run_once(processed)

        except StopCompositionProcessing:
            pass

    def reset_all(self):
        self.reset()
        for child in self._children:
            child.reset_all()
