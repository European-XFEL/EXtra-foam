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

from ...metadata import MetaProxy
from ...config import AnalysisType


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

    Exception raise in the process() method which will be called by the
    parent processor to stop the rest child processors.
    """
    pass


class MetaProcessor(type):
    def __new__(mcs, name, bases, class_dict):
        for key, value in class_dict.items():
            if isinstance(value, SharedProperty):
                value.name = key

        cls = type.__new__(mcs, name, bases, class_dict)
        return cls


_analysis_types = {v: 0 for v in AnalysisType if v != AnalysisType.UNDEFINED}


class _BaseProcessor(_RedisParserMixin, metaclass=MetaProcessor):
    """Data processor interface."""

    def __init__(self):

        self._parent = None
        self._data = None

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

    @staticmethod
    def _register_analysis(analysis_type):
        if analysis_type in _analysis_types:
            _analysis_types[analysis_type] += 1

    @staticmethod
    def _unregister_analysis(analysis_type):
        if analysis_type in _analysis_types:
            _analysis_types[analysis_type] -= 1

    @staticmethod
    def _has_analysis(analysis_type):
        return _analysis_types[analysis_type] > 0

    def _update_analysis(self, analysis_type):
        if analysis_type != self.analysis_type:
            self._unregister_analysis(self.analysis_type)
            self.analysis_type = analysis_type
            self._register_analysis(analysis_type)

    @abstractmethod
    def run_once(self, processed, raw=None):
        """Composition interface.

        :param ProcessedData processed: processed data.
        :param dict raw: raw data.
        """
        pass

    def process(self, processed, raw):
        """Process data."""
        pass

    @abstractmethod
    def reset_all(self):
        pass

    def reset(self):
        pass


class LeafProcessor(_BaseProcessor):

    def run_once(self, processed, raw=None):
        # self._state = self._state.next()
        # self._state.update(self)
        self.process(processed, raw)
        # self._state = self._state.next()

    def reset_all(self):
        self.reset()


class CompositeProcessor(_BaseProcessor):
    def __init__(self):
        super().__init__()

        self._children = []

    def add(self, child):
        self._children.append(child)
        child._parent = self
        child._data = self._data

    def remove(self, child):
        self._children.remove(child)

    def pop(self):
        """Remove and return the last child."""
        return self._children.pop(-1)

    def run_once(self, processed, raw=None):
        self.update()

        params = copy.deepcopy(self._params)  # froze all the shared properties
        # StopCompositionProcessing it raises will be handled by its
        # parent processor
        self.process(processed, raw)

        for child in self._children:
            child._params.update(params)
            try:
                child.run_once(processed, raw)
            except StopCompositionProcessing:
                break

    def update(self):
        """Update metadata."""
        pass

    def reset_all(self):
        self.reset()
        for child in self._children:
            child.reset_all()
