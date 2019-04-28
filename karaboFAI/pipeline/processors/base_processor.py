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


class MetaProcessor(type):
    def __new__(mcs, name, bases, class_dict):
        for key, value in class_dict.items():
            if isinstance(value, SharedProperty):
                value.name = key

        cls = type.__new__(mcs, name, bases, class_dict)
        return cls


class _BaseProcessor(metaclass=MetaProcessor):
    """Data processor interface."""

    def __init__(self):

        self._parent = None

        self._state = StateOn()

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

    @abstractmethod
    def process(self, processed, raw=None):
        """Composition interface.

        :param ProcessedData processed: processed data.
        :param dict raw: raw data.
        """
        pass

    def run(self, processed, raw):
        """Process data."""
        pass

    def set_parent(self, parent):
        if parent is not None and not isinstance(parent, _BaseProcessor):
            raise ValueError(f"Invalid parent: {type(parent)}")
        self._parent = parent


class LeafProcessor(_BaseProcessor):
    def __init__(self):
        super().__init__()

    def process(self, processed, raw=None):
        # self._state = self._state.next()
        # self._state.update(self)
        self.run(processed, raw)
        # self._state = self._state.next()


class CompositeProcessor(_BaseProcessor):
    def __init__(self):
        super().__init__()

        self._children = []

    def add(self, child):
        self._children.append(child)
        child.set_parent(self)

    def remove(self, child):
        self._children.remove(child)

    def process(self, processed, raw=None):
        for child in self._children:
            child._params.update(self._params)
            child.process(processed, raw)
