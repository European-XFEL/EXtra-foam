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


class AbstractProcessor(ABC):
    """Data processor interface."""
    def __init__(self, scheduler):

        self._state = StateOn()

        self.on_hanlder = None
        self.processing_handler = None

        if scheduler is not None:
            scheduler.register_processor(self)

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


class LeafProcessor(AbstractProcessor):
    def __init__(self, scheduler=None):
        super().__init__(scheduler)

    def process(self, processed, raw=None):
        # self._state = self._state.next()
        # self._state.update(self)
        self.run(processed, raw)
        # self._state = self._state.next()


class CompositeProcessor(AbstractProcessor):
    def __init__(self, scheduler=None):
        super().__init__(scheduler)

        self._childrens = set()

    def process(self, processed, raw=None):
        self.run(processed, raw)
        for proc in self._childrens:
            proc.process(processed, raw)
