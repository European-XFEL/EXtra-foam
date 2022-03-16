import os
import re
import queue
import random
import string
import operator
import traceback
from enum import Enum
from itertools import chain
from threading import Thread, Event
from collections import defaultdict, namedtuple

from metropc.core import Path as MetroPath
from metropc.euxfel import KaraboPath
from metropc.client import decode_protocol, IndexViewEntry
from metropc.frontend import ThreadedFrontend, ProcessRunner, Context, ContextError

import zmq
import numpy as np
import xarray as xr
from PyQt5.QtCore import pyqtSignal

from . import logger
from .special_analysis_base import QThreadWorker, ClientType

def traverse_slots(current, prefix=""):
    """
    This generator yields the name and value of every slot in an object,
    recursively. The object is traversed depth-first.
    """
    # Get all slots declared in `current`'s class, and from any
    # inherited classes.
    slots = chain.from_iterable(getattr(cls, "__slots__", []) for cls in type(current).__mro__)

    # Iterate over all the slots. Some classes repeat their inherited
    # slots, so we remove duplicates with set().
    for slot in set(slots):
        # Ignore 'hidden' slots
        if not slot.startswith("_"):
            yield f"{prefix}{slot}", getattr(current, slot)
            yield from traverse_slots(getattr(current, slot), f"{prefix}{slot}.")


class FoamPath(MetroPath, alias="foam"):
    """
    A MetroPath subclass to add support for accessing data from extra-foam's
    ProcessedData objects.
    """
    foam_re = re.compile(r"^foam#(\w+\.?)*\w+$")

    def parse(self, path):
        if not self.foam_re.match(path):
            raise ValueError("Malformed extra-foam path")

        return super().parse(path)

    def extract(self, data):
        return operator.attrgetter(self.strip_type(self._full_path))(data)


class ViewEntry(IndexViewEntry):
    __slots__ = ["annotations"]

    def __init__(self, annotations, entry):
        super().__init__(entry.counts, entry.rate, entry.output, entry.stage)

        self.annotations = annotations


# This is a helper type to hold useful data about a path, to be displayed in a
# client.
PathData = namedtuple("PathData",
                      ["type", "shape", "dtype"],
                      defaults=[None, None])


# Helper enum to distinguish types of events that can be waited for
class MetroEvent(Enum):
    INDEX = 0
    DATA = 0


class MetroPipeline(ThreadedFrontend):
    """
    Simple frontend to interact with metropc.
    """
    def __init__(self, log):
        self._log = log
        self._ready_event = Event()

        # Create a random prefix for the abstract namespace, so it doesn't
        # conflict with any other instances of this suite.
        prefix = "".join(random.choices(string.ascii_lowercase, k=8))
        self.output_addr = f"ipc://@{prefix}-output"

        super().__init__(control_addr=f"ipc://@{prefix}-control",
                         reduce_addr=f"ipc://@{prefix}-reduce",
                         output_addr=self.output_addr,
                         queue_addr=f"inproc://{prefix}-ctrl-queue",
                         num_pool_stages=max(2, os.cpu_count() // 2),
                         sr_cls=ProcessRunner)

    def send_train(self, data: dict):
        """
        Send a train to metropc.

        :param dict data: Data for the train.
        """
        self.queue_to_any(b"event", data)

    def set_context(self, ctx: Context):
        """
        Update the metropc context.

        :param Context ctx: The new context.
        """
        self.queue_to_all(b"context", ctx.to_dict())

    def on_pipeline_ready(self):
        """Overide"""
        self._ready_event.set()
        self._log.info("Pipeline ready")

    def on_pipeline_error(self, msg):
        """Override"""
        self._log.error(msg)

    def on_pipeline_panic(self, msg):
        """Override"""
        self._log.error(msg)

    def on_pipeline_debug(self, msg):
        """Override"""
        # Todo: figure out a better way to handle this than sending it straight
        # to the debug logger (it's far too spammy).
        pass

    def wait_till_ready(self):
        """
        Block until the pipeline is ready.
        """
        self._ready_event.wait()

    def send_index(self):
        """
        Send the current index.
        """
        self.queue_to('reduce', b'index')


class CorrelatorProcessor(QThreadWorker):
    # Emitted when the views are changed
    updated_views_sgn = pyqtSignal(dict)

    # Emitted when incoming data is processed, only contains paths generated
    # from the incoming data (not including e.g. view paths).
    updated_data_paths_sgn = pyqtSignal(dict)

    # Emitted upon a ContextError
    context_error_sgn = pyqtSignal(object)

    # Emitted upon a successful reload
    reloaded_sgn = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ctx = None
        self.client_type = None

        self._index_event = Event()
        self._data_event = Event()
        self._pipeline = MetroPipeline(self.log)
        self._next_ctx_version = 0
        self._output_queue = queue.Queue(maxsize=100)

        # Set up the subscriber socket to the pipeline output
        self._subscriptions = { }
        self._subscriber = zmq.Context().socket(zmq.SUB)
        self._subscriber.connect(self._pipeline.output_addr)
        self._subscriber.subscribe(b"index")

        # Start the monitoring thread to handle output
        self._monitoring = True
        self._monitor_thread = Thread(target=self.monitor)
        self._monitor_thread.start()

    def monitor(self):
        """
        Monitors the pipelines output queue. Index and data events are handled.
        """
        while self._monitoring:
            readable, _, _ = zmq.select([self._subscriber], [], [], timeout=0.2)
            if not readable:
                continue

            path, data = decode_protocol(self._subscriber.recv_multipart())
            if path == b"index":
                self.handleIndex(data)
            else:
                self.handleData(path, data)

        self._subscriber.unsubscribe(b"index")
        for s in self._subscriptions:
            self._subscriber.unsubscribe(s.encode())
        self._subscriber.close()

    def handleData(self, path, data):
        """
        Push incoming view outputs onto the internal output queue.

        :param bytes path: The name of the view.
        :param dict data:  The output of the view.
        """
        path_str = path.decode()
        output = data["data"]

        try:
            self._output_queue.put_nowait((path_str, output))
        except queue.Full:
            self.log.warning("Metropc output queue is full, ignoring results from pipeline")
            return

        # Notify threads that a data event was processed
        self._data_event.set()

    def handleIndex(self, index):
        """
        Handle subscriptions to the pipeline.

        :param dict index: Dictionary of names to their IndexEntry.
        """
        views = set(p for p, v in index.items() if isinstance(v, IndexViewEntry))
        subscribed_views = set(self._subscriptions.keys())
        new_views = views - subscribed_views
        old_views = subscribed_views - views

        # Subscribe to new views and update existing ones
        for s in views:
            if s in new_views:
                # If it's a new view, subscribe to it
                self._subscriber.subscribe(s.encode())

            # Create a custom index entry that stores annotations
            view = self._ctx.views[s.split("#")[1]]
            view_entry = ViewEntry(getattr(view, "annotations", []), index[s])

            self._subscriptions[s] = view_entry
            self.log.debug(f"Subscribed to {s}")

        # Unsubscribe from old ones
        for s in old_views:
            self._subscriber.unsubscribe(s.encode())
            del self._subscriptions[s]
            self.log.debug(f"Unsubscribed to {s}")

        # We make a copy of this dict so that any users of it will know when it
        # will be changed. Otherwise when they receive this signal and compare
        # their dict to this one, they will be the same object and any possible
        # update that depends on them being different will not occur.
        self.updated_views_sgn.emit(self._subscriptions.copy())

        # Notify threads that an index event was processed
        self._index_event.set()

    def waitUntil(self, event_type: MetroEvent):
        """
        Helper function to wait until the next pipeline event is received.

        Note that index events are treated specially. Since it is assumed that
        the caller wants to wait as little as possible, this function gets the
        pipeline to send an index event immediately.
        """
        if event_type == MetroEvent.INDEX:
            event = self._index_event
        elif event_type == MetroEvent.DATA:
            event = self._data_event
        else:
            raise RuntimeError(f"Unsupported event type: {event_type}")

        event.clear()

        if event_type == MetroEvent.INDEX:
            self._pipeline.send_index()

        event.wait()

    def set_parameter(self, name, value):
        self._pipeline.queue_to_all(b"params", {name: value})

    # Helper function to inspect an object and create a PathData object
    # for it.
    def inspect_data(self, data):
        if isinstance(data, np.ndarray) or isinstance(data, xr.DataArray):
            return PathData(type(data).__name__, shape=data.shape, dtype=data.dtype)
        else:
            return PathData(type(data).__name__)

    def initProcessorData(self, tid):
        """
        Create the initial train_data and path_data objects when processing new
        data.

        Train data is sent to metropc, and path data is sent to the GUI to be
        used for display, autocompletion, etc.

        Returns a tuple of train_data and path_data. train_data is a dictionary
        of metropc paths to their data, and path_data is a dictionary of all
        possible paths to their PathData objects.
        """
        train_data = { "internal#train_id": tid,
                       "internal#event_id": tid,
                       "internal#sequence_id": -1,
                       "internal#ctx_version": self._ctx.version }
        path_data = { path: self.inspect_data(data) for path, data in train_data.items() }

        return train_data, path_data

    def extractFoamData(self, data):
        """
        Extract all data from an object sent by EXtra-foam.
        """
        raw = data["raw"]
        processed = data["processed"]

        # Set the internal paths
        tid = raw.pop("META timestamp.tid")
        train_data, path_data = self.initProcessorData(tid)

        # Add the raw data
        raw_train_data, raw_path_data = self.extractRawData(data)
        train_data.update(raw_train_data)
        path_data.update(raw_path_data)

        # And the processed data
        foam_paths = [p for p in self._paths if isinstance(p, FoamPath)]
        for path in foam_paths:
            path_str = str(path)
            train_data[path_str] = path.extract(processed)

        # Inspect the paths from extra-foam
        for slot, data in traverse_slots(processed):
            path_data[f"foam#{slot}"] = self.inspect_data(data)

        return train_data, path_data

    def extractRawData(self, data):
        """
        Extract all data from an object sent by a Karabo bridge.
        """
        raw = data["raw"]
        meta = data["meta"]

        # If there is no raw data, just return empty dicts
        if len(raw) == 0:
            return { }, { }

        tid = list(meta.values())[0]["train_id"]
        train_data, path_data = self.initProcessorData(tid)

        karabo_paths = [str(p) for p in self._paths if isinstance(p, KaraboPath)]
        for key, raw_data in raw.items():
            # If this data comes from extra-foam, then the detector data will
            # always be None.
            if raw_data is None:
                continue

            parts = key.split()
            if ":" in key:
                path = f"karabo#{parts[0]}[{parts[1]}]"
            else:
                path = f"karabo#{parts[0]}.{parts[1]}"

            if path in karabo_paths:
                train_data[path] = raw_data

            path_data[path] = self.inspect_data(raw_data)

        return train_data, path_data

    def process(self, data):
        """
        Process each new datum from the incoming stream (may be either
        EXtra-foam or a Karabo bridge). But because it needs to return something
        for the GUI, it also gets the latest output from the pipeline and
        returns that.
        """
        if self.client_type == ClientType.EXTRA_FOAM:
            train_data, path_data = self.extractFoamData(data)
        elif self.client_type == ClientType.KARABO_BRIDGE:
            train_data, path_data = self.extractRawData(data)
        else:
            raise RuntimeError(f"Unknown client type: {self.client_type}")

        # Send the train data to the pipeline
        if len(train_data) != 0:
            self._pipeline.send_train(train_data)

        # Emit the path data
        self.updated_data_paths_sgn.emit(path_data)

        tid = train_data["internal#train_id"]
        self.log.info(f"Received train {tid}")

        # Finally, we take any outputs from the pipeline and return them for display
        return self.getOutputs()

    def getOutputs(self):
        """
        Retrieve all outputs from metropc.

        Returns a dictionary of lists, where each view name maps to a list of
        outputs for that view. Note: multiple threads should not call this
        function concurrently.
        """
        outputs = defaultdict(list)
        while not self._output_queue.empty():
            path, data = self._output_queue.get_nowait()
            outputs[path].append(data)

        return outputs

    def setContext(self, source: str):
        """
        Update the pipeline context with the given source code.

        :param str source: The updated source code.
        """
        try:
            self._ctx = Context(source, version=self._next_ctx_version,
                                features=["karabo"], event_alias="train")
        except ContextError as e:
            # ContextError's have their own functions for pretty printing
            logger.error(e.format_for_context(source))
            self.context_error_sgn.emit(e)
            return
        except Exception as e:
            # For all other exceptions we log the traceback and error
            logger.error("".join([*traceback.format_tb(e.__traceback__), repr(e)]))
            self.context_error_sgn.emit(e)
            return

        self._next_ctx_version += 1
        self._paths = self._ctx.get_paths()
        self._pipeline.set_context(self._ctx)

        self.reloaded_sgn.emit()
        self.log.info("Reloaded")

    def close(self, timeout=1):
        """
        Shutdown the processor and metropc pipeline.
        """
        self._monitoring = False
        self._monitor_thread.join()
        self._pipeline.close(timeout=timeout)
