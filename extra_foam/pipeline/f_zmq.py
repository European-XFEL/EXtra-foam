"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque
import pickle
from threading import Event

import zmq

from karabo_bridge import Client

from ..config import config
from ..utils import run_in_thread


class BridgeProxy:
    """A proxy bridge which can connect to more than one server.

    The current implementation has the following limits:
    1. All the connections must be alive;
    2. It is blocked, which means if one connection is fast and the other
       is slow, the overall performance is limited by the slow one.
    """

    POLL_TIMEOUT = 100  # timeout of the poller in milliseconds

    def __init__(self):

        self._context = None

        self._client = None
        self._frontend = None
        self._backend = dict()
        self._backend_ready = deque()

        self._running = False
        self._stopped = Event()
        self._stopped.set()

    @property
    def client(self):
        return self._client

    def connect(self, endpoints):
        """Connect the backend to one or more endpoints.

        :param str/list/tuple endpoints: addresses of endpoints.
        """
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        elif not isinstance(endpoints, (tuple, list)):
            raise ValueError("Endpoints must be either a string or "
                             "a tuple/list of string!")

        context = zmq.Context()

        for end in endpoints:
            backend = context.socket(zmq.DEALER)
            backend.connect(end)
            self._backend[end] = backend

        frontendpoint = "inproc://frontend"
        self._frontend = context.socket(zmq.ROUTER)
        self._frontend.bind(frontendpoint)

        self._client = Client(frontendpoint,
                              context=context,
                              timeout=config['BRIDGE_TIMEOUT'])

        self._context = context

    @run_in_thread()
    def start(self):
        """Run the proxy in a thread."""
        if self._running:
            raise RuntimeError(f"{self.__class__} is already running!")

        frontend = self._frontend

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        for address, bk in self._backend.items():
            poller.register(bk, zmq.POLLIN)
            self._backend_ready.append(address)

        self._stopped.clear()
        self._running = True
        while self._running:
            socks = dict(poller.poll(timeout=self.POLL_TIMEOUT))

            if socks.get(frontend) == zmq.POLLIN:
                message = frontend.recv_multipart()
                if len(self._backend_ready) > 0:
                    address = self._backend_ready.popleft()
                    self._backend[address].send_multipart(message)

            for address, bk in self._backend.items():
                if socks.get(bk) == zmq.POLLIN:
                    message = bk.recv_multipart()
                    frontend.send_multipart(message)
                    self._backend_ready.append(address)

        # clean up and close all sockets to avoid problems with buffer

        poller.unregister(frontend)
        for bk in self._backend.values():
            poller.unregister(bk)

        for bk in self._backend.values():
            bk.setsockopt(zmq.LINGER, 0)
        self._backend.clear()
        self._backend_ready.clear()
        self._frontend.setsockopt(zmq.LINGER, 0)
        self._frontend = None
        self._client = None
        self._context.destroy(linger=0)
        self._context = None

        self._stopped.set()

    def stop(self):
        """Stop the proxy running in a thread."""
        self._running = False
        if not self._stopped.is_set():
            self._stopped.wait()


class FoamZmqServer:
    """Internal zmq server for EXtra-foam."""
    def __init__(self):

        self._ctx = None
        self._socket = None

    def bind(self, endpoint):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REP)
        self._socket.bind(endpoint)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)

    def stop(self):
        if self._socket is not None:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket = None

        if self._ctx is not None:
            self._ctx.destroy(linger=0)
            self._ctx = None

    def send(self, data):
        try:
            self._socket.recv()
        except zmq.error.Again:
            raise TimeoutError
        self._socket.send(pickle.dumps(data), copy=False)


class FoamZmqClient:
    """Internal zmq client for EXtra-foam.

    It uses pickle to serialize and deserialize data.

    It keeps the same interface as karabo_bridge.Client.
    """
    def __init__(self, endpoint, timeout=None):

        self._ctx = zmq.Context()

        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

        if timeout is not None:
            self._socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self._recv_ready = False

    def next(self):
        """Request next data container.

        This function call is blocking.

        :return: data, meta
        :rtype: dict, dict

        :raise TimeoutError: If timeout is reached before receiving data.
        """
        if not self._recv_ready:
            self._socket.send(b'')
            self._recv_ready = True

        try:
            msg = self._socket.recv(copy=False)
        except zmq.error.Again:
            raise TimeoutError(
                'No data received from {} in the last {} ms'.format(
                    self._socket.getsockopt_string(zmq.LAST_ENDPOINT),
                    self._socket.getsockopt(zmq.RCVTIMEO)))
        self._recv_ready = False
        return pickle.loads(msg)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._ctx.destroy(linger=0)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
