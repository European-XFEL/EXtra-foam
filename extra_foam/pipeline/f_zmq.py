"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import deque
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

        self._context = zmq.Context()

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

        for end in endpoints:
            backend = self._context.socket(zmq.DEALER)
            backend.connect(end)
            self._backend[end] = backend

        frontendpoint = "inproc://frontend"
        self._frontend = self._context.socket(zmq.ROUTER)
        self._frontend.bind(frontendpoint)

        self._client = Client(frontendpoint,
                              context=self._context,
                              timeout=config['BRIDGE_TIMEOUT'])

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
            bk.close()
        self._backend.clear()
        self._backend_ready.clear()
        self._frontend.setsockopt(zmq.LINGER, 0)
        self._frontend.close()
        del self._client
        self._client = None

        self._stopped.set()

    def stop(self):
        """Stop the proxy running in a thread."""
        self._running = False
        if not self._stopped.is_set():
            self._stopped.wait()
