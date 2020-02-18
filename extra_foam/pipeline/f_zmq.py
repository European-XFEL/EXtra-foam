"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from threading import Event

import zmq

from karabo_bridge import Client

from ..config import config
from ..utils import run_in_thread


class BridgeProxy:
    def __init__(self):

        self._context = zmq.Context()

        self._client = None
        self._frontend = None
        self._backend = None

        self._running = False
        self._stopped = Event()
        self._stopped.set()

    @property
    def client(self):
        return self._client

    def connect(self, endpoints):
        if isinstance(endpoints, str):
            endpoints = [endpoints]
        elif not isinstance(endpoints, (tuple, list)):
            raise ValueError("Endpoints must be either a string or "
                             "a tuple/list of string!")

        self._backend = self._context.socket(zmq.DEALER)
        self._backend.setsockopt(zmq.LINGER, 0)
        for end in endpoints:
            self._backend.connect(end)

        frontendpoint = "inproc://frontend"
        self._frontend = self._context.socket(zmq.ROUTER)
        self._frontend.setsockopt(zmq.LINGER, 0)
        self._frontend.bind(frontendpoint)

        self._client = Client(frontendpoint,
                              context=self._context,
                              timeout=config['BRIDGE_TIMEOUT'])

    @run_in_thread()
    def start(self):
        if self._running:
            raise RuntimeError("Proxy is already running!")

        frontend = self._frontend
        backend = self._backend

        # Initialize poll set
        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        self._stopped.clear()
        self._running = True
        while self._running:
            socks = dict(poller.poll(timeout=100))

            if socks.get(frontend) == zmq.POLLIN:
                message = frontend.recv_multipart()
                backend.send_multipart(message)

            if socks.get(backend) == zmq.POLLIN:
                message = backend.recv_multipart()
                frontend.send_multipart(message)

        poller.unregister(frontend)
        poller.unregister(backend)

        # close all sockets to avoid problems with buffer
        self._backend.close()
        self._frontend.close()
        del self._client

        self._stopped.set()

    def stop(self):
        self._running = False
        if not self._stopped.is_set():
            self._stopped.wait()
