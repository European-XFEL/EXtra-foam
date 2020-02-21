import unittest
from threading import Thread

import zmq
import msgpack

from extra_foam.pipeline.f_zmq import BridgeProxy


def _simple_data_in_karabo(src):
    data = {'a': 1, 'b': 2}  # property: value
    meta = {'timestamp.tid': 1001, 'source': src, 'content': 'msgpack'}
    return meta, data


class TestZmq(unittest.TestCase):

    class Server(Thread):
        def __init__(self, ctx, endpoint, *, src='A'):
            super().__init__()
            self._socket = ctx.socket(zmq.REP)
            self._socket.bind(endpoint)
            self.dumps = msgpack.Packer(use_bin_type=True).pack

            self._src = src

        def run(self):
            for i in range(4):
                #  Wait for next request from client
                message = self._socket.recv()
                if message == b"next":
                    #  Send reply back to client
                    meta, data = _simple_data_in_karabo(self._src)
                    self._socket.send_multipart([self.dumps(meta), self.dumps(data)])

    def testMultiServerConnection(self):
        endpoints = []

        proxy = BridgeProxy()
        ctx = proxy._context

        # start 3 'REP' server
        for src in ['A', 'B', 'C']:
            endpoint = f"inproc://server{src}"
            server = self.Server(ctx, endpoint, src=src)
            server.start()
            endpoints.append(endpoint)

        for _ in range(2):
            proxy.connect(endpoints)
            proxy.start()  # run in thread

            data = []
            for i in range(6):
                data.append(proxy.client.next())

            # data in different servers will arrive in turn
            self.assertEqual(
                [({'A': {'a': 1, 'b': 2}}, {'A': {}}),
                 ({'B': {'a': 1, 'b': 2}}, {'B': {}}),
                 ({'C': {'a': 1, 'b': 2}}, {'C': {}}),
                 ({'A': {'a': 1, 'b': 2}}, {'A': {}}),
                 ({'B': {'a': 1, 'b': 2}}, {'B': {}}),
                 ({'C': {'a': 1, 'b': 2}}, {'C': {}})], data)

            # test stop and connect again
            proxy.stop()
            self.assertIsNone(proxy._client)
