import unittest
from threading import Thread

import zmq
import msgpack

from extra_foam.pipeline.f_zmq import BridgeProxy, KaraboBridgeServer

from karabo_bridge import Client


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
        import socket

        def _get_free_tcp_port():
            tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp.bind(('', 0))
            _, port = tcp.getsockname()
            tcp.close()
            return port

        endpoints = []

        proxy = BridgeProxy()
        ctx = zmq.Context()

        # start 3 'REP' server
        for src in ['A', 'B', 'C']:
            endpoint = f"tcp://127.0.0.1:{_get_free_tcp_port()}"
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

        ctx.destroy(linger=0)

    def testKaraboBridge(self):
        endpoint = "ipc://karabo-bridge"

        # Create server and client
        server = KaraboBridgeServer()
        server.bind(endpoint)
        client = Client(endpoint, timeout=5)

        # Generate fake data
        data = []
        for src in ["foo", "bar", "baz"]:
            meta, d = _simple_data_in_karabo(src)
            for key in d:
                d[key] = {
                    "value": d[key],
                    "metadata": meta
                }
            data.append(d)

        # Helper function to stream data in order
        def run_server():
            for d in data:
                server.send(d)
            server.stop()

        # Test server
        server_thread = Thread(target=run_server)
        server_thread.start()

        # Check we receive the right data
        for d in data:
            received, _ = client.next()
            self.assertEqual(received, d)

        server_thread.join(timeout=1)
