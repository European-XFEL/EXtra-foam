import unittest

import karabo_bridge as kb

from extra_foam.pipeline.pipe import KaraboBridgePipeIn, MpQueuePipeIn, MpQueuePipeOut


class TestMpPipes(unittest.TestCase):
    def setUp(self):
        self._in = MpQueuePipeIn()
        self._out = MpQueuePipeOut()

    def testConnectInputToOutput(self):
        with self.assertRaises(TypeError):
            self._in.connect(kb.Client("tcp://localhost:12345"))

    def testPassingData(self):
        self._in.connect(self._out)

        data = [1, 2, 4]
        self._out._client.put(data)
        self.assertListEqual(data, self._in._client.get())


class TestKaraboBridgePipe(unittest.TestCase):
    def setUp(self):
        self._in = KaraboBridgePipeIn()

    def testConnectInputToOutput(self):
        with self.assertRaises(TypeError):
            self._in.connect(MpQueuePipeOut())
        self._in.connect(kb.Client("tcp://localhost:12345"))
