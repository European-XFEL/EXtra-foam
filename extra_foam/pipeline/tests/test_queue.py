import unittest
from unittest.mock import patch
from queue import Empty, Full

from extra_foam.pipeline.f_queue import CorrelateQueue


class TestCorrelateQueue(unittest.TestCase):
    def _create_data(self, tid):
        data = {"meta": dict()}
        data["meta"]["abc"] = {'tid': tid}
        return data

    @patch('extra_foam.ipc.ProcessLogger.warning')
    def testCorrelation(self, warning):
        queue = CorrelateQueue(maxsize=2)
        self.assertTrue(queue.empty())

        data = self._create_data(1001)
        queue.put(data)
        self.assertEqual(1, queue.qsize())

        data = self._create_data(1002)
        queue.put(data)
        self.assertEqual(2, queue.qsize())

        data = self._create_data(1002)
        queue.put(data)
        warning.assert_called_once()
        self.assertEqual(2, queue.qsize())

        data = self._create_data(1003)
        with self.assertRaises(Full):
            queue.put_nowait(data)
        self.assertTrue(queue.full())

        out = queue.get()
        self.assertEqual(1001, out["processed"].tid)
        self.assertEqual(1, queue.qsize())
        out = queue.get_nowait()
        self.assertEqual(1002, out["processed"].tid)
        with self.assertRaises(Empty):
            queue.get_nowait()
