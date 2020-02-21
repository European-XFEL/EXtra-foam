import unittest
from unittest.mock import patch, PropertyMock
from queue import Empty, Full
import random
from threading import Thread

from extra_foam.pipeline.f_queue import CorrelateQueue, SimpleQueue
from extra_foam.database import SourceCatalog, SourceItem
from extra_foam.config import config


class TestSimpleQueue(unittest.TestCase):
    def testGeneral(self):
        queue = SimpleQueue(maxsize=2)
        self.assertTrue(queue.empty())
        queue.put_nowait(1)
        queue.put(2)
        with self.assertRaises(Full):
            queue.put(3)
        self.assertTrue(queue.full())
        self.assertEqual(2, queue.qsize())
        self.assertEqual(1, queue.get())
        self.assertEqual(2, queue.get())
        with self.assertRaises(Empty):
            queue.get()
        self.assertTrue(queue.empty())

        # test no maxsize
        queue = SimpleQueue()
        for i in range(100):
            queue.put(i)
        self.assertEqual(100, queue.qsize())
        self.assertFalse(queue.full())
        queue.clear()
        self.assertTrue(queue.empty())

    def testMultiThreads(self):
        def worker1(queue):
            for i in range(100):
                queue.put(i)

        def worker2(queue):
            for _ in range(100):
                queue.get()

        queue = SimpleQueue()
        t1 = Thread(target=worker1, args=(queue,))
        t2 = Thread(target=worker2, args=(queue,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertTrue(queue.empty())


@patch.dict(config._data, {"DETECTOR": "ABC"})
class TestCorrelateQueue(unittest.TestCase):
    def _create_data(self, tid, mapping, extra=None):
        meta = dict()
        raw = dict()

        for ctg, src in mapping.items():
            # all sources have a property name "ppt" since it does not matter
            meta[src + " ppt"] = {'tid': tid}
            raw[src + " ppt"] = random.randint(0, 100)  # whatever data

        return {"meta": meta, "raw": raw}

    def _create_catalog(self, mapping):
        catalog = SourceCatalog()
        for ctg, src in mapping.items():
            catalog.add_item(SourceItem(ctg, src, [], "ppt", None, None))
        return catalog

    @patch('extra_foam.ipc.ProcessLogger.warning')
    def testCorrelation(self, warning):
        catalog = self._create_catalog({"ABC": "a"})

        queue = CorrelateQueue(catalog, maxsize=2)
        self.assertTrue(queue.empty())

        data = self._create_data(1001, {"ABC": "a"})
        queue.put(data)
        # correlated, single source
        self.assertEqual(1, queue.qsize())

        data = self._create_data(1002, {"Motor": "b"})
        queue.put(data)
        # not correlated
        self.assertEqual(1, queue.qsize())

        data = self._create_data(1002, {"ABC": "a"}, extra={"Motor": "b"})
        queue.put(data)
        # correlated
        self.assertEqual(2, queue.qsize())

        data = self._create_data(1003, {"ABC": "a", "Motor": "b"})
        with self.assertRaises(Full):
            # correlated, but queue is full
            queue.put_nowait(data)
        self.assertTrue(queue.full())

        # pop the first train
        out = queue.get()
        self.assertEqual(1001, out["processed"].tid)
        self.assertIn("a ppt", out["raw"])
        self.assertEqual(1, queue.qsize())

        # pop the second train
        out = queue.get_nowait()
        self.assertEqual(1002, out["processed"].tid)
        self.assertIn("a ppt", out["raw"])
        self.assertIn("b ppt", out["raw"])

        with self.assertRaises(Empty):
            # queue is empty
            queue.get_nowait()

    @patch('extra_foam.ipc.ProcessLogger.warning')
    @patch("extra_foam.pipeline.f_queue.CorrelateQueue._cache_size",
           new_callable=PropertyMock, return_value=3)
    def testMaximumCacheSize(self, cache_size, warning):
        # test when not all data are found
        catalog = self._create_catalog({"ABC": "a", "Motor": "b", "Motor": "c"})
        queue = CorrelateQueue(catalog, maxsize=2)
        for i in range(cache_size() + 1):
            data = self._create_data(1000 + i, {"ABC": "a", "Motor": "b"})
            queue.put(data)
        warning.assert_called_once()
        self.assertEqual(cache_size(), len(queue._cached))
