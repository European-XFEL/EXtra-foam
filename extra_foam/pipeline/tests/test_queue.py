import unittest
from queue import Empty, Full
from threading import Thread

from extra_foam.pipeline.f_queue import SimpleQueue


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
