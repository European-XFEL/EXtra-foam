import unittest
from unittest.mock import MagicMock
from threading import Thread
import time

from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.processors.base_processor import (
    LeafProcessor, CompositeProcessor, SharedProperty,
    StopCompositionProcessing
)


class _DummyLeafProcessor1(LeafProcessor):
    pass


class _DummyLeafProcessor2(LeafProcessor):
    pass


class _DummyLeafProcessor3(LeafProcessor):
    def process(self, processed, raw=None):
        raise StopCompositionProcessing


class _DummyLeafProcessor4(LeafProcessor):
    pass


class _DummyCompProcessor1(CompositeProcessor):
    param1 = SharedProperty()


class _DummyCompProcessor2(CompositeProcessor):
    param1 = SharedProperty()
    param2 = SharedProperty()

    def process(self, processed, raw=None):
        time.sleep(0.02)  # simulating data processing


class _DummyCompProcessor3(CompositeProcessor):
    def process(self, processed, raw=None):
        raise StopCompositionProcessing


class _DummyCompProcessor4(CompositeProcessor):
    pass


class TestBaseProcessor(unittest.TestCase):
    def setUp(self):
        self._leaf1 = _DummyLeafProcessor1()
        self._leaf2 = _DummyLeafProcessor2()
        self._leaf3 = _DummyLeafProcessor3()
        self._leaf4 = _DummyLeafProcessor4()

        self._comp1 = _DummyCompProcessor1()
        self._comp2 = _DummyCompProcessor2()
        self._comp3 = _DummyCompProcessor3()
        self._comp4 = _DummyCompProcessor4()

        # comp2 -> comp1 -> leaf1
        #       -> leaf2
        self._comp1.add(self._leaf1)
        self._comp2.add(self._comp1)
        self._comp2.add(self._leaf2)

        self._processed = ProcessedData(1)
        self._raw = {"timestamp.tid": 1000000}

    def testComposition(self):
        self.assertIs(self._leaf1._parent, self._comp1)
        self.assertIs(self._leaf2._parent, self._comp2)
        self.assertIs(self._comp1._parent, self._comp2)

        self.assertListEqual([self._leaf1], self._comp1._children)
        # sequence matters
        self.assertListEqual([self._comp1, self._leaf2], self._comp2._children)

        leaf1 = self._comp1.pop()
        self.assertListEqual([], self._comp1._children)
        self.assertIs(leaf1, self._leaf1)

        leaf2 = self._comp2.pop()
        self.assertListEqual([self._comp1], self._comp2._children)
        self.assertIs(leaf2, self._leaf2)

        self._comp2.remove(self._comp1)
        self.assertListEqual([], self._comp2._children)

    def testSharedPropertyInitialization(self):
        self.assertEqual(0, len(self._comp1._params))
        # if a key does not exist, it will be assigned to None when visited
        self.assertEqual(None, self._comp1.param1)
        self.assertEqual(None, self._comp1._params['param1'])
        self.assertEqual(1, len(self._comp1._params))

        self.assertEqual(0, len(self._comp2._params))
        self.assertEqual(None, self._comp2.param1)
        self.assertEqual(None, self._comp2._params['param1'])
        self.assertEqual(None, self._comp2.param2)
        self.assertEqual(None, self._comp2._params['param2'])
        self.assertEqual(2, len(self._comp2._params))

    def testProcessInterface(self):
        self._leaf1.process = MagicMock()
        self._leaf1.run_once(self._processed, self._raw)
        self._leaf1.process.assert_called_once_with(self._processed, self._raw)
        self._leaf1.process.reset_mock()

        self._leaf2.process = MagicMock()
        self._comp1.process = MagicMock()
        self._comp2.process = MagicMock()
        self._comp2.run_once(self._processed, self._raw)
        self._leaf1.process.assert_called_once_with(self._processed, self._raw)
        self._leaf2.process.assert_called_once_with(self._processed, self._raw)
        self._comp1.process.assert_called_once_with(self._processed, self._raw)
        self._comp2.process.assert_called_once_with(self._processed, self._raw)

    def testResetInterface(self):
        self._leaf1.reset = MagicMock()
        self._leaf1.reset_all()
        self._leaf1.reset.assert_called_once_with()
        self._leaf1.reset.reset_mock()

        self._leaf2.reset = MagicMock()
        self._comp1.reset = MagicMock()
        self._comp2.reset = MagicMock()
        self._comp2.reset_all()
        self._leaf1.reset.assert_called_once_with()
        self._leaf2.reset.assert_called_once_with()
        self._comp1.reset.assert_called_once_with()
        self._comp2.reset.assert_called_once_with()

    def testSharedPropertyPropagation(self):
        self._comp1.param1 = 1
        self._comp1.run_once(self._processed)
        self.assertEqual(1, self._leaf1.param1)

        self._comp2.param1 = 2
        self._comp2.param2 = 3
        self._comp2.run_once(self._processed)
        # overridden by the parent class
        self.assertEqual(2, self._leaf1.param1)
        self.assertEqual(3, self._leaf1.param2)
        self.assertEqual(2, self._comp1.param1)
        self.assertEqual(3, self._comp1.param2)
        self.assertEqual(2, self._leaf2.param1)
        self.assertEqual(3, self._leaf2.param2)

        def setter(obj, attr):
            for i in range(10):
                setattr(obj, attr, i)
                time.sleep(0.005)

        # if the process() method of a parent CompositionProcessor
        # takes some time, shared properties could be modified during
        # this period. After the processing is finished, the modified
        # shared properties will be passed to its children processors.
        # The following code tests this race condition!!!
        t = Thread(target=setter, args=(self._comp2, 'param2'))
        t.start()
        value = self._comp2.param2
        self._comp2.run_once(self._processed)
        self.assertEqual(value, self._comp1.param2)
        self.assertEqual(value, self._leaf2.param2)
        self.assertEqual(value, self._leaf1.param2)
        t.join()

    def testStopProcessing1(self):
        # comp4 -> leaf3
        #       -> leaf4
        #
        # a LeafProcessor stops the following LeafProcessors after
        # raising StopCompositionProcessing
        self._comp4.add(self._leaf3)
        self._comp4.add(self._leaf4)

        self._leaf4.process = MagicMock()
        self.assertFalse(self._leaf4.process.called)

    def testStopProcessing2(self):
        # comp2 -> comp1 -> leaf1
        #       -> comp4 -> comp3 -> leaf3
        #                -> leaf4
        #       -> leaf2
        self._comp2.remove(self._leaf2)
        self.assertListEqual([self._comp1], self._comp2._children)

        self._comp2.add(self._comp4)
        self._comp4.add(self._comp3)
        self._comp4.add(self._leaf4)
        self._comp3.add(self._leaf3)
        self._comp2.add(self._leaf2)
        self._leaf2.process = MagicMock()
        self._leaf4.process = MagicMock()
        # comp3 raises StopCompositionProcessing, Leaf4 will not be called,
        # but leaf2 will still be called
        self._comp2.run_once(self._processed, self._raw)
        self._leaf2.process.assert_called_once_with(self._processed, self._raw)
        self.assertFalse(self._leaf4.process.called)


class TestRedisParserMixin(unittest.TestCase):
    def testStr2Tuple(self):
        proc = CompositeProcessor()

        self.assertTupleEqual((1.0, 2.0), proc.str2tuple('(1, 2)'))

    def testStr2List(self):
        proc = CompositeProcessor()

        self.assertListEqual([1.0, 2.0], proc.str2list('[1, 2]'))
        self.assertListEqual([1], proc.str2list('[1]'))
        self.assertListEqual([], proc.str2list('[]'))
