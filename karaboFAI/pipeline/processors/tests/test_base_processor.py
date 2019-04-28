import unittest
from unittest.mock import MagicMock

from karaboFAI.pipeline.data_model import ProcessedData
from karaboFAI.pipeline.processors.base_processor import (
    LeafProcessor, CompositeProcessor, SharedProperty
)


class _DummyLeafProcessor1(LeafProcessor):
    pass


class _DummyLeafProcessor2(LeafProcessor):
    pass


class _DummyCompProcessor1(CompositeProcessor):
    param1 = SharedProperty()


class _DummyCompProcessor2(CompositeProcessor):
    param1 = SharedProperty()
    param2 = SharedProperty()


class TestBaseProcessor(unittest.TestCase):
    def setUp(self):
        self._leaf1 = _DummyLeafProcessor1()
        self._leaf2 = _DummyLeafProcessor2()

        self._comp1 = _DummyCompProcessor1()
        self._comp2 = _DummyCompProcessor2()

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

    def testProcess(self):
        self._leaf1.run = MagicMock()
        self._leaf1.process(self._processed, self._raw)
        self._leaf1.run.assert_called_once_with(self._processed, self._raw)
        self._leaf1.run.reset_mock()

        self._leaf2.run = MagicMock()
        self._comp2.process(self._processed, self._raw)
        self._leaf2.run.assert_called_once_with(self._processed, self._raw)
        self._leaf1.run.assert_called_once_with(self._processed, self._raw)

    def testSharedPropertyPropagation(self):
        self._comp1.param1 = 1
        self._comp1.process(self._processed)
        self.assertEqual(1, self._leaf1.param1)

        self._comp2.param1 = 2
        self._comp2.param2 = 3
        self._comp2.process(self._processed)
        # overridden by the parent class
        self.assertEqual(2, self._leaf1.param1)
        self.assertEqual(3, self._leaf1.param2)
        self.assertEqual(2, self._comp1.param1)
        self.assertEqual(3, self._comp1.param2)
        self.assertEqual(2, self._leaf2.param1)
        self.assertEqual(3, self._leaf2.param2)
