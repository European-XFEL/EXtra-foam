import unittest

from extra_foam.algorithms.data_structures import OrderedSet, Stack


class TestDataStructures(unittest.TestCase):
    def testStack(self):
        stack = Stack()

        self.assertTrue(stack.empty())
        stack.push(4)
        stack.push(6)
        stack.push(8)
        self.assertFalse(stack.empty())
        self.assertEqual(3, len(stack))
        self.assertEqual(8, stack.top())
        self.assertEqual(8, stack.pop())
        self.assertEqual(2, len(stack))
        self.assertEqual(6, stack.pop())
        stack.push(1)
        self.assertEqual(1, stack.pop())
        self.assertEqual(4, stack.top())
        self.assertEqual(4, stack.pop())
        self.assertTrue(stack.empty())

    def testOrderedSet(self):
        x = OrderedSet([1, 3, 0])
        self.assertEqual('OrderedSet([1, 3, 0])', repr(x))
        x.add(0)  # add an existing item
        x.add(100)
        x.add('A')

        self.assertIn(1, x)
        self.assertEqual(5, len(x))
        self.assertListEqual([1, 3, 0, 100, 'A'], list(x))

        # delete a non-existing item
        x.discard(4)
        self.assertListEqual([1, 3, 0, 100, 'A'], list(x))

        # delete an existing item
        x.discard(3)
        self.assertListEqual([1, 0, 100, 'A'], list(x))

        # test 'remove' mixin method
        with self.assertRaises(KeyError):
            x.remove('B')
        x.remove(100)

        self.assertEqual(1, x.pop())
        self.assertEqual(0, x.pop())
        self.assertListEqual(['A'], list(x))
        x.clear()
        self.assertEqual(0, len(x))

        # test comparison

        x.add(1)
        x.add(2)

        y = set()
        y.add(1)
        y.add(2)

        # compare with a normal Set
        self.assertEqual(y, x)

        # TODO: more
