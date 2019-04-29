import unittest

from karaboFAI.algorithms.data_structures import Stack


class TestDataStructures(unittest.TestCase):
    def testStack(self):
        stack = Stack()

        self.assertTrue(stack.empty())
        stack.push(4)
        stack.push(6)
        stack.push(8)
        self.assertFalse(stack.empty())
        self.assertEqual(3, len(stack))
        self.assertEqual(8, stack.pop())
        self.assertEqual(2, len(stack))
        self.assertEqual(6, stack.pop())
        stack.push(1)
        self.assertEqual(1, stack.pop())
        self.assertEqual(4, stack.pop())
        self.assertTrue(stack.empty())
