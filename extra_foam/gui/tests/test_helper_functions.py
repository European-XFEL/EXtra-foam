import unittest
import math

import numpy as np

from extra_foam.gui.gui_helpers import (
    parse_boundary, parse_id, parse_slice, parse_slice_inv
)


class TestGUI(unittest.TestCase):
    def test_parserboundary(self):
        invalid_inputs = ["1., 2., 3.0", "1, ", ", 2", "a a", ", 1,,2, ",
                          "2.0, 0.1", "1, 1"]
        for v in invalid_inputs:
            with self.assertRaises(ValueError):
                parse_boundary(v)

        self.assertEqual(parse_boundary("0.1, 2"), (0.1, 2))
        self.assertEqual(parse_boundary(" 0.1, 2.0 "), (0.1, 2))

        # test parse -Inf and Inf
        self.assertEqual(parse_boundary(" -inf, inf "), (-np.inf, np.inf))
        lb, ub = parse_boundary(" -Inf, 0 ")
        self.assertTrue(math.isinf(lb))
        self.assertTrue(math.isfinite(ub))
        lb, ub = parse_boundary(" -100, INF ")
        self.assertTrue(math.isfinite(lb))
        self.assertTrue(math.isinf(ub))

    def test_parseids(self):
        self.assertEqual([-1], parse_id(":"))
        self.assertEqual([-1], parse_id("  : "))
        self.assertEqual([], parse_id("  "))
        self.assertEqual([1, 2, 3], parse_id("1, 2, 3"))
        self.assertEqual([1, 2, 3], parse_id("1, 2, ,3"))
        self.assertEqual([1], parse_id("1, 1, ,1"))

        self.assertEqual([1, 2], parse_id("1:3"))
        self.assertEqual([1, 2, 5], parse_id("1:3, 5"))
        self.assertEqual([1, 2, 5], parse_id("1:3, , 5"))
        self.assertEqual([1, 2, 5], parse_id(" 1 : 3 , , 5"))
        self.assertEqual([1, 2, 5], parse_id(",, 1 : 3 , , 5,, "))
        self.assertEqual([1, 2, 5], parse_id("1:3, 5"))
        self.assertEqual([0, 1, 2, 5, 6, 7], parse_id("0:3, 5, 6:8"))
        self.assertEqual([0, 1, 2, 3, 4], parse_id("0:3, 1:5"))
        self.assertEqual([0, 1, 2, 3, 4], parse_id("0:3, 1:5"))
        self.assertEqual([], parse_id("4:4"))
        self.assertEqual([0, 2, 4, 6, 8], parse_id("0:10:2"))

        invalid_inputs = ["1, 2, ,a", "1:", ":1", "-1:3", "2:a", "a:b",
                          "1:2:3:4", "4:1:-1"]
        for v in invalid_inputs:
            with self.assertRaises(ValueError):
                parse_id(v)

    def test_parseslice(self):
        with self.assertRaises(ValueError):
            parse_slice("")

        with self.assertRaises(ValueError):
            parse_slice(":::")

        with self.assertRaises(ValueError):
            parse_slice("1:5:2:1")

        self.assertListEqual([None, 2], parse_slice('2'))
        self.assertListEqual([None, 2], parse_slice(':2'))
        self.assertListEqual([2, 3], parse_slice('2:3'))
        self.assertListEqual([-3, -1], parse_slice('-3:-1'))
        self.assertListEqual([None, None], parse_slice(":"))
        self.assertListEqual([2, None], parse_slice("2:"))
        self.assertListEqual([None, 3], parse_slice(':3'))
        self.assertListEqual([1, 4, 2], parse_slice('1:4:2'))
        # input with space in between
        self.assertListEqual([1, 4, 2], parse_slice(' 1 :  4 : 2   '))
        self.assertListEqual([1, -4, 2], parse_slice('1:-4:2'))
        self.assertListEqual([2, None, 4], parse_slice('2::4'))
        self.assertListEqual([1, 3, None], parse_slice('1:3:'))
        self.assertListEqual([None, None, 4], parse_slice('::4'))
        self.assertListEqual([2, None, None], parse_slice('2::'))
        self.assertListEqual([None, None, None], parse_slice('::'))

        with self.assertRaises(ValueError):
            parse_slice('2.0')

        with self.assertRaises(ValueError):
            parse_slice('2:3.0:2.0')

    def testParseSliceInv(self):
        with self.assertRaises(ValueError):
            parse_slice_inv("")

        with self.assertRaises(ValueError):
            parse_slice_inv(":::")

        with self.assertRaises(ValueError):
            parse_slice_inv("1:a")

        for s in [':2', '2:3', '-3:-1', ":", "2:", ':3', '1:4:2',
                  '1:-4:2', '2::4', '1:3:', '::4', '2::', '::']:
            self.assertEqual(s, parse_slice_inv(str(parse_slice(s))))
