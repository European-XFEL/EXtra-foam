import unittest
import math

import numpy as np

from karaboFAI.gui.gui_helpers import parse_boundary, parse_ids


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
        self.assertEqual([-1], parse_ids(":"))
        self.assertEqual([-1], parse_ids("  : "))
        self.assertEqual([], parse_ids("  "))
        self.assertEqual([1, 2, 3], parse_ids("1, 2, 3"))
        self.assertEqual([1, 2, 3], parse_ids("1, 2, ,3"))
        self.assertEqual([1], parse_ids("1, 1, ,1"))

        self.assertEqual([1, 2], parse_ids("1:3"))
        self.assertEqual([1, 2, 5], parse_ids("1:3, 5"))
        self.assertEqual([1, 2, 5], parse_ids("1:3, , 5"))
        self.assertEqual([1, 2, 5], parse_ids(" 1 : 3 , , 5"))
        self.assertEqual([1, 2, 5], parse_ids(",, 1 : 3 , , 5,, "))
        self.assertEqual([1, 2, 5], parse_ids("1:3, 5"))
        self.assertEqual([0, 1, 2, 5, 6, 7], parse_ids("0:3, 5, 6:8"))
        self.assertEqual([0, 1, 2, 3, 4], parse_ids("0:3, 1:5"))
        self.assertEqual([0, 1, 2, 3, 4], parse_ids("0:3, 1:5"))
        self.assertEqual([], parse_ids("4:4"))
        self.assertEqual([0, 2, 4, 6, 8], parse_ids("0:10:2"))

        invalid_inputs = ["1, 2, ,a", "1:", ":1", "-1:3", "2:a", "a:b",
                          "1:2:3:4", "4:1:-1"]
        for v in invalid_inputs:
            with self.assertRaises(ValueError):
                parse_ids(v)
