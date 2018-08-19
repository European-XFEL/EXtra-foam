import unittest

from src.fxeAzimuthalIntegration.gui import MainGUI


class TestGUI(unittest.TestCase):
    def testParserboundary(self):
        invalid_inputs = ["1, 2, 3", "1, ", ", 2", "a a", ", 1,,2, "]
        for v in invalid_inputs:
            with self.assertRaises(ValueError):
                MainGUI._parse_boundary(v)

        self.assertEqual(MainGUI._parse_boundary("1, 2"), (1, 2))
        self.assertEqual(MainGUI._parse_boundary(" 1, 2 "), (1, 2))

    def testParseids(self):
        self.assertEqual(MainGUI._parse_ids("  "), [])
        self.assertEqual(MainGUI._parse_ids("1, 2, 3"), [1, 2, 3])
        self.assertEqual(MainGUI._parse_ids("1, 2, ,3"), [1, 2, 3])
        self.assertEqual(MainGUI._parse_ids("1, 1, ,1"), [1])

        self.assertEqual(MainGUI._parse_ids("1:3"), [1, 2])
        self.assertEqual(MainGUI._parse_ids("1:3, 5"), [1, 2, 5])
        self.assertEqual(MainGUI._parse_ids("1:3, , 5"), [1, 2, 5])
        self.assertEqual(MainGUI._parse_ids(" 1 : 3 , , 5"), [1, 2, 5])
        self.assertEqual(MainGUI._parse_ids(",, 1 : 3 , , 5,, "), [1, 2, 5])
        self.assertEqual(MainGUI._parse_ids("1:3, 5"), [1, 2, 5])
        self.assertEqual(MainGUI._parse_ids("0:3, 5, 6:8"), [0, 1, 2, 5, 6, 7])
        self.assertEqual(MainGUI._parse_ids("0:3, 1:5"), [0, 1, 2, 3, 4])
        self.assertEqual(MainGUI._parse_ids("0:3, 1:5"), [0, 1, 2, 3, 4])

        self.assertEqual(MainGUI._parse_ids("0:10:2"), [0, 2, 4, 6, 8])

        invalid_inputs = ["1, 2, ,a", "1:", ":1", "-1:3", "4:4", "4:3", "2:a",
                          "a:b", "1:2:3:4"]
        for v in invalid_inputs:
            with self.assertRaises(ValueError):
                MainGUI._parse_ids(v)
