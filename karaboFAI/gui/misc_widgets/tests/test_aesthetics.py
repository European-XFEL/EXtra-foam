import unittest

from karaboFAI.gui.misc_widgets.aesthetics import SequentialColors


class TestSequentialColors(unittest.TestCase):
    def testGeneral(self):

        sc = SequentialColors()

        s = sc.s1(1)

        self.assertTupleEqual(sc.r[0], s[0])

        s = sc.s1(20)
        self.assertTupleEqual(sc.r[0], s[0])
        self.assertTupleEqual(sc.g[-1], s[-1])

        s = sc.s1(21)
        self.assertTupleEqual(sc.r[0], s[0])
        self.assertTupleEqual(sc.r[0], s[-1])
        self.assertTupleEqual(sc.g[-1], s[-2])

        s = sc.s1(128)
        self.assertTupleEqual(sc.b[-3], s[-1])
