import unittest

from PyQt5.QtCore import Qt

from extra_foam.gui.misc_widgets.aesthetics import FColor, SequentialColor


class TestColor(unittest.TestCase):
    def testGeneral(self):
        # test make color
        FColor.mkColor('r', alpha=0)

        # test make pen
        pen = FColor.mkPen(None)
        self.assertEqual(Qt.NoPen, pen.style())
        pen = FColor.mkPen('k')
        self.assertEqual(Qt.SolidLine, pen.style())
        self.assertEqual(1, pen.width())
        pen = FColor.mkPen('b', alpha=100, width=2, style=Qt.DashDotLine)
        self.assertEqual(Qt.DashDotLine, pen.style())
        self.assertEqual(2, pen.width())
        self.assertTrue(pen.isCosmetic())

        # test make brush
        brush = FColor.mkBrush(None)
        self.assertEqual(Qt.NoBrush, brush.style())
        brush = FColor.mkColor('y', alpha=20)


class TestSequentialColor(unittest.TestCase):
    def testGeneral(self):
        SC = SequentialColor

        with self.assertRaises(ValueError):
            for i in SC.mkColor(0):
                pass

        for p in SC.mkColor(3, alpha=20):
            pass
        self.assertEqual(SC.r[2][0], p.red())
        self.assertEqual(SC.r[2][1], p.green())
        self.assertEqual(SC.r[2][2], p.blue())

        for p in SC.mkPen(6, alpha=20, width=2, style=Qt.DashDotLine):
            pass
        self.assertEqual(Qt.DashDotLine, p.style())
        self.assertEqual(SC.b[0][0], p.color().red())
        self.assertEqual(SC.b[0][1], p.color().green())
        self.assertEqual(SC.b[0][2], p.color().blue())
        self.assertEqual(20, p.color().alpha())
        self.assertEqual(2, p.width())
        self.assertTrue(p.isCosmetic())

        for p in SC.mkBrush(42, alpha=30):
            pass
        self.assertEqual(SC.r[1][0], p.color().red())
        self.assertEqual(SC.r[1][1], p.color().green())
        self.assertEqual(SC.r[1][2], p.color().blue())
        self.assertEqual(30, p.color().alpha())
