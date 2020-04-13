import unittest

from extra_foam.gui.items import GeometryItem


class TestGeometryItem(unittest.TestCase):

    def testSingleton(self):
        g1 = GeometryItem()
        g2 = GeometryItem()
        self.assertEqual(g1, g2)
