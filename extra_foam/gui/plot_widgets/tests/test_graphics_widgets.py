import unittest
from unittest.mock import MagicMock, patch

from PyQt5.QtWidgets import QMenu

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.graphics_widgets import (
    HistogramLUTItem, PlotArea
)
from extra_foam.gui.plot_widgets.image_items import (
    ImageItem, MaskItem, RectROI
)
from extra_foam.gui.plot_widgets.plot_items import (
    CurvePlotItem, BarGraphItem, StatisticsBarItem
)
from extra_foam.gui import pyqtgraph as pg
from extra_foam.logger import logger


app = mkQApp()

logger.setLevel("CRITICAL")


class TestPlotArea(unittest.TestCase):
    def setUp(self):
        self._view = pg.GraphicsView()
        self._area = PlotArea()
        self._area.removeAllItems()  # axis items etc. should not be affected

        # FIXME: need the following line because of the implementation of ViewBox
        self._view.setCentralWidget(self._area)

    def testAxes(self):
        area = self._area

        self.assertEqual(4, len(area._axes))
        for name, pos in [('left', (2, 0)), ('bottom', (3, 1))]:
            left_axis = area._axes[name]
            self.assertIsInstance(left_axis['item'], pg.AxisItem)
            self.assertTrue(left_axis['item'].isVisible())
            self.assertTupleEqual(pos, left_axis['pos'])
            self.assertIs(area.getAxis(name), left_axis['item'])

            with patch.object(left_axis['item'], "show") as mocked:
                area.showAxis(name)
                mocked.assert_called_once()

            with patch.object(left_axis['item'], "hide") as mocked:
                area.showAxis(name, False)
                mocked.assert_called_once()

            with patch.object(left_axis['item'], "setLabel") as mocked:
                area.setLabel(name, "abc")
                mocked.assert_called_once_with(text="abc", units=None)

            with patch.object(left_axis['item'], "showLabel") as mocked:
                area.showLabel(name)
                mocked.assert_called_once()

        for name in ['top', 'right']:
            self.assertFalse(area.getAxis(name).isVisible())

    def testLegend(self):
        area = self._area
        self.assertIsNone(area._legend)

        legend = area.addLegend((-30, -30))
        self.assertIsInstance(area._legend, pg.LegendItem)
        self.assertIs(legend, area._legend)

        # test addLegend when legend already exists
        area.addLegend((-10, -10))
        self.assertIsInstance(area._legend, pg.LegendItem)
        self.assertIs(legend, area._legend)

    def testTitle(self):
        area = self._area
        self.assertIsInstance(area._title, pg.LabelItem)

        self.assertEqual(0, area._title.maximumHeight())
        self.assertFalse(area._title.isVisible())

        area.setTitle("abcdefg")
        self.assertGreater(area._title.maximumHeight(), 0)
        self.assertTrue(area._title.isVisible())

    def testForwardMethod(self):
        area = self._area

        for method in ["setAspectLocked", "invertY", "invertX"]:
            with patch.object(area._vb, method) as mocked:
                getattr(area, method)()
                mocked.assert_called_once()

    def testPlotItemManipulation(self):
        area = self._area
        area.addLegend()

        image_item = ImageItem()
        area.addItem(image_item)
        area.addItem(MaskItem(image_item))
        area.addItem(RectROI(0))
        bar_graph_item = BarGraphItem()
        area.addItem(bar_graph_item)
        area.addItem(StatisticsBarItem())
        area.addItem(CurvePlotItem())
        area.addItem(pg.PlotCurveItem())
        area.addItem(pg.ScatterPlotItem())

        self.assertEqual(8, len(area._items))
        self.assertEqual(8, len(area._vb.addedItems))
        self.assertEqual(5, len(area._legend.items))

        # remove an item which does not exist
        area.removeItem(BarGraphItem())
        self.assertEqual(8, len(area._items))
        self.assertEqual(8, len(area._vb.addedItems))
        self.assertEqual(5, len(area._legend.items))

        area.removeItem(bar_graph_item)
        self.assertEqual(7, len(area._items))
        self.assertEqual(7, len(area._vb.addedItems))
        self.assertEqual(4, len(area._legend.items))

        area.removeItem(image_item)
        self.assertEqual(6, len(area._items))
        self.assertEqual(6, len(area._vb.addedItems))
        self.assertEqual(4, len(area._legend.items))

        area.removeAllItems()
        self.assertEqual(0, len(area._items))
        self.assertEqual(0, len(area._vb.addedItems))
        self.assertEqual(0, len(area._legend.items))

    def testContextMenu(self):
        event = object()
        menu = self._area.getContextMenus(event)
        for row in menu:
            self.assertIsInstance(row, QMenu)


class TestHistogramLUTItem(unittest.TestCase):
    def testGeneral(self):
        image_item = ImageItem()
        item = HistogramLUTItem(image_item, parent=None)
