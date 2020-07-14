import unittest

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets import PlotWidgetF, RingItem
from extra_foam.gui.plot_widgets.image_items import (
    ImageItem
)
from extra_foam.logger import logger

from . import _display

app = mkQApp()

logger.setLevel("CRITICAL")


class TestImageItem(unittest.TestCase):
    def testSetImage(self):
        item = ImageItem()

        # TODO: check test in TestImageView


class TestGeometryItem(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls._widget = PlotWidgetF()
        if _display():
            cls._widget.show()

    def testRingItem(self):
        item = RingItem()
        self._widget.addItem(item)
        _display()

        item.setGeometry(100, 100, [50, 100])
        _display()

        item.clearGeometry()
        _display()
