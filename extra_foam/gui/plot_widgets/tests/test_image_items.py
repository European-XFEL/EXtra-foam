import unittest
from unittest.mock import MagicMock

import numpy as np

from extra_foam.gui import mkQApp
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
