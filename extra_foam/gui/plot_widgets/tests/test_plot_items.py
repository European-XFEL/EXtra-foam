import unittest
from unittest.mock import MagicMock

import numpy as np

from extra_foam.gui import mkQApp
from extra_foam.gui.plot_widgets.plot_items import (
    ImageItem, HistogramLUTItem
)
from extra_foam.logger import logger


app = mkQApp()

logger.setLevel("CRITICAL")


class TestImageItem(unittest.TestCase):
    pass