import unittest
from unittest.mock import MagicMock, patch
from collections import Counter

import pytest
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy, QTest

from extra_foam.logger import logger_suite as logger
from extra_foam.gui import mkQApp
from extra_foam.special_suite.xas_tim_proc import XasTimProcessor
from extra_foam.special_suite.xas_tim_w import (
    XasTimWindow
)


app = mkQApp()

logger.setLevel('CRITICAL')


class TestXasTimWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("extra_foam.special_suite.special_analysis_base._SpecialAnalysisBase.startWorker"):
            cls._win = XasTimWindow('SCS')

    @classmethod
    def tearDown(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._win.close()

    def testWindow(self):
        win = self._win

        self.assertEqual(0, len(win._plot_widgets))
        counter = Counter()
        for key in win._plot_widgets:
            counter[key.__class__] += 1

        # self.assertEqual(1, counter[])

        win.updateWidgetsF()


class TestXasTimProcessor:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = XasTimProcessor(object(), object())

    def testGeneral(self):
        pass
