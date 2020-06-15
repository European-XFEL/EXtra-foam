import unittest
import unittest
from unittest.mock import MagicMock, patch
from collections import Counter

import pytest
import numpy as np

from extra_foam.special_suite import logger, mkQApp
from extra_foam.special_suite.module_scan_proc import ModuleScanProcessor
from extra_foam.special_suite.module_scan_w import (
    ModuleScanWindow
)


app = mkQApp()

logger.setLevel('CRITICAL')


class TestModuleScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._win = ModuleScanWindow('DET')

    @classmethod
    def tearDownClass(cls):
        # explicitly close the MainGUI to avoid error in GuiLogger
        cls._win.close()

    def testWindow(self):
        win = self._win

        self.assertEqual(1, len(win._plot_widgets_st))
        counter = Counter()
        for key in win._plot_widgets_st:
            counter[key.__class__] += 1

        # self.assertEqual(1, counter[GotthardImageView])
        # self.assertEqual(1, counter[GotthardAvgPlot])
        # self.assertEqual(1, counter[GotthardPulsePlot])
        # self.assertEqual(1, counter[GotthardHist])
        #
        # win.updateWidgetsST()

    def testCtrl(self):
        win = self._win
        widget = win._ctrl_widget_st
        proc = win._worker_st


class TestModuleScanProcessor:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self._proc = ModuleScanProcessor(object(), object())

    def testGeneral(self):
        pass
