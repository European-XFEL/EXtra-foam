import unittest
import tempfile
import os

from karaboFAI.logger import logger
from karaboFAI.config import _Config, ConfigWrapper
from karaboFAI.gui.main_gui import MainGUI
from karaboFAI.gui import mkQApp
from karaboFAI.gui.windows import (
    CorrelationWindow, ImageToolWindow, OverviewWindow, PumpProbeWindow,
    XasWindow
)


class TestMainGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.setLevel('CRITICAL')
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        config = ConfigWrapper()  # ensure file

        config.load('LPD')

        mkQApp()
        cls.gui = MainGUI()

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()
        ImageToolWindow._reset()

    def testOpenCloseWindows(self):
        actions = self.gui._tool_bar.actions()
        imagetool_action = actions[2]
        overview_action = actions[3]
        pp_action = actions[4]
        correlation_action = actions[5]
        xas_action = actions[6]

        imagetool_window = self._check_open_window(imagetool_action)
        self.assertIsInstance(imagetool_window, ImageToolWindow)

        overview_window = self._check_open_window(overview_action)
        self.assertIsInstance(overview_window, OverviewWindow)

        correlation_window = self._check_open_window(correlation_action)
        self.assertIsInstance(correlation_window, CorrelationWindow)

        pp_window = self._check_open_window(pp_action)
        self.assertIsInstance(pp_window, PumpProbeWindow)

        # open one window twice
        xas_window = self._check_open_window(xas_action)
        self.assertIsInstance(xas_window, XasWindow)
        self._check_open_window(xas_action, registered=False)

        self._check_close_window(overview_window)
        self._check_close_window(correlation_window)
        self._check_close_window(pp_window)
        self._check_close_window(xas_window)
        self._check_close_window(imagetool_window)

        # if a plot window is closed, it can be re-openned and a new instance
        # will be created
        pp_window_new = self._check_open_window(pp_action)
        self.assertIsInstance(pp_window_new, PumpProbeWindow)
        self.assertIsNot(pp_window_new, pp_window)

        # imagetool_window is a singleton, therefore, the re-openned window
        # is the same instance
        imagetool_window_new = self._check_open_window(imagetool_action)
        self.assertIs(imagetool_window_new, imagetool_window)

    def _check_open_window(self, action, registered=True):
        """Check triggering action about opening a window.

        :param bool registered: True for the new window is expected to be
            registered; False for the old window will be activate and thus
            no new window will be registered.
        """
        n_registered = len(self.gui._windows)
        action.trigger()
        if registered:
            window = list(self.gui._windows.keys())[-1]
            self.assertEqual(n_registered+1, len(self.gui._windows))
            return window

        self.assertEqual(n_registered, len(self.gui._windows))

    def _check_close_window(self, window):
        n_registered = len(self.gui._windows)
        window.close()
        self.assertEqual(n_registered-1, len(self.gui._windows))
