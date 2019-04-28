import unittest

from karaboFAI.services import FaiServer
from karaboFAI.gui.windows import (
    CorrelationWindow, ImageToolWindow, OverviewWindow, PumpProbeWindow,
    XasWindow
)


class TestMainGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fai = FaiServer('LPD')
        cls.gui = fai.gui
        cls.app = fai.app

    def testOpenCloseWindows(self):
        actions = self.gui._tool_bar.actions()
        imagetool_action = actions[2]
        overview_action = actions[3]
        pp_action = actions[4]
        correlation_action = actions[5]
        xas_action = actions[6]

        # TODO: there must be some bug
        # imagetool_window = self._check_open_window(imagetool_action)
        # # TODO: ImageToolWindow is a SingletonWindow
        # # self.assertIsInstance(window, ImageToolWindow)
        # self._check_reopen_singleton_window(imagetool_action)

        overview_window = self._check_open_window(overview_action)
        self.assertIsInstance(overview_window, OverviewWindow)

        correlation_window = self._check_open_window(correlation_action)
        self.assertIsInstance(correlation_window, CorrelationWindow)

        pp_window = self._check_open_window(pp_action)
        self.assertIsInstance(pp_window, PumpProbeWindow)

        # open one window twice
        xas_window1 = self._check_open_window(xas_action)
        self.assertIsInstance(xas_window1, XasWindow)
        xas_window2 = self._check_open_window(xas_action)
        self.assertIsInstance(xas_window2, XasWindow)
        self.assertNotEqual(xas_window1, xas_window2)

        self._check_close_window(overview_window)
        self._check_close_window(correlation_window)
        self._check_close_window(pp_window)
        self._check_close_window(xas_window1)
        self._check_close_window(xas_window2)
        # self._check_close_window(imagetool_window)

    def _check_open_window(self, action):
        n_registered = len(self.gui._windows)
        action.trigger()
        window = list(self.gui._windows.keys())[-1]
        self.assertEqual(n_registered+1, len(self.gui._windows))
        return window

    def _check_reopen_singleton_window(self, action):
        n_registered = len(self.gui._windows)
        action.trigger()
        self.assertEqual(n_registered, len(self.gui._windows))

    def _check_close_window(self, window):
        n_registered = len(self.gui._windows)
        window.close()
        self.assertEqual(n_registered-1, len(self.gui._windows))
