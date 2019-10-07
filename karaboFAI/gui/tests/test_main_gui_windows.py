import unittest
import tempfile
import os

from karaboFAI.logger import logger
from karaboFAI.config import _Config, ConfigWrapper
from karaboFAI.gui.main_gui import MainGUI
from karaboFAI.gui import mkQApp
from karaboFAI.gui.windows import (
    AzimuthalIntegrationWindow, Bin1dWindow, Bin2dWindow, CorrelationWindow,
    DarkRunWindow, ImageToolWindow, OverviewWindow, StatisticsWindow,
    PulseOfInterestWindow, PumpProbeWindow
)

app = mkQApp()

logger.setLevel('CRITICAL')


class TestMainGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        config = ConfigWrapper()  # ensure file

        config.load('LPD')

        ImageToolWindow.reset()
        cls.gui = MainGUI()

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

    def testOpenCloseWindows(self):
        actions = self.gui._tool_bar.actions()
        imagetool_action = actions[2]
        overview_action = actions[3]
        pp_action = actions[4]
        statistics_action = actions[5]
        correlation_action = actions[6]
        bin1d_action = actions[7]
        bin2d_action = actions[8]
        poi_action = actions[9]
        ai_action = actions[10]
        # TODO: add ROI
        darkrun_action = actions[12]

        # ImageToolWindow is opened together with the MainGUI
        imagetool_window = list(self.gui._windows.keys())[-1]
        self.assertIsInstance(imagetool_window, ImageToolWindow)

        darkrun_window = self._check_open_window(darkrun_action)
        self.assertIsInstance(darkrun_window, DarkRunWindow)

        overview_window = self._check_open_window(overview_action)
        self.assertIsInstance(overview_window, OverviewWindow)

        pp_window = self._check_open_window(pp_action)
        self.assertIsInstance(pp_window, PumpProbeWindow)

        statistics_window = self._check_open_window(statistics_action)
        self.assertIsInstance(statistics_window, StatisticsWindow)

        correlation_window = self._check_open_window(correlation_action)
        self.assertIsInstance(correlation_window, CorrelationWindow)

        bin1d_window = self._check_open_window(bin1d_action)
        self.assertIsInstance(bin1d_window, Bin1dWindow)

        bin2d_window = self._check_open_window(bin2d_action)
        self.assertIsInstance(bin2d_window, Bin2dWindow)

        # open one window twice
        poi_window = self._check_open_window(poi_action)
        self.assertIsInstance(poi_window, PulseOfInterestWindow)
        self._check_open_window(poi_action, registered=False)

        ai_window = self._check_open_window(ai_action)
        self.assertIsInstance(ai_window, AzimuthalIntegrationWindow)
        self._check_open_window(ai_action, registered=False)

        self._check_close_window(imagetool_window)
        self._check_close_window(darkrun_window)
        self._check_close_window(overview_window)
        self._check_close_window(pp_window)
        self._check_close_window(statistics_window)
        self._check_close_window(correlation_window)
        self._check_close_window(bin1d_window)
        self._check_close_window(bin2d_window)
        self._check_close_window(poi_window)
        self._check_close_window(ai_window)

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
