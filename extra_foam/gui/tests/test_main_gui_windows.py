import unittest
import tempfile
import os

from extra_foam.logger import logger
from extra_foam.config import config
from extra_foam.gui.main_gui import MainGUI
from extra_foam.gui import mkQApp
from extra_foam.gui.windows import (
    BinningWindow, StatisticsWindow, PulseOfInterestWindow, PumpProbeWindow,
    FileStreamControllerWindow, AboutWindow,
)

app = mkQApp()

logger.setLevel('CRITICAL')

_tmp_cfg_dir = tempfile.mkdtemp()


def setup_module(module):
    from extra_foam import config
    module._backup_ROOT_PATH = config.ROOT_PATH
    config.ROOT_PATH = _tmp_cfg_dir


def teardown_module(module):
    os.rmdir(_tmp_cfg_dir)
    from extra_foam import config
    config.ROOT_PATH = module._backup_ROOT_PATH


class TestOpenCloseWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.load('LPD', 'FXE')

        cls.gui = MainGUI()

    @classmethod
    def tearDownClass(cls):
        cls.gui.close()

        os.remove(config.config_file)

    def testOpenCloseWindows(self):
        actions = self.gui._tool_bar.actions()

        poi_action = actions[4]
        self.assertEqual("Pulse-of-interest", poi_action.text())
        pp_action = actions[5]
        self.assertEqual("Pump-probe", pp_action.text())
        statistics_action = actions[6]
        self.assertEqual("Statistics", statistics_action.text())
        bin1d_action = actions[7]
        bin2d_action = actions[8]
        # TODO: add ROI

        pp_window = self._check_open_window(pp_action)
        self.assertIsInstance(pp_window, PumpProbeWindow)

        statistics_window = self._check_open_window(statistics_action)
        self.assertIsInstance(statistics_window, StatisticsWindow)

        binning_window = self._check_open_window(bin1d_action)
        self.assertIsInstance(binning_window, BinningWindow)

        poi_window = self._check_open_window(poi_action)
        self.assertIsInstance(poi_window, PulseOfInterestWindow)
        # open one window twice
        self._check_open_window(poi_action, registered=False)

        self._check_close_window(pp_window)
        self._check_close_window(statistics_window)
        self._check_close_window(binning_window)
        self._check_close_window(poi_window)

        # if a plot window is closed, it can be re-openned and a new instance
        # will be created
        pp_window_new = self._check_open_window(pp_action)
        self.assertIsInstance(pp_window_new, PumpProbeWindow)
        self.assertIsNot(pp_window_new, pp_window)

    def testOpenCloseSatelliteWindows(self):
        actions = self.gui._tool_bar.actions()
        about_action = actions[-1]
        streamer_action = actions[-2]
        monitor_action = actions[-3]

        about_window = self._check_open_satellite_window(about_action)
        self.assertIsInstance(about_window, AboutWindow)

        streamer_window = self._check_open_satellite_window(streamer_action)
        self.assertIsInstance(streamer_window, FileStreamControllerWindow)

        # open one window twice
        self._check_open_satellite_window(about_action, registered=False)

        self._check_close_satellite_window(about_window)
        self._check_close_satellite_window(streamer_window)

        # if a window is closed, it can be re-opened and a new instance
        # will be created
        about_window_new = self._check_open_satellite_window(about_action)
        self.assertIsInstance(about_window_new, AboutWindow)
        self.assertIsNot(about_window_new, about_window)

    def _check_open_window(self, action, registered=True):
        """Check triggering action about opening a window.

        :param bool registered: True for the new window is expected to be
            registered; False for the old window will be activate and thus
            no new window will be registered.
        """
        n_registered = len(self.gui._plot_windows)
        action.trigger()
        if registered:
            window = list(self.gui._plot_windows.keys())[-1]
            self.assertEqual(n_registered+1, len(self.gui._plot_windows))
            return window

        self.assertEqual(n_registered, len(self.gui._plot_windows))

    def _check_close_window(self, window):
        n_registered = len(self.gui._plot_windows)
        window.close()
        self.assertEqual(n_registered-1, len(self.gui._plot_windows))

    def _check_open_satellite_window(self, action, registered=True):
        """Check triggering action about opening a satellite window.

        :param bool registered: True for the new window is expected to be
            registered; False for the old window will be activate and thus
            no new window will be registered.
        """
        n_registered = len(self.gui._satellite_windows)
        action.trigger()
        if registered:
            window = list(self.gui._satellite_windows.keys())[-1]
            self.assertEqual(n_registered+1, len(self.gui._satellite_windows))
            return window

        self.assertEqual(n_registered, len(self.gui._satellite_windows))

    def _check_close_satellite_window(self, window):
        n_registered = len(self.gui._satellite_windows)
        window.close()
        self.assertEqual(n_registered-1, len(self.gui._satellite_windows))
