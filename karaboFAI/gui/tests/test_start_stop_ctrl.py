import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os

from PyQt5 import QtCore
from PyQt5.QtTest import QTest, QSignalSpy
from PyQt5.QtCore import Qt

from karaboFAI.gui import mkQApp
from karaboFAI.services import Fai
from karaboFAI.logger import logger


class TestMainGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from karaboFAI.config import _Config, ConfigWrapper

        # do not use the config file in the current computer
        _Config._filename = os.path.join(tempfile.mkdtemp(), "config.json")
        ConfigWrapper()  # ensure file

        cls.fai = Fai('LPD')
        cls.fai.init()
        cls.app = mkQApp()
        cls.gui = cls.fai.gui
        cls.scheduler = cls.fai._scheduler
        cls.bridge = cls.fai._bridge

        cls._actions = cls.gui._tool_bar.actions()
        cls._start_action = cls._actions[0]
        cls._stop_action = cls._actions[1]

    @classmethod
    def tearDownClass(cls):
        cls.fai.shutdown()
        cls.gui.close()

    @patch('karaboFAI.gui.ctrl_widgets.PumpProbeCtrlWidget.'
           'updateSharedParameters', MagicMock(return_value=True))
    @patch('karaboFAI.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.'
           'updateSharedParameters', MagicMock(return_value=True))
    @patch('karaboFAI.gui.ctrl_widgets.PumpProbeCtrlWidget.onStart', Mock())
    @patch('karaboFAI.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.onStart', Mock())
    @patch('karaboFAI.gui.ctrl_widgets.PumpProbeCtrlWidget.onStop', Mock())
    @patch('karaboFAI.gui.ctrl_widgets.AzimuthalIntegCtrlWidget.onStop', Mock())
    @patch('karaboFAI.pipeline.bridge.Bridge.activate', Mock())
    @patch('karaboFAI.pipeline.bridge.Bridge.pause', Mock())
    def testStartStop(self):
        logger.setLevel("CRITICAL")

        start_spy = QSignalSpy(self.gui.start_sgn)
        stop_spy = QSignalSpy(self.gui.stop_sgn)

        # -------------------------------------------------------------
        # test when the start action button is clicked
        # -------------------------------------------------------------

        self._start_action.trigger()

        self.gui.pump_probe_ctrl_widget.updateSharedParameters. \
            assert_called_once()
        self.gui.azimuthal_integ_ctrl_widget.updateSharedParameters. \
            assert_called_once()

        self.assertEqual(1, len(start_spy))

        self.gui.azimuthal_integ_ctrl_widget.onStart.assert_called_once()
        self.gui.pump_probe_ctrl_widget.onStart.assert_called_once()

        self.assertFalse(self._start_action.isEnabled())
        self.assertTrue(self._stop_action.isEnabled())

        # FIXME
        # self.bridge.activate.assert_called_once()

        # -------------------------------------------------------------
        # test when the start action button is clicked
        # -------------------------------------------------------------

        self._stop_action.trigger()

        self.gui.azimuthal_integ_ctrl_widget.onStop.assert_called_once()
        self.gui.pump_probe_ctrl_widget.onStop.assert_called_once()

        self.assertEqual(1, len(stop_spy))

        self.assertTrue(self._start_action.isEnabled())
        self.assertFalse(self._stop_action.isEnabled())

        # FIXME
        # self.bridge.pause.assert_called_once()
