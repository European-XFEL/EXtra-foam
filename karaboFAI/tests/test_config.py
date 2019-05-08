import unittest
import os
import tempfile
import json

from karaboFAI.config import ConfigWrapper, _Config
from karaboFAI.logger import logger


class TestLaserOnOffWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        _Config._filename = os.path.join(cls.dir, "config.json")

        cls.expected_keys = set(_Config._system_readonly_config.keys()).union(
            set(_Config._system_reconfigurable_config.keys())).union(
            set(_Config._detector_readonly_config['DEFAULT'].keys())).union(
            set(_Config._detector_reconfigurable_config['DEFAULT'].keys()))

    def setUp(self):
        self._cfg = ConfigWrapper()

    def tearDown(self):
        os.remove(_Config._filename)

    def testGeneral(self):
        # test readonly
        with self.assertRaises(TypeError):
            self._cfg['DETECTOR'] = "ABC"

    def testNewFileGeneration(self):
        with open(_Config._filename, 'r') as fp:
            cfg = json.load(fp)

        # check system configuration
        self.assertEqual(set(cfg.keys()),
                         set(_Config._system_reconfigurable_config.keys())
                         .union(set(_Config.detectors)))

        for key, value in _Config._system_reconfigurable_config.items():
            if key not in _Config.detectors:
                self.assertEqual(value, cfg[key])

        # check configuration for each detector
        for det in _Config.detectors:
            self.assertEqual(
                set(cfg[det].keys()),
                set(_Config._detector_reconfigurable_config['DEFAULT'].keys()))

    def testLoadLPD(self):
        cfg = self._cfg

        detector = "LPD"
        cfg.load(detector)

        # test keys
        self.assertEqual(self.expected_keys, set(cfg.keys()))

        # test values
        self.assertEqual(cfg["DETECTOR"], detector)
        self.assertEqual(cfg["PULSE_RESOLVED"], True)
        self.assertEqual(cfg["SOURCE_NAME_BRIDGE"],
                         _Config._detector_reconfigurable_config[
                             detector]["SOURCE_NAME_BRIDGE"])
        self.assertEqual(cfg["SOURCE_NAME_FILE"],
                         _Config._detector_reconfigurable_config[
                             detector]["SOURCE_NAME_FILE"])

    def testLoadJungFrau(self):
        cfg = self._cfg

        detector = "JungFrau"
        cfg.load(detector)

        # test keys
        self.assertEqual(self.expected_keys, set(cfg.keys()))

        # test values
        self.assertEqual(cfg["DETECTOR"], detector)
        self.assertEqual(cfg["PULSE_RESOLVED"], False)
        self.assertEqual(cfg["SERVER_ADDR"],
                         _Config._detector_reconfigurable_config[
                             detector]["SERVER_ADDR"])
        self.assertEqual(cfg["SERVER_PORT"],
                         _Config._detector_reconfigurable_config[
                             detector]["SERVER_PORT"])

    def testInvalidSysKeys(self):
        # invalid keys in system config
        with open(_Config._filename, 'r') as fp:
            cfg = json.load(fp)

        with open(_Config._filename, 'w') as fp:
            cfg['UNKNOWN1'] = 1
            cfg['UNKNOWN2'] = 2
            json.dump(cfg, fp, indent=4)

        detector = "LPD"
        with self.assertRaisesRegex(ValueError, 'UNKNOWN1, UNKNOWN2'):
            with self.assertLogs(logger, "ERROR"):
                self._cfg.load(detector)

    def testInvalidDetectorKeys(self):
        # invalid keys in detector config
        with open(_Config._filename, 'r') as fp:
            cfg = json.load(fp)

        detector = "LPD"

        with open(_Config._filename, 'w') as fp:
            cfg[detector]['UNKNOWN'] = 1
            cfg[detector]['TIMEOUT'] = 2
            json.dump(cfg, fp, indent=4)

        with self.assertRaisesRegex(
                ValueError, f'{detector}.UNKNOWN, {detector}.TIMEOUT'):
            with self.assertLogs(logger, "ERROR"):
                self._cfg.load(detector)
