import unittest
import os
import tempfile
import json

from karaboFAI.config import Config
from karaboFAI.logger import logger


class TestLaserOnOffWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        Config._filename = os.path.join(cls.dir, "config.json")

    def setUp(self):
        self._cfg = Config()

    def tearDown(self):
        os.remove(Config._filename)

    def testNewFileGeneration(self):
        with open(Config._filename, 'r') as fp:
            cfg = json.load(fp)

        # check system configuration
        self.assertEqual(set(cfg.keys()),
                         set(Config._sys_reconfigurable_config.keys()).union(
                             set(Config.detectors)))

        for key, value in Config._sys_reconfigurable_config.items():
            if key not in Config.detectors:
                self.assertEqual(value, cfg[key])

        # check configuration for each detector
        for detector in Config._default_detector_configs:
            self.assertEqual(set(cfg[detector].keys()),
                             set(Config._detector_default_config.keys()))

    def testLoadLPD(self):
        cfg = self._cfg

        detector = "LPD"
        cfg.load(detector)

        # test keys
        expected_keys = set(Config._sys_readonly_config.keys()).union(
            set(Config._sys_reconfigurable_config.keys())).union(
            set(Config._detector_default_config.keys()))
        self.assertEqual(expected_keys, set(cfg.keys()))

        # test values
        self.assertEqual(cfg["DETECTOR"], detector)
        self.assertEqual(cfg["PULSE_RESOLVED"], True)
        self.assertEqual(cfg["SOURCE_NAME_BRIDGE"],
                         Config._default_detector_configs[detector]["SOURCE_NAME_BRIDGE"])
        self.assertEqual(cfg["SOURCE_NAME_FILE"],
                         Config._default_detector_configs[detector]["SOURCE_NAME_FILE"])

    def testLoadJungFrau(self):
        cfg = self._cfg

        detector = "JungFrau"
        cfg.load(detector)

        # test keys
        expected_keys = set(Config._sys_readonly_config.keys()).union(
            set(Config._sys_reconfigurable_config.keys())).union(
            set(Config._detector_default_config.keys()))
        self.assertEqual(expected_keys, set(cfg.keys()))

        # test values
        self.assertEqual(cfg["DETECTOR"], detector)
        self.assertEqual(cfg["PULSE_RESOLVED"], False)
        self.assertEqual(cfg["SERVER_ADDR"],
                         Config._default_detector_configs[detector]["SERVER_ADDR"])
        self.assertEqual(cfg["SERVER_PORT"],
                         Config._default_detector_configs[detector]["SERVER_PORT"])

    def testInvalidSysKeys(self):
        # invalid keys in system config
        with open(Config._filename, 'r') as fp:
            cfg = json.load(fp)

        with open(Config._filename, 'w') as fp:
            cfg['UNKNOWN1'] = 1
            cfg['UNKNOWN2'] = 2
            json.dump(cfg, fp, indent=4)

        detector = "LPD"
        with self.assertRaisesRegex(ValueError, 'UNKNOWN1, UNKNOWN2'):
            with self.assertLogs(logger, "ERROR"):
                self._cfg.load(detector)

    def testInvalidDetectorKeys(self):
        # invalid keys in detector config
        with open(Config._filename, 'r') as fp:
            cfg = json.load(fp)

        detector = "LPD"

        with open(Config._filename, 'w') as fp:
            cfg[detector]['UNKNOWN'] = 1
            cfg[detector]['TIMEOUT'] = 2
            json.dump(cfg, fp, indent=4)

        with self.assertRaisesRegex(
                ValueError, f'{detector}.UNKNOWN, {detector}.TIMEOUT'):
            with self.assertLogs(logger, "ERROR"):
                self._cfg.load(detector)
