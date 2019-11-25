import unittest
from unittest import mock
import os
import os.path as osp
import tempfile
import json

from extra_foam.config import ConfigWrapper, _Config
from extra_foam.logger import logger


class TestLaserOnOffWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        _Config._filename = osp.join(cls.dir, "config.json")

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

            for key, value in _Config._detector_reconfigurable_config[det].items():
                self.assertEqual(value, cfg[det][key])

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
        # test raise if the answer is 'no'
        with mock.patch('builtins.input', return_value='n'):
            with self.assertRaisesRegex(ValueError, 'UNKNOWN1, UNKNOWN2'):
                self._cfg.load(detector)

        # test generating backup file if the answer is 'yes'
        with mock.patch('builtins.input', return_value='y'):
            self._cfg.load(detector)

        self.assertTrue(osp.exists(_Config._filename + '.bak'))
        # check the content of the backup file
        with open(_Config._filename + '.bak', 'r') as fp:
            self.assertEqual(cfg, json.load(fp))
        # nothing should happen since there is a valid config file
        self._cfg.load(detector)

        # mess up the new config file again
        with open(_Config._filename, 'w') as fp:
            cfg['UNKNOWN3'] = 1
            json.dump(cfg, fp, indent=4)

        with mock.patch('builtins.input', return_value='y'):
            self._cfg.load(detector)

        self.assertTrue(osp.exists(_Config._filename + '.bak'))
        # check the content of the backup file
        with open(_Config._filename + '.bak', 'r') as fp:
            self.assertEqual(cfg, json.load(fp))

        # test that the second level backup file was generated
        self.assertTrue(osp.exists(_Config._filename + '.bak.bak'))
        # check the content of the second level backup file
        with open(_Config._filename + '.bak.bak', 'r') as fp:
            del cfg['UNKNOWN3']
            self.assertEqual(cfg, json.load(fp))

    def testInvalidDetectorKeys(self):
        # invalid keys in detector config
        with open(_Config._filename, 'r') as fp:
            cfg = json.load(fp)

        detector = "LPD"

        with open(_Config._filename, 'w') as fp:
            cfg[detector]['UNKNOWN'] = 1
            cfg[detector]['TIMEOUT'] = 2
            json.dump(cfg, fp, indent=4)

        with mock.patch('builtins.input', return_value='n'):
            with self.assertRaisesRegex(ValueError, 'LPD.UNKNOWN, LPD.TIMEOUT'):
                self._cfg.load(detector)

    def testInvalidDetectors(self):
        # detector config is missing in config
        with open(_Config._filename, 'r') as fp:
            cfg = json.load(fp)

        det = "LPD"

        del cfg[det]
        with open(_Config._filename, 'w') as fp:
            json.dump(cfg, fp, indent=4)

        with mock.patch('builtins.input', return_value='n'):
            # nothing will happen since the default configuration will be used
            self._cfg.load(det)
            self.assertFalse(osp.exists(_Config._filename + '.bak'))
            # check the current config
            self.assertEqual(_Config._detector_reconfigurable_config[det]['GEOMETRY_FILE'],
                             self._cfg['GEOMETRY_FILE'])

        with mock.patch('builtins.input', return_value='y'):
            self._cfg.load(det)
            # check the backup file has been generated
            self.assertTrue(osp.exists(_Config._filename + '.bak'))
            with open(_Config._filename, 'r') as fp:
                cfg = json.load(fp)
                # test the detector config is in the new config file
                self.assertTrue(det in cfg)
