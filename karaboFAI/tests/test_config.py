import unittest
import os

from karaboFAI.config import Config, UpperCaseConfigParser


class TestLaserOnOffWindow(unittest.TestCase):
    def setUp(self):
        Config._filename = "mock_settings.ini"
        self._expected_keys = set(Config._default_sys_config.keys()).union(
                              set(Config._detector_reconfigurable_keys)).union(
                              set(Config._detector_readonly_config_keys)
        )

    def testNewFileGeneration(self):
        Config()  # generate the file

        cfg = UpperCaseConfigParser()
        cfg.read(Config._filename)

        # "DEFAULT" section should be empty
        self.assertTrue(not cfg["DEFAULT"])
        for detector in Config._default_detector_configs:
            # only reconfigurable keys are allowed to be defined in the file
            self.assertEqual(set(cfg[detector].keys()),
                             set(Config._detector_reconfigurable_keys))
            # values are all ''
            self.assertEqual(set(cfg[detector].values()), {''})

    def testLoad(self):
        detector = "AGIPD"

        config = Config()
        config.load(detector)

        # test keys
        self.assertEqual(self._expected_keys, set(config.keys()))

        # test values
        self.assertEqual(config["DETECTOR"], detector)
        self.assertEqual(config["SOURCE_NAME"],
                         Config._default_detector_configs[detector]["SOURCE_NAME"])
        self.assertEqual(config["SOURCE_TYPE"],
                         Config._default_detector_configs[detector]["SOURCE_TYPE"])

        # *************************************************************
        # now we change the content of the file
        with open(Config._filename) as fp:
            lines = fp.readlines()
            lines[3] = 'SOURCE_NAME = changed\n'
            lines[4] = 'SOURCE_TYPE = also changed\n'

        with open(Config._filename, 'w') as fp:
            for line in lines:
                fp.write(line)

        config.load(detector)

        # test keys
        self.assertEqual(self._expected_keys, set(config.keys()))

        # test values which should be overwritten by new values in the file
        self.assertEqual(config["DETECTOR"], detector)
        self.assertEqual(config["SOURCE_NAME"], "changed")
        self.assertEqual(config["SOURCE_TYPE"], "also changed")

        # *************************************************************
        # now we add an unknown key to the file
        with open(Config._filename) as fp:
            lines = fp.readlines()
            lines.insert(1, 'SOURCE_PLACE = \n')

        with open(Config._filename, 'w') as fp:
            for line in lines:
                fp.write(line)

        config.load(detector)  # unknown keys will not raise

        # *************************************************************
        # now we add a read-only key to the file
        with open(Config._filename) as fp:
            lines = fp.readlines()
            lines[1] = Config._detector_readonly_config_keys[-1] + ' = \n'

        with open(Config._filename, 'w') as fp:
            for line in lines:
                fp.write(line)

        # read-only config is not allowed to be in the file
        with self.assertRaises(KeyError):
            config.load(detector)

        # *************************************************************
        # now we add a system config key to the file
        with open(Config._filename) as fp:
            lines = fp.readlines()
            lines[1] = list(Config._default_sys_config.keys())[-1] + ' = \n'

        with open(Config._filename, 'w') as fp:
            for line in lines:
                fp.write(line)

        # system config is not allowed to be in the file
        with self.assertRaises(KeyError):
            config.load(detector)

    def tearDown(self):
        os.remove(Config._filename)
