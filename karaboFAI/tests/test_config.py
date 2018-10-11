import unittest
import os

from karaboFAI.config import Config, UpperCaseConfigParser


class TestLaserOnOffWindow(unittest.TestCase):
    def setUp(self):
        Config._filename = "mock_settings.ini"
        self._expected_keys = set(Config._default_sys_config.keys()).union(
                              set(Config._allowed_topic_config_keys))

    def testNewFileGeneration(self):
        Config()

        cfg = UpperCaseConfigParser()
        cfg.read(Config._filename)

        # "DEFAULT" section should be empty
        self.assertTrue(not cfg["DEFAULT"])
        for topic in Config._default_topic_configs:
            self.assertEqual(set(cfg[topic].keys()),
                             set(Config._allowed_topic_config_keys))
            # values are all ''
            self.assertEqual(set(cfg[topic].values()), {''})

    def testLoad(self):
        topic = "SPB"

        config = Config()
        config.load(topic)

        # test keys
        self.assertEqual(self._expected_keys, set(config.keys()))

        # test values
        self.assertEqual(config["TOPIC"], topic)
        self.assertEqual(config["SOURCE_NAME"],
                         Config._default_topic_configs["SPB"]["SOURCE_NAME"])
        self.assertEqual(config["SOURCE_TYPE"],
                         Config._default_topic_configs["SPB"]["SOURCE_TYPE"])

        # now we change the content of the file
        with open(Config._filename) as fp:
            lines = fp.readlines()
            lines[3] = 'SOURCE_NAME = changed\n'
            lines[4] = 'SOURCE_TYPE = also changed\n'

        with open(Config._filename, 'w') as fp:
            for line in lines:
                fp.write(line)

        config.load(topic)

        # test keys
        self.assertEqual(self._expected_keys, set(config.keys()))

        # test values which should be overwritten by new values in the file
        self.assertEqual(config["TOPIC"], topic)
        self.assertEqual(config["SOURCE_NAME"], "changed")
        self.assertEqual(config["SOURCE_TYPE"], "also changed")

    def tearDown(self):
        os.remove(Config._filename)
