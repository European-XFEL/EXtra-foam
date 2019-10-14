import unittest

from karaboFAI.config import config
from karaboFAI.database.data_categories import (
    DATA_SOURCE_CATEGORIES, _MISC_CATEGORIES
)


class TestGUI(unittest.TestCase):

    def test_general(self):
        for topic in DATA_SOURCE_CATEGORIES:
            topic_data = DATA_SOURCE_CATEGORIES[topic]
            for ctg in _MISC_CATEGORIES:
                ctg_data = topic_data[ctg]
                self.assertEqual(sorted(ctg_data), ctg_data)

            for det in config.detectors:
                if topic == "UNKNOWN":
                    self.assertIn(det, topic_data)

                if det in topic_data:
                    self.assertEqual(set(), topic_data[det])

            if topic == "FXE":
                self.assertNotIn("AGIPD", topic_data)
                self.assertIn("LPD", topic_data)
            elif topic == "SCS":
                self.assertIn("DSSC", topic_data)
                self.assertIn("FastCCD", topic_data)
                self.assertNotIn("JungFrau", topic_data)
