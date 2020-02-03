import unittest
from unittest import mock
import os
import os.path as osp
import tempfile
import yaml

import pytest

from extra_foam.config import ConfigWrapper, _Config
from extra_foam.logger import logger

logger.setLevel("CRITICAL")

_tmp_cfg_dir = tempfile.mkdtemp()


@mock.patch("extra_foam.config.ROOT_PATH", _tmp_cfg_dir)
class TestConfig:
    def teardown_class(self):
        os.rmdir(_tmp_cfg_dir)

    def setup_method(self):
        self._cfg = ConfigWrapper()
        self.detectors = self._cfg.detectors
        self.topics = self._cfg.topics

    def testGeneral(self):
        # test readonly
        with pytest.raises(TypeError):
            self._cfg['DETECTOR'] = "ABC"

        assert 'TOPIC' in self._cfg
        assert 'REDIS_PORT' in self._cfg
        assert 'SOURCE_DEFAULT_TYPE' in self._cfg

    @pytest.mark.parametrize("detector, topic",
                             [('AGIPD', 'SPB'), ('AGIPD', 'MID'),
                              ('DSSC', 'SCS'),
                              ('LPD', 'FXE'),
                              ('JungFrau', 'FXE'),
                              ('FastCCD', 'SCS'),
                              ('BaslerCamera', 'SCS')])
    def testFileGeneration(self, detector, topic):
        # TODO: add SQS and HED tests when detectors are added

        # test load config from file

        cfg = self._cfg
        cfg.load(detector, topic)

        # test config file generation and the correctness of the default
        # config files

        filepath = cfg.config_file
        with open(filepath, 'r') as fp:
            cfg_from_file = yaml.load(fp, Loader=yaml.Loader)

        orig_filepath = osp.join(_Config._abs_dirpath, f'configs/{topic.lower()}.config.yaml')
        assert orig_filepath != filepath
        with open(orig_filepath, 'r') as fp:
            cfg_gt = yaml.load(fp, Loader=yaml.Loader)
        assert cfg_from_file == cfg_gt

        os.remove(filepath)

    def testLoadConfigFromFile(self):
        cfg = self._cfg
        cfg.load('DSSC', 'SCS')

        filepath = cfg.config_file
        with open(filepath, 'r') as fp:
            cfg_from_file = yaml.load(fp, Loader=yaml.Loader)

        # test only one "main detector" category will be loaded
        assert "FastCCD" in cfg_from_file["SOURCE"]["CATEGORY"]
        assert "FastCCD" not in cfg

        # test main detector config were loaded
        assert (128, 512) == cfg["MODULE_SHAPE"]
        assert 16 == cfg["NUMBER_OF_MODULES"]

        # test source items were loaded
        dssc_srcs = cfg_from_file["SOURCE"]["CATEGORY"]["DSSC"]["PIPELINE"]
        for device_id, ppts in dssc_srcs.items():
            assert cfg.pipeline_sources["DSSC"][device_id] == ppts

        xgm_pipeline_srcs = cfg_from_file["SOURCE"]["CATEGORY"]["XGM"]["PIPELINE"]
        for device_id, ppts in xgm_pipeline_srcs.items():
            assert cfg.pipeline_sources["XGM"][device_id] == ppts
        xgm_control_srcs = cfg_from_file["SOURCE"]["CATEGORY"]["XGM"]["CONTROL"]
        for device_id, ppts in xgm_control_srcs.items():
            assert cfg.control_sources["XGM"][device_id] == ppts

        os.remove(cfg.config_file)

    def testInvalidSourceCategory(self):
        cfg = self._cfg
        cfg.load('DSSC', 'SCS')

        filepath = cfg.config_file
        with open(filepath, 'r') as fp:
            cfg_from_file = yaml.load(fp, Loader=yaml.Loader)

        # test invalid source category
        cfg_from_file["SOURCE"]["CATEGORY"]["XXGM"] = cfg_from_file["SOURCE"]["CATEGORY"]["XGM"]
        with open(filepath, 'w') as fp:
            yaml.dump(cfg_from_file, fp, Dumper=yaml.Dumper)
        with pytest.raises(ValueError, match="Invalid source category"):
            cfg.load('DSSC', 'SCS')

        os.remove(filepath)

    def testInvalidFileFormat(self):
        cfg = self._cfg
        cfg.load('DSSC', 'SCS')

        filepath = cfg.config_file
        with open(filepath, 'r') as fp:
            content = fp.read()

        with open(filepath, 'w') as fp:
            # add a line at the end of the file to make it an invalid YAML
            content += "\nABCDEFEA"
            fp.write(content)

        with pytest.raises(IOError):
            cfg.load('DSSC', 'SCS')

        with open(filepath, 'w') as fp:
            # make an empty config file
            content = ""
            fp.write(content)

        with pytest.raises(ValueError, match="empty"):
            cfg.load('DSSC', 'SCS')

        os.remove(filepath)


class TestPlotLabel(unittest.TestCase):
    def testGeneral(self):
        from extra_foam.config import AnalysisType, plot_labels

        self.assertTupleEqual(("", ""), plot_labels[AnalysisType.UNDEFINED])
        self.assertTupleEqual(("", ""), plot_labels[AnalysisType.ROI_FOM])
        self.assertTrue(bool(plot_labels[AnalysisType.ROI_PROJ].x))
        self.assertTrue(bool(plot_labels[AnalysisType.ROI_PROJ].y))

        with self.assertRaises(KeyError):
            plot_labels['abc']

        with self.assertRaises(TypeError):
            # test not assignable
            plot_labels[AnalysisType.ROI_PROJ] = 'abc'
