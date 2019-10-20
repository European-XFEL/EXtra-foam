"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold configuration related objects.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import copy
from enum import IntEnum
import json
import os.path as osp
import collections
import shutil

import numpy as np

from . import ROOT_PATH
from .logger import logger
from .utils import query_yes_no


class DataSource(IntEnum):
    FILE = 0  # data from files (run directory)
    BRIDGE = 1  # real-time data from the bridge


class PumpProbeMode(IntEnum):
    UNDEFINED = 0
    PRE_DEFINED_OFF = 1  # use pre-defined reference image
    SAME_TRAIN = 2  # on-/off- pulses in the same train
    EVEN_TRAIN_ON = 3  # on-/off- pulses have even/odd train IDs, respectively
    ODD_TRAIN_ON = 4  # on/-off- pulses have odd/even train IDs, respectively


class AnalysisType(IntEnum):
    UNDEFINED = 0
    PUMP_PROBE = 1
    ROI1 = 11
    ROI2 = 12
    ROI1_SUB_ROI2 = 13
    ROI1_ADD_ROI2 = 14
    PROJ_ROI1 = 21
    PROJ_ROI2 = 22
    PROJ_ROI1_SUB_ROI2 = 23
    PROJ_ROI1_ADD_ROI2 = 24
    AZIMUTHAL_INTEG = 41
    PULSE = 2700
    ROI1_PULSE = 2711
    ROI2_PULSE = 2712
    ROI1_SUB_ROI2_PULSE = 2713
    ROI1_ADD_ROI2_PULSE = 2714
    PROJ_ROI1_PULSE = 2721
    PROJ_ROI2_PULSE = 2722
    PROJ_ROI1_SUB_ROI2_PULSE = 2723
    PROJ_ROI1_ADD_ROI2_PULSE = 2724
    AZIMUTHAL_INTEG_PULSE = 2741


class BinMode(IntEnum):
    ACCUMULATE = 0
    AVERAGE = 1


class VFomNormalizer(IntEnum):
    AUC = 0  # area under curve
    XGM = 1
    ROI3 = 2
    ROI4 = 3
    ROI3_SUB_ROI4 = 4
    ROI3_ADD_ROI4 = 5


# a simple class saves the trouble when the attribute needs to be read/write
# from/at Redis.
class MaskState:
    UNMASK = 0
    MASK = 1
    CLEAR_MASK = -1


class _Config(dict):
    """Config implementation."""

    _system_readonly_config = {
        # detector name, leave it empty
        "DETECTOR": "",
        # Instrument name
        "TOPIC": "",
        # QTimer interval for updating plots, in milliseconds
        "PLOT_UPDATE_INTERVAL": 10,
        # QTimer interval for monitoring processes, in milliseconds
        "PROCESS_MONITOR_HEART_BEAT": 5000,
        # Maximum attempts to ping the Redis server before shutting down GUI
        "MAX_REDIS_PING_ATTEMPTS": 10,
        # timeout when cleaning up remnant processes, in second
        "PROCESS_CLEANUP_TIMEOUT": 1,
        # max number of pulses per pulse train
        "MAX_N_PULSES_PER_TRAIN": 2700,
        # maximum length of a queue in data pipeline
        "MAX_QUEUE_SIZE": 5,
        # blocking time (s) in get/put method of Queue
        "TIMEOUT": 0.1,
        # maximum number of trains in a dark run
        "MAX_DARK_TRAIN_COUNT": 999999,
        # colors of for ROI 1 to 4
        "ROI_COLORS": ['b', 'r', 'g', 'o'],
        # colors for correlation parameters 1 to 4
        "CORRELATION_COLORS": ['b', 'o', 'g', 'r'],
        # color of the bounding box used in masking and unmasking
        "MASK_BOUNDING_BOX_COLOR": 'b',
        # full path of the Redis server executable
        "REDIS_EXECUTABLE": osp.join(osp.abspath(
            osp.dirname(__file__)), "thirdparty/bin/redis-server"),
        # default REDIS port used in testing. Each detector has its
        # dedicated REDIS port.
        "REDIS_PORT": 6379,
        # maximum allowed REDIS memory (fraction of system memory)
        "REDIS_MAX_MEMORY_FRAC": 0.2,  # must <= 0.5
        # password to access the Redis server
        "REDIS_PASSWORD": "karaboFAI",  # FIXME
    }

    # system configurations which will appear in the config file so that
    # users can modify them
    _system_reconfigurable_config = {
        # Source of data: FILES or BRIDGE
        "DEFAULT_SOURCE_TYPE": DataSource.BRIDGE,
        # host name if source is FILES
        "LOCAL_HOST": "localhost",
        # port number if source is FILES
        "LOCAL_PORT": 45454,
        "GUI": {
            # color map in contour plots, valid options are: thermal, flame,
            # yellowy, bipolar, spectrum, cyclic, greyclip, grey
            "COLOR_MAP": 'thermal',
            # (width, height) of large plot window
            "PLOT_WINDOW_SIZE": [1440, 1080],
        }
    }

    _detector_readonly_config = {
        # set all default value to None
        "DEFAULT": {
            # image data dtype
            # Note: the cpp code must have the corresponding overload!!!
            #       np.float32 -> float
            #       np.float64 -> double
            "IMAGE_DTYPE": np.float32,
            # whether the data is pulse resolved
            "PULSE_RESOLVED": None,
            # whether geometry is required to assemble the detector
            "REQUIRE_GEOMETRY": None,
            # number of modules
            "NUMBER_OF_MODULES": None,
            # shape of a single module
            "MODULE_SHAPE": [None, None],
            # detector pixel size, in meter
            "PIXEL_SIZE": None,
        },
        "AGIPD": {
            "REDIS_PORT": 6378,
            "PULSE_RESOLVED": True,
            "REQUIRE_GEOMETRY": True,
            "NUMBER_OF_MODULES": 16,
            "MODULE_SHAPE": [512, 128],
            "PIXEL_SIZE": 0.2e-3,
        },
        "LPD": {
            "REDIS_PORT": 6379,
            "PULSE_RESOLVED": True,
            "REQUIRE_GEOMETRY": True,
            "NUMBER_OF_MODULES": 16,
            "MODULE_SHAPE": [256, 256],
            "PIXEL_SIZE": 0.5e-3,
        },
        "JungFrauPR": {
            "REDIS_PORT": 6383,
            "PULSE_RESOLVED": True,
            "REQUIRE_GEOMETRY": False,
            "NUMBER_OF_MODULES": 2,
            "MODULE_SHAPE": [512, 1024],
            "PIXEL_SIZE": 0.075e-3,
        },
        "JungFrau": {
            "REDIS_PORT": 6380,
            "PULSE_RESOLVED": False,
            "REQUIRE_GEOMETRY": False,
            "NUMBER_OF_MODULES": 1,
            "MODULE_SHAPE": [512, 1024],
            "PIXEL_SIZE": 0.075e-3,
        },
        "FastCCD": {
            "REDIS_PORT": 6381,
            "PULSE_RESOLVED": False,
            "REQUIRE_GEOMETRY": False,
            "NUMBER_OF_MODULES": 1,
            "MODULE_SHAPE": [1934, 960],
            "PIXEL_SIZE": 0.030e-3,
        },
        "BaslerCamera": {
            "REDIS_PORT": 6390,
            "PULSE_RESOLVED": False,
            "REQUIRE_GEOMETRY": False,
            "NUMBER_OF_MODULES": 1,
            # BaslerCamera has quite a few different models with different
            # module shapes and pixel sizes. Therefore, we do not apply the
            # module shape check.
            "MODULE_SHAPE": [-1, -1],
            # TODO: how to deal with the pixel size?
            "PIXEL_SIZE": 0.0022e-3,
        },
        "DSSC": {
            "REDIS_PORT": 6382,
            "PULSE_RESOLVED": True,
            "REQUIRE_GEOMETRY": True,
            "NUMBER_OF_MODULES": 16,
            "MODULE_SHAPE": [128, 512],
            # Hexagonal pixels, 236 μm step in fast-scan axis, 204 μm in slow-scan
            "PIXEL_SIZE": 0.22e-3,
        },

    }

    _detector_reconfigurable_config = {
        "DEFAULT": {
            # TCP address of the ZMQ bridge
            "SERVER_ADDR": "",
            # TCP port of the ZMQ bridge
            "SERVER_PORT": 0,
            # Karabo device ID + output channel name
            "SOURCE_NAME_BRIDGE": ["", ],
            # instrument source name in HDF5 files
            "SOURCE_NAME_FILE": ["", ],
            # path of the geometry file
            "GEOMETRY_FILE": "",
            # quadrant coordinates for assembling detector modules,
            # ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            "QUAD_POSITIONS": [[0, 0], [0, 0], [0, 0], [0, 0]],
            # distance from sample to detector plan (orthogonal distance,
            # not along the beam), in meter
            "SAMPLE_DISTANCE": 1.0,
            # coordinate of the point of normal incidence along the detector's
            # first dimension, in pixels, PONI1 in pyFAI
            "CENTER_Y": 0,
            # coordinate of the point of normal incidence along the detector's
            # second dimension, in pixels, PONI2 in pyFAI
            "CENTER_X": 0,
            # azimuthal integration methods supported in pyFAI
            "AZIMUTHAL_INTEG_METHODS": [
                'nosplit_csr', 'csr_ocl', 'csr', 'BBox', 'splitpixel', 'lut',
                'lut_ocl'
            ],
            # range (lower, upper) of the radial unit of azimuthal integration
            "AZIMUTHAL_INTEG_RANGE": [1e-3, 0.1],
            # number of points of azimuthal integration
            "AZIMUTHAL_INTEG_POINTS": 512,
            # pixels with values outside the (lower, upper) range will be masked
            "MASK_RANGE": [-1e5, 1e5],
            # photon energy, in keV
            "PHOTON_ENERGY": 12.4,
        },
        "AGIPD": {
            "SERVER_ADDR": '10.253.0.51',
            "SERVER_PORT": 45012,
            "SOURCE_NAME_BRIDGE": [
                'SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED',
                'MID_DET_AGIPD1M-1/CAL/APPEND_CORRECTED'],
            "SOURCE_NAME_FILE": [
                'SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED',
                'MID_DET_AGIPD1M-1/CAL/APPEND_CORRECTED'],
            "GEOMETRY_FILE": osp.join(osp.dirname(osp.abspath(__file__)),
                                      'geometries/agipd_mar18_v11.geom'),
            "QUAD_POSITIONS": [[-526, 630],
                               [-549, -4],
                               [522, -157],
                               [543, 477]],
            "AZIMUTHAL_INTEG_METHODS": [
                'BBox', 'splitpixel', 'csr', 'nosplit_csr', 'csr_ocl',
                'lut',
                'lut_ocl'
            ],
            "AZIMUTHAL_INTEG_RANGE": [1e-3, 0.1],
            "SAMPLE_DISTANCE": 5.5,
            "CENTER_Y": 490,
            "CENTER_X": 590,
            "PHOTON_ENERGY": 9.3,
        },
        "LPD": {
            "SERVER_ADDR": "10.253.0.53",
            "SERVER_PORT": 4501,
            "SOURCE_NAME_BRIDGE": ["FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED"],
            "SOURCE_NAME_FILE": ["FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED"],
            "GEOMETRY_FILE": osp.join(osp.dirname(osp.abspath(__file__)),
                                      'geometries/lpd_mar_18_axesfixed.h5'),
            # For lpd_mar_18.h5 and LPDGeometry in karabo_data
            # "QUAD_POSITIONS": [[-13.0, -299.0],
            #                    [11.0, -8.0],
            #                    [-254.0, 16.0],
            #                    [-278.0, -275.0]],
            # For lpd_mar18_axesfixed.h5 and LPD_1MGeometry in karabo_data
            # The geometry uses XFEL standard coordinate directions.
            "QUAD_POSITIONS": [[11.4, 299],
                               [-11.5, 8],
                               [254.5, -16],
                               [278.5, 275]],
            "AZIMUTHAL_INTEG_METHODS": [
                'BBox', 'splitpixel', 'csr', 'nosplit_csr', 'csr_ocl',
                'lut',
                'lut_ocl'
            ],
            "AZIMUTHAL_INTEG_RANGE": [0.2, 5],
            "SAMPLE_DISTANCE": 0.4,
            "CENTER_Y": 620,
            "CENTER_X": 570,
            "PHOTON_ENERGY": 9.3,
        },
        "JungFrauPR": {
            "SERVER_ADDR": "10.253.0.53",
            "SERVER_PORT": 4501,
            # Source name from bridge not yet known.
            "SOURCE_NAME_BRIDGE": [
                "FXE_XAD_JF1M/CAL/APPEND",
                "FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput",
                "FXE_XAD_JF1M/DET/RECEIVER-2:daqOutput",
                "FXE_XAD_JF500K/DET/RECEIVER:display",
            ],
            # Source name from files not yet known.
            "SOURCE_NAME_FILE": [
                "FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput",
                "FXE_XAD_JF1M/DET/RECEIVER-2:daqOutput",
            ],
            "AZIMUTHAL_INTEG_RANGE": [0.05, 0.4],
            "SAMPLE_DISTANCE": 2.0,
            "CENTER_Y": 512,
            "CENTER_X": 1400,
            "PHOTON_ENERGY": 9.3,
        },
        "JungFrau": {
            "SERVER_ADDR": "10.253.0.53",
            "SERVER_PORT": 4501,
            "SOURCE_NAME_BRIDGE": [
                "FXE_XAD_JF1M/DET/RECEIVER-1:display",
                "FXE_XAD_JF1M/DET/RECEIVER-2:display",
            ],
            "SOURCE_NAME_FILE": [
                "FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput",
                "FXE_XAD_JF1M/DET/RECEIVER-2:daqOutput",
                "FXE_XAD_JF500K/DET/RECEIVER:daqOutput",
                "FXE_XAD_JF1M1/DET/RECEIVER:daqOutput",
            ],
            "AZIMUTHAL_INTEG_RANGE": [0.05, 0.4],
            "SAMPLE_DISTANCE": 2.0,
            "CENTER_Y": 512,
            "CENTER_X": 1400,
            "PHOTON_ENERGY": 9.3,
        },
        "FastCCD": {
            "SERVER_ADDR": "10.253.0.140",
            "SERVER_PORT": 4502,
            "SOURCE_NAME_BRIDGE": [
                "SCS_CDIDET_FCCD2M/DAQ/FCCD:output",
            ],
            "SOURCE_NAME_FILE": [
                "SCS_CDIDET_FCCD2M/DAQ/FCCD:daqOutput",
            ],
            "AZIMUTHAL_INTEG_RANGE": [1e-3, 0.02],
            "SAMPLE_DISTANCE": 0.6,
            "CENTER_Y": 967,
            "CENTER_X": 480,
            "PHOTON_ENERGY": 0.780,
        },
        "BaslerCamera": {
            "SERVER_ADDR": "localhost",
            "SERVER_PORT": 45454,
            "SAMPLE_DISTANCE": 1.0,
            "CENTER_Y": 512,
            "CENTER_X": 512,
        },
        "DSSC": {
            "SERVER_ADDR": '10.253.0.140',
            "SERVER_PORT": 4511,
            "SOURCE_NAME_BRIDGE": [
                'SCS_CDIDET_DSSC/CAL/APPEND_CORRECTED',
                'SCS_CDIDET_DSSC/CAL/APPEND_RAW',
            ],
            "SOURCE_NAME_FILE": [
                'SCS_CDIDET_DSSC/CAL/APPEND_CORRECTED',
                'SCS_CDIDET_DSSC/CAL/APPEND_RAW',
            ],
            "GEOMETRY_FILE": osp.join(osp.dirname(osp.abspath(__file__)),
                                      'geometries/dssc_geo_june19.h5'),
            "QUAD_POSITIONS": [[-124.100,    3.112],
                               [-133.068, -110.604],
                               [   0.988, -125.236],
                               [   4.528,   -4.912]],
            "AZIMUTHAL_INTEG_METHODS": [
                'BBox', 'splitpixel', 'csr', 'nosplit_csr', 'csr_ocl',
                'lut',
                'lut_ocl'
            ],
            "AZIMUTHAL_INTEG_RANGE": [0, 0.18],
            "SAMPLE_DISTANCE": 0.6,
            "CENTER_Y": 686,
            "CENTER_X": 550,
            "PHOTON_ENERGY": 0.780,
        },
    }

    _filename = osp.join(ROOT_PATH, "config.json")

    detectors = list(_detector_readonly_config.keys())
    detectors.remove('DEFAULT')

    def __init__(self):
        super().__init__()

        self.update(self._system_readonly_config)
        self.update(self._system_reconfigurable_config)
        # to allow unittest without specifying detector
        self.update(self._detector_readonly_config['DEFAULT'])
        self.update(self._detector_reconfigurable_config['DEFAULT'])

        self.ensure_file()

    def ensure_file(self):
        """Generate the config file if it does not exist."""
        if not osp.isfile(self._filename):
            cfg = copy.deepcopy(self._system_reconfigurable_config)

            for det in self.detectors:
                cfg[det] = copy.deepcopy(
                    self._detector_reconfigurable_config['DEFAULT'])
                cfg[det].update(self._detector_reconfigurable_config[det])
            with open(self._filename, 'w') as fp:
                json.dump(cfg, fp, indent=4)

    def load(self, detector):
        """Update the config from the config file.

        :param str detector: detector name.
        """
        self.__setitem__("DETECTOR", detector)
        # config (self) does not have a detector hierarchy!
        self.update(self._detector_readonly_config[detector])
        self.update(self._detector_reconfigurable_config[detector])
        self.from_file(detector)

    def set_topic(self, topic):
        """Set the topic key in system read only config

        :param str topic: topic name.
        """
        self.__setitem__("TOPIC", topic)

    def from_file(self, detector):
        """Update the config dictionary from the config file."""
        with open(self._filename, 'r') as fp:
            try:
                # FIXME: what if the file is wrong formatted
                config_from_file = json.load(fp)
            except json.decoder.JSONDecodeError as e:
                logger.error(f"Invalid config file: {self._filename}")
                raise

        # check system configuration
        invalid_keys = []
        for key in config_from_file:
            if key not in self.detectors and \
                    key not in self._system_reconfigurable_config:
                invalid_keys.append(key)

        # check detector configuration
        if detector not in config_from_file:
            msg = f"\nThe detector config cannot be found in the config " \
                  f"file:\n\n{self._filename}.\n\nThis could be caused by a " \
                  f"version update.\nCreate a new config file? The default " \
                  f"config will be used otherwise."

            if query_yes_no(msg):
                self._generate_new_config_file()
                with open(self._filename, 'r') as fp:
                    config_from_file = json.load(fp)

            # Whether the new config file is generated or not, the default
            # configuration has already been applied.
            return

        # validate keys in config file
        for key in config_from_file[detector]:
            if key not in self._detector_reconfigurable_config['DEFAULT']:
                invalid_keys.append(f"{detector}.{key}")

        if invalid_keys:
            msg = f"\nThe following invalid keys were found in " \
                f"{self._filename}:\n\n{', '.join(invalid_keys)}.\n\n" \
                f"This could be caused by a version update.\n" \
                f"Create a new config file?"

            if not query_yes_no(msg):
                raise ValueError(
                    f"Invalid config keys: {', '.join(invalid_keys)}")

            self._generate_new_config_file()

            # We use the default configuration and there is no need to read
            # the newly generated config file.
            return

        # update system configuration
        for key in self._system_reconfigurable_config:
            if key in config_from_file:
                self.__setitem__(key, config_from_file[key])

        # update detector configuration
        if detector in config_from_file:
            self.update(config_from_file[detector])

    def _generate_new_config_file(self):
        backup_file = self._filename + ".bak"
        if not osp.exists(backup_file):
            shutil.move(self._filename, backup_file)
        else:
            # up to two backup files
            shutil.move(backup_file, backup_file + '.bak')
            shutil.move(self._filename, backup_file)

        # generate a new config file
        self.ensure_file()


class ConfigWrapper(collections.Mapping):
    """Readonly config."""
    def __init__(self):
        self._data = _Config()

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def load(self, detector):
        self._data.load(detector)

    def set_topic(self, topic):
        self._data.set_topic(topic)

    @property
    def detectors(self):
        return _Config.detectors

    @staticmethod
    def parse_detector_name(detector):
        if detector == 'JUNGFRAU':
            return 'JungFrau'

        if detector == 'FASTCCD':
            return 'FastCCD'

        if detector == 'BASLERCAMERA':
            return 'BaslerCamera'

        if detector == 'JUNGFRAUPR':
            return 'JungFrauPR'

        return detector.upper()


config = ConfigWrapper()  # global configuration
