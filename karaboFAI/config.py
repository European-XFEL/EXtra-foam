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
import os
import os.path as osp
import collections

import redis

from . import ROOT_PATH
from .logger import logger


class DataSource(IntEnum):
    FILE = 0  # data from files (run directory)
    BRIDGE = 1  # real-time data from the bridge


class PumpProbeMode(IntEnum):
    UNDEFINED = 0
    PRE_DEFINED_OFF = 1  # use pre-defined reference image
    SAME_TRAIN = 2  # on-/off- pulses in the same train
    EVEN_TRAIN_ON = 3  # on-/off- pulses have even/odd train IDs, respectively
    ODD_TRAIN_ON = 4  # on/-off- pulses have odd/even train IDs, respectively


class PumpProbeType(IntEnum):
    AZIMUTHAL_INTEG = 0
    ROI = 1
    ROI_PROJECTION_X = 2
    ROI_PROJECTION_Y = 3
    ROI1_BY_ROI2 = 4


class AnalysisType(IntEnum):
    UNDEFINED = 0
    AZIMUTHAL_INTEG = 1
    ROI_PROJECTION_X = 2
    ROI_PROJECTION_Y = 3


class BinMode(IntEnum):
    ACCUMULATE = 0
    AVERAGE = 1


class CorrelationFom(IntEnum):
    UNDEFINED = 0
    PUMP_PROBE_FOM = 1
    ROI_SUB = 2  # ROI1 - ROI2
    ROI1 = 3
    ROI2 = 4
    ROI_SUM = 5  # ROI1 + ROI2
    # Calculate the FOM based on the azimuthal integration of the mean
    # of the assembled image(s).
    AZIMUTHAL_INTEG_MEAN = 6


class AiNormalizer(IntEnum):
    # Normalize the azimuthal integration curve by the area under the curve.
    AUC = 0
    # Normalize the azimuthal integration curve by the sum of ROI(s).
    ROI_SUB = 1  # ROI1 - ROI2
    ROI1 = 2
    ROI2 = 3
    ROI_SUM = 4  # ROI1 + ROI2


class Projection1dNormalizer(IntEnum):
    AUC = 0


class RoiFom(IntEnum):
    SUM = 0  # monitor sum of ROI
    MEAN = 1  # monitor mean of ROI


class _Config(dict):
    """Config implementation."""

    _system_readonly_config = {
        # detector name, leave it empty
        "DETECTOR": "",
        # QTimer interval in milliseconds
        "TIMER_INTERVAL": 20,
        # maximum length of a queue in data pipeline
        "MAX_QUEUE_SIZE": 2,
        # blocking time (s) in get/put method of Queue
        "TIMEOUT": 0.1,
        # colors of for ROI 1 to 4
        "ROI_COLORS": ['c', 'b', 'o', 'y'],
        # colors for correlation parameters 1 to 4
        "CORRELATION_COLORS": ['c', 'b', 'o', 'y'],
        # colors for binning plots 1 to 3
        "BIN_COLORS": ['c', 'b', 'o'],
        "REDIS": {
            # full path of the Redis server executable
            "EXECUTABLE": osp.join(osp.abspath(
                osp.dirname(__file__)), "thirdparty/bin/redis-server"),
            # password to access the Redis server
            "PASSWORD": "karaboFAI",  # FIXME
            # default port of the Redis server
            "PORT": 6379,
        }
    }

    # system configuration which users are allowed to modify
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
            "PULSE_RESOLVED": True,
            "REQUIRE_GEOMETRY": True,
            "NUMBER_OF_MODULES": 16,
            "MODULE_SHAPE": [512, 128],
            "PIXEL_SIZE": 0.2e-3,
        },
        "LPD": {
            "PULSE_RESOLVED": True,
            "REQUIRE_GEOMETRY": True,
            "NUMBER_OF_MODULES": 16,
            "MODULE_SHAPE": [256, 256],
            "PIXEL_SIZE": 0.5e-3,
        },
        "JungFrau": {
            "PULSE_RESOLVED": False,
            "REQUIRE_GEOMETRY": False,
            "NUMBER_OF_MODULES": 1,
            "MODULE_SHAPE": [512, 1024],
            "PIXEL_SIZE": 0.075e-3,
        },
        "FastCCD": {
            "PULSE_RESOLVED": False,
            "REQUIRE_GEOMETRY": False,
            "NUMBER_OF_MODULES": 1,
            "MODULE_SHAPE": [1934, 960],
            "PIXEL_SIZE": 0.030e-3,
        },
        "BaslerCamera": {
            "PULSE_RESOLVED": False,
            "REQUIRE_GEOMETRY": False,
            "NUMBER_OF_MODULES": 1,
            "MODULE_SHAPE": [1024, 1024],
            "PIXEL_SIZE": 0.0022e-3,
        }
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
            # data folder if streamed from files
            "DATA_FOLDER": "",
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
                'SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED'],
            "SOURCE_NAME_FILE": ['SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED'],
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
                                      'geometries/lpd_mar_18.h5'),
            "QUAD_POSITIONS": [[-13.0, -299.0],
                               [11.0, -8.0],
                               [-254.0, 16.0],
                               [-278.0, -275.0]],
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
        }
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
        self.update(self._detector_readonly_config[detector])
        self.from_file(detector)

    def from_file(self, detector):
        """Update the config dictionary from the config file."""
        with open(self._filename, 'r') as fp:
            try:
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
        for key in config_from_file[detector]:
            if key not in self._detector_reconfigurable_config['DEFAULT']:
                invalid_keys.append(f"{detector}.{key}")

        if invalid_keys:
            msg = f"The following invalid keys were found in " \
                f"{self._filename}:\n{', '.join(invalid_keys)}"
            logger.error(msg)
            raise ValueError(msg)

        # update system configuration
        for key in self._system_reconfigurable_config:
            if key in config_from_file:
                self.__setitem__(key, config_from_file[key])

        # update detector configuration
        self.update(config_from_file[detector])


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

    def load(self, detector, **kwargs):
        self._data.load(detector, **kwargs)

    @property
    def detectors(self):
        return _Config.detectors


config = ConfigWrapper()  # global configuration

REDIS_CONNECTION = None


def redis_connection():
    """Return a Redis connection."""
    global REDIS_CONNECTION
    if REDIS_CONNECTION is None:
        redis_cfg = config['REDIS']
        connection = redis.Redis('localhost', redis_cfg['PORT'],
                                 password=redis_cfg['PASSWORD'],
                                 decode_responses=True)
        REDIS_CONNECTION = connection

    return REDIS_CONNECTION


REDIS_CONNECTION_BYTES = None


def redis_connection_bytes():
    """Return a Redis connection."""
    global REDIS_CONNECTION_BYTES
    if REDIS_CONNECTION_BYTES is None:
        redis_cfg = config['REDIS']
        connection = redis.Redis('localhost', redis_cfg['PORT'],
                                 password=redis_cfg['PASSWORD'],
                                 decode_responses=False)
        REDIS_CONNECTION_BYTES = connection

    return REDIS_CONNECTION_BYTES
