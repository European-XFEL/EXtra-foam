"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from enum import Enum, IntEnum
import os.path as osp
import shutil
from collections import abc, namedtuple, OrderedDict

import yaml
from yaml.scanner import ScannerError
from yaml.parser import ParserError

import numpy as np

from . import ROOT_PATH
from .logger import logger


_MAX_INT32 = np.iinfo(np.int32).max
_MIN_INT32 = np.iinfo(np.int32).min


class ExtensionType(Enum):
    ALL_OUTPUT = "endpoint"
    DETECTOR_OUTPUT = "detector_endpoint"


class DataSource(IntEnum):
    FILE = 0  # data from files (run directory)
    BRIDGE = 1  # real-time data from the bridge
    UNKNOWN = 2  # not specified


class KaraboType(IntEnum):
    CONTROL_DATA = 0
    PIPELINE_DATA = 1


class PumpProbeMode(IntEnum):
    UNDEFINED = 0
    REFERENCE_AS_OFF = 1  # use pre-defined reference image
    SAME_TRAIN = 2  # on-/off- pulses in the same train
    EVEN_TRAIN_ON = 3  # on-/off- pulses have even/odd train IDs, respectively
    ODD_TRAIN_ON = 4  # on/-off- pulses have odd/even train IDs, respectively


class RoiCombo(IntEnum):
    UNDEFINED = 0
    ROI1 = 1
    ROI2 = 2
    ROI1_SUB_ROI2 = 3
    ROI1_ADD_ROI2 = 4
    ROI1_DIV_ROI2 = 5
    ROI3 = 11
    ROI4 = 12
    ROI3_SUB_ROI4 = 13
    ROI3_ADD_ROI4 = 14


class RoiFom(IntEnum):
    SUM = 1
    MEAN = 2
    MEDIAN = 3
    MAX = 4
    MIN = 5
    STD = 6  # standard deviation
    VAR = 7  # variance
    N_STD = 16  # normalized standard deviation (speckle contrast)
    N_VAR = 17  # normalized variance (speckle contrast)


class RoiProjType(IntEnum):
    SUM = 1
    MEAN = 2


class AnalysisType(IntEnum):
    UNDEFINED = 0
    PUMP_PROBE = 1
    ROI_FOM = 11
    ROI_NORM = 12
    ROI_PROJ = 21
    AZIMUTHAL_INTEG = 41
    AZIMUTHAL_INTEG_PEAK = 42
    AZIMUTHAL_INTEG_PEAK_Q = 43
    AZIMUTHAL_INTEG_COM = 44
    PULSE = 2700
    PUMP_PROBE_PULSE = 2701
    ROI_FOM_PULSE = 2711
    ROI_NORM_PULSE = 2712
    ROI_PROJ_PULSE = 2721
    AZIMUTHAL_INTEG_PULSE = 2741


class GeomAssembler(IntEnum):
    OWN = 1  # use Extra-foam own geometry assembler
    EXTRA_GEOM = 2  # use Extra-geom geometry assembler


class ImageTransformType(IntEnum):
    CONCENTRIC_RINGS = 0
    FOURIER_TRANSFORM = 1
    EDGE_DETECTION = 2
    BRAGG_PEAK_ANALYSIS = 3
    UNDEFINED = 255


class PipelineSlowPolicy(IntEnum):
    DROP = 0
    WAIT = 1


class CalibrationOffsetPolicy(IntEnum):
    UNDEFINED = 0
    INTRA_DARK = 1


_user_analysis_types = OrderedDict({
    "": AnalysisType.UNDEFINED,
    "pump-probe": AnalysisType.PUMP_PROBE,
    "ROI FOM": AnalysisType.ROI_FOM,
    "ROI proj": AnalysisType.ROI_PROJ,
    "azimuthal integ (sum)": AnalysisType.AZIMUTHAL_INTEG,
    "azimuthal integ (peak)": AnalysisType.AZIMUTHAL_INTEG_PEAK,
    "azimuthal integ (peak q)": AnalysisType.AZIMUTHAL_INTEG_PEAK_Q,
    "azimuthal integ (CoM)": AnalysisType.AZIMUTHAL_INTEG_COM
})


def get_analysis_types(without=None):
    if without is not None:
        return { k: v for k, v in _user_analysis_types.items()
                 if v != without }
    else:
        return _user_analysis_types


def list_azimuthal_integ_methods(detector):
    """Return a list of available azimuthal integration methos.

    :param str detector: detector name
    """
    if detector in ['AGIPD', 'DSSC', 'LPD']:
        return ['BBox', 'splitpixel', 'csr', 'nosplit_csr', 'csr_ocl',
                'lut', 'lut_ocl']
    return ['nosplit_csr', 'csr_ocl', 'csr', 'BBox', 'splitpixel',
            'lut', 'lut_ocl']


_PlotLabelItem = namedtuple("_PlotLabel", ['x', 'y'])


class PlotLabel(abc.Mapping):
    """Labels used in data visualization."""
    _labels = {
        AnalysisType.ROI_PROJ: _PlotLabelItem("x", "Projection"),
        AnalysisType.AZIMUTHAL_INTEG: _PlotLabelItem(
            "Momentum transfer (1/A)", "Scattering signal (arb. u.)")
    }

    def __init__(self):
        super().__init__()

        for item in AnalysisType:
            if item not in self._labels:
                self._labels[item] = _PlotLabelItem("", "")

    def __contains__(self, analysis_type):
        """Override."""
        return analysis_type in self._labels

    def __getitem__(self, analysis_type):
        """Override."""
        return self._labels[analysis_type]

    def __iter__(self):
        """Override."""
        return iter(self._labels)

    def __len__(self):
        """Override."""
        return len(self._labels)


plot_labels = PlotLabel()


class BinMode(IntEnum):
    ACCUMULATE = 0
    AVERAGE = 1


class Normalizer(IntEnum):
    UNDEFINED = 0
    AUC = 1  # area under curve
    XGM = 2
    DIGITIZER = 3
    ROI = 4


# a simple class saves the trouble when the attribute needs to be read/write
# from/at Redis.
class MaskState:
    UNMASK = 0
    MASK = 1
    CLEAR_MASK = -1


class _Config(dict):
    """Config implementation."""

    _abs_dirpath = osp.abspath(osp.dirname(__file__))

    _opts = {
        # instrument name
        "TOPIC": "",
        # -------------------------------------------------------------
        # Main detector
        # -------------------------------------------------------------
        # main detector name
        "DETECTOR": "",
        # Whether pulse-resolved analysis setup is used
        "PULSE_RESOLVED": False,
        # whether a geometry is required to assemble the detector modules
        "REQUIRE_GEOMETRY": False,
        # absolute path of geometry file
        "GEOMETRY_FILE": "",
        # Note: due to the historical reason, we have "QUAD_POSITIONS" in the
        #       config file for a long time. However, it is only used in
        #       1M detectors geometry, while "MODULE_POSITIONS", which is
        #       introduced in version 0.8.3, is used in general detector
        #       geometry.
        # quadrant coordinates for assembling detector modules,
        # ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
        "QUAD_POSITIONS": ((0, 0), ) * 4,
        # module coordinates for assembling detector modules,
        # a tuple of (x, y), we assume 12 modules will be used at maximum
        "MODULE_POSITIONS": ((0, 0),) * 12,
        # number of modules
        "NUMBER_OF_MODULES": 1,
        # shape of a single module
        "MODULE_SHAPE": (-1, -1),
        # detector pixel size, in meter
        "PIXEL_SIZE": 1.e-3,
        # whether the detector supports masking edge pixels of tiles
        "MASK_TILE_EDGE": False,
        # whether the detector supports masking edge pixels of asics
        "MASK_ASIC_EDGE": False,
        # Default TCP address when data source is BRIDGE
        "BRIDGE_ADDR": "127.0.0.1",
        # Default TCP port when data source is BRIDGE
        "BRIDGE_PORT": 4501,
        # Default TCP address when data source is FILE
        "LOCAL_ADDR": "127.0.0.1",
        # Default TCP port when data source is FILE
        "LOCAL_PORT": 45454,
        # distance from sample to detector plan (orthogonal distance,
        # not along the beam), in meter
        "SAMPLE_DISTANCE": 1.0,
        # photon energy, in keV
        "PHOTON_ENERGY": 12.4,
        # -------------------------------------------------------------
        # Process
        # -------------------------------------------------------------
        # timeout when cleaning up remnant processes, in second
        "PROCESS_CLEANUP_TIMEOUT": 1,
        # interval for updating the status of all processes, in milliseconds
        "PROCESS_MONITOR_UPDATE_TIMER": 5000,
        # -------------------------------------------------------------
        # Pipeline
        # -------------------------------------------------------------
        # maximum length of the queue in data pipeline (the smaller the queue
        # size, the smaller the latency)
        "PIPELINE_MAX_QUEUE_SIZE": 2,
        "PIPELINE_SLOW_POLICY": PipelineSlowPolicy.DROP,
        # timeout of the zmq bridge, in second
        "BRIDGE_TIMEOUT": 0.1,
        # default extension port
        "EXTENSION_PORT": 5555,
        # size of cache when correlating data arriving at different time
        # with train ID
        "TRANSFORMER_CACHE_SIZE": 20,
        # -------------------------------------------------------------
        # Source of data
        # -------------------------------------------------------------
        # default source type: online or from file
        "SOURCE_DEFAULT_TYPE": int(DataSource.BRIDGE),
        # dtype of the processed image data
        # Note: the cpp code must have the corresponding overload!!!
        #       np.float32 -> float
        #       np.float64 -> double
        "SOURCE_PROC_IMAGE_DTYPE": np.float32,
        # dtype of the raw image data
        "SOURCE_RAW_IMAGE_DTYPE": np.uint16,
        # interval for updating available data sources, in milliseconds
        "SOURCE_AVAIL_UPDATE_TIMER": 1000,
        # After how long the available sources key expires, in seconds
        "SOURCE_EXPIRATION_TIMER": 2,
        "SOURCE_USER_DEFINED_CATEGORY": "User-defined",
        # -------------------------------------------------------------
        # REDIS
        # -------------------------------------------------------------
        # absolute path of the Redis server executable
        "REDIS_EXECUTABLE": osp.join(_abs_dirpath, "thirdparty/bin/redis-server"),
        # default Redis port used in testing. Each detector has its
        # dedicated Redis port.
        "REDIS_PORT": 6379,
        # maximum attempts to ping the Redis server before shutting down the app
        "REDIS_MAX_PING_ATTEMPTS": 3,
        # interval for pinging the Redis server from the main GUI, in milliseconds
        "REDIS_PING_ATTEMPT_INTERVAL": 5000,
        # If the path is given, redis-py will use UnixDomainSocketConnection.
        # Otherwise, normal TCP socket connection.
        "REDIS_UNIX_DOMAIN_SOCKET_PATH": "",
        # maximum allowed REDIS memory (fraction of system memory)
        "REDIS_MAX_MEMORY_FRAC": 0.2,  # must <= 0.5
        # password to access the Redis server
        "REDIS_PASSWORD": "karaboFAI",  # FIXME
        "REDIS_LOGFILE": osp.join(ROOT_PATH, "redis.log"),
        # -------------------------------------------------------------
        # GUI
        # -------------------------------------------------------------
        # interval for polling new processed data, in milliseconds
        "GUI_PLOT_UPDATE_TIMER": 10,
        # interval for updating plots with state, in milliseconds
        # (for example, correlation, heatmap, etc.)
        "GUI_PLOT_WITH_STATE_UPDATE_TIMER": 1000,
        # initial (width, height) of the main GUI window
        "GUI_MAIN_GUI_SIZE": (1720, 1080),
        # initial (width, height) of the image tool window
        "GUI_IMAGE_TOOL_SIZE": (1720, 1080),
        # initial (width, height) of a large plot window
        "GUI_PLOT_WINDOW_SIZE": (1520, 1080),
        # color map in contour plots, valid options are: thermal, flame,
        # yellowy, bipolar, spectrum, cyclic, greyclip, grey, viridis,
        # inferno, plasma, magma
        "GUI_COLOR_MAP": 'plasma',
        # foreground/background color (r, g, b, alpha)
        "GUI_FOREGROUND_COLOR": (0, 0, 0, 255),
        "GUI_BACKGROUND_COLOR": (225, 225, 225, 255),
        # colors of for ROI bounding boxes 1 to 4
        "GUI_ROI_COLORS": ('b', 'r', 'g', 'o'),
        # colors (master, slave) for correlation plots 1 and 2
        "GUI_CORRELATION_COLORS": (('b', 'r'), ('g', 'p')),
        # color of the image mask bounding box while drawing
        "GUI_MASK_BOUNDING_BOX_COLOR": 'b',
        # color of the masked area for MaskItem
        "GUI_MASK_FILL_COLOR": 'g',
        # -------------------------------------------------------------
        # Misc
        # -------------------------------------------------------------
        # max number of pulses per pulse train
        "MAX_N_PULSES_PER_TRAIN": 2700,
    }

    _AreaDetectorConfig = namedtuple("_AreaDetectorConfig", [
        "REDIS_PORT", "PULSE_RESOLVED", "REQUIRE_GEOMETRY",
        "NUMBER_OF_MODULES", "MODULE_SHAPE", "PIXEL_SIZE",
        "MASK_TILE_EDGE", "MASK_ASIC_EDGE",
    ])

    # "name" is only used for labeling, it is ignored in the pipeline code.
    StreamerEndpointItem = namedtuple("StreamerEndpointItem",
                                      ["name", "type", "address", "port"])

    for key in _AreaDetectorConfig._fields:
        assert key in _opts

    _detector_config = {
        "AGIPD": _AreaDetectorConfig(
            REDIS_PORT=6378,
            PULSE_RESOLVED=True,
            REQUIRE_GEOMETRY=True,
            NUMBER_OF_MODULES=16,
            MODULE_SHAPE=(512, 128),
            PIXEL_SIZE=0.2e-3,
            MASK_TILE_EDGE=True,
            MASK_ASIC_EDGE=False,
        ),
        "LPD": _AreaDetectorConfig(
            REDIS_PORT=6379,
            PULSE_RESOLVED=True,
            REQUIRE_GEOMETRY=True,
            NUMBER_OF_MODULES=16,
            MODULE_SHAPE=(256, 256),
            PIXEL_SIZE=0.5e-3,
            MASK_TILE_EDGE=True,
            MASK_ASIC_EDGE=False,
        ),
        # DSSC has hexagonal pixels:
        # 236 μm step in fast-scan axis, 204 μm in slow-scan
        "DSSC": _AreaDetectorConfig(
            REDIS_PORT=6380,
            PULSE_RESOLVED=True,
            REQUIRE_GEOMETRY=True,
            NUMBER_OF_MODULES=16,
            MODULE_SHAPE=(128, 512),
            PIXEL_SIZE=0.22e-3,
            MASK_TILE_EDGE=True,
            MASK_ASIC_EDGE=False,
        ),
        "JungFrau": _AreaDetectorConfig(
            REDIS_PORT=6381,
            PULSE_RESOLVED=True,
            REQUIRE_GEOMETRY=False,
            NUMBER_OF_MODULES=1,
            MODULE_SHAPE=(512, 1024),
            PIXEL_SIZE=0.075e-3,
            MASK_TILE_EDGE=False,
            MASK_ASIC_EDGE=True,
        ),
        "FastCCD": _AreaDetectorConfig(
            REDIS_PORT=6383,
            PULSE_RESOLVED=False,
            REQUIRE_GEOMETRY=False,
            NUMBER_OF_MODULES=1,
            MODULE_SHAPE=(1934, 960),
            PIXEL_SIZE=0.030e-3,
            MASK_TILE_EDGE=False,
            MASK_ASIC_EDGE=False,
        ),
        "ePix100": _AreaDetectorConfig(
            REDIS_PORT=6384,
            PULSE_RESOLVED=False,
            REQUIRE_GEOMETRY=False,
            NUMBER_OF_MODULES=1,
            MODULE_SHAPE=(708, 768),
            PIXEL_SIZE=0.050e-3,
            MASK_TILE_EDGE=False,
            MASK_ASIC_EDGE=True,
        ),
        "BaslerCamera": _AreaDetectorConfig(
            REDIS_PORT=6389,
            PULSE_RESOLVED=False,
            REQUIRE_GEOMETRY=False,
            NUMBER_OF_MODULES=1,
            # BaslerCamera has quite a few different models with different
            # module shapes and pixel sizes.
            MODULE_SHAPE=(-1, -1),
            PIXEL_SIZE=0.0022e-3,
            MASK_TILE_EDGE=False,
            MASK_ASIC_EDGE=False,
        ),
    }

    _misc_source_categories = (
        'XGM', 'Digitizer', 'Magnet', 'Motor', 'Monochromator')

    def __init__(self):
        super().__init__()

        self.update(self._opts)

        # ctrl data sources listed in the DataSourceWidget
        self.control_sources = dict()
        # pipeline data sources listed in the DataSourceWidget
        self.pipeline_sources = dict()

        self.appendix_streamers = []

    @classmethod
    def topics(cls):
        return 'SPB', 'FXE', 'SCS', 'SQS', 'MID', 'HED'

    @classmethod
    def detectors(cls):
        return tuple(cls._detector_config.keys())

    @classmethod
    def config_file(cls, topic):
        return osp.join(ROOT_PATH, f"{topic.lower()}.config.yaml")

    def ensure_file(self, topic):
        """Generate the config file if it does not exist."""
        config_file = self.config_file(topic)
        if not osp.isfile(config_file):
            shutil.copyfile(osp.join(
                self._abs_dirpath, f"configs/{osp.basename(config_file)}"),
                config_file)

    def load(self, det, topic, **kwargs):
        """Update configs from the config file.

        :param str det: detector name.
        :param str topic: topic name
        """
        self.ensure_file(topic)

        self.__setitem__("DETECTOR", det)
        self.__setitem__("TOPIC", topic)

        # update the configurations of the main detector
        for k, v in self._detector_config[det]._asdict().items():
            self[k] = v

        self.from_file(det, topic)

        # kwargs may contain the configuration for the main detector
        for k, v in kwargs.items():
            if k in self:
                self.__setitem__(k, v)
            else:
                raise KeyError

    def from_file(self, detector, topic):
        """Read configs from the config file."""
        config_file = self.config_file(topic)
        with open(config_file, 'r') as fp:
            try:
                cfg = yaml.load(fp, Loader=yaml.Loader)
            except (ScannerError, ParserError) as e:
                msg = f"Invalid config file: {config_file}\n{repr(e)}"
                logger.error(msg)
                raise OSError(msg)

        if cfg is None:
            raise ValueError(f"Config file {config_file} is empty: "
                             f"returned 'None' from the loader!")

        # update the main detector config
        # TODO: may add "MODULE_POSITIONS" in the future
        det_cfg = cfg.get("DETECTOR", dict()).get(detector, dict())
        for key in ["GEOMETRY_FILE", "QUAD_POSITIONS", "BRIDGE_ADDR",
                    "BRIDGE_PORT", "LOCAL_ADDR", "LOCAL_PORT",
                    "SAMPLE_DISTANCE", "PHOTON_ENERGY"]:
            if key in det_cfg:
                # convert 'GEOMETRY_FILE' to absolute path if it is not
                if key == "GEOMETRY_FILE":
                    geom_path = det_cfg[key]
                    if not osp.isabs(geom_path):
                        det_cfg[key] = osp.join(
                            self._abs_dirpath, 'geometries', geom_path)
                elif key == "QUAD_POSITIONS":
                    pos = det_cfg[key]
                    det_cfg[key] = ((pos['x1'], pos['y1']),
                                    (pos['x2'], pos['y2']),
                                    (pos['x3'], pos['y3']),
                                    (pos['x4'], pos['y4']))

                self[key] = det_cfg[key]

        # update data sources
        src_cfg = cfg.get("SOURCE", dict())

        self["SOURCE_DEFAULT_TYPE"] = src_cfg["DEFAULT_TYPE"]
        for ctg, srcs in src_cfg.get("CATEGORY", dict()).items():
            if ctg == "JungFrauPR":
                logger.warning(
                    f"Found 'JungFrauPR' in {config_file}\n\n"
                    f"Please modify your config file: \n"
                    f"1. Move non-duplicated data sources under 'JungFrauPR' "
                    f"into 'JungFrau';\n"
                    f"2. Remove JungFrauPR from 'SOURCE' and 'DETECTOR'.\n")

                if "JungFrau" not in src_cfg.get("CATEGORY", dict()):
                    # try to import sources under JungFrauPR later
                    ctg = "JungFrau"

            # We assume the main detector type is exclusive, i.e., only one
            # of them will be needed.
            if ctg == detector or ctg in self._misc_source_categories:
                self.control_sources[ctg] = srcs.get("CONTROL", dict())
                self.pipeline_sources[ctg] = srcs.get("PIPELINE", dict())

                if ctg == "JungFrau":
                    # try to merge "JungFrauPR"
                    self.pipeline_sources[ctg].update(
                        src_cfg.get("CATEGORY", dict()).get(
                            "JungFrauPR", dict()).get(
                            "PIPELINE", dict()))

            elif ctg not in self.detectors() and ctg not in ("JungFrauPR",):
                raise ValueError(
                    f"Invalid source category: {ctg}!\n"
                    f"The valid source categories are: "
                    f"{self._misc_source_categories + self.detectors()}")

        # update connection
        con_cfg = cfg.get("STREAMER", dict())
        stream_cfg = con_cfg.get("ZMQ", dict())
        for name, opts in stream_cfg.items():
            self.appendix_streamers.append(
                self.StreamerEndpointItem(
                    name, opts["DEFAULT_TYPE"], opts["ADDR"], opts["PORT"]))


class ConfigWrapper(abc.Mapping):
    """Readonly config."""
    def __init__(self):
        super().__init__()
        self._data = _Config()

    def __contains__(self, key):
        """Override."""
        return self._data.__contains__(key)

    def __getitem__(self, key):
        """Override."""
        return self._data.__getitem__(key)

    def __len__(self):
        """Override."""
        return self._data.__len__()

    def __iter__(self):
        """Override."""
        return self._data.__iter__()

    def load(self, detector, topic, **kwargs):
        self._data.load(detector, topic, **kwargs)

    @property
    def detectors(self):
        return _Config.detectors()

    @property
    def topics(self):
        return _Config.topics()

    @property
    def config_file(self):
        topic = self._data['TOPIC']
        if topic:
            return _Config.config_file(topic)
        raise ValueError("TOPIC is not specified!")

    @property
    def setup_file(self):
        detector = self._data['DETECTOR']
        return osp.join(ROOT_PATH, f".{detector.lower()}.setup.yaml")

    @property
    def control_sources(self):
        return self._data.control_sources

    @property
    def pipeline_sources(self):
        return self._data.pipeline_sources

    @property
    def appendix_streamers(self):
        return self._data.appendix_streamers

    @staticmethod
    def parse_detector_name(detector):
        if detector == 'JUNGFRAU':
            return 'JungFrau'

        if detector == 'FASTCCD':
            return 'FastCCD'

        if detector == 'BASLERCAMERA':
            return 'BaslerCamera'

        if detector == 'EPIX100':
            return 'ePix100'

        return detector.upper()


config = ConfigWrapper()  # global configuration
