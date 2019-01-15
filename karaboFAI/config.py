"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold configuration related objects.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import configparser


# root path for storing config and log files
ROOT_PATH = os.path.join(os.path.expanduser("~"), ".karaboFAI")
if not os.path.isdir(ROOT_PATH):
    os.makedirs(ROOT_PATH)


class UpperCaseConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr.upper()


class Config(dict):
    """Config class.

    The default topic config, e.g. _default_spb_config, should be the
    config used in the corresponding experimental hutch on the online
    cluster.

    The config file should hold the config used in users' local PCs,
    typically for offline analysis and tests.
    """
    # miscellaneous
    # -------------
    # COLOR_MAP str: color map in contour plots, valid options are:
    #                thermal, flame, yellowy, bipolar, spectrum, cyclic,
    #                greyclip, grey
    #
    # data pipeline setup
    # -------------------
    # MASK_RANGE tuple: pixels with values outside the (lower, upper) range
    #                   will be masked
    # TIMER_INTERVAL int: QTimer interval in milliseconds.
    # MAX_QUEUE_SIZE int: maximum length of data acquisition and processing
    #                     queues in data pipeline
    # TIMEOUT int: block time (s) in Queue.get() and Queue.put() methods
    #
    # networking
    # ----------
    # SERVER_ADDR str: TCP address of the ZMQ bridge
    # SERVER_PORT int: TCP port of the ZMQ bridge
    # SOURCE_NAME str: PipeToZeroMQ device ID / folder of the HDF5 data files
    # SOURCE_TYPE int: see data_processing.data_model.DataSource
    #
    # azimuthal integration
    # ---------------------
    # QUAD_POSITIONS tuple: quadrant coordinates for assembling detector
    #                       modules, ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    # GEOMETRY_FILE str: path of the geometry file of the detector
    # INTEGRATION_METHODS list: azimuthal integration methods supported
    #                           in pyFAI
    # INTEGRATION_RANGE tuple: (lower, upper) range of the radial unit of
    #                          azimuthal integration
    # INTEGRATION_POINTS int: number of points in the output pattern of
    #                         azimuthal integration
    #
    # experiment setup
    # ----------------
    # PHOTON_ENERGY float: photon energy, in keV
    # DISTANCE float: distance from sample - detector plan (orthogonal
    #                 distance, not along the beam), in meter
    # CENTER_Y int: coordinate of the point of normal incidence along the
    #               detector's first dimension, in pixels
    # CENTER_X int: coordinate of the point of normal incidence along the
    #               detector's second dimension, in pixels
    # PIXEL_SIZE float: detector pixel size, in meter

    # system config should not appear in the topic config
    _default_sys_config = {
        "TOPIC": '',  # topic name, leave it empty
        "TIMER_INTERVAL": 20,
        "MAX_QUEUE_SIZE": 2,
        "TIMEOUT": 0.1,
        "COLOR_MAP": 'thermal',
    }

    # this is to guard again the topic config defined in the file modifying
    # the system config
    _allowed_topic_config_keys = (
        "SERVER_ADDR",
        "SERVER_PORT",
        "SOURCE_NAME",
        "SOURCE_TYPE",
        "GEOMETRY_FILE",
        "QUAD_POSITIONS",
        "INTEGRATION_METHODS",
        "INTEGRATION_RANGE",
        "INTEGRATION_POINTS",
        "PHOTON_ENERGY",
        "DISTANCE",
        "CENTER_Y",
        "CENTER_X",
        "PIXEL_SIZE",
        "COLOR_MAP",
        "MASK_RANGE"
    )

    # In order to pass the test, the default topic config must include
    # all the keys in '_allowed_topic_config_keys'.

    _default_spb_config = {
        "SERVER_ADDR": '10.253.0.51',
        "SERVER_PORT": 45012,
        "SOURCE_NAME": 'SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED',
        "SOURCE_TYPE": 1,
        "GEOMETRY_FILE": '/home/spbonc/xfel_geom_AgBehenate_20181012.geom',
        "QUAD_POSITIONS": ((0, 0),
                           (0, 0),
                           (0, 0),
                           (0, 0)),
        "INTEGRATION_METHODS": ['BBox', 'numpy', 'cython', 'splitpixel', 'lut',
                                'csr', 'nosplit_csr', 'lut_ocl', 'csr_ocl'],
        "INTEGRATION_RANGE": (1e-3, 0.1),
        "INTEGRATION_POINTS": 512,
        "PHOTON_ENERGY": 9.3,
        "DISTANCE": 5.5,
        "CENTER_Y": 490,
        "CENTER_X": 590,
        "PIXEL_SIZE": 0.2e-3,
        "COLOR_MAP": 'flame',
        "MASK_RANGE": (0, 2500)
    }

    _default_fxe_config = {
        "SERVER_ADDR": "10.253.0.53",
        "SERVER_PORT": 4501,
        "SOURCE_NAME": "FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED",
        "SOURCE_TYPE": 1,
        "GEOMETRY_FILE": os.path.join(os.path.expanduser("~"), "lpd_mar_18.h5"),
        "QUAD_POSITIONS": ((-13.0, -299.0),
                           (11.0, -8.0),
                           (-254.0, 16.0),
                           (-278.0, -275.0)),
        "INTEGRATION_METHODS": ['BBox', 'numpy', 'cython', 'splitpixel', 'lut',
                                'csr', 'nosplit_csr', 'lut_ocl', 'csr_ocl'],
        "INTEGRATION_RANGE": (0.2, 5),
        "INTEGRATION_POINTS": 512,
        "PHOTON_ENERGY": 9.3,
        "DISTANCE": 0.2,
        "CENTER_Y": 620,
        "CENTER_X": 580,
        "PIXEL_SIZE": 0.5e-3,
        "COLOR_MAP": 'thermal',
        "MASK_RANGE": (0, 2500)
    }

    _default_jfrau_config = {
        "SERVER_ADDR": "",
        "SERVER_PORT": 2583,
        "SOURCE_NAME": "",
        "SOURCE_TYPE": 1,
        "GEOMETRY_FILE": "",
        "QUAD_POSITIONS": ((0, 0),
                           (0, 0),
                           (0, 0),
                           (0, 0)),
        "INTEGRATION_METHODS": ['BBox', 'numpy', 'cython', 'splitpixel', 'lut',
                                'csr', 'nosplit_csr', 'lut_ocl', 'csr_ocl'],
        "INTEGRATION_RANGE": (0.2, 5),
        "INTEGRATION_POINTS": 512,
        "PHOTON_ENERGY": 9.3,
        "DISTANCE": 0.2,
        "CENTER_Y": 620,
        "CENTER_X": 580,
        "PIXEL_SIZE": 0.5e-3,
        "COLOR_MAP": 'thermal',
        "MASK_RANGE": (0, 2500)
    }

    _default_topic_configs = {
        "SPB": _default_spb_config,
        "FXE": _default_fxe_config,
        "JungFrau": _default_jfrau_config
    }

    _filename = os.path.join(ROOT_PATH, "settings.ini")

    def __init__(self):
        super().__init__(self._default_sys_config)

        self.ensure_file()

    def ensure_file(self):
        """Generate the config file if it does not exist."""
        if not os.path.isfile(self._filename):
            cfg = UpperCaseConfigParser()
            for topic in self._default_topic_configs.keys():
                cfg[topic] = {k: "" for k
                              in self._default_topic_configs[topic].keys()}
            with open(self._filename, 'w') as fp:
                cfg.write(fp)

    def load(self, topic):
        """Update the global config.

        The default config will be overwritten by the valid config in
        the config file.

        :param str topic: detector topic, allowed options "SPB", "FXE".
        """
        self.__setitem__("TOPIC", topic)
        self.update(self._default_topic_configs[topic])
        self.from_file(topic)

    def from_file(self, topic):
        """Update the config dictionary from the config file.

        The parameters in the config file are grouped by topics, e.g.

        [SPB]
        SERVER_ADDR = localhost
        SERVER_PORT = 12345
        SOURCE_NAME = /hdf5/data/file/folder/
        SOURCE_TYPE = 0

        [FXE]
        SERVER_ADDR = 10.253.0.53
        SERVER_PORT = 4501
        SOURCE_NAME: FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED
        SOURCE_TYPE: 1

        where the "SPB" topic defines a local file server while the
        "FXE" topic defines an online server.

        Invalid keys or keys with empty entries will be ignored.
        """
        cfg = UpperCaseConfigParser()
        cfg.read(self._filename)

        if topic in cfg:
            invalid_keys = []
            for key in cfg[topic]:
                if key not in self._allowed_topic_config_keys:
                    invalid_keys.append(key)
                else:
                    if cfg[topic][key]:
                        self.__setitem__(key, cfg[topic][key])

            if invalid_keys:
                msg = "The following invalid keys were found in '{}':\n".\
                    format(self._filename)
                msg += ", ".join(invalid_keys)
                print(msg)


config = Config()  # global configuration
