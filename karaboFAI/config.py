"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold configuration related objects.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import sys
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

    The local 'settings.ini' file should contain the config used in
    users' local PCs, typically for offline analysis and tests.
    """
    # miscellaneous
    # -------------
    # TITLE str: title of the GUI
    # COLOR_MAP str: color map in contour plots, valid options are: thermal,
    #                flame, yellowy, bipolar, spectrum, cyclic, greyclip, grey
    # MAX_LOGGING int: maximum number of lines in the logging window of GUI
    #
    # networking
    # ----------
    # SERVER_ADDR str: TCP address of the ZMQ bridge
    # SERVER_PORT int: TCP port of the ZMQ bridge
    # SOURCE_NAME str: PipeToZeroMQ device ID
    #
    # azimuthal integration
    # ---------------------
    # QUAD_POSITIONS tuple: quadrant coordinates for assembling detector
    #                       modules, ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
    # GEOMETRY_FILE str: path of the geometry file of the detector
    # INTEGRATION_METHODS list: azimuthal integration methods supported
    #                           by pyFAI
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
    #
    # data pipeline setup
    # -------------------
    # MASK_RANGE tuple: pixels with values outside the (lower, upper) range
    #                   will be masked
    # MAX_QUEUE_SIZE int: maximum length of data acuisition and processing
    #                     queues in data pipeline
    # TIMEOUT int: block time (s) in Queue.get() and Queue.put() methods
    #
    _default_config = {
        "TITLE": 'Karabo Azimuthal Integration',
        "COLOR_MAP": 'flame',
        "MAX_LOGGING": 1000,
        "SERVER_ADDR": '',
        "SERVER_PORT": '',
        "SOURCE_NAME": '',
        "FILE_SERVER_FOLDER": '',
        "GEOMETRY_FILE": '',
        "QUAD_POSITIONS": '',
        "INTEGRATION_METHODS": ['BBox', 'numpy', 'cython', 'splitpixel', 'lut',
                                'csr', 'nosplit_csr', 'lut_ocl', 'csr_ocl'],
        "INTEGRATION_RANGE": '',
        "INTEGRATION_POINTS": '',
        "PHOTON_ENERGY": '',
        "DISTANCE": '',
        "CENTER_Y": '',
        "CENTER_X": '',
        "PIXEL_SIZE": '',
        "MASK_RANGE": (0, 2500),
        "MAX_QUEUE_SIZE": 2,
        "TIMEOUT": 5,
    }

    _default_spb_config = {
        "TITLE": "SPB Azimuthal Integration",
        "SERVER_ADDR": '',
        "SERVER_PORT": '',
        "SOURCE_NAME": "FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED",
        "GEOMETRY_FILE": '',
        "QUAD_POSITIONS": '',
        "INTEGRATION_RANGE": (0.2, 5),
        "INTEGRATION_POINTS": 512,
        "PHOTON_ENERGY": 9.3,
        "DISTANCE": 0.2,
        "CENTER_Y": 620,
        "CENTER_X": 580,
        "PIXEL_SIZE": 0.5e-3,
    }

    _default_fxe_config = {
        "TITLE": "FXE Azimuthal Integration",
        "SERVER_ADDR": "10.253.0.53",
        "SERVER_PORT": 4501,
        "SOURCE_NAME": "FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED",
        "GEOMETRY_FILE": '',
        "QUAD_POSITIONS": ((-13.0, -299.0),
                           (11.0, -8.0),
                           (-254.0, 16.0),
                           (-278.0, -275.0)),
        "INTEGRATION_RANGE": (0.2, 5),
        "INTEGRATION_POINTS": 512,
        "PHOTON_ENERGY": 9.3,
        "DISTANCE": 0.2,
        "CENTER_Y": 620,
        "CENTER_X": 580,
        "PIXEL_SIZE": 0.5e-3,
    }

    _default_topic_configs = {
        "SPB": _default_spb_config,
        "FXE": _default_fxe_config
    }

    def __init__(self):
        super().__init__(self._default_config)

        self._filename = os.path.join(ROOT_PATH, "settings.ini")
        self.ensure_file()

    def ensure_file(self):
        """Generate the 'settings.ini' file if it does not exist."""
        if not os.path.isfile(self._filename):
            cfg = UpperCaseConfigParser()
            cfg["DEFAULT"] = {k: "" for k in self._default_config.keys()}
            with open(self._filename, 'w') as fp:
                cfg.write(fp)

    def load(self, topic):
        """Update the global config.

        non-empty file config > topic config > default config

        :param str topic: detector topic, allowed options "SPB", "FXE".
        """
        cfg = self._default_topic_configs[topic].copy()
        for key in self._default_topic_configs[topic]:
            if key not in self._default_config:
                sys.stderr.write("'{}' in default topic {} config is not found "
                                 "in default config!\n".format(key, topic))
                del cfg[key]

        self.update(cfg)
        self.from_file()

    def from_file(self):
        """Update the config dictionary from the settings.ini file.

        keys with empty entries or not found in '_default_config' will be
        ignored!
        """
        cfg = UpperCaseConfigParser()
        cfg.read(self._filename)

        for key in cfg["DEFAULT"]:
            if key in self._default_config and cfg["DEFAULT"][key]:
                self.__setitem__(key, cfg["DEFAULT"][key])


config = Config()  # global configuration
