"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Hold configuration related objects.

TODO:: implement key check, define sections in the file, etc.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import configparser

import numpy as np


# root path for storing config and log files
ROOT_PATH = os.path.join(os.path.expanduser("~"), ".karaboFAI")


class UpperCaseConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr.upper()


class Config(dict):
    # [DEFAULT]
    # TITLE str: title of the GUI
    # COLOR_MAP str: color map in contour plot
    # MAX_LOGGING int: maximum number of lines in the logging window of GUI
    #
    # [NETWORKING]
    # SERVER_ADDR str: TCP address of the ZMQ bridge
    # SERVER_PORT int: TCP port of the ZMQ bridge
    # SOURCE_NAME str: PipeToZeroMQ device ID
    #
    # [AZIMUTHAL INTEGRATION]
    # GEOMETRY_FILE str: path of the geometry file of the detector
    # INTEGRATION_METHODS list: 1D Azimuthal integration methods supported
    #                           by pyFAI
    # INTEGRATION_RANGE tuple: (lower, upper) range of the radial unit of
    #                          azimuthal integration
    # INTEGRATION_POINTS int: number of points in the output pattern of
    #                         azimuthal integration
    #
    # [EXPERIMENT]
    # PHOTON_ENERGY float: photon energy, in keV
    # DISTANCE float: distance from sample - detector plan (orthogonal
    #                 distance, not along the beam), in meter
    # CENTER_Y int: coordinate of the point of normal incidence along the
    #               detector's first dimension, in pixels
    # CENTER_X int: coordinate of the point of normal incidence along the
    #               detector's second dimension, in pixels
    # PIXEL_SIZE float: detector pixel size, in meter
    #
    # [DATA PIPELINE]
    # MASK_RANGE tuple: pixel with values outside the (lower, upper) range
    #                   will be masked
    # MAX_QUEUE_SIZE int: maximum length of Queue in data pipeline
    # TIMEOUT int: block time (s) in Queue.get() and Queue.put()
    _default_config = {
        "TITLE": "",
        "MAX_QUEUE_SIZE": 2,
        "TIMEOUT": 5,
        "COLOR_MAP": "flame",
        "MAX_LOGGING": 1000,
        "QUAD_POSITIONS": None,
        "SERVER_ADDR": None,
        "SERVER_PORT": 1234,
        "SOURCE_NAME": None,
        "FILE_SERVER_FOLDER": os.path.expanduser("~"),
        "GEOMETRY_FILE": os.path.expanduser("~"),
        "INTEGRATION_METHODS": ['BBox', 'numpy', 'cython', 'splitpixel', 'lut',
                                'csr', 'nosplit_csr', 'lut_ocl', 'csr_ocl'],
        "INTEGRATION_RANGE": None,
        "INTEGRATION_POINTS": None,
        "PHOTON_ENERGY": 100,
        "DISTANCE": None,
        "CENTER_Y": None,
        "CENTER_X": None,
        "PIXEL_SIZE": None,
        "MASK_RANGE": (-np.inf, np.inf),
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

    def update_global(self, cfg):
        """Update the global config.

        Non-empty config in the file > cfg > default
        """
        self.update(cfg)
        self.update(self.from_file())

    def from_file(self):
        """Update the config dictionary from the settings.ini file."""
        cfg = UpperCaseConfigParser()
        cfg.read(self._filename)

        for key in cfg["DEFAULT"]:
            if cfg["DEFAULT"][key]:
                self.__setitem__(key, cfg["DEFAULT"][key])
        return dict()


config = Config()  # global configuration
