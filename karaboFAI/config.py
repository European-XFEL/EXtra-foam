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

    The default detector config, e.g. _default_agipd_config, should be
    the config used in the corresponding experimental hutch on the
    online cluster.

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
    # TIMER_INTERVAL int: QTimer interval in milliseconds (sys)
    # MAX_QUEUE_SIZE int: maximum length of data acquisition and processing
    #                     queues in data pipeline (sys)
    # TIMEOUT int: block time (s) in Queue.get() and Queue.put() methods (sys)
    #
    # networking
    # ----------
    # SERVER_ADDR str: TCP address of the ZMQ bridge
    # SERVER_PORT int: TCP port of the ZMQ bridge
    # SOURCE_NAME str: PipeToZeroMQ device ID
    # SOURCE_TYPE int: see data_processing.data_model.DataSource
    # DATA_FOLDER str: data folder if streamed from files
    # PULSE_RESOLVED bool: whether the data is pulse resolved (readonly)
    #
    # azimuthal integration
    # ---------------------
    # EXPECTED_SHAPE tuple: shape (modules, y, x) of the detector image data
    #                       (readonly)
    # REQUIRE_GEOMETRY tuple: whether geometry is required to assemble the
    #                         detector (readonly)
    # GEOMETRY_FILE str: path of the geometry file of the detector
    # QUAD_POSITIONS tuple: quadrant coordinates for assembling detector
    #                       modules, ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
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

    # system config
    _default_sys_config = {
        "DETECTOR": '',  # detector name, leave it empty
        "TIMER_INTERVAL": 20,
        "MAX_QUEUE_SIZE": 2,
        "TIMEOUT": 0.1,
        "ROI_COLORS": ('purple', 'green')
    }

    _detector_readonly_config_keys = (
        "PULSE_RESOLVED",
        "REQUIRE_GEOMETRY",
        "EXPECTED_SHAPE",
        "SOURCE_NAME",
    )

    _detector_reconfigurable_keys = (
        "SERVER_ADDR",
        "SERVER_PORT",
        "SOURCE_TYPE",
        "DATA_FOLDER",
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

    # In order to pass the test, the default detector config must include
    # all the keys in '_allowed_detector_config_keys'.

    # the read-only config keys should come first, e.g. PULSE_RESOLVED,
    # REQUIRED_GEOMETRY, EXPECTED_SHAPE, SOURCE_NAME
    _default_agipd_config = {
        "PULSE_RESOLVED": True,
        "REQUIRE_GEOMETRY": True,
        "EXPECTED_SHAPE": (16, 512, 128),
        "SOURCE_NAME": ('SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED',),
        "SERVER_ADDR": '10.253.0.51',
        "SERVER_PORT": 45012,
        "SOURCE_TYPE": 1,
        "DATA_FOLDER": "",
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

    _default_lpd_config = {
        "PULSE_RESOLVED": True,
        "REQUIRE_GEOMETRY": True,
        "EXPECTED_SHAPE": (16, 256, 256),
        "SOURCE_NAME": ("FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED",),
        "SERVER_ADDR": "10.253.0.53",
        "SERVER_PORT": 4501,
        "SOURCE_TYPE": 1,
        "DATA_FOLDER": "",
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
        "PULSE_RESOLVED": False,
        "REQUIRE_GEOMETRY": False,
        "EXPECTED_SHAPE": (1, 512, 1024),
        "SOURCE_NAME": ("FXE_XAD_JF1M1/DET/RECEIVER:daqOutput",),
        "SERVER_ADDR": "10.253.0.53",
        "SERVER_PORT": 4501,
        "SOURCE_TYPE": 1,
        "DATA_FOLDER": "",
        "GEOMETRY_FILE": "",
        "QUAD_POSITIONS": (),
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

    _default_fastccd_config = {
        "PULSE_RESOLVED": False,
        "REQUIRE_GEOMETRY": False,
        "EXPECTED_SHAPE": (1934, 960),
        "SOURCE_NAME": ("SCS_CDIDET_FCCD2M/DAQ/FCCD:daqOutput",),
        "SERVER_ADDR": "",
        "SERVER_PORT": 4501,
        "SOURCE_TYPE": 1,
        "DATA_FOLDER": "",
        "GEOMETRY_FILE": "",
        "QUAD_POSITIONS": (),
        "INTEGRATION_METHODS": ['BBox', 'numpy', 'cython', 'splitpixel', 'lut',
                                'csr', 'nosplit_csr', 'lut_ocl', 'csr_ocl'],
        "INTEGRATION_RANGE": (1e-3, 0.02),
        "INTEGRATION_POINTS": 512,
        "PHOTON_ENERGY": 1,
        "DISTANCE": 5.0,
        "CENTER_Y": 1000,
        "CENTER_X": 480,
        "PIXEL_SIZE": 0.03e-3,
        "COLOR_MAP": 'thermal',
        "MASK_RANGE": (0, 2500)
    }

    _default_detector_configs = {
        "AGIPD": _default_agipd_config,
        "LPD": _default_lpd_config,
        "JungFrau": _default_jfrau_config,
        "FastCCD": _default_fastccd_config
    }

    _filename = os.path.join(ROOT_PATH, "settings.ini")

    def __init__(self):
        super().__init__(self._default_sys_config)

        self.ensure_file()

    def ensure_file(self):
        """Generate the config file if it does not exist."""
        if not os.path.isfile(self._filename):
            cfg = UpperCaseConfigParser()
            for detector in self._default_detector_configs.keys():
                # only write the reconfigurable keys to the file
                cfg[detector] = \
                    {k: "" for k in
                     self._detector_reconfigurable_keys}
            with open(self._filename, 'w') as fp:
                cfg.write(fp)

    def load(self, detector):
        """Update the global config.

        The default config will be overwritten by the valid config in
        the config file.

        :param str detector: detector detector, allowed options "LPD", "AGIPD",
            "JungFrau", "FastCCD".
        """
        self.__setitem__("DETECTOR", detector)
        self.update(self._default_detector_configs[detector])
        self.from_file(detector)

    def from_file(self, detector):
        """Update the config dictionary from the config file.

        The parameters in the config file are grouped by detectors, e.g.

        [AGIPD]
        SERVER_ADDR = localhost
        SERVER_PORT = 12345
        SOURCE_NAME = /hdf5/data/file/folder/
        SOURCE_TYPE = 0

        [LPD]
        SERVER_ADDR = 10.253.0.53
        SERVER_PORT = 4501
        SOURCE_NAME: FXE_DET_LPD1M-1/CAL/APPEND_CORRECTED
        SOURCE_TYPE: 1

        where the "AGIPD" detector defines a local file server while the
        "LPD" detector defines an online server.

        Invalid keys or keys with empty entries will be ignored.
        """
        cfg = UpperCaseConfigParser()
        cfg.read(self._filename)

        if detector in cfg:
            invalid_keys = []
            for key in cfg[detector]:
                if key in self._default_sys_config:
                    raise KeyError("Found system config key in the file: '{}'".
                                   format(key))
                elif key in self._detector_readonly_config_keys:
                    raise KeyError("Found read-only key in the file: '{}'".
                                   format(key))
                elif key not in self._detector_reconfigurable_keys:
                    invalid_keys.append(key)
                else:
                    if cfg[detector][key]:
                        self.__setitem__(key, cfg[detector][key])

            if invalid_keys:
                msg = "The following unknown keys were found in '{}':\n".\
                    format(self._filename)
                msg += ", ".join(invalid_keys)
                print(msg)


config = Config()  # global configuration
