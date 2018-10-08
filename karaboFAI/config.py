import os

import numpy as np


class Config(dict):
    def __init__(self, default=None):
        super().__init__(default or {})


# TITLE str: title of the GUI
# MAX_QUEUE_SIZE int: maximum length of Queue in data pipeline
# TIMEOUT int: block time (s) in Queue.get() and Queue.put()
#
# SERVER_ADDR str: TCP address of the ZMQ bridge
# SERVER_PORT int: TCP port of the ZMQ bridge
# SOURCE_NAME str: PipeToZeroMQ device ID
#
# GEOMETRY_FILE str: path of the geometry file of the detector
# INTEGRATION_METHODS list: 1D Azimuthal integration methods supported by pyFAI
# INTEGRATION_RANGE tuple: (lower, upper) range of the radial unit of azimuthal
#                          integration
# INTEGRATION_POINTS int: number of points in the output pattern of azimuthal
#                         integration
# PHOTON_ENERGY float: photon energy, in keV
# DISTANCE float: distance from sample - detector plan (orthogonal distance, not
#                 along the beam), in meter
# CENTER_Y int: coordinate of the point of normal incidence along the detector's
#               first dimension, in pixels
# CENTER_X int: coordinate of the point of normal incidence along the detector's
#               second dimension, in pixels
# PIXEL_SIZE float: detector pixel size, in meter
# MASK_RANGE tuple: pixel with values outside the (lower, upper) range will be
#                   masked
# COLOR_MAP str: color map in contour plot
# MAX_LOGGING int: maximum number of lines in the logging window of GUI
config = Config({
    "TITLE": "",
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
    "MAX_QUEUE_SIZE": 2,
    "TIMEOUT": 5,
    "COLOR_MAP": "flame",
    "MAX_LOGGING": 1000,
})  # global configuration
