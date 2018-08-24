from enum import IntEnum
import os


class DataSource(IntEnum):
    CALIBRATED_FILE = 0  # calibrated data from files
    CALIBRATED = 1  # calibrated data from Karabo-bridge
    ASSEMBLED = 2  # assembled data from Karabo-bridge
    PROCESSED = 3  # processed data from the Middle-layer device


class Config:
    # distance sample - detector plan (orthogonal distance, not along the
    # beam), in meter
    DIST = 0.2
    # coordinate of the point of normal incidence along the detector's first
    # dimension, in pixels
    CENTER_Y = 620
    # coordinate of the point of normal incidence along the detector's second
    # dimension, in pixels
    CENTER_X = 580
    PIXEL_SIZE = 0.5e-3  # in meter

    PHOTON_ENERGY = 9.30  # in, keV

    PULSES_PER_TRAIN = 16

    # 'full_csr" crashes the app
    INTEGRATION_METHODS = ['BBox', 'numpy', 'cython', 'splitpixel', 'lut',
                           'csr', 'nosplit_csr', 'lut_ocl', 'csr_ocl']

    # the lower and upper range of the radial unit
    INTEGRATION_RANGE = (0.2, 5)
    INTEGRATION_POINTS = 512  # number of points in the output pattern

    MASK_RANGE = (0, 1e4)  # image pixels beyond this range will be masked

    # The following is valid-ish for the 20180318 geometry
    QUAD_POSITIONS = [(-11.4, -229), (11.5, -8), (-254.5, 16), (-278.5, -275)]

    # DEFAULT_SERVER_SRC = DataSource.CALIBRATED
    # DEFAULT_SERVER_ADDR = "10.253.0.53"
    # DEFAULT_SERVER_PORT = "4501"

    DEFAULT_SERVER_SRC = DataSource.CALIBRATED_FILE
    DEFAULT_SERVER_ADDR = "localhost"
    DEFAULT_SERVER_PORT = "12345"

    DEFAULT_GEOMETRY_FILE = os.path.join(os.path.expanduser("~"),
                                         "fxe-data/lpd_mar_18.h5")

    # *********************************************************************
    MAX_QUEUE_SIZE = 10

    # *********************************************************************
    UPDATE_FREQUENCY = 10  # in Hz

    MAIN_WINDOW_HEIGHT = 960
    MAIN_WINDOW_WIDTH = 1320
    MAIN_LINE_PLOT_HEIGHT = 440

    MAX_LOGGING = 1000
    LOGGER_FONT_SIZE = 12

    LINE_PLOT_WIDTH = 600
    LINE_PLOT_HEIGHT = 260
    LINE_PLOT_LEGEND_OFFSET = (-10, -50)

    MA_PLOT_WIDTH = 800
    MA_PLOT_HEIGHT = 900

    # *********************************************************************
    DEFAULT_FILE_SERVER_PORT = "12345"
    DEFAULT_FILE_SERVER_FOLDER = os.path.join(os.path.expanduser("~"),
                                              "fxe-data/r0078")