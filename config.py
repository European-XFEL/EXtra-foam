FROM_ASSEMBLED_DATA = False

# distance sample - detector plan (orthogonal distance, not along the
# beam), in meter
DIST = 0.2
# coordinate of the point of normal incidence along the detector's first
# dimension, in meter
CENTER_Y = 620
# coordinate of the point of normal incidence along the detector's second
# dimension, in meter
CENTER_X = 580
PIXEL_SIZE = 0.5e-3  # in meter

ENERGY = 9.30  # in, keV
LAMBDA_R = 12.3984 / ENERGY * 1e-10  # in m

PULSES_PER_TRAIN = 16

INTEGRATION_METHOD = "BBox"
RADIAL_RANGE = (0.2, 5)  # the lower and upper range of the radial unit
N_POINTS = 512  # number of points in the output pattern
MASK_RANGE = (0, 1e4)  # image pixels beyond this range will be masked

ON_PULSES = [0, 5, 10]
VIEW_PULSE = 0

# The following is valid-ish for the 20180318 geometry
QUAD_POSITIONS = [(-11.4, -229), (11.5, -8), (-254.5, 16), (-278.5, -275)]
GEOMETRY_FILE = "data/lpd_mar_18.h5"

# *********************************************************************
WINDOW_HEIGHT = 900
WINDOW_WIDTH = 600

MAX_LOGGING = 1000
LOGGER_FONT_SIZE = 10

LINE_PLOT_WIDTH = WINDOW_WIDTH - 20
LINE_PLOT_HEIGHT = 200
