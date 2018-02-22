import numpy as np

distance = 275 * 1e-3  # Meter: 160 Millimeters
center_x = 546
center_y = 532
pixel_size = 0.5e-3  # Meter: 0.5 Millimeter
energy = 9.33  # keV
wavelength_lambda = 12.3984 / energy * 1e-10  # Meter: 0.132 Nanometer

muSi = 91  # silicon sensor absorption coefficient
tSi = 500 * 1e-4  # sensor thickness
mus = 6.3  # sample absorption coefficien
ts = 100 * 1e-4  # sample thickness

# `hole_size` is the size, in meters, of the gap at the center of the
# LPD's supermodules
hole_size = -26.28e-3  # in Meter: 26.28 milimeters
hole_pixel_size = np.abs(np.int(np.ceil(hole_size / pixel_size)))

qnorm_min = 0.5
qnorm_max = 3
q_offset = 3

# SM is the size in pixels of a supermodule,
# which is a group of four modules
SM = 256

# These are the offset of each module.
# They are referred to by their index.
# module 0, then, looks at index 0 in both
# maps to figure out where to be placed in the
# full image.
dx_map = [0, 0, SM, SM, 0, 0, SM, SM, SM*2, SM*2,
          SM*3, SM*3, SM*2, SM*2, SM*3, SM*3]
dy_map = [0, SM, SM, 0, SM*2, SM*3, SM*3, SM*2,
          SM*2, SM*3, SM*3, SM*2, 0, SM, SM, 0]

