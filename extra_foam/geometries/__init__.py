"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from itertools import product

import numpy as np
import h5py

from ..algorithms.geometry import LPD_1MGeometry as _LPD_1MGeometry
from ..algorithms.geometry import DSSC_1MGeometry as _DSSC_1MGeometry


class _1MGeometryPyMixin:
    def output_array_for_position_fast(self, extra_shape, dtype):
        """Match the EXtra-geom signature."""
        shape = extra_shape + tuple(self.assembledShape())
        return np.full(shape, np.nan, dtype=dtype)

    def position_all_modules(self, modules, out):
        """Match the EXtra-geom signature."""
        if isinstance(modules, np.ndarray):
            self.positionAllModules(modules, out)
        else:  # extra_data.StackView
            ml = []
            for i in range(self.n_modules):
                ml.append(modules[:, i, ...])
            self.positionAllModules(ml, out)


class DSSC_1MGeometryFast(_DSSC_1MGeometry, _1MGeometryPyMixin):
    """DSSC_1MGeometryFast.

    Extend the functionality of DSSC_1MGeometry implementation in C++.
    """
    @classmethod
    def from_h5_file_and_quad_positions(cls, filepath, positions):
        modules = []
        with h5py.File(filepath, 'r') as f:
            for Q, M in product(range(1, cls.n_quads + 1),
                                range(1, cls.n_modules_per_quad + 1)):
                quad_pos = np.array(positions[Q - 1])
                mod_grp = f['Q{}/M{}'.format(Q, M)]
                mod_offset = mod_grp['Position'][:2]

                # Which way round is this quadrant
                x_orient = cls.quad_orientations[Q - 1][0]
                y_orient = cls.quad_orientations[Q - 1][1]

                tiles = []
                for T in range(1, cls.n_tiles_per_module+1):
                    first_pixel_pos = np.zeros(3)
                    tile_offset = mod_grp['T{:02}/Position'.format(T)][:2]
                    # mm -> m
                    first_pixel_pos[:2] = 0.001 * (quad_pos + mod_offset + tile_offset)

                    # Corner position is measured at low-x, low-y corner (bottom
                    # right as plotted). We want the position of the corner
                    # with the first pixel, which is either high-x low-y or
                    # low-x high-y.
                    if x_orient == 1:
                        first_pixel_pos[1] += cls.pixelSize()[1] * cls.tile_shape[0]
                    if y_orient == 1:
                        first_pixel_pos[0] += cls.pixelSize()[0] * cls.tile_shape[1]

                    tiles.append(list(first_pixel_pos))
                modules.append(tiles)

        return cls(modules)


class LPD_1MGeometryFast(_LPD_1MGeometry, _1MGeometryPyMixin):
    """LPD_1MGeometryFast.

    Extend the functionality of LPD_1MGeometry implementation in C++.
    """
    @classmethod
    def from_h5_file_and_quad_positions(cls, filepath, positions):
        modules = []
        with h5py.File(filepath, 'r') as f:
            for Q, M in product(range(1, cls.n_quads + 1),
                                range(1, cls.n_modules_per_quad + 1)):
                quad_pos = np.array(positions[Q - 1])
                mod_grp = f['Q{}/M{}'.format(Q, M)]
                mod_offset = mod_grp['Position'][:2]

                tiles = []
                for T in range(1, cls.n_tiles_per_module+1):
                    first_pixel_pos = np.zeros(3)
                    tile_offset = mod_grp['T{:02}/Position'.format(T)][:2]
                    # mm -> m
                    first_pixel_pos[:2] = 0.001 * (quad_pos + mod_offset + tile_offset)

                    # LPD geometry is measured to the last pixel of each tile.
                    # Subtract tile dimensions for the position of 1st pixel.
                    first_pixel_pos[0] -= cls.pixelSize()[0] * cls.tile_shape[1]
                    first_pixel_pos[1] -= cls.pixelSize()[1] * cls.tile_shape[0]

                    tiles.append(list(first_pixel_pos))
                modules.append(tiles)

        return cls(modules)
