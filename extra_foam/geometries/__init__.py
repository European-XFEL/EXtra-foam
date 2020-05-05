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

from extra_geom import AGIPD_1MGeometry as _geom_AGIPD_1MGeometry
from extra_geom import LPD_1MGeometry as _geom_LPD_1MGeometry
from extra_geom import DSSC_1MGeometry as _geom_DSSC_1MGeometry

from ..algorithms.geometry_1m import AGIPD_1MGeometry as _AGIPD_1MGeometry
from ..algorithms.geometry_1m import LPD_1MGeometry as _LPD_1MGeometry
from ..algorithms.geometry_1m import DSSC_1MGeometry as _DSSC_1MGeometry
from ..config import config, GeomAssembler


_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']


class _1MGeometryPyMixin:
    def output_array_for_position_fast(self, extra_shape=(), dtype=_IMAGE_DTYPE):
        """Make an array with the shape of assembled data filled with nan.

        Match the EXtra-geom signature.
        """
        shape = extra_shape + tuple(self.assembledShape())
        if dtype == np.bool:
            return np.full(shape, 0, dtype=dtype)
        return np.full(shape, np.nan, dtype=dtype)

    def position_all_modules(self, modules, out, *, ignore_tile_edge=False):
        """Assemble data in modules according to where the pixels are.

        Match the EXtra-geom signature.

        :param numpy.ndarray/list modules: data in modules.
            Shape = (memory cells, modules, y x) / (modules, y, x)
        :param numpy.ndarray out: assembled data.
            Shape = (memory cells, y, x) / (y, x)
        :param ignore_tile_edge: True for ignoring the pixels at the edges
            of tiles. If 'out' is pre-filled with nan, it it equivalent to
            masking the tile edges. This is an extra feature which does not
            exist in EXtra-geom.
        """
        if isinstance(modules, np.ndarray):
            self.positionAllModules(modules, out, ignore_tile_edge)
        else:  # extra_data.StackView
            ml = []
            for i in range(self.n_modules):
                ml.append(modules[:, i, ...])
            self.positionAllModules(ml, out, ignore_tile_edge)

    def output_array_for_dismantle_fast(self, extra_shape=(), dtype=_IMAGE_DTYPE):
        """Make an array with the shape of data in modules filled with nan."""
        shape = extra_shape + (self.n_modules, *self.module_shape)
        if dtype == np.bool:
            return np.full(shape, 0, dtype=dtype)
        return np.full(shape, np.nan, dtype=dtype)

    def dismantle_all_modules(self, assembled, out):
        """Dismantle assembled data into data in modules.

        :param numpy.ndarray out: assembled data.
            Shape = (memory cells, y, x) / (y, x)
        :param numpy.ndarray out: data in modules.
            Shape = (memory cells, modules, y x) / (modules, y, x)
        """
        self.dismantleAllModules(assembled, out)


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


class AGIPD_1MGeometryFast(_AGIPD_1MGeometry, _1MGeometryPyMixin):
    """AGIPD_1MGeometryFast.

    Extend the functionality of AGIPD_1MGeometry implementation in C++.
    """
    @classmethod
    def from_crystfel_geom(cls, filename):
        from cfelpyutils.crystfel_utils import load_crystfel_geometry
        from extra_geom.detectors import GeometryFragment

        geom_dict = load_crystfel_geometry(filename)
        modules = []
        for p in range(cls.n_modules):
            tiles = []
            modules.append(tiles)
            for a in range(cls.n_tiles_per_module):
                d = geom_dict['panels']['p{}a{}'.format(p, a)]
                tiles.append(GeometryFragment.from_panel_dict(d).corner_pos)

        return cls(modules)

# patches for geometry classes from EXtra-geom

class AGIPD_1MGeometry(_geom_AGIPD_1MGeometry):
    def position_all_modules(self, modules, out, *args, **kwargs):
        super().position_all_modules(modules, out)


class LPD_1MGeometry(_geom_LPD_1MGeometry):
    def position_all_modules(self, modules, out, *args, **kwargs):
        super().position_all_modules(modules, out)


class DSSC_1MGeometry(_geom_DSSC_1MGeometry):
    def position_all_modules(self, modules, out, *args, **kwargs):
        super().position_all_modules(modules, out)


def load_geometry(detector, filepath, *,
                  assembler=GeomAssembler.OWN,
                  quad_positions=None,
                  stack_only=False):
    if detector == 'AGIPD':
        if assembler == GeomAssembler.OWN:
            if stack_only:
                return AGIPD_1MGeometryFast()

            return AGIPD_1MGeometryFast.from_crystfel_geom(filepath)

        else:
            return AGIPD_1MGeometry.from_crystfel_geom(filepath)

    if detector == 'LPD':
        if assembler == GeomAssembler.OWN:

            if stack_only:
                return LPD_1MGeometryFast()
            return LPD_1MGeometryFast.from_h5_file_and_quad_positions(
                filepath, quad_positions)

        else:
            return LPD_1MGeometry.from_h5_file_and_quad_positions(
                filepath, quad_positions)

    if detector == 'DSSC':
        if assembler == GeomAssembler.OWN:
            if stack_only:
                return DSSC_1MGeometryFast()

            return DSSC_1MGeometryFast.from_h5_file_and_quad_positions(
                filepath, quad_positions)
        else:
            return DSSC_1MGeometry.from_h5_file_and_quad_positions(
                filepath, quad_positions)

    raise ValueError(f"Unknown detector {detector}!")
