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
from extra_geom import Epix100Geometry as _geom_Epix100Geometry

from ..algorithms.geometry import JungFrauGeometry
from ..algorithms.geometry import EPix100Geometry as _EPix100Geometry
from ..algorithms.geometry_1m import AGIPD_1MGeometry as _AGIPD_1MGeometry
from ..algorithms.geometry_1m import LPD_1MGeometry as _LPD_1MGeometry
from ..algorithms.geometry_1m import DSSC_1MGeometry as _DSSC_1MGeometry
from ..config import config, GeomAssembler


_IMAGE_DTYPE = config['SOURCE_PROC_IMAGE_DTYPE']


def module_indices(n_modules, *, detector=None, topic=None):
    """Return the indices of a given number of modules.

    :param int n_modules: number of modules
    :param str detector: detector name
    :param str topic: topic
    """
    if detector == "JungFrau":
        if n_modules == 6:
            return [1, 2, 3, 6, 7, 8]
        return [*range(1, n_modules + 1)]

    if detector == "ePix100":
        return [*range(1, n_modules + 1)]

    return [*range(n_modules)]


def module_grid_shape(n_modules, *, detector=None, topic=None):
    """Return grid shape (n_rows, n_columns) of a given number of modules.

    :param int n_modules: number of modules
    :param str detector: detector name
    :param str topic: topic
    """
    if n_modules == 8:
        return 4, 2
    if n_modules == 6:
        return 3, 2
    if n_modules == 4:
        return 2, 2
    if n_modules == 2:
        return 2, 1
    if n_modules == 1 and detector == "ePix100":
        return 1, 1

    raise NotImplementedError(
        f"Grid layout with {n_modules} modules is not supported!")


class _1MGeometryPyMixin:
    def output_array_for_position_fast(self, extra_shape=(), dtype=_IMAGE_DTYPE):
        """Make an array with the shape of assembled data filled with nan.

        Match the EXtra-geom signature.
        """
        shape = extra_shape + tuple(self.assembledShape())
        if dtype == bool:
            return np.full(shape, 0, dtype=dtype)
        return np.full(shape, np.nan, dtype=dtype)

    def position_all_modules(self, modules, out, *,
                             ignore_tile_edge=False, ignore_asic_edge=False):
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
        :param ignore_asic_edge: placeholder. Not used.
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
        if dtype == bool:
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
        from cfelpyutils.geometry import load_crystfel_geometry
        from extra_geom.detectors import GeometryFragment

        geom_dict = load_crystfel_geometry(filename).detector
        modules = []
        for i_p in range(cls.n_modules):
            tiles = []
            modules.append(tiles)
            for i_a in range(cls.n_tiles_per_module):
                d = geom_dict['panels'][f'p{i_p}a{i_a}']
                tiles.append(GeometryFragment.from_panel_dict(d).corner_pos)

        return cls(modules)

# Patch geometry classes from EXtra-geom since EXtra-foam passes
# extra arguments.

class AGIPD_1MGeometry(_geom_AGIPD_1MGeometry):
    def position_all_modules(self, modules, out, *args, **kwargs):
        super().position_all_modules(modules, out)


class LPD_1MGeometry(_geom_LPD_1MGeometry):
    def position_all_modules(self, modules, out, *args, **kwargs):
        super().position_all_modules(modules, out)


class DSSC_1MGeometry(_geom_DSSC_1MGeometry):
    def position_all_modules(self, modules, out, *args, **kwargs):
        super().position_all_modules(modules, out)


class _GeometryPyMixin:
    def output_array_for_position_fast(self, extra_shape=(), dtype=_IMAGE_DTYPE):
        """Make an array with the shape of assembled data filled with nan.

        Match the EXtra-geom signature.
        """
        shape = extra_shape + tuple(self.assembledShape())
        if dtype == bool:
            return np.full(shape, 0, dtype=dtype)
        return np.full(shape, np.nan, dtype=dtype)

    def position_all_modules(self, modules, out, *,
                             ignore_tile_edge=False, ignore_asic_edge=False):
        """Assemble data in modules according to where the pixels are.

        Match the EXtra-geom signature.

        :param numpy.ndarray/list modules: data in modules.
            Shape = (memory cells, modules, y x) / (modules, y, x)
        :param numpy.ndarray out: assembled data.
            Shape = (memory cells, y, x) / (y, x)
        :param ignore_tile_edge: placeholder. Not used.
        :param ignore_asic_edge: True for ignoring the pixels at the edges
            of asics. If 'out' is pre-filled with nan, it it equivalent to
            masking the asic edges.
        """
        if isinstance(modules, np.ndarray):
            self.positionAllModules(modules, out, ignore_asic_edge)
        else:  # extra_data.StackView
            ml = []
            for i in range(self.nModules()):
                ml.append(modules[..., i, :, :])
            self.positionAllModules(ml, out, ignore_asic_edge)

    def output_array_for_dismantle_fast(self, extra_shape=(), dtype=_IMAGE_DTYPE):
        """Make an array with the shape of data in modules filled with nan."""
        shape = extra_shape + (self.nModules(), *self.module_shape)
        if dtype == bool:
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

    @classmethod
    def mask_module_py(cls, image):
        """Mask the ASIC edges of a single module.

        :param numpy.ndarray image: image data of a single module.
            Shape = (y, x) or (pulses, y, x)
        """
        ah, aw = cls.asic_shape
        ny, nx = cls.asic_grid_shape
        for i in range(ny):
            image[..., i * ah, :] = np.nan
            image[..., (i + 1) * ah - 1, :] = np.nan
        for j in range(nx):
            image[..., :, j * aw] = np.nan
            image[..., :, (j + 1) * aw - 1] = np.nan


class JungFrauGeometryFast(JungFrauGeometry, _GeometryPyMixin):
    """JungFrauGeometryFast.

    Extend the functionality of JungFrauGeometry implementation in C++.
    """
    @classmethod
    def from_crystfel_geom(cls, n_rows, n_columns, filename):
        from cfelpyutils.geometry import load_crystfel_geometry
        from extra_geom.detectors import GeometryFragment

        geom_dict = load_crystfel_geometry(filename).detector
        modules = []
        for i_p in module_indices(n_rows * n_columns, detector="JungFrau"):
            i_a = 1 if i_p > 4 else 8
            d = geom_dict['panels'][f'p{i_p}a{i_a}']
            modules.append(GeometryFragment.from_panel_dict(d).corner_pos)
        return cls(n_rows, n_columns, modules)


class EPix100GeometryFast(_geom_Epix100Geometry, _GeometryPyMixin):
    """EPix100GeometryFast.

    Extend the functionality of EPix100Geometry implementation in C++.
    """
    _asic_mask = _geom_Epix100Geometry.asic_seams()

    @property
    def module_shape(self):
        return (704, 768)

    def nModules(self):
        return 1

    def dismantleAllModules(self, assembled, out):
        half_height = self.module_shape[0] // 2
        half_width = self.module_shape[1] // 2
        v_gap = 24
        h_gap = 5

        # Top left
        out[0, :half_height, :half_width] = assembled[:half_height, :half_width]
        # Bottom left
        out[0, half_height:, :half_width] = assembled[half_height + v_gap:, :half_width]
        # Top right
        out[0, :half_height, half_width:] = assembled[:half_height, half_width + h_gap:]
        # Bottom right
        out[0, half_height:, half_width:] = assembled[half_height + v_gap:, half_width + h_gap:]

    @classmethod
    def mask_module_py(cls, image):
        """Override.

        :param numpy.ndarray image: image data of a single module.
            Shape = (y, x)
        """
        image[0, :] = np.nan
        image[-1, :] = np.nan

    def position_all_modules(self, modules, out, ignore_tile_edge=False, ignore_asic_edge=False):
        modules = super().normalize_data(modules)

        # Raw data can be of type int16, which doesn't support nans, so we need
        # to convert it to a float value first.
        if modules.dtype != np.float32 or modules.dtype != np.float16:
            modules = modules.astype(np.float32)

        if ignore_asic_edge:
            modules[self._asic_mask] = np.nan

        super().position_modules(modules, out=out)


def load_geometry(detector, *,
                  stack_only=False,
                  filepath=None,
                  coordinates=None,
                  n_modules=None,
                  assembler=GeomAssembler.OWN):
    """A geometry factory which generate geometry instance.

    :param str detector: name of the detector.
    :param bool stack_only: True for stacking detector modules without
        geometry file.
    :param str filepath: path of the geometry file. Ignored if stack_only
        is True.
    :param coordinates: quadrant/module coordinates. Ignored if stack_only
        is True of a CFEL geometry file is used.
    :param int n_modules: number of modules.
    :param GeomAssembler assembler: assembler type. Ignored for detectors
        which does not support external assembler.
    """
    if not stack_only and not filepath:
        raise ValueError(f"Geometry file is required for a "
                         f"non-stack-only geometry!")

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
                filepath, coordinates)

        else:
            return LPD_1MGeometry.from_h5_file_and_quad_positions(
                filepath, coordinates)

    if detector == 'DSSC':
        if assembler == GeomAssembler.OWN:
            if stack_only:
                return DSSC_1MGeometryFast()

            return DSSC_1MGeometryFast.from_h5_file_and_quad_positions(
                filepath, coordinates)
        else:
            return DSSC_1MGeometry.from_h5_file_and_quad_positions(
                filepath, coordinates)

    if detector == "JungFrau":
        shape = module_grid_shape(n_modules, detector=detector)
        if stack_only:
            return JungFrauGeometryFast(*shape)

        return JungFrauGeometryFast.from_crystfel_geom(
            *shape, filepath)

    if detector == "ePix100":
        shape = module_grid_shape(n_modules, detector=detector)
        if stack_only:
            return EPix100GeometryFast.from_relative_positions(
                top=[386.5, 364.5, 0.], bottom=[386.5, -12.5, 0.]
            )

        raise NotImplementedError(
            "ePix100 detector does not support loading geometry from file!")

    raise ValueError(f"Unknown detector {detector}!")


def maybe_mask_asic_edges(image, detector):
    """Helper function to mask the edges of ASICs of a single module."""
    if detector == "JungFrau":
        JungFrauGeometryFast.mask_module_py(image)
        return

    if detector == "ePix100":
        EPix100GeometryFast.mask_module_py(image)
        return
